"""Numerical checks for the opt-in native Flash-Next DFlash target adapter."""

from __future__ import annotations

import mlx.core as mx
from mlx.utils import tree_map
import pytest

from test_mlx_vlm_qwen4_exp_compat import _tiny_config
from omlx.patches.mlx_lm_mtp import batch_generator as bg
from mlx_vlm.models.qwen4_exp import language


def _target(monkeypatch):
    monkeypatch.setenv("OMLX_QWEN4_DFLASH_TRIAL", "1")
    config = _tiny_config()
    config.text_config.hc_count = 4
    mx.random.seed(412)
    model = language.LanguageModel(config.text_config, config)
    model.update(tree_map(
        lambda value: value.astype(mx.bfloat16) if mx.issubdtype(value.dtype, mx.floating) else value,
        model.parameters(),
    ))
    model.prepare_dflash_target()
    return model


@pytest.mark.parametrize("deferred", [False, True])
def test_native_dflash_capture_matches_completed_hc_stream_oracle(monkeypatch, deferred):
    model = _target(monkeypatch)
    monkeypatch.setattr(language.hc_fused, "_WRITE_DISABLED", not deferred)
    captured_layers = []
    original = language.Qwen4ExpDecoderLayer.__call__

    def record(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        completed = language._hc_write(result[0], *result[1:]) if isinstance(result, tuple) else result
        captured_layers.append(completed)
        return result

    monkeypatch.setattr(language.Qwen4ExpDecoderLayer, "__call__", record)
    inputs = mx.array([[2, 3, 4]], dtype=mx.int32)
    baseline_cache, captured_cache = model.make_cache(), model.make_cache()
    model.model._omlx_dflash_hc_mean = False
    baseline = model(inputs, cache=baseline_cache)
    captured_layers.clear()
    model.model._omlx_dflash_hc_mean = True
    captured = model(inputs, cache=captured_cache, capture_layer_ids=[0, 1])
    mx.eval(baseline.logits, captured.logits, captured.hidden_states, captured_layers)
    assert mx.array_equal(baseline.logits, captured.logits).item()
    for actual, completed in zip(captured.hidden_states, captured_layers):
        expected = completed.reshape(*completed.shape[:-1], model.args.hc_count, model.args.hidden_size).mean(axis=-2)
        assert mx.array_equal(actual, expected).item()
        assert actual.shape == (1, 3, model.args.hidden_size)

    # One-row continuation must retain ordinary decode arithmetic and cache.
    captured_layers.clear()
    token = mx.array([[5]], dtype=mx.int32)
    model.model._omlx_dflash_hc_mean = False
    baseline = model(token, cache=baseline_cache)
    captured_layers.clear()
    model.model._omlx_dflash_hc_mean = True
    captured = model(token, cache=captured_cache, return_hidden=True, capture_layer_ids=[0, 1])
    mx.eval(baseline.logits, captured.logits, captured.hidden_states)
    assert mx.array_equal(baseline.logits, captured.logits).item()
    assert captured.gdn_states is None
    assert len(captured.hidden_states) == 3
    for actual, completed in zip(captured.hidden_states, captured_layers):
        expected = completed.reshape(*completed.shape[:-1], model.args.hc_count, model.args.hidden_size).mean(axis=-2)
        assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("block,accepted", [(block, accepted) for block in (2, 3, 4) for accepted in range(block)])
def test_native_dflash_verify_and_every_rollback_prefix_match_serial(monkeypatch, block, accepted):
    model = _target(monkeypatch)
    verify_cache, serial_cache = model.make_cache(), model.make_cache()
    prefix = mx.array([[2, 3, 4]], dtype=mx.int32)
    model(prefix, cache=verify_cache)
    model(prefix, cache=serial_cache)
    tokens = mx.array([[5, 6, 7, 8][:block]], dtype=mx.int32)
    logits, _, transaction, captures = bg._call_backbone_captured(
        model, tokens, verify_cache, n_confirmed=1, capture_layer_ids=[0, 1]
    )
    serial_logits = []
    for index in range(accepted + 1):
        serial_logits.append(model(tokens[:, index:index + 1], cache=serial_cache).logits)
    serial_logits = mx.concatenate(serial_logits, axis=1)
    mx.eval(logits, serial_logits, captures)
    assert transaction is not None and transaction.active
    assert len(captures) == 2
    assert all(value.shape == (1, block, model.args.hidden_size) for value in captures)
    assert mx.array_equal(logits[:, :accepted + 1], serial_logits).item()
    model.rollback_speculative_cache(verify_cache, transaction, accepted, block)
    assert not transaction.active
    assert verify_cache[1].offset == serial_cache[1].offset == 4 + accepted
    assert getattr(verify_cache[0], "_qwen4_exp_ple_speculative_state", None) is None
    for actual_cache, expected_cache in zip(verify_cache, serial_cache):
        actual_state, expected_state = actual_cache.state, expected_cache.state
        assert len(actual_state) == len(expected_state)
        for actual, expected in zip(actual_state, expected_state):
            if actual is None or expected is None:
                assert actual is expected
            else:
                mx.eval(actual, expected)
                assert mx.array_equal(actual, expected).item()
    # QSA's index state is separate from its ordinary K/V cache state.
    for name in ("index_keys", "index_position_ids"):
        assert mx.array_equal(getattr(verify_cache[1], name), getattr(serial_cache[1], name)).item()
    next_token = mx.array([[9]], dtype=mx.int32)
    actual = model(next_token, cache=verify_cache).logits
    expected = model(next_token, cache=serial_cache).logits
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected).item()
