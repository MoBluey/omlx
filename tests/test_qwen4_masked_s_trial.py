"""Opt-in s-class masked SDPA: exact dispatch, output and rollback."""

import importlib

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch

apply_mlx_vlm_qwen4_exp_compat_patch()
qsa = importlib.import_module("mlx_vlm.models.qwen4_exp.qsa_fast")


@pytest.mark.parametrize("value", [None, "", "0", "true", "1"])
def test_process_environment_opt_in_scope(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("OMLX_QWEN4_MASKED_S", raising=False)
    else:
        monkeypatch.setenv("OMLX_QWEN4_MASKED_S", value)
    spec = importlib.util.spec_from_file_location(qsa.__name__, qsa.__file__)
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    assert fresh.MASKED_S_ENABLED is (value == "1")
    monkeypatch.setattr(fresh, "_gpu_class", lambda: "s")
    assert fresh._two_pass_blocks(12, 28287) == (256 if value == "1" else None)
    assert fresh._two_pass_blocks(12, 8192) is None
    assert fresh._two_pass_blocks(4, 28287) is None
    monkeypatch.setattr(fresh, "_gpu_class", lambda: "unknown")
    assert fresh._two_pass_blocks(12, 28287) is None
    monkeypatch.setattr(fresh, "_gpu_class", lambda: "d")
    assert fresh._two_pass_blocks(12, 28287) == 512


def test_partition_rules_and_default_off(monkeypatch):
    monkeypatch.setattr(qsa, "_gpu_class", lambda: "s")
    monkeypatch.setattr(qsa, "MASKED_S_ENABLED", False)
    assert qsa._two_pass_blocks(12, 28287) is None
    monkeypatch.setattr(qsa, "MASKED_S_ENABLED", True)
    for length, expected in [(8192, None), (8193, 256), (32768, 256),
                             (32769, 512), (65536, 512), (65537, 1024)]:
        assert qsa._two_pass_blocks(12, length) == expected
    assert qsa._two_pass_blocks(4, 28287) is None
    monkeypatch.setattr(qsa, "_gpu_class", lambda: "d")
    for length, expected in [(8192, 128), (16383, 128), (16384, 512),
                             (65535, 512), (65536, 1024)]:
        assert qsa._two_pass_blocks(12, length) == expected


@pytest.mark.parametrize("length", [8193, 16383, 16384, 32768, 32769, 65536, 65537, 79093])
@pytest.mark.parametrize("kind", ["qsa", "sparse", "all", "none"])
def test_masked_s_matches_mlx_bits(monkeypatch, length, kind):
    if qsa._gpu_class() != "s":
        pytest.skip("actual s-class Metal required")
    monkeypatch.setattr(qsa, "MASKED_S_ENABLED", True)
    monkeypatch.setattr(qsa, "_DECODE_SDPA_DISABLED", False)
    capacity = length + 257
    keys = mx.random.normal((1, 2, capacity, 256), key=mx.random.key(1)).astype(mx.bfloat16)
    values = mx.random.normal((1, 2, capacity * 2, 256), key=mx.random.key(2)).astype(mx.bfloat16)
    keys = keys[:, :, :length]
    values = values[:, :, 1 : 2 * length + 1 : 2]
    queries = (mx.random.normal((1, 24, 2, 256), key=mx.random.key(3)) * 4).astype(mx.bfloat16)[:, :, :1]
    selected = np.zeros(length, dtype=bool)
    if kind == "qsa":
        rng = np.random.default_rng(length)
        blocks = length // 4
        for block in rng.choice(blocks, 512, replace=False):
            selected[block * 4 : block * 4 + 4] = True
        selected[blocks * 4 :] = True
    elif kind == "sparse":
        selected[:37] = True
    elif kind == "all":
        selected[:] = True
    mask = mx.array(np.repeat(selected, 2))[::2].reshape(1, 1, 1, length)
    before = qsa.MASKED_S_CALLS
    actual = qsa.masked_decode_sdpa(queries, keys, values, mask, 0.0625)
    expected = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=0.0625, mask=mask)
    mx.eval(actual, expected)
    assert actual is not None
    assert qsa.MASKED_S_CALLS == before + 1
    assert not qsa._DECODE_SDPA_DISABLED
    assert mx.array_equal(actual.view(mx.uint16), expected.view(mx.uint16)).item()


def test_masked_s_initial_launch_failure_and_off(monkeypatch, caplog):
    monkeypatch.setattr(qsa, "_gpu_class", lambda: "s")
    monkeypatch.setattr(qsa, "MASKED_S_ENABLED", False)
    monkeypatch.setattr(qsa, "_DECODE_SDPA_DISABLED", False)
    queries = mx.zeros((1, 24, 1, 256), dtype=mx.bfloat16)
    keys = mx.zeros((1, 2, 8193, 256), dtype=mx.bfloat16)
    mask = mx.ones((1, 1, 1, 8193), dtype=mx.bool_)
    before = qsa.MASKED_S_CALLS
    assert qsa.masked_decode_sdpa(queries, keys, keys, mask, 0.0625) is None
    assert not caplog.records
    monkeypatch.setattr(qsa, "MASKED_S_ENABLED", True)

    def fail(**kwargs):
        raise RuntimeError("deliberate launch failure")

    monkeypatch.setattr(qsa, "_DECODE_SDPA_KERNELS", [fail, fail])
    assert qsa.masked_decode_sdpa(queries, keys, keys, mask, 0.0625) is None
    assert qsa._DECODE_SDPA_DISABLED
    assert qsa.MASKED_S_CALLS == before
    assert qsa.masked_decode_sdpa(queries, keys, keys, mask, 0.0625) is None
    assert [record.message for record in caplog.records] == [
        "Qwen M4 masked attention disabled after kernel failure"
    ]


def test_masked_s_logs_first_success_only(monkeypatch, caplog):
    if qsa._gpu_class() != "s":
        pytest.skip("actual s-class Metal required")
    monkeypatch.setattr(qsa, "MASKED_S_ENABLED", True)
    monkeypatch.setattr(qsa, "MASKED_S_CALLS", 0)
    monkeypatch.setattr(qsa, "_DECODE_SDPA_DISABLED", False)
    caplog.set_level("INFO", logger=qsa.__name__)
    queries = mx.zeros((1, 24, 1, 256), dtype=mx.bfloat16)
    keys = mx.zeros((1, 2, 8193, 256), dtype=mx.bfloat16)
    mask = mx.ones((1, 1, 1, 8193), dtype=mx.bool_)
    for _ in range(2):
        actual = qsa.masked_decode_sdpa(queries, keys, keys, mask, 0.0625)
        assert actual is not None
        mx.eval(actual)
    assert qsa.MASKED_S_CALLS == 2
    assert [record.message for record in caplog.records] == [
        "Qwen M4 masked attention active"
    ]


def test_masked_s_verify_rollback_and_continuation(monkeypatch):
    if qsa._gpu_class() != "s":
        pytest.skip("actual s-class Metal required")
    import test_qwen4_verify_attention_rows as fixture

    monkeypatch.setattr(qsa, "_DECODE_SDPA_DISABLED", False)
    attention = fixture._attention(seed=13)
    cache = fixture._prefill(attention, 8192, seed=14, chunk=512)
    baseline_cache, candidate_cache = fixture._copy_cache(cache), fixture._copy_cache(cache)
    x = mx.random.normal((1, 2, 2560), key=mx.random.key(15)).astype(mx.bfloat16)
    monkeypatch.setattr(qsa, "MASKED_S_ENABLED", False)
    baseline = fixture._verify(attention, baseline_cache, x)
    monkeypatch.setattr(qsa, "MASKED_S_ENABLED", True)
    before = qsa.MASKED_S_CALLS
    candidate = fixture._verify(attention, candidate_cache, x)
    assert qsa.MASKED_S_CALLS == before + 2
    assert mx.array_equal(baseline.view(mx.uint16), candidate.view(mx.uint16)).item()
    for left, right in zip(fixture._state(baseline_cache), fixture._state(candidate_cache)):
        assert mx.array_equal(left, right).item()
    baseline_cache.trim(1)
    candidate_cache.trim(1)
    next_input = x[:, :1]
    monkeypatch.setattr(qsa, "MASKED_S_ENABLED", False)
    baseline = fixture._serial(attention, baseline_cache, next_input)
    monkeypatch.setattr(qsa, "MASKED_S_ENABLED", True)
    candidate = fixture._serial(attention, candidate_cache, next_input)
    assert qsa.MASKED_S_CALLS == before + 3
    assert not qsa._DECODE_SDPA_DISABLED
    assert mx.array_equal(baseline.view(mx.uint16), candidate.view(mx.uint16)).item()
    assert baseline_cache.offset == candidate_cache.offset == 8194
    for left, right in zip(fixture._state(baseline_cache), fixture._state(candidate_cache)):
        assert mx.array_equal(left, right).item()
