"""Exact decode K/V copies, including reserved-capacity cache views."""

import importlib

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch

apply_mlx_vlm_qwen4_exp_compat_patch()
qsa = importlib.import_module("mlx_vlm.models.qwen4_exp.qsa_fast")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")

@pytest.fixture(autouse=True)
def _candidate(monkeypatch):
    monkeypatch.setattr(qsa, "_FUSED_DECODE_KV_GATHER", True)
    monkeypatch.setattr(qsa, "_FUSED_DECODE_KV_GATHER_DISABLED", False)
    monkeypatch.setattr(qsa, "_FUSED_DECODE_KV_GATHER_PROVEN", False)
    monkeypatch.setattr(qsa, "_FUSED_DECODE_KV_GATHER_CALLS", 0)


def _raw_bf16(shape, offset=0):
    # Include all 65,536 BF16 encodings: signed zero, infinities and NaN
    # payloads must survive as bits, not merely compare numerically equal.
    count = int(np.prod(shape))
    bits = (mx.arange(count, dtype=mx.uint32) + offset).astype(mx.uint16)
    return bits.view(mx.bfloat16).reshape(shape)


def _assert_bits(actual, expected):
    mx.eval(actual, expected)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert mx.array_equal(actual.view(mx.uint8), expected.view(mx.uint8)).item()

@pytest.mark.parametrize("tokens", [2443, 28287])
@pytest.mark.parametrize("tail", range(4))
@pytest.mark.parametrize("strided", [False, True])
def test_actual_decode_shapes_preserve_every_bf16_bit(tokens, tail, strided):
    width = 2048 + tail
    capacity = ((tokens + 255) // 256) * 256
    if strided:
        keys = _raw_bf16((1, 4, capacity, 512))[:, ::2, :tokens, ::2]
        values = _raw_bf16((1, 2, capacity * 2, 256), 103)[
            :, :, 1 : 2 * tokens + 1 : 2, :
        ]
    else:
        keys = _raw_bf16((1, 2, capacity, 256))[:, :, :tokens]
        values = _raw_bf16((1, 2, capacity, 256), 103)[:, :, :tokens]
    ids = np.linspace(0, tokens - 1, width, dtype=np.int32)
    # Index views can also have a non-unit stride.
    indices = mx.array(np.repeat(ids, 2).reshape(1, -1))[:, ::2]
    expected = (qsa._gather_kv_rows(keys, indices), qsa._gather_kv_rows(values, indices))
    actual = qsa._gather_decode_kv_pair(keys, values, indices)
    assert qsa._FUSED_DECODE_KV_GATHER_PROVEN
    assert qsa._FUSED_DECODE_KV_GATHER_CALLS == 1
    assert not qsa._FUSED_DECODE_KV_GATHER_DISABLED
    for a, b in zip(actual, expected):
        _assert_bits(a, b)


def test_negative_duplicate_unordered_indices_and_request_toggle(monkeypatch):
    keys = _raw_bf16((1, 2, 17, 256))
    values = _raw_bf16((1, 2, 17, 256), 331)
    indices = mx.array([[-17, -1, 7, 0, 7, 16]], dtype=mx.int32)
    monkeypatch.setattr(qsa, "_FUSED_DECODE_KV_GATHER", False)
    baseline = qsa._gather_decode_kv_pair(keys, values, indices)
    monkeypatch.setattr(qsa, "_FUSED_DECODE_KV_GATHER", True)
    candidate = qsa._gather_decode_kv_pair(keys, values, indices)
    monkeypatch.setattr(qsa, "_FUSED_DECODE_KV_GATHER", False)
    restored = qsa._gather_decode_kv_pair(keys, values, indices)
    assert qsa._FUSED_DECODE_KV_GATHER_PROVEN
    for a, b, c in zip(baseline, candidate, restored):
        _assert_bits(a, b)
        _assert_bits(a, c)

@pytest.mark.parametrize("kind", ["empty", "float32", "batch2", "per_query", "int64"])
def test_unsupported_or_empty_shapes_keep_existing_gather(kind):
    shape = (2 if kind == "batch2" else 1, 2, 17, 256)
    keys = _raw_bf16(shape)
    values = _raw_bf16(shape, 19)
    if kind == "float32":
        keys, values = keys.astype(mx.float32), values.astype(mx.float32)
    ids = [] if kind == "empty" else [0, 7, 16]
    indices = mx.array([ids] * shape[0], dtype=mx.int64 if kind == "int64" else mx.int32)
    if kind == "per_query":
        indices = indices[:, None]
    actual = qsa._gather_decode_kv_pair(keys, values, indices)
    expected = (qsa._gather_kv_rows(keys, indices), qsa._gather_kv_rows(values, indices))
    assert not qsa._FUSED_DECODE_KV_GATHER_PROVEN
    assert qsa._FUSED_DECODE_KV_GATHER_CALLS == 0
    assert not qsa._FUSED_DECODE_KV_GATHER_DISABLED
    for a, b in zip(actual, expected):
        _assert_bits(a, b)


def test_kernel_failure_disables_candidate_and_preserves_copy(monkeypatch):
    keys = _raw_bf16((1, 2, 17, 256))
    values = _raw_bf16((1, 2, 17, 256), 7)
    indices = mx.array([[0, 7, 16]], dtype=mx.int32)

    def failed():
        raise RuntimeError("deliberate build failure")

    monkeypatch.setattr(qsa, "_decode_kv_pair_kernel", failed)
    actual = qsa._gather_decode_kv_pair(keys, values, indices)
    expected = (qsa._gather_kv_rows(keys, indices), qsa._gather_kv_rows(values, indices))
    assert qsa._FUSED_DECODE_KV_GATHER_DISABLED
    assert qsa._FUSED_DECODE_KV_GATHER_CALLS == 0
    for a, b in zip(actual, expected):
        _assert_bits(a, b)


def test_attention_verify_rollback_and_continuation_keep_exact_state(monkeypatch):
    # Reuse the existing actual-geometry attention and cache helpers. This
    # exercises the production gather consumer, not just the copy in isolation.
    import test_qwen4_verify_attention_rows as fixture

    attn = fixture._attention(seed=31)
    cache = fixture.language.QSAKVCache()
    mx.random.seed(32)
    for start in range(0, 2060, 512):
        size = min(512, 2060 - start)
        x = mx.random.normal((1, size, 2560)).astype(mx.bfloat16)
        prefix = attn(x, mask="causal", cache=cache,
                      position_ids=mx.arange(start, start + size, dtype=mx.int32)[None])
        mx.eval(prefix)
    control_cache, candidate_cache = fixture._copy_cache(cache), fixture._copy_cache(cache)
    verify_input = mx.random.normal((1, 2, 2560)).astype(mx.bfloat16)

    def run(current_cache, enabled, x, verify):
        monkeypatch.setattr(qsa, "_FUSED_DECODE_KV_GATHER", enabled)
        fixture.qwen35_verify_qmm.set_verify_qmm_armed(verify, row_exact=verify)
        try:
            positions = mx.arange(current_cache.offset, current_cache.offset + x.shape[1],
                                  dtype=mx.int32)[None]
            result = attn(x, mask="causal", cache=current_cache,
                          position_ids=positions, target_verify=verify)
            mx.eval(result)
            return result
        finally:
            fixture.qwen35_verify_qmm.set_verify_qmm_armed(False)

    control = run(control_cache, False, verify_input, True)
    candidate = run(candidate_cache, True, verify_input, True)
    assert qsa._FUSED_DECODE_KV_GATHER_PROVEN
    _assert_bits(control, candidate)
    for a, b in zip(fixture._state(control_cache), fixture._state(candidate_cache)):
        _assert_bits(a, b)
    control_cache.trim(1)
    candidate_cache.trim(1)
    continuation = mx.random.normal((1, 1, 2560)).astype(mx.bfloat16)
    _assert_bits(run(control_cache, False, continuation, False),
                 run(candidate_cache, True, continuation, False))
    assert control_cache.offset == candidate_cache.offset == 2062
    for a, b in zip(fixture._state(control_cache), fixture._state(candidate_cache)):
        _assert_bits(a, b)
