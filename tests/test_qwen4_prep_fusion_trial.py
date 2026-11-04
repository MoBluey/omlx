"""Independent q/k preparation, with exact rotary/cache fallback coverage."""

import importlib

import mlx.core as mx
import pytest

from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch

apply_mlx_vlm_qwen4_exp_compat_patch()
fused = importlib.import_module("mlx_vlm.models.qwen4_exp.attn_fused")


def _bits(left, right):
    mx.eval(left, right)
    assert left.shape == right.shape and left.dtype == right.dtype
    assert mx.array_equal(left.view(mx.uint8), right.view(mx.uint8)).item()


@pytest.mark.parametrize('rows', [1, 2, 16, 300])
@pytest.mark.parametrize('rank', [2, 3])
@pytest.mark.parametrize('yarn', [False, True])
@pytest.mark.parametrize('dtype', [mx.bfloat16, mx.float32])
def test_prep_exact_raw_norm_and_rope(monkeypatch, rows, rank, yarn, dtype):
    import test_qwen4_attn_fused as fixture

    monkeypatch.setattr(fused, "PREP_FUSION_ENABLED", True)
    monkeypatch.setattr(fused, "PREP_FUSION_DISABLED", False)
    config = fixture._YARN_CONFIG if yarn else fixture._CONFIG
    attention = fixture._attention(seed=9, config=config)
    q = (mx.random.normal((rows, 24 * 512), key=mx.random.key(17)) * 4).astype(dtype)
    k = (mx.random.normal((rows, 2 * 256), key=mx.random.key(18)) * 4).astype(dtype)
    positions = fixture._positions(40000 + rows, rows)
    if rank == 2:
        positions = positions[0]
    queries = attention.q_norm(q.reshape(1, rows, 24, 512)[..., :256]).transpose(0, 2, 1, 3)
    keys = attention.k_norm(k.reshape(1, rows, 2, 256)).transpose(0, 2, 1, 3)
    expected_q, expected_k = attention.rotary_emb.apply_rotary(queries, keys, positions, unsqueeze_dim=1)
    assert attention._prep_ready(q[None], rank)
    before = fused.PREP_FUSION_CALLS
    actual_q, actual_k = attention._prep_qk(q, k, positions)
    assert fused.PREP_FUSION_CALLS == before + 1
    assert not fused.PREP_FUSION_DISABLED
    _bits(actual_q, expected_q)
    _bits(actual_k, expected_k)


@pytest.mark.parametrize('yarn', [False, True])
@pytest.mark.parametrize('position_kind', ["rank2", "rank3", "none"])
def test_prep_attention_prefill_verify_rollback_continuation(monkeypatch, yarn, position_kind):
    import test_qwen4_attn_fused as fixture
    attention = fixture._attention(seed=23, config=fixture._YARN_CONFIG if yarn else fixture._CONFIG)

    def positions(offset, rows):
        ids = fixture._positions(offset, rows)
        return None if position_kind == "none" else ids[0] if position_kind == "rank2" else ids

    def prefill(enabled):
        monkeypatch.setattr(fused, "PREP_FUSION_ENABLED", enabled)
        cache = fixture.language.QSAKVCache()
        outputs = []
        for start, rows in [(0, 1500), (1500, 700)]:
            x = (mx.random.normal((1, rows, 2560), key=mx.random.key(24 + start)) * 0.5).astype(mx.bfloat16)
            out = attention(x, mask="causal", cache=cache, position_ids=positions(start, rows))
            mx.eval(out)
            outputs.append(out)
        return outputs, cache

    monkeypatch.setattr(fused, "PREP_FUSION_DISABLED", False)
    before = fused.PREP_FUSION_CALLS
    baseline_outputs, baseline_cache = prefill(False)
    assert fused.PREP_FUSION_CALLS == before
    candidate_outputs, candidate_cache = prefill(True)
    assert fused.PREP_FUSION_CALLS > before
    for left, right in zip(baseline_outputs, candidate_outputs):
        _bits(left, right)
    for left, right in zip(fixture._state(baseline_cache), fixture._state(candidate_cache)):
        _bits(left, right)

    def step(cache, enabled, x, verify):
        monkeypatch.setattr(fused, "PREP_FUSION_ENABLED", enabled)
        fixture.qwen35_verify_qmm.set_verify_qmm_armed(verify, row_exact=verify)
        try:
            out = attention(x, mask="causal", cache=cache,
                            position_ids=positions(cache.offset, x.shape[1]), target_verify=verify)
            mx.eval(out)
            return out
        finally:
            fixture.qwen35_verify_qmm.set_verify_qmm_armed(False)

    x = mx.random.normal((1, 2, 2560), key=mx.random.key(29)).astype(mx.bfloat16)
    _bits(step(baseline_cache, False, x, True), step(candidate_cache, True, x, True))
    for left, right in zip(fixture._state(baseline_cache), fixture._state(candidate_cache)):
        _bits(left, right)
    baseline_cache.trim(1)
    candidate_cache.trim(1)
    _bits(step(baseline_cache, False, x[:, :1], False),
          step(candidate_cache, True, x[:, :1], False))
    assert baseline_cache.offset == candidate_cache.offset == 2202
    assert not fused.PREP_FUSION_DISABLED
    for left, right in zip(fixture._state(baseline_cache), fixture._state(candidate_cache)):
        _bits(left, right)


def test_prep_readiness_failure_is_independent_and_off_preserves_plan(monkeypatch):
    monkeypatch.setattr(fused, "PREP_FUSION_ENABLED", False)
    monkeypatch.setattr(fused, "PREP_FUSION_DISABLED", False)
    monkeypatch.setattr(fused, "_PREP_READY", set())
    assert not fused.prep_ready(mx.bfloat16, 24, 2, 256, 64, 3)
    before = fused.PREP_FUSION_CALLS
    original_rows = fused.rows_available()
    original_disabled = fused._DISABLED

    def fail(*args):
        raise RuntimeError("deliberate preparation failure")

    monkeypatch.setattr(fused, "_tiny_prep", fail)
    monkeypatch.setattr(fused, "PREP_FUSION_ENABLED", True)
    assert not fused.prep_ready(mx.bfloat16, 24, 2, 256, 64, 3)
    assert fused.PREP_FUSION_DISABLED
    assert fused.PREP_FUSION_CALLS == before
    assert fused._DISABLED == original_disabled
    assert fused.rows_available() == original_rows


@pytest.mark.parametrize('batch,external_rope', [(2, False), (1, True)])
def test_prep_batch_two_and_position_embedding_fallback(monkeypatch, batch, external_rope):
    import test_qwen4_attn_fused as fixture

    attention = fixture._attention(seed=31)
    monkeypatch.setattr(fused, "PREP_FUSION_DISABLED", False)
    q = mx.random.normal((batch, 2, 24 * 512), key=mx.random.key(32)).astype(mx.bfloat16)
    k = mx.random.normal((batch, 2, 2 * 256), key=mx.random.key(33)).astype(mx.bfloat16)
    v = mx.random.normal((batch, 2, 2 * 256), key=mx.random.key(34)).astype(mx.bfloat16)
    positions = mx.broadcast_to(mx.arange(2)[None, None], (3, batch, 2))
    embeddings = (mx.ones((batch, 2, 256)), mx.zeros((batch, 2, 256))) if external_rope else None
    mask = mx.ones((1, 1, 2, 2), dtype=mx.bool_)
    before = fused.PREP_FUSION_CALLS
    monkeypatch.setattr(fused, "PREP_FUSION_ENABLED", False)
    expected = attention._prepare_projected_qkv(q, k, v, fixture.language.QSAKVCache(), positions, embeddings, mask)
    monkeypatch.setattr(fused, "PREP_FUSION_ENABLED", True)
    actual = attention._prepare_projected_qkv(q, k, v, fixture.language.QSAKVCache(), positions, embeddings, mask)
    for left, right in zip(actual, expected):
        _bits(left, right)
    assert fused.PREP_FUSION_CALLS == before
