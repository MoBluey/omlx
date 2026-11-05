# SPDX-License-Identifier: Apache-2.0
"""Long-context exactness of the reused private QSA selector trial."""

from __future__ import annotations

import mlx.core as mx
import pytest

from test_qwen4_qsa_decode_gather import (
    HEAD_DIM,
    RATIO,
    TOPK,
    _attention_and_caches,
    _head_scores,
    _official_gathered_tokens,
    _official_mask,
    language,
    qsa_fast,
)


@pytest.mark.parametrize("key_len", [64003, 79003])
@pytest.mark.parametrize("kind", ["normal", "magnitudes", "cutoff_ties", "specials"])
@pytest.mark.parametrize("tokens", [False, True])
def test_long_reduced_selector_is_exact_and_engages(monkeypatch, key_len, kind, tokens):
    blocks = key_len // RATIO
    shape = (1, 1, 4, blocks) if tokens else (1, 4, 1, blocks)
    scores = mx.array(_head_scores(blocks, kind, key_len).reshape(shape))
    monkeypatch.setattr(qsa_fast, "_DECODE_SELECT_REDUCED_BARRIERS_ENABLED", True)
    monkeypatch.setattr(qsa_fast, "_DECODE_SELECT_FAILED", {})
    monkeypatch.setattr(qsa_fast, "_DECODE_SELECT_REDUCED_BARRIERS_STATS", {
        "mask": 0, "tokens": 0, "failures": 0,
    })
    actual = qsa_fast._decode_block_selection(scores, HEAD_DIM, key_len, RATIO, TOPK, tokens)
    assert actual is not None, qsa_fast._DECODE_SELECT_FAILED
    expected = _official_gathered_tokens(scores, key_len) if tokens else _official_mask(scores, key_len)
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected).item()
    assert qsa_fast._DECODE_SELECT_REDUCED_BARRIERS_STATS == {
        "mask": int(not tokens), "tokens": int(tokens), "failures": 0,
    }
    assert not qsa_fast._DECODE_SELECT_FAILED
    monkeypatch.setattr(qsa_fast, "_DECODE_SELECT_REDUCED_BARRIERS_ENABLED", False)
    baseline = qsa_fast._decode_block_selection(scores, HEAD_DIM, key_len, RATIO, TOPK, tokens)
    assert baseline is not None, qsa_fast._DECODE_SELECT_FAILED
    assert mx.array_equal(actual, baseline).item()


def _long_verify_continuation(monkeypatch, key_len, position_rank, flag, extra_flags=()):
    """Exercise real attention methods and raw-cache prefix persistence.

    The helper is shared by the independent direct-cache candidate tests.
    """
    config, attention, caches = _attention_and_caches(key_len)
    monkeypatch.setattr(language, "_row_exact_verify_armed", lambda: True)
    gather = qsa_fast._gather_kv_rows
    arm = [False]
    gathers = {False: 0, True: 0}

    def counted(kv, tokens):
        gathers[arm[0]] += 1
        return gather(kv, tokens)

    monkeypatch.setattr(qsa_fast, "_gather_kv_rows", counted)

    def run(hidden, cache, enabled, verify=False):
        arm[0] = enabled
        monkeypatch.setattr(qsa_fast, flag, enabled)
        for extra_flag in extra_flags:
            monkeypatch.setattr(qsa_fast, extra_flag, enabled)
        positions = mx.arange(cache.offset, cache.offset + hidden.shape[1], dtype=mx.int32)[None]
        if position_rank == 3:
            positions = mx.broadcast_to(positions[None], (3, 1, hidden.shape[1]))
        result = attention(hidden, cache=cache, position_ids=positions, target_verify=verify)
        mx.eval(result, cache.state)
        return result

    def compare(hidden, verify=False):
        actual = run(hidden, caches[0], True, verify)
        baseline = run(hidden, caches[1], False, verify)
        assert mx.array_equal(actual.view(mx.uint16), baseline.view(mx.uint16)).item()
        for a, b in zip(caches[0].state, caches[1].state):
            assert mx.array_equal(a, b).item()
        assert caches[0].nbytes == caches[1].nbytes

    window = mx.random.normal((1, 4, config.hidden_size)).astype(mx.bfloat16)
    compare(window, verify=True)
    assert caches[0].trim(2) == caches[1].trim(2) == 2
    assert caches[0]._pooled_index_offset == caches[1]._pooled_index_offset == caches[0].offset // RATIO
    # First continue from the retained accepted-prefix banks, then restore the
    # persisted raw prefix and continue with the same private candidate switch.
    for step in range(5):
        if step == 2:
            restored = []
            for cache in caches:
                other = language.QSAKVCache()
                other.prefix_cache_restore(cache.prefix_cache_snapshot())
                assert other._pooled_index_keys is None
                restored.append(other)
            caches = restored
        hidden = mx.random.normal((1, 1, config.hidden_size)).astype(mx.bfloat16)
        compare(hidden)
    return gathers


@pytest.mark.parametrize("key_len", [64003, 79003])
@pytest.mark.parametrize("position_rank", [2, 3])
def test_long_selector_verify_accepted_prefix_and_continuation(monkeypatch, key_len, position_rank):
    monkeypatch.setattr(qsa_fast, "_DECODE_SELECT_REDUCED_BARRIERS_STATS", {
        "mask": 0, "tokens": 0, "failures": 0,
    })
    monkeypatch.setattr(qsa_fast, "_DECODE_SELECT_FAILED", {})
    _long_verify_continuation(monkeypatch, key_len, position_rank, "_DECODE_SELECT_REDUCED_BARRIERS_ENABLED")
    output = "tokens" if position_rank == 2 else "mask"
    assert qsa_fast._DECODE_SELECT_REDUCED_BARRIERS_STATS[output] == 9
    assert qsa_fast._DECODE_SELECT_REDUCED_BARRIERS_STATS["failures"] == 0
    assert not any(key[0] == "reduced" for key in qsa_fast._DECODE_SELECT_FAILED)
