"""Smaller DFlash2 block shapes and production sampled prefix verification."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.patches.mlx_lm_mtp import batch_generator as bg
from omlx.utils.sampling import make_sampler
from test_dflash_batched import _captured, _tiny_drafter
from test_mlx_lm_mtp_patch import _drafted_counting_model, generate


@pytest.mark.parametrize("block", [2, 3])
def test_smaller_dflash2_blocks_match_checkpoint_forward(block):
    drafter = _tiny_drafter()
    drafter.block_size, drafter.depth = block, block - 1
    for length in (1, 3, drafter.window - 1):
        context = mx.concatenate(_captured(length, seed=80 + length), axis=-1)
        anchor = mx.array([7], dtype=mx.int32)
        reference_cache = drafter.model.make_cache()
        for entry in reference_cache:
            entry.offset = 0
        expected = drafter.model.draft_block(
            anchor, context, reference_cache, block, lambda logits: mx.argmax(logits, axis=-1)
        )
        # A fresh uid avoids feeding the same context twice into the ring.
        actual, _ = drafter._draft_batched(
            [(SimpleNamespace(uid=length), drafter._row(length), context, anchor, None)]
        )[0]
        assert actual.shape == (1, block - 1)
        assert actual.tolist() == expected.tolist()


@pytest.mark.parametrize("depth,accepted", [(depth, accepted) for depth in (1, 2) for accepted in range(depth + 1)])
def test_production_sparse_acceptance_and_residual_for_every_smaller_prefix(monkeypatch, depth, accepted):
    """Force each reject position and inspect the actual sparse p-q sampler.

    q also assigns mass outside target top-k. The residual must exclude that
    mass while preserving p = min(p,q) + positive(p-q) on every target row.
    """
    sampler = make_sampler(temp=1.0, top_p=0.95, top_k=20)
    logits = mx.broadcast_to(mx.linspace(0.0, -3.0, 64), (depth + 1, 64))
    lp = bg._logprobs(logits)
    q = bg.SparseDraftQ(
        mx.broadcast_to(mx.array([0, 63], dtype=mx.int32), (2, 2)),
        mx.broadcast_to(mx.log(mx.array([0.8, 0.2])), (2, 2)),
        64,
    )
    state = SimpleNamespace(drafts=mx.zeros((2,), dtype=mx.uint32), draft_lps=[], draft_accept_lps=q)
    bg._clip_block_drafts(state, depth)
    assert len(state.draft_accept_lps) == depth
    categorical_logits = []

    def categorical(values):
        categorical_logits.append(values)
        return mx.argmax(values, axis=-1)

    rolls = mx.array([0.0 if index < accepted else 1.0 for index in range(depth)])
    monkeypatch.setattr(mx.random, "uniform", lambda **kwargs: rolls)
    monkeypatch.setattr(mx.random, "categorical", categorical)
    result = bg._stochastic_verify_tokens(sampler, lp, state.drafts, state.draft_accept_lps)
    assert result[0].item() == accepted
    assert len(categorical_logits) == 2  # residual rows, then bonus

    target_ids = mx.argsort(-lp, axis=-1)[:, :20]
    target = mx.exp(bg._accept_lp_for(sampler, lp))
    q_dense = mx.exp(mx.stack(bg._dense_q_rows(state.draft_accept_lps)))
    residual = mx.maximum(target[:depth] - q_dense, 0.0)
    assert mx.allclose(mx.minimum(target[:depth], q_dense) + residual, target[:depth], atol=1e-6).item()
    expected = mx.take_along_axis(residual, target_ids[:depth], axis=-1)
    expected /= expected.sum(axis=-1, keepdims=True)
    actual = mx.softmax(categorical_logits[0], axis=-1)
    assert mx.allclose(actual, expected, atol=1e-6).item()
    bonus = mx.take_along_axis(target[depth:depth + 1], target_ids[depth:depth + 1], axis=-1)
    assert mx.allclose(mx.softmax(categorical_logits[1], axis=-1), bonus, atol=1e-6).item()

@pytest.mark.parametrize("block,adaptive", [(2, False), (3, False), (3, True)])
def test_smaller_sampled_block_routes_respect_eos_and_token_limit(monkeypatch, block, adaptive):
    sampled_widths = []
    verify = bg._stochastic_verify_tokens

    def recorded(sampler, lp, drafts, q):
        assert len(q) == drafts.shape[0]
        sampled_widths.append(lp.shape[0])
        return verify(sampler, lp, drafts, q)

    monkeypatch.setattr(bg, "_stochastic_verify_tokens", recorded)
    model = _drafted_counting_model(block - 1, adaptive=adaptive)
    sampler = make_sampler(temp=1.0, top_p=0.95, top_k=20)
    output, terminal = generate(
        model, [[1, 2]], [20], samplers=[sampler], stop_tokens=[[6]]
    )
    assert output[0] == [3, 4, 5, 6]
    assert terminal[0].finish_reason == "stop"
    assert sampled_widths and max(sampled_widths) == block

    # A one-token remainder must truncate an accepted run, including its cache.
    model = _drafted_counting_model(block - 1, adaptive=adaptive)
    output, terminal = generate(model, [[1, 2]], [5], samplers=[sampler])
    assert output[0] == [3, 4, 5, 6, 7]
    assert terminal[0].finish_reason == "length"
    cache = terminal[0].prompt_cache[0]
    assert cache.keys[0, 0, :cache.offset, 0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
