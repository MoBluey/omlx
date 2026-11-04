"""Exact BF16 head arithmetic against installed one-row/verify dispatch."""

import mlx.core as mx
import pytest

from omlx.patches import qwen4_bf16_head as head
from mlx_vlm.models.exact_speculative_verify import exact_speculative_verify_weight


@pytest.mark.parametrize('rows', [1, 2])
@pytest.mark.parametrize('strided', [False, True])
@pytest.mark.parametrize('kind', ["random", "zero", "wide"])
def test_head_tile_bits(rows, strided, kind):
    width = 256
    weight = mx.random.normal((width, 5120 if strided else 2560), key=mx.random.key(11)).astype(mx.bfloat16)
    x = mx.random.normal((rows, 5120 if strided else 2560), key=mx.random.key(12)).astype(mx.bfloat16)
    if strided:
        weight, x = weight[:, ::2], x[:, ::2]
    x = x[None]
    if kind == "zero":
        x = mx.zeros_like(x)
    elif kind == "wide":
        x = (x * 1000).astype(mx.bfloat16)
    actual = head._project(x, weight)
    expected = x @ weight.T if rows == 1 else exact_speculative_verify_weight(weight, x)
    assert expected is not None
    mx.eval(actual, expected)
    assert mx.array_equal(actual.view(mx.uint16), expected.view(mx.uint16)).item()
    # The verifier's reference must itself retain serial-decode bits.
    serial = mx.concatenate([x[:, i:i+1] @ weight.T for i in range(rows)], axis=1)
    mx.eval(serial)
    assert mx.array_equal(actual.view(mx.uint16), serial.view(mx.uint16)).item()


def test_default_off_unsupported_and_failure_fallback(monkeypatch):
    weight = mx.broadcast_to(mx.array(0, dtype=mx.bfloat16), (248320, 2560))
    x = mx.zeros((1, 2, 2560), dtype=mx.bfloat16)
    monkeypatch.setattr(head, "BF16_HEAD_ENABLED", False)
    monkeypatch.setattr(head, "BF16_HEAD_DISABLED", False)
    monkeypatch.setattr(head, "BF16_HEAD_CALLS", {1: 0, 2: 0})
    assert head.try_bf16_head(x, weight) is None
    monkeypatch.setattr(head, "BF16_HEAD_ENABLED", True)
    assert head.try_bf16_head(x[:, :1].astype(mx.float32), weight) is None
    assert head.try_bf16_head(mx.zeros((1, 3, 2560), dtype=mx.bfloat16), weight) is None
    assert head.try_bf16_head(x, weight[:128]) is None
    assert not head.BF16_HEAD_DISABLED

    def fail(*args):
        raise RuntimeError("deliberate build failure")

    monkeypatch.setattr(head, "_project", fail)
    assert head.try_bf16_head(x, weight) is None
    assert head.BF16_HEAD_DISABLED
    assert head.BF16_HEAD_CALLS == {1: 0, 2: 0}
