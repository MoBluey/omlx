# SPDX-License-Identifier: Apache-2.0
"""Exact direct-cache SDPA, independently of the reduced-selector trial."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from test_qwen4_qsa_decode_gather import qsa_fast
from test_qwen4_qsa_long_selector_trial import _long_verify_continuation


@pytest.fixture
def direct_trial(monkeypatch):
    monkeypatch.setattr(qsa_fast, "_GATHERED_SDPA_ENABLED", True)
    monkeypatch.setattr(qsa_fast, "_GATHERED_SDPA_DISABLED", False)
    monkeypatch.setattr(qsa_fast, "_GATHERED_SDPA_CALLS", 0)
    monkeypatch.setattr(qsa_fast, "_GATHERED_SDPA_FAILED", {})


@pytest.mark.parametrize("key_len", [64003, 79003])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("layout", ["capacity", "sequence", "element", "offset_element", "unaligned"])
def test_direct_cache_long_strides_and_tails_match_native_copies(key_len, dtype, layout, direct_trial):
    capacity = key_len + 777
    seq = 2 * capacity if layout == "sequence" else capacity
    dim = 512 if layout in {"element", "offset_element"} else 256
    if layout == "unaligned":
        dim = 264
    keys = mx.random.normal((1, 2, seq, dim), key=mx.random.key(21)).astype(dtype)
    values = mx.random.normal((1, 2, seq, dim), key=mx.random.key(22)).astype(dtype)
    if layout == "sequence":
        keys, values = keys[:, :, ::2], values[:, :, ::2]
    elif layout in {"element", "offset_element"}:
        start = int(layout == "offset_element")
        keys, values = keys[..., start::2], values[..., start::2]
    elif layout == "unaligned":
        # Unit element stride and row/head strides divisible by eight, but the
        # starting BF16/FP16 address is two bytes off vec<T, 8> alignment.
        keys, values = keys[..., 1:257], values[..., 1:257]
    keys, values = keys[:, :, :key_len], values[:, :, :key_len]
    queries = mx.random.normal((1, 24, 1, 512), key=mx.random.key(23)).astype(dtype)[..., ::2]
    rows = 2051
    positions = np.sort(np.random.default_rng(key_len).choice(key_len, rows, replace=False))
    packed = np.repeat(positions.astype(np.int32), 2).reshape(1, -1)
    tokens = mx.array(packed)[:, ::2]
    actual = qsa_fast.gathered_decode_sdpa(queries, keys, values, tokens, 256**-0.5)
    assert actual is not None, qsa_fast._GATHERED_SDPA_FAILED
    expected = qsa_fast._decode_qsa_sdpa(
        mx.contiguous(queries),
        qsa_fast._gather_kv_rows(keys, tokens),
        qsa_fast._gather_kv_rows(values, tokens),
        256**-0.5,
    )
    mx.eval(actual, expected)
    assert mx.array_equal(actual.view(mx.uint16), expected.view(mx.uint16)).item()
    assert qsa_fast._GATHERED_SDPA_CALLS == 1
    assert not qsa_fast._GATHERED_SDPA_FAILED
    assert not qsa_fast._GATHERED_SDPA_DISABLED


@pytest.mark.parametrize("key_len", [64003, 79003])
def test_direct_cache_long_verify_accepted_prefix_and_continuation(monkeypatch, key_len, direct_trial):
    gathers = _long_verify_continuation(monkeypatch, key_len, 2, "_GATHERED_SDPA_ENABLED")
    assert gathers == {False: 18, True: 0}
    assert qsa_fast._GATHERED_SDPA_CALLS == 9
    assert not qsa_fast._GATHERED_SDPA_FAILED
    assert not qsa_fast._GATHERED_SDPA_DISABLED


def test_direct_cache_keeps_rank_three_masked_positions_on_the_existing_route(monkeypatch, direct_trial):
    gathers = _long_verify_continuation(monkeypatch, 64003, 3, "_GATHERED_SDPA_ENABLED")
    assert gathers == {False: 0, True: 0}
    assert qsa_fast._GATHERED_SDPA_CALLS == 0


def test_direct_cache_failure_is_visible_and_does_not_disable_baseline(monkeypatch, direct_trial):
    keys = mx.ones((1, 2, 3000, 256), dtype=mx.bfloat16)
    values = keys * 2
    queries = mx.ones((1, 24, 1, 256), dtype=mx.bfloat16)
    tokens = mx.arange(2051, dtype=mx.int32)[None]

    def failed(**kwargs):
        raise ValueError("direct-cache trial pipeline unavailable")

    monkeypatch.setattr(qsa_fast, "_GATHERED_SDPA_KERNELS", [failed, failed])
    assert qsa_fast.gathered_decode_sdpa(queries, keys, values, tokens, 256**-0.5) is None
    assert qsa_fast._GATHERED_SDPA_DISABLED
    assert qsa_fast._GATHERED_SDPA_CALLS == 0
    assert list(qsa_fast._GATHERED_SDPA_FAILED.values()) == ["direct-cache trial pipeline unavailable"]
    expected = qsa_fast._decode_qsa_sdpa(
        queries, qsa_fast._gather_kv_rows(keys, tokens), qsa_fast._gather_kv_rows(values, tokens), 256**-0.5
    )
    mx.eval(expected)
    assert expected.shape == queries.shape


@pytest.mark.parametrize("unsupported", ["off", "disabled", "gpu", "native", "short", "dtype", "head_dim"])
def test_direct_cache_fails_closed_outside_its_plan(monkeypatch, direct_trial, unsupported):
    dtype = mx.float32 if unsupported == "dtype" else mx.bfloat16
    dim = 128 if unsupported == "head_dim" else 256
    queries = mx.ones((1, 24, 1, dim), dtype=dtype)
    keys = mx.ones((1, 2, 3000, dim), dtype=dtype)
    tokens = mx.arange(1000 if unsupported == "short" else 2051, dtype=mx.int32)[None]
    if unsupported == "off":
        monkeypatch.setattr(qsa_fast, "_GATHERED_SDPA_ENABLED", False)
    elif unsupported == "disabled":
        monkeypatch.setattr(qsa_fast, "_GATHERED_SDPA_DISABLED", True)
    elif unsupported == "gpu":
        monkeypatch.setattr(qsa_fast, "_gpu_class", lambda: "d")
    elif unsupported == "native":
        monkeypatch.setattr(qsa_fast, "_native_decode_sdpa_supported", lambda *args: False)
    assert qsa_fast.gathered_decode_sdpa(queries, keys, keys, tokens, dim**-0.5) is None
    assert qsa_fast._GATHERED_SDPA_CALLS == 0
    assert not qsa_fast._GATHERED_SDPA_FAILED
