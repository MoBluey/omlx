"""Combined candidates preserve the installed masked path and exact cache state."""

import pytest

from test_qwen4_qsa_long_selector_trial import _long_verify_continuation, qsa_fast


@pytest.mark.parametrize("key_len", [9003, 64003, 79003])
@pytest.mark.parametrize("position_rank", [2, 3])
def test_combined_verify_rollback_restore_and_continuation(monkeypatch, key_len, position_rank):
    monkeypatch.setattr(qsa_fast, "MASKED_S_ENABLED", True)
    monkeypatch.setattr(qsa_fast, "MASKED_S_CALLS", 0)
    monkeypatch.setattr(qsa_fast, "_DECODE_SDPA_DISABLED", False)
    monkeypatch.setattr(qsa_fast, "_DECODE_SELECT_FAILED", {})
    monkeypatch.setattr(qsa_fast, "_DECODE_SELECT_REDUCED_BARRIERS_STATS", {
        "mask": 0, "tokens": 0, "failures": 0,
    })
    monkeypatch.setattr(qsa_fast, "_FUSED_DECODE_KV_GATHER_DISABLED", False)
    monkeypatch.setattr(qsa_fast, "_FUSED_DECODE_KV_GATHER_PROVEN", False)
    monkeypatch.setattr(qsa_fast, "_FUSED_DECODE_KV_GATHER_CALLS", 0)
    _long_verify_continuation(
        monkeypatch, key_len, position_rank,
        "_DECODE_SELECT_REDUCED_BARRIERS_ENABLED",
        extra_flags=("_FUSED_DECODE_KV_GATHER",),
    )
    output = "tokens" if position_rank == 2 else "mask"
    assert qsa_fast._DECODE_SELECT_REDUCED_BARRIERS_STATS == {
        "mask": 9 if output == "mask" else 0,
        "tokens": 9 if output == "tokens" else 0,
        "failures": 0,
    }
    assert qsa_fast._FUSED_DECODE_KV_GATHER_CALLS == (9 if position_rank == 2 else 0)
    assert not qsa_fast._FUSED_DECODE_KV_GATHER_DISABLED
    assert not qsa_fast._DECODE_SELECT_FAILED
    assert not qsa_fast._DECODE_SDPA_DISABLED
    if position_rank == 3 and qsa_fast._gpu_class() == "s":
        assert qsa_fast.MASKED_S_CALLS == 18
