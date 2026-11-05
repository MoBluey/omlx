"""Marked-idle policy checks against production scheduler entry points."""

from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest

from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler


def context():
    return SimpleNamespace(_admission_paused=False, waiting=deque(), running={},
        prefilling=deque(), requests={}, _pending_abort_ids=set(),
        _decode_contention=lambda:False,
        config=SimpleNamespace(prefill_step_size=4096, paged_cache_block_size=4096))


def request():
    result = Request(request_id='idle', prompt=list(range(4097)),
        sampling_params=SamplingParams(max_tokens=1), idle_preparation_limit=512)
    result.prompt_token_ids = list(range(4097))
    result.cached_tokens = 3996
    return result


def test_only_small_useful_actual_cache_gap_is_admitted():
    scheduler, req = context(), request()
    assert Scheduler._idle_preparation_skip_reason(scheduler, req, req.prompt_token_ids[3996:]) is None
    req.cached_tokens = 0
    assert Scheduler._idle_preparation_skip_reason(scheduler, req, req.prompt_token_ids) is not None
    req.cached_tokens = 3500
    assert Scheduler._idle_preparation_skip_reason(scheduler, req, req.prompt_token_ids[3500:]) is not None
    req.cached_tokens = 4096
    assert Scheduler._idle_preparation_skip_reason(scheduler, req, req.prompt_token_ids[-1:]) is not None


@pytest.mark.parametrize('condition', ['pressure','waiting','running','cancel','contention'])
def test_idle_job_yields_to_foreground_and_pressure(condition):
    scheduler, req = context(), request()
    if condition == 'pressure': scheduler._admission_paused = True
    if condition == 'waiting': scheduler.waiting.append(object())
    if condition == 'running': scheduler.running['foreground'] = object()
    if condition == 'cancel': scheduler._pending_abort_ids.add(req.request_id)
    if condition == 'contention': scheduler._decode_contention = lambda:True
    assert Scheduler._idle_preparation_skip_reason(scheduler, req, req.prompt_token_ids[3996:]) == 'foreground_or_pressure'
    req.idle_preparation_limit = 0
    assert Scheduler._idle_preparation_skip_reason(scheduler, req, req.prompt_token_ids[3996:]) is None


def test_idle_add_never_queues_behind_foreground():
    scheduler, req = context(), request()
    scheduler.waiting.append(object())
    with pytest.raises(RuntimeError, match='foreground or pressure'):
        Scheduler.add_request(scheduler, req)
    assert list(scheduler.requests) == []


def test_idle_cancel_cannot_store_cache_but_normal_cancel_can():
    scheduler, req = context(), request()
    scheduler.block_aware_cache = object()
    scheduler._model_has_unreconstructible_cache = lambda:False
    assert not Scheduler._abort_keeps_cache(scheduler, req)
    req.idle_preparation_limit = 0
    assert Scheduler._abort_keeps_cache(scheduler, req)


def test_late_idle_abort_sets_persistent_store_suppression_only_for_idle():
    scheduler, req = context(), request()
    scheduler.requests[req.request_id] = req
    scheduler._abort_keep_cache_ids = set()
    Scheduler.abort_request(scheduler, req.request_id, keep_cache=True)
    assert req.skip_cache_store
    assert req.request_id not in scheduler._abort_keep_cache_ids
    scheduler._pending_abort_ids.clear()
    assert Scheduler._idle_preparation_store_cancelled(scheduler, req.request_id)
    req.idle_preparation_limit, req.skip_cache_store = 0, False
    Scheduler.abort_request(scheduler, req.request_id, keep_cache=True)
    assert not req.skip_cache_store
    assert req.request_id in scheduler._abort_keep_cache_ids


def test_idle_cancel_during_store_worker_sync_does_not_begin_new_store():
    scheduler, req = context(), request()
    scheduler.requests[req.request_id] = req
    scheduler._abort_keep_cache_ids = set()
    scheduler._phase_timer = lambda *a: nullcontext()
    scheduler._stream = None
    scheduler.paged_cache_manager = None
    scheduler.block_aware_cache = MagicMock()
    scheduler._idle_preparation_store_cancelled = lambda rid: Scheduler._idle_preparation_store_cancelled(scheduler, rid)
    with patch('omlx.scheduler._safe_sync_stream', side_effect=lambda *a: Scheduler.abort_request(scheduler, req.request_id, True)):
        Scheduler._async_store_cache_worker(scheduler, req.request_id, [1, 2], [], None, None, None, None, None)
    scheduler.block_aware_cache.store_cache.assert_not_called()
    scheduler.block_aware_cache.clear_request_entry.assert_called_once_with(req.request_id)
