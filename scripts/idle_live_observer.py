"""Private real-HTTP qualification observations, never enabled for timing."""

import asyncio
from functools import wraps
import json
import os
from pathlib import Path
import threading
import time


def install():
    if os.environ.get('QWEN_IDLE_LIVE_OBSERVE') != '1':
        return
    from omlx import idle_preparation
    from omlx.engine_core import EngineCore
    from omlx.scheduler import Scheduler

    directory = Path(os.environ['QWEN_PROBE_DIR'])
    output = directory / 'idle-live-owner.jsonl'
    control = directory / 'idle-live-control.json'
    lock = threading.Lock()

    def event(kind, **fields):
        with lock, output.open('a') as f:
            f.write(json.dumps({'kind':kind, 'at':time.time(), **fields})+'\n')

    def queued_gate():
        try:
            return bool(json.loads(control.read_text()).get('queued_gate'))
        except FileNotFoundError:
            return False

    original_run = idle_preparation._run

    @wraps(original_run)
    async def run(*args, **kwargs):
        if queued_gate():
            event('queued_gate_enter')
            try:
                while queued_gate():
                    await asyncio.sleep(0.01)
            finally:
                event('queued_gate_exit')
        return await original_run(*args, **kwargs)

    idle_preparation._run = run
    original_prefill = Scheduler._do_external_prefill

    @wraps(original_prefill)
    def prefill(self, request, *args, **kwargs):
        marked = bool(getattr(request, 'idle_preparation_limit', 0))
        if marked:
            event('idle_prefill_enter', request_id=request.request_id,
                  cached_tokens=request.cached_tokens, limit=request.idle_preparation_limit)
        try:
            result = original_prefill(self, request, *args, **kwargs)
        except BaseException as exc:
            if marked:
                event('idle_prefill_exit', request_id=request.request_id, error=type(exc).__name__)
            raise
        if marked:
            event('idle_prefill_exit', request_id=request.request_id, error=None)
        return result

    Scheduler._do_external_prefill = prefill
    original_abort = EngineCore.abort_request

    @wraps(original_abort)
    async def abort(self, request_id):
        request = self.scheduler.requests.get(request_id)
        if getattr(request, 'idle_preparation_limit', 0):
            event('idle_cancel_requested', request_id=request_id)
        return await original_abort(self, request_id)

    EngineCore.abort_request = abort
    original_cleanup = Scheduler._do_abort_request

    @wraps(original_cleanup)
    def cleanup(self, request_id, *args, **kwargs):
        marked = bool(getattr(self.requests.get(request_id), 'idle_preparation_limit', 0))
        result = original_cleanup(self, request_id, *args, **kwargs)
        if marked:
            event('idle_abort_cleanup', request_id=request_id,
                  request_retained=request_id in self.requests,
                  transient_snapshot_retained=bool(self._boundary_cache_snapshots.get(request_id)))
        return result

    Scheduler._do_abort_request = cleanup
    original_released = idle_preparation.released

    @wraps(original_released)
    def released(engine, ticket):
        result = original_released(engine, ticket)
        state = getattr(engine, '_omlx_idle_preparation', None)
        if state is not None:
            event('foreground_released', active_foreground=len(state.active),
                  pending_idle=state.pending is not None,
                  idle_task_active=state.task is not None and not state.task.done())
        return result

    idle_preparation.released = released
    event('installed')
