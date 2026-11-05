"""Default-off, bounded preparation of one existing Qwen cache boundary."""

import asyncio
from contextlib import suppress
from copy import deepcopy
import json
import logging
import os
import time
import uuid

logger = logging.getLogger(__name__)
MAX_UNCACHED_TOKENS = 512
_STATE = '_omlx_idle_preparation'


def _core(engine):
    wrapper = getattr(engine, '_engine', None)
    return getattr(wrapper, 'engine', wrapper)


def _enabled(engine):
    if os.environ.get('OMLX_EXPERIMENTAL_QWEN_IDLE_BOUNDARY') != '1':
        return False
    if getattr(engine, 'model_type', None) != 'qwen4_exp':
        return False
    core = _core(engine)
    config = getattr(getattr(core, 'scheduler', None), 'config', None)
    return bool(getattr(engine, '_loaded', False) and config is not None
                and config.paged_cache_block_size == 4096 and config.max_num_seqs == 1)


def _eligible(engine, messages, kwargs):
    ct = kwargs.get('chat_template_kwargs') or {}
    return (_enabled(engine) and not kwargs.get('tools') and not kwargs.get('compiled_grammar')
            and not kwargs.get('skip_cache_store')
            and not kwargs.get('is_partial') and not kwargs.get('preserve_reasoning')
            and ct.get('preserve_thinking') is False
            and ct.get('enable_thinking', getattr(engine, '_enable_thinking', None)) is True
            and all(m.get('role') in ('system', 'user', 'assistant')
                    and isinstance(m.get('content'), str) and not m.get('tool_calls')
                    and not m.get('partial') for m in messages))


class _Preparation:
    def __init__(self):
        self.epoch = 0
        self.task = None
        self.active = set()
        self.pending = None
        self.last = {}

    def record(self, status, **fields):
        self.last = {'event': 'idle_prefix_preparation', 'status': status, **fields}
        logger.info('%s', json.dumps(self.last, sort_keys=True))


async def foreground(engine, *, track=False):
    """Cancel background work before any foreground route submits preflight."""
    state = getattr(engine, _STATE, None)
    if state is None:
        if not _enabled(engine):
            return
        state = _Preparation()
        setattr(engine, _STATE, state)
    state.epoch += 1
    ticket = state.epoch
    if track:
        state.active.add(ticket)
    if state.pending is not None:
        state.pending = None
        state.record('cancelled', reason='foreground_before_start')
    task = state.task
    try:
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            if state.last.get('status') != 'cancelled':
                state.record('cancelled', reason='foreground_before_start')
    except BaseException:
        state.active.discard(ticket)
        raise
    return ticket


def released(engine, ticket):
    """The normal lease lifecycle accounts for preflight as well as decode."""
    state = getattr(engine, _STATE, None)
    if state is None:
        return
    state.active.discard(ticket)
    if not state.active and state.pending is not None:
        history, settings, expected_epoch = state.pending
        state.pending = None
        if expected_epoch == state.epoch:
            state.task = asyncio.create_task(_run(engine, history, settings, state, expected_epoch))


async def shutdown(engine):
    """Drop the pending task before its loaded tokenizer/core are torn down."""
    if getattr(engine, _STATE, None) is not None:
        await foreground(engine)
        delattr(engine, _STATE)


def epoch(engine):
    state = getattr(engine, _STATE, None)
    return state.epoch if state is not None else None


def _prepare(engine, history, kwargs):
    # Runs on the existing model-owner executor. Use the loaded tokenizer and
    # the same renderer as normal VLM text requests, without loading anything.
    from .idle_prefix import aligned_next_turn_prefix

    class Renderer:
        tokenizer = engine.tokenizer

        def prepare_chat_prompt(self, messages, *, tools=None,
                                chat_template_kwargs=None, is_partial=False,
                                add_generation_prompt=True):
            text = engine._apply_chat_template(
                messages, tools, chat_template_kwargs=chat_template_kwargs,
                is_partial=is_partial, add_generation_prompt=add_generation_prompt,
            )
            return text, list(self.tokenizer.encode(text))

    renderer = Renderer()
    ct = kwargs.get('chat_template_kwargs')
    text, ids, selection = aligned_next_turn_prefix(renderer, history, chat_template_kwargs=ct)
    if ids is None:
        return None, selection
    _, previous = renderer.prepare_chat_prompt(deepcopy(history[:-1]),
        chat_template_kwargs=ct, add_generation_prompt=False)
    selection['known_uncached_tokens'] = len(ids)-len(previous)
    if not 0 < selection['known_uncached_tokens'] <= MAX_UNCACHED_TOKENS:
        return None, {**selection, 'skipped_reason': 'known_span_exceeds_limit'}
    sentinel = list(renderer.tokenizer.encode('<|im_start|>'))
    if len(sentinel) != 1 or list(renderer.tokenizer.encode(text+'<|im_start|>')) != ids+sentinel:
        return None, {**selection, 'skipped_reason': 'sentinel_not_exact'}
    return ids+sentinel, selection


async def _run(engine, history, kwargs, state, expected_epoch):
    started = time.monotonic()
    request_id = 'idle-prefix-' + uuid.uuid4().hex
    selection = {}
    try:
        await asyncio.sleep(0)
        core = _core(engine)
        prompt, selection = await asyncio.get_running_loop().run_in_executor(
            core._mlx_executor, _prepare, engine, history, kwargs)
        if prompt is None:
            state.record('skipped', request_id=request_id, **selection)
            return
        if state.epoch != expected_epoch or getattr(core, '_closed', False):
            state.record('cancelled', request_id=request_id, reason='foreground_during_render')
            return
        from .request import SamplingParams
        state.record('preparing', request_id=request_id, **selection)
        output = await core.generate(prompt=prompt,
            sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
            request_id=request_id, specprefill=False,
            idle_preparation_limit=MAX_UNCACHED_TOKENS)
        state.record('completed', request_id=request_id, elapsed_s=time.monotonic()-started,
                     cached_tokens=output.cached_tokens, **selection)
    except asyncio.CancelledError:
        state.record('cancelled', request_id=request_id,
                     elapsed_s=time.monotonic()-started, **selection)
        raise
    except Exception as exc:
        # Background work is best-effort and must never fail a real response.
        state.record('skipped', request_id=request_id, reason=str(exc), **selection)


def _schedule(engine, messages, content, kwargs, expected_epoch):
    state = getattr(engine, _STATE, None)
    if (not isinstance(content, str) or not content or not _eligible(engine, messages, kwargs) or state is None
            or expected_epoch != state.epoch
            or bool(state.active - {expected_epoch})
            or (state.task is not None and not state.task.done())):
        return
    history = deepcopy(messages) + [{'role': 'assistant', 'content': content}]
    state.record('scheduled')
    settings = {'chat_template_kwargs': deepcopy(kwargs.get('chat_template_kwargs'))}
    state.pending = (history, settings, expected_epoch)
    if not state.active:
        released(engine, expected_epoch)


def observe_json(body, engine, messages, kwargs, expected_epoch):
    """Observe the actual canonical content returned to a nonstreaming caller."""
    if _eligible(engine, messages, kwargs):
        try:
            response = json.loads(body)
            choice = response['choices'][0]
            message = choice['message']
            if choice.get('finish_reason') == 'stop' and not message.get('tool_calls'):
                _schedule(engine, messages, message.get('content'), kwargs, expected_epoch)
        except (ValueError, KeyError, IndexError, TypeError):
            pass
    return body


def observe_stream(stream, engine, messages, kwargs, expected_epoch):
    """Pass SSE through unchanged; capture only actual emitted content deltas."""
    if not _eligible(engine, messages, kwargs):
        return stream

    async def observed():
        content = []
        finish = None
        failed = False
        try:
            async for chunk in stream:
                data = chunk.removeprefix('data:').strip()
                if data == '[DONE]':
                    if finish == 'stop' and not failed:
                        _schedule(engine, messages, ''.join(content), kwargs, expected_epoch)
                else:
                    try:
                        payload = json.loads(data)
                        failed = failed or bool(payload.get('error'))
                        for choice in payload.get('choices', []):
                            delta = choice.get('delta') or {}
                            failed = failed or bool(delta.get('tool_calls'))
                            if isinstance(delta.get('content'), str):
                                content.append(delta['content'])
                            finish = choice.get('finish_reason') or finish
                    except (ValueError, KeyError, TypeError, AttributeError):
                        failed = True
                yield chunk
        finally:
            await stream.aclose()

    return observed()
