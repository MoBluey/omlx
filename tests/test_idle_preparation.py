"""CPU-only protocol/controller checks; model checks use the private server."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omlx import idle_preparation as idle


class Tokenizer:
    def encode(self, text):
        parts = text.split('<|im_start|>')
        ids = list(parts[0].encode())
        for part in parts[1:]:
            ids.extend([1000, *part.encode()])
        return ids

    def decode(self, ids, **kwargs):
        return bytes(ids).decode()


class Core:
    def __init__(self):
        self._mlx_executor = ThreadPoolExecutor(max_workers=1)
        self.scheduler = SimpleNamespace(config=SimpleNamespace(paged_cache_block_size=4096, max_num_seqs=1))
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = []
        self.cancelled = False

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return SimpleNamespace(cached_tokens=3996)


class Engine:
    model_type = 'qwen4_exp'
    _loaded = True
    tokenizer = Tokenizer()
    previous_length = 3996
    unsupported = False

    def __init__(self):
        self.core = Core()
        self._engine = SimpleNamespace(engine=self.core)
        self.rendered = []

    def _apply_chat_template(self, messages, tools, **kwargs):
        self.rendered.append(deepcopy(messages))
        if self.unsupported:
            return 'unsupported suffix'
        if messages[-1]['content'] == '' and not kwargs['add_generation_prompt']:
            return 'x'*4200 + '<|im_start|>user\n<|im_end|>\n'
        return 'p'*self.previous_length + ('GEN' if kwargs['add_generation_prompt'] else '')


class IdlePreparationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.env = patch.dict('os.environ', {'OMLX_EXPERIMENTAL_QWEN_IDLE_BOUNDARY': '1'})
        self.env.start()
        self.engine = Engine()
        self.messages = [{'role': 'user', 'content': 'original user text'}]
        self.kwargs = {'chat_template_kwargs': {'enable_thinking': True, 'preserve_thinking': False},
                       'preserve_reasoning': False}
        await idle.foreground(self.engine)
        self.mark = idle.epoch(self.engine)

    async def asyncTearDown(self):
        await idle.foreground(self.engine)
        self.engine.core._mlx_executor.shutdown(wait=True)
        self.env.stop()

    def response(self, **message):
        return json.dumps({'choices': [{'finish_reason': 'stop', 'message': {
            'content': 'canonical visible content', 'reasoning_content': 'private reasoning', **message}}]})

    async def started(self):
        await asyncio.wait_for(self.engine.core.started.wait(), 2)

    async def test_nonstream_uses_actual_visible_content_and_preserves_history(self):
        original = deepcopy(self.messages)
        body = self.response()
        self.assertEqual(idle.observe_json(body, self.engine, self.messages, self.kwargs, self.mark), body)
        await self.started()
        self.assertEqual(self.messages, original)
        history = next(m for m in self.engine.rendered if len(m) == 3)
        self.assertEqual(history[-2], {'role': 'assistant', 'content': 'canonical visible content'})
        call = self.engine.core.calls[0]
        self.assertEqual(call['idle_preparation_limit'], 512)
        self.assertEqual(call['sampling_params'].temperature, 0.0)
        self.assertEqual(call['sampling_params'].max_tokens, 1)
        self.engine.core.release.set()
        await self.engine._omlx_idle_preparation.task
        self.assertEqual(self.engine._omlx_idle_preparation.last['status'], 'completed')

    async def test_stream_sse_is_unchanged_and_reasoning_is_excluded(self):
        chunks = [
            'data: '+json.dumps({'choices':[{'delta':{'reasoning_content':'reasoning'},'finish_reason':None}]})+'\n\n',
            'data: '+json.dumps({'choices':[{'delta':{'content':'visible '},'finish_reason':None}]})+'\n\n',
            'data: '+json.dumps({'choices':[{'delta':{'content':'answer'},'finish_reason':'stop'}]})+'\n\n',
            'data: [DONE]\n\n']
        async def source():
            for chunk in chunks:
                yield chunk
        stream = idle.observe_stream(source(), self.engine, self.messages, self.kwargs, self.mark)
        self.assertEqual([chunk async for chunk in stream], chunks)
        await self.started()
        history = next(m for m in self.engine.rendered if len(m) == 3)
        self.assertEqual(history[-2]['content'], 'visible answer')

    async def test_foreground_cancels_queued_work_before_any_generation(self):
        idle.observe_json(self.response(), self.engine, self.messages, self.kwargs, self.mark)
        await idle.foreground(self.engine)
        self.assertEqual(self.engine.core.calls, [])
        self.assertEqual(self.engine._omlx_idle_preparation.last['status'], 'cancelled')

    async def test_foreground_cancels_active_core_task(self):
        idle.observe_json(self.response(), self.engine, self.messages, self.kwargs, self.mark)
        await self.started()
        await idle.foreground(self.engine)
        self.assertTrue(self.engine.core.cancelled)
        self.assertTrue(self.engine._omlx_idle_preparation.task.done())
        self.assertEqual(self.engine._omlx_idle_preparation.last['status'], 'cancelled')

    async def test_stale_completed_response_cannot_start_idle_work(self):
        await idle.foreground(self.engine)
        idle.observe_json(self.response(), self.engine, self.messages, self.kwargs, self.mark)
        self.assertIsNone(self.engine._omlx_idle_preparation.task)

    async def test_tools_partial_cached_output_and_failed_replies_skip(self):
        for fields in ({'tools': [{'type':'function'}]}, {'is_partial':True}, {'preserve_reasoning':True}, {'skip_cache_store':True},
                       {'chat_template_kwargs': {'enable_thinking':False,'preserve_thinking':False}}):
            idle.observe_json(self.response(), self.engine, self.messages, {**self.kwargs, **fields}, self.mark)
            self.assertIsNone(self.engine._omlx_idle_preparation.task)
        idle.observe_json(self.response(tool_calls=[{}]), self.engine, self.messages, self.kwargs, self.mark)
        self.assertIsNone(self.engine._omlx_idle_preparation.task)
        idle.observe_json(self.response().replace('"stop"','"length"'), self.engine, self.messages, self.kwargs, self.mark)
        self.assertIsNone(self.engine._omlx_idle_preparation.task)

    async def test_unknown_template_and_large_known_span_fail_closed(self):
        for unsupported, previous in ((True,3996),(False,2000)):
            self.engine.unsupported, self.engine.previous_length = unsupported, previous
            idle.observe_json(self.response(), self.engine, self.messages, self.kwargs, self.mark)
            await self.engine._omlx_idle_preparation.task
            self.assertEqual(self.engine._omlx_idle_preparation.last['status'], 'skipped')
            self.assertEqual(self.engine.core.calls, [])

    async def test_disabled_route_returns_original_objects(self):
        with patch.dict('os.environ', {'OMLX_EXPERIMENTAL_QWEN_IDLE_BOUNDARY':'0'}):
            self.assertEqual(idle.observe_json('not json', self.engine, self.messages, self.kwargs, self.mark), 'not json')
            marker = object()
            self.assertIs(idle.observe_stream(marker, self.engine, self.messages, self.kwargs, self.mark), marker)

    async def test_shutdown_cancels_before_model_references_are_released(self):
        idle.observe_json(self.response(), self.engine, self.messages, self.kwargs, self.mark)
        await self.started()
        await idle.shutdown(self.engine)
        self.assertTrue(self.engine.core.cancelled)
        self.assertFalse(hasattr(self.engine, '_omlx_idle_preparation'))

    async def test_disconnected_or_malformed_stream_never_schedules(self):
        async def source():
            yield 'data: not-json\n\n'
            yield 'data: '+json.dumps({'choices':[{'delta':{'content':'partial'},'finish_reason':None}]})+'\n\n'
        stream = idle.observe_stream(source(), self.engine, self.messages, self.kwargs, self.mark)
        async for _ in stream:
            pass
        self.assertIsNone(self.engine._omlx_idle_preparation.task)

    async def test_unsupported_model_does_not_schedule(self):
        self.engine.model_type = 'unrelated_model'
        idle.observe_json(self.response(), self.engine, self.messages, self.kwargs, self.mark)
        self.assertIsNone(self.engine._omlx_idle_preparation.task)

    async def test_other_foreground_preflight_blocks_newer_response_scheduling(self):
        older = await idle.foreground(self.engine, track=True)
        newer = await idle.foreground(self.engine, track=True)
        idle.observe_json(self.response(), self.engine, self.messages, self.kwargs, newer)
        state = self.engine._omlx_idle_preparation
        self.assertIsNone(state.task)
        self.assertIsNone(state.pending)
        idle.released(self.engine, newer)
        idle.released(self.engine, older)
        self.assertIsNone(state.task)

    async def test_own_lease_defers_preparation_until_release(self):
        ticket = await idle.foreground(self.engine, track=True)
        idle.observe_json(self.response(), self.engine, self.messages, self.kwargs, ticket)
        self.assertIsNone(self.engine._omlx_idle_preparation.task)
        idle.released(self.engine, ticket)
        await self.started()

    async def test_overlapping_acquisitions_receive_distinct_captured_tickets(self):
        idle.observe_json(self.response(), self.engine, self.messages, self.kwargs, self.mark)
        await self.started()
        a, b = await asyncio.gather(idle.foreground(self.engine, track=True),
                                    idle.foreground(self.engine, track=True))
        self.assertNotEqual(a, b)
        idle.observe_json(self.response(), self.engine, self.messages, self.kwargs, a)
        self.assertIsNone(self.engine._omlx_idle_preparation.pending)
        idle.released(self.engine, a)
        idle.released(self.engine, b)


if __name__ == '__main__':
    unittest.main()
