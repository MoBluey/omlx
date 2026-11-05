"""Small CPU guard checks; actual tokenizer proof is check_idle_prefix.py."""

from pathlib import Path
import hashlib
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from omlx.idle_prefix import aligned_next_turn_prefix, canonical_next_turn_prefix
from idle_state_probe import _different
from compare_idle_audits import load_npy, logical_text_positions
import prepare_idle_prefix


class Renderer:
    tokenizer = None

    def __init__(self):
        self.tokenizer = self
        self.kwargs = None

    def encode(self, text):
        return list(text.encode())

    def prepare_chat_prompt(self, messages, **kwargs):
        self.kwargs = kwargs
        messages[0]['content'] = 'mutated copy'
        return 'canonical<|im_start|>user\n<|im_end|>\n', []


class IdlePrefixGuards(unittest.TestCase):
    def test_history_is_not_mutated(self):
        history = [{'role': 'user', 'content': 'original'},
                   {'role': 'assistant', 'content': 'answer'}]
        engine = Renderer()
        text, ids = canonical_next_turn_prefix(engine, history,
            chat_template_kwargs={'preserve_thinking': False})
        self.assertEqual(history[0]['content'], 'original')
        self.assertEqual(len(history), 2)
        self.assertEqual(text, 'canonical')
        self.assertEqual(ids, list(b'canonical'))
        self.assertFalse(engine.kwargs['add_generation_prompt'])
        self.assertFalse(engine.kwargs['is_partial'])

    def test_incomplete_history_refused(self):
        with self.assertRaises(ValueError):
            canonical_next_turn_prefix(Renderer(), [{'role': 'user', 'content': 'hi'}])

    def test_unknown_suffix_refused(self):
        engine = Renderer()
        engine.prepare_chat_prompt = lambda *a, **k: ('different template', [])
        with self.assertRaises(ValueError):
            canonical_next_turn_prefix(engine, [{'role': 'assistant', 'content': 'answer'}])

    def test_state_comparison_identifies_ple_history_and_offset(self):
        captured = {'cache': [{'offset': 42, 'state': [None, None, {'sha256': 'conv'}, {'sha256': 'tokens'}]}]}
        restored = {'cache': [{'offset': 43, 'state': [None, None, {'sha256': 'conv'}, {'sha256': 'wrong'}]}]}
        self.assertEqual(_different(captured, restored),
            ['.cache[0].offset', '.cache[0].state[3].sha256'])

    def test_equal_state_accepted(self):
        state = {'shape': [1, 42, 3], 'dtype': 'bfloat16', 'sha256': 'bits'}
        self.assertEqual(_different(state, dict(state)), [])

    def test_state_shape_difference_rejected(self):
        self.assertEqual(_different({'shape': [1, 2]}, {'shape': [1]}), ['.shape.length'])

    def test_text_position_broadcast_normalised_only_after_byte_verification(self):
        raw = struct.pack('<iii', 0, 1, 2)
        def state(shape, data):
            return [{'class': 'QSAKVCache', 'state': [None, None, None,
                {'shape': shape, 'dtype': 'mlx.core.int32', 'sha256': hashlib.sha256(data).hexdigest()}]}]
        a, ai = logical_text_positions(state([1, 3], raw), 3)
        b, bi = logical_text_positions(state([3, 1, 3], raw*3), 3)
        self.assertEqual(a, b)
        self.assertEqual(ai, bi)
        bad, indices = logical_text_positions(state([3, 1, 3], b'wrong'), 3)
        self.assertEqual(indices, [])
        self.assertNotEqual(a, bad)

    def test_fortran_single_row_npy_is_read_without_reordering(self):
        header = repr({'descr': '<f4', 'fortran_order': True, 'shape': (1, 2)}).encode()
        data = b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header + struct.pack('<ff', 1.5, -2)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'row.npy'
            path.write_bytes(data)
            self.assertEqual(list(load_npy(path)), [1.5, -2])

    def test_aligned_prefix_and_non_roundtrip_refusal(self):
        class BoundaryRenderer(Renderer):
            def prepare_chat_prompt(self, messages, **kwargs):
                if kwargs['add_generation_prompt']:
                    return 'old', [1, 2, 3]
                return 'x'*5000 + '<|im_start|>user\n<|im_end|>\n', []

            def decode(self, ids, **kwargs):
                return bytes(ids).decode()

        engine = BoundaryRenderer()
        history = [{'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': 'answer'}]
        text, ids, decision = aligned_next_turn_prefix(engine, history)
        self.assertEqual(len(ids), 4096)
        self.assertEqual(text, 'x'*4096)
        self.assertTrue(decision['roundtrip_exact'])
        engine.decode = lambda *a, **k: 'wrong'
        text, ids, decision = aligned_next_turn_prefix(engine, history)
        self.assertIsNone(ids)
        self.assertEqual(decision['skipped_reason'], 'boundary_not_roundtrip_exact')

    def test_old_prompt_not_advanced_is_skipped(self):
        class ShortRenderer(Renderer):
            def prepare_chat_prompt(self, messages, **kwargs):
                if kwargs['add_generation_prompt']:
                    return 'old', list(range(4500))
                return 'x'*5000 + '<|im_start|>user\n<|im_end|>\n', []

        history = [{'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': 'answer'}]
        _, ids, decision = aligned_next_turn_prefix(ShortRenderer(), history)
        self.assertIsNone(ids)
        self.assertEqual(decision['skipped_reason'], 'boundary_does_not_advance_previous_prompt')

    def test_boundary_skip_never_reads_key_or_contacts_server(self):
        history = [{'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': 'answer'}]
        with patch.dict('os.environ', {'QWEN_TRIAL_IDLE_PREFIX': '1', 'QWEN_TRIAL_IDLE_BOUNDARY': '1'}), \
             patch.object(prepare_idle_prefix, '_renderer', return_value=Renderer()), \
             patch.object(prepare_idle_prefix.urllib.request, 'build_opener', side_effect=AssertionError('network')):
            result = prepare_idle_prefix.prepare_idle_prefix(
                history, 'http://127.0.0.1:8011/v1', '/nonexistent-idle-cpu-key',
                chat_template_kwargs={'preserve_thinking': False})
        self.assertFalse(result['enabled'])
        self.assertEqual(result['warm_request_s'], 0.0)


if __name__ == '__main__':
    unittest.main()
