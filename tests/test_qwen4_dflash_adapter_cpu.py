"""Host-only checks of the native DFlash capture and transaction methods.

Execute the production methods with NumPy arrays and small fake layers. These
checks do not import MLX or initialise Metal; full-model qualification is separate.
"""

from __future__ import annotations

import ast
from contextvars import ContextVar
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/language.py"


def _production_class(name, base, namespace, methods):
    tree = ast.parse(SOURCE.read_text())
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    cls = ast.ClassDef(
        name=name,
        bases=[ast.Name(id="Base", ctx=ast.Load())],
        keywords=[],
        body=[node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in methods],
        decorator_list=[],
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls],
        type_ignores=[],
    )
    namespace = dict(namespace, Base=base)
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace[name]


def _write(hidden, branch, weights):
    return hidden + (branch[..., None, :] * weights[..., None]).reshape(hidden.shape)


class CaptureTests(unittest.TestCase):
    def test_completed_hc_mean_does_not_consume_deferred_write(self):
        for deferred in (False, True):
            with self.subTest(deferred=deferred):
                namespace = dict(
                    mx=SimpleNamespace(array=np.ndarray, tile=np.tile),
                    _create_qwen3_5_attention_mask=lambda *args: None,
                    q35_language=SimpleNamespace(_create_qwen3_5_ssm_mask=lambda *args: None),
                    hc_fused=SimpleNamespace(write_enabled=lambda: deferred),
                    _EAGER_DISPATCH=False,
                    _MTP_ONE_ROW_STEP=ContextVar("one_row", default=False),
                    _hc_write=_write,
                )
                cls = _production_class("Qwen4ExpModel", object, namespace, {"__call__"})
                model = cls()
                model.args = SimpleNamespace(hc_count=2, hidden_size=2)
                model.embed_tokens = lambda ids: np.stack([ids, ids + 1], axis=-1).astype(np.float32)
                model.ssm_idx, model.fa_idx = 0, 1
                completed_layers = []

                class Layer:
                    is_linear = False

                    def __call__(self, hidden, ids, *, write, defer_write, **kwargs):
                        hidden = _write(hidden, *write) if write is not None else hidden
                        branch = np.ones((*hidden.shape[:-1], 2), dtype=np.float32)
                        weights = np.broadcast_to(np.array([2, 4], dtype=np.float32), (*hidden.shape[:-1], 2))
                        completed_layers.append(_write(hidden, branch, weights))
                        return (hidden, branch, weights) if defer_write else completed_layers[-1]

                model.layers = [Layer(), Layer(), Layer()]
                mixer_calls = []

                def mixer(hidden, *, write=None, **kwargs):
                    mixer_calls.append(True)
                    completed = _write(hidden, *write) if write is not None else hidden
                    mixed = completed[..., :2] * 3 + completed[..., 2:]
                    return (mixed, completed) if write is not None else mixed

                model.hyper_connection_mixer = mixer
                inputs = np.array([[2, 3]])
                baseline = model(inputs, gdn_sink=[])
                completed_layers.clear()
                mixer_calls.clear()
                model._omlx_dflash_hc_mean = True
                captures = []
                actual = model(inputs, capture_layer_ids=[0, 2], hidden_sink=captures, gdn_sink=[])
                np.testing.assert_array_equal(actual, baseline)
                self.assertEqual(len(mixer_calls), 1)
                for captured, index in zip(captures, [0, 2]):
                    np.testing.assert_array_equal(captured, completed_layers[index].reshape(1, 2, 2, 2).mean(axis=-2))


class TransactionTests(unittest.TestCase):
    def setUp(self):
        self.transactions = []

        class Transaction:
            aborted = False

            def abort(self):
                self.aborted = True

        def start(cache, length):
            transaction = Transaction()
            self.transactions.append((cache, length, transaction))
            return transaction

        class Parent:
            def __call__(self, inputs, inputs_embeds, mask, cache, **kwargs):
                self.last_kwargs = kwargs
                if getattr(self, "fail", False):
                    raise RuntimeError("forward failed")
                captures = [f"capture-{i}" for i in kwargs.get("capture_layer_ids", [])]
                if kwargs.get("return_hidden"):
                    captures.append("head")
                return SimpleNamespace(hidden_states=captures, gdn_states=None)

        cls = _production_class(
            "LanguageModel", Parent,
            dict(start_speculative_cache=start, _MTP_ONE_ROW_STEP=ContextVar("one_row", default=False), os=os),
            {"__call__", "prepare_dflash_target"},
        )
        self.model = cls()
        self.model.model = SimpleNamespace(_omlx_dflash_hc_mean=True)
        self.cache = [SimpleNamespace()]

    def test_capture_verify_starts_native_transaction(self):
        output = self.model(np.zeros((1, 4)), cache=self.cache, return_hidden=True, capture_layer_ids=[5, 15])
        self.assertEqual(output.hidden_states, ["capture-5", "capture-15", "head"])
        self.assertIs(output.gdn_states, self.transactions[0][2])
        self.assertEqual(self.transactions[0][1], 4)

    def test_one_row_capture_uses_normal_decode(self):
        output = self.model(np.zeros((1, 1)), cache=self.cache, return_hidden=True, capture_layer_ids=[5])
        self.assertIsNone(output.gdn_states)
        self.assertEqual(self.transactions, [])

    def test_prefill_capture_does_not_start_transaction(self):
        self.model(np.zeros((1, 20)), cache=self.cache, capture_layer_ids=[5])
        self.assertEqual(self.transactions, [])

    def test_failed_verify_aborts_transaction(self):
        self.model.fail = True
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            self.model(np.zeros((1, 4)), cache=self.cache, return_hidden=True, capture_layer_ids=[5])
        self.assertTrue(self.transactions[0][2].aborted)

    def test_ordinary_decode_observes_committed_token(self):
        observed = []
        self.model._omlx_drafter = SimpleNamespace(scope_uids=("request",), target_layer_ids=[5, 15], observe=lambda *args: observed.append(args))
        self.model(np.zeros((1, 1)), cache=self.cache)
        self.assertEqual(observed, [(("request",), ["capture-5", "capture-15"])])
        self.assertEqual(self.transactions, [])

    def test_native_mtp_retains_all_streams(self):
        self.model.model._omlx_dflash_hc_mean = False
        output = self.model(np.zeros((1, 4)), cache=self.cache, return_hidden=True)
        self.assertEqual(self.model.last_kwargs["capture_layer_ids"], [])
        self.assertEqual(output.hidden_states, ["head"])
        self.assertIsNotNone(output.gdn_states)

    def test_prepare_requires_explicit_trial_gate(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OMLX_QWEN4_DFLASH_TRIAL=1"):
                self.model.prepare_dflash_target()
        with patch.dict(os.environ, {"OMLX_QWEN4_DFLASH_TRIAL": "1"}):
            self.model.prepare_dflash_target()
        self.assertTrue(self.model.model._omlx_dflash_hc_mean)


class GateTests(unittest.TestCase):
    def test_only_vlm_target_is_allowed_during_trial(self):
        source = ROOT / "omlx/utils/model_loading.py"
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "dflash_batched_supported")
        namespace = dict(os=os, _DFLASH_BATCHED_MODEL_TYPES={"vlm": ("qwen3_5",)})
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(source), "exec"), namespace)
        supported = namespace["dflash_batched_supported"]
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(supported("qwen4_exp", "vlm"))
            self.assertTrue(supported("qwen3_5", "vlm"))
        with patch.dict(os.environ, {"OMLX_QWEN4_DFLASH_TRIAL": "1"}):
            self.assertTrue(supported("qwen4_exp", "vlm"))
            self.assertFalse(supported("qwen4_exp", "batched"))


if __name__ == "__main__":
    unittest.main()
