"""Exact raw-row checks for the opt-in prompt assembly trial, without MLX."""

import json
import struct
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import test_qwen4_runtime_ple_fork_cpu as runtime


class PrefillDedupTests(unittest.TestCase):
    def embedding(self, directory, affine):
        tensors, specs, cursor, data = {}, {}, 0, []
        random = np.random.default_rng(42)
        for shard, rows in enumerate((11, 9)):
            keys = []
            for family in range(3 if affine else 1):
                key = f"shard{shard}.family{family}"
                width = 8 if family == 0 else 2
                dtype = "U32" if affine and family == 0 else "BF16"
                values = random.integers(0, 65536, (rows, width), dtype=np.uint32)
                values = values.astype("<u4" if dtype == "U32" else "<u2")
                raw = values.tobytes()
                tensors[key] = {"dtype": dtype, "shape": [rows, width],
                                "data_offsets": [cursor, cursor + len(raw)]}
                cursor += len(raw)
                data.append(raw)
                keys.append(key)
            specs[shard] = tuple(keys) + ((5, 32) if affine else (None,) * 4)
        header = json.dumps(tensors).encode()
        path = Path(directory) / "rows.safetensors"
        path.write_bytes(struct.pack("<Q", len(header)) + header + b"".join(data))
        reader = runtime.NS["_SafeTensorMMap"](path)
        embedding = object.__new__(runtime.NS["DiskBackedShardedEmbedding"])
        embedding.shard_offsets = (0, 11, 20)
        embedding._shard_specs = specs
        embedding._tensor_readers = dict.fromkeys(tensors, reader)
        embedding._prefill_dedup_lock = threading.Lock()
        embedding.prefill_dedup_stats = dict(calls=0, rows=0, unique_rows=0, failures=0)
        self.addCleanup(reader.close)
        return embedding, reader

    def test_raw_dense_and_affine_rows_restore_exact_order(self):
        with tempfile.TemporaryDirectory() as directory:
            for affine in (False, True):
                with self.subTest(affine=affine):
                    embedding, reader = self.embedding(directory, affine)
                    host = np.array([19, 0, 11, 4, 19, 4, 0, 17, 11, 19], np.int64)
                    plan = embedding._plan(host)
                    baseline = embedding._assemble(host, plan)
                    with patch.object(reader, "rows_np", wraps=reader.rows_np) as reads:
                        candidate = embedding._assemble_prefill_deduplicated(host, plan)
                    for family, values in baseline.items():
                        np.testing.assert_array_equal(candidate[family], values)
                        self.assertTrue(candidate[family].flags.owndata)
                    expected = np.unique(host).size * (3 if affine else 1)
                    self.assertEqual(sum(len(call.args[1]) for call in reads.call_args_list), expected)
                    self.assertEqual(embedding.prefill_dedup_stats,
                                     dict(calls=1, rows=10, unique_rows=5, failures=0))
                    reader.close()
                    for family, values in baseline.items():
                        np.testing.assert_array_equal(candidate[family], values)

    def test_unique_rows_use_existing_assembly_and_failure_propagates(self):
        with tempfile.TemporaryDirectory() as directory:
            embedding, _ = self.embedding(directory, False)
            host = np.array([19, 0, 11, 4], np.int64)
            plan = embedding._plan(host)
            with patch.object(embedding, "_assemble", wraps=embedding._assemble) as assemble:
                embedding._assemble_prefill_deduplicated(host, plan)
                self.assertIs(assemble.call_args.args[0], host)
                self.assertIs(assemble.call_args.args[1], plan)
            error = OSError("injected row failure")
            with patch.object(embedding, "_assemble", side_effect=error):
                with self.assertRaises(OSError) as caught:
                    embedding._assemble_prefill_deduplicated(host, plan)
            self.assertIs(caught.exception, error)
            self.assertEqual(embedding.prefill_dedup_stats["failures"], 1)

    def test_switch_only_selects_prompt_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            embedding, _ = self.embedding(directory, True)
            with patch.dict(runtime.NS, {"_PLE_PREFILL_DEDUP_TRIAL": False}):
                self.assertEqual(embedding._assembler(types.SimpleNamespace(shape=(1, 4096, 16))),
                                 embedding._assemble)
            with patch.dict(runtime.NS, {"_PLE_PREFILL_DEDUP_TRIAL": True}):
                for shape in ((32,), (1, 1, 16), (1, 2, 16)):
                    self.assertEqual(embedding._assembler(types.SimpleNamespace(shape=shape)),
                                     embedding._assemble)
                self.assertEqual(embedding._assembler(types.SimpleNamespace(shape=(1, 3, 16))),
                                 embedding._assemble_prefill_deduplicated)
