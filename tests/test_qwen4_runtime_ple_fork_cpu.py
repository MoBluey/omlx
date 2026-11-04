"""Host-only real-fork regressions for the vendored runtime PLE reader."""

import ast
import json
import logging
import math
import mmap
import os
import select
import signal
import struct
import tempfile
import threading
import time
import types
import unittest
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError, wait
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/language.py"
)


class NoMLX:
    def __getattr__(self, name):
        raise AssertionError("MLX touched: " + name)


def load_runtime():
    tree = ast.parse(SOURCE.read_text())
    selected = []
    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and (
                node.name.startswith("_ple_")
                or node.name in {"_SafeTensorMMap", "DiskBackedShardedEmbedding"}
            )
            or isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name)
                and t.id.startswith("_PLE_")
                and t.id not in {"_PLE_RUNTIME_MODEL_PATH", "_PLE_RUNTIME_MODE"}
                for t in node.targets
            )
            or isinstance(node, ast.If)
            and "register_at_fork" in ast.unparse(node.test)
        ):
            selected.append(node)
    namespace = dict(
        os=os,
        Lock=threading.Lock,
        RLock=threading.RLock,
        ThreadPoolExecutor=ThreadPoolExecutor,
        Path=Path,
        struct=struct,
        json=json,
        mmap=mmap,
        np=np,
        math=math,
        time=time,
        wait=wait,
        register_ple_resource=lambda *a, **k: None,
        logger=logging.getLogger(__name__),
        mx=NoMLX(),
        nn=types.SimpleNamespace(Module=object),
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            )
        ]
        + selected,
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace


NS = load_runtime()


def fork_check(callback, timeout=3):
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        try:
            callback()
            result = b"ok"
        except BaseException as exc:
            result = repr(exc).encode()
        os.write(write_fd, result)
        os._exit(0)
    os.close(write_fd)
    try:
        if not select.select([read_fd], [], [], timeout)[0]:
            os.kill(pid, signal.SIGKILL)
            raise AssertionError("fork child timed out")
        result = os.read(read_fd, 8192)
        if result != b"ok":
            raise AssertionError(result.decode())
    finally:
        os.close(read_fd)
        os.waitpid(pid, 0)


class RuntimePLEForkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "rows.safetensors"
        self.values = np.arange(64, dtype=np.float32).reshape(16, 4)
        header = json.dumps(
            {
                "weight": {
                    "shape": [16, 4],
                    "dtype": "F32",
                    "data_offsets": [0, self.values.nbytes],
                }
            }
        ).encode()
        self.path.write_bytes(
            struct.pack("<Q", len(header)) + header + self.values.tobytes()
        )
        self.reader = NS["_SafeTensorMMap"](self.path)

    def tearDown(self):
        self.reader.close()
        self.temp.cleanup()

    def test_selected_rows_own_storage_after_reader_close(self):
        for indices in ([], [3], [7, 2, 7, 0]):
            with self.subTest(indices=indices):
                copied, dtype = self.reader.rows_np("weight", indices)
                self.assertEqual(dtype, "F32")
                self.assertTrue(copied.flags.owndata)
                np.testing.assert_array_equal(copied, self.values[indices])
                if indices:
                    copied[0, 0] = -1
                    np.testing.assert_array_equal(
                        self.reader.rows_np("weight", indices)[0], self.values[indices]
                    )
        copied, _ = self.reader.rows_np("weight", [7, 2, 7, 0])
        self.reader.close()
        np.testing.assert_array_equal(copied, self.values[[7, 2, 7, 0]])

    def test_page_batches_bound_tasks_and_retry_short_reads(self):
        pages = np.arange(4 * NS["_PLE_IO_WORKERS"] + 5)
        page_size = NS["_PLE_PAGE_SIZE"]
        self.reader._seen_pages = bytearray(len(pages))
        reads = []

        def pread(fd, size, offset):
            count = min(size, page_size // 2)
            reads.append((offset, count))
            return b"x" * count

        pool = MagicMock(wraps=NS["_PLE_IO_POOL"])
        with patch.dict(NS, {"_PLE_IO_POOL": pool}), patch.object(os, "pread", pread):
            self.assertFalse(self.reader._prefetch_missing_pages(pages, 0, page_size))
            self.assertEqual(pool.submit.call_count, NS["_PLE_IO_WORKERS"])
            self.assertEqual(
                sorted(reads),
                [
                    (offset, page_size // 2)
                    for offset in range(0, len(pages) * page_size, page_size // 2)
                ],
            )
            self.assertEqual(self.reader._seen_pages, bytes([1]) * len(pages))
            pool.submit.reset_mock()
            self.assertTrue(self.reader._prefetch_missing_pages(pages, 0, page_size))
            pool.submit.assert_not_called()

    def test_failed_page_batch_leaves_pages_unmarked(self):
        pages = np.arange(2 * NS["_PLE_IO_WORKERS"])
        self.reader._seen_pages = bytearray(len(pages))

        def pread(fd, size, offset):
            if offset == 0:
                raise OSError("injected batch failure")
            return b"x" * size

        with (
            patch.object(os, "pread", pread),
            self.assertRaisesRegex(OSError, "injected batch failure"),
        ):
            self.reader._prefetch_missing_pages(pages, 0, NS["_PLE_PAGE_SIZE"])
        self.assertFalse(any(self.reader._seen_pages))

    def embedding(self):
        e = object.__new__(NS["DiskBackedShardedEmbedding"])
        e._owner_pid = os.getpid()
        e._prefetch_lock = threading.Lock()
        e._prefetch_closed = False
        e._reuse_indices = None
        e._pending = {b"pending": (None, Future())}
        e._prefetch_executor = ThreadPoolExecutor(max_workers=1)
        e._readers = {"rows": self.reader}
        e._tensor_readers = {"weight": self.reader}
        e._shard_specs = {}
        return e

    def test_inherited_locks_future_reject_before_mlx_and_child_close(self):
        e = self.embedding()
        future = e._pending[b"pending"][1]
        e._prefetch_lock.acquire()
        self.reader._resource_lock.acquire()

        def child():
            # Negative controls demonstrate the inherited hazards are real.
            self.assertFalse(e._prefetch_lock.acquire(timeout=0.03))
            with self.assertRaises(TimeoutError):
                future.result(timeout=0.03)
            for call in (
                lambda: e.prefetch(object()),
                lambda: e.prefetch_reuse(object()),
                lambda: e.clear_index_reuse(drain=True),
                lambda: NS["_ple_prepare_decode"](
                    [object()], types.SimpleNamespace(shape=(1, 1)), [object()]
                ),
                lambda: e(object()),
                lambda: self.reader.rows_np("weight", [0]),
                lambda: NS["_SafeTensorMMap"](self.path),
                lambda: NS["DiskBackedShardedEmbedding"]("", "", 16, 4, 1),
            ):
                with self.assertRaisesRegex(RuntimeError, "after fork"):
                    call()
            errors = []

            def close():
                try:
                    e.close()
                    self.reader.close()
                except BaseException as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=close) for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(0.5)
            self.assertFalse(any(thread.is_alive() for thread in threads))
            self.assertEqual(errors, [])
            e.close()

        try:
            fork_check(child)
        finally:
            self.reader._resource_lock.release()
            e._prefetch_lock.release()
        np.testing.assert_array_equal(
            self.reader.rows_np("weight", [2])[0], self.values[[2]]
        )
        e.close()

    def test_exact_index_ticket_is_one_use_and_mismatch_evaluates_again(self):
        e = self.embedding()
        e._pending.clear()
        e._plan = lambda host: None
        e._gather_per_shard = lambda host, shape: (host, shape)
        first = types.SimpleNamespace(shape=(1, 2))
        equal_but_distinct = types.SimpleNamespace(shape=(1, 2))
        host = np.array([2, 7], dtype=np.int64)
        e._host_indices = MagicMock(return_value=host)
        try:
            e.prefetch_reuse(first)
            self.assertIs(e._reuse_indices[0], first)
            self.assertEqual(e(first), ([2, 7], (1, 2)))
            self.assertEqual(e._host_indices.call_count, 1)
            self.assertIsNone(e._reuse_indices)
            e(first)
            self.assertEqual(e._host_indices.call_count, 2)
            e.prefetch_reuse(first)
            e(equal_but_distinct)
            self.assertEqual(e._host_indices.call_count, 4)
            self.assertIsNone(e._reuse_indices)
        finally:
            e.close()

    def test_boundary_clear_drains_running_and_cancels_queued_rows(self):
        e = self.embedding()
        e._pending.clear()
        entered, release, cleared = (threading.Event() for _ in range(3))
        host = np.array([2, 7], dtype=np.int64)
        def read():
            entered.set()
            self.assertTrue(release.wait(2))
            return {"weight": host.copy()}
        running = e._prefetch_executor.submit(read)
        self.assertTrue(entered.wait(1))
        queued = e._prefetch_executor.submit(read)
        e._pending = {b"running": (None, running), host.tobytes(): (None, queued)}
        e._reuse_indices = (object(), host)
        thread = threading.Thread(target=lambda: (e.clear_index_reuse(drain=True), cleared.set()))
        thread.start()
        try:
            self.assertFalse(cleared.wait(0.03))
            release.set()
            thread.join(2)
            self.assertTrue(cleared.is_set())
            self.assertTrue(running.done())
            self.assertTrue(queued.cancelled())
            self.assertEqual(e._pending, {})
            self.assertIsNone(e._reuse_indices)
        finally:
            release.set()
            thread.join(2)
            e.close()

    def test_failed_reused_rows_do_not_leave_an_index_ticket(self):
        e = self.embedding()
        host = np.array([2, 7], dtype=np.int64)
        indices = types.SimpleNamespace(shape=(1, 2))
        failure = Future()
        failure.set_exception(OSError("injected row failure"))
        e._pending = {host.tobytes(): (None, failure)}
        e._reuse_indices = (indices, host)
        try:
            with self.assertRaisesRegex(OSError, "injected row failure"):
                e(indices)
            self.assertIsNone(e._reuse_indices)
            self.assertFalse(e._pending)
        finally:
            e.close()

    def test_initialized_global_pool_cannot_be_used_by_new_child_reader(self):
        self.assertEqual(NS["_PLE_IO_POOL"].submit(lambda: 7).result(timeout=1), 7)

        def child():
            with self.assertRaisesRegex(RuntimeError, "after fork"):
                NS["_SafeTensorMMap"](self.path)
            # Grandchildren also see a fresh cleanup mutex (callbacks resolve globals).
            fork_check(lambda: self.reader.close())

        fork_check(child)
        np.testing.assert_array_equal(
            self.reader.rows_np("weight", np.arange(16))[0], self.values
        )

    def test_close_waits_for_active_reader_without_blocking_other_reader(self):
        entered, release, closed = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        original = self.reader._rows_np_owned

        def slow(*args):
            entered.set()
            self.assertTrue(release.wait(2))
            return original(*args)

        self.reader._rows_np_owned = slow
        worker = threading.Thread(target=lambda: self.reader.rows_np("weight", [0]))
        worker.start()
        self.assertTrue(entered.wait(1))
        closer = threading.Thread(target=lambda: (self.reader.close(), closed.set()))
        closer.start()
        try:
            self.assertFalse(closed.wait(0.03))
            other = NS["_SafeTensorMMap"](self.path)
            np.testing.assert_array_equal(
                other.rows_np("weight", [1])[0], self.values[[1]]
            )
            other.close()
        finally:
            release.set()
            worker.join(2)
            closer.join(2)
        self.assertTrue(closed.is_set())
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.reader.rows_np("weight", [0])

    def test_exported_view_close_retry_and_descriptor_reuse(self):
        view = memoryview(self.reader._mapping)
        with self.assertRaises(BufferError):
            self.reader.close()
        self.assertIsNone(self.reader._file)
        self.assertIsNotNone(self.reader._mapping)
        with self.path.open("rb") as replacement:
            view.release()
            self.reader.close()
            self.reader.close()
            self.assertEqual(len(replacement.read(8)), 8)

    def test_child_exported_view_close_retry_does_not_close_reused_fd(self):
        e = self.embedding()
        view = memoryview(self.reader._mapping)

        def child():
            with self.assertRaises(BufferError):
                e.close()
            self.assertIsNone(self.reader._file)
            with self.path.open("rb") as replacement:
                view.release()
                e.close()
                e.close()
                self.assertEqual(len(replacement.read(8)), 8)

        try:
            fork_check(child)
        finally:
            view.release()
        np.testing.assert_array_equal(
            self.reader.rows_np("weight", [4])[0], self.values[[4]]
        )
        e.close()

    def test_resource_lock_allows_same_thread_fork_reentry(self):
        def child():
            with NS["_PLE_RESOURCE_LOCK"]:
                fork_check(lambda: self.reader.close())

        fork_check(child)
        np.testing.assert_array_equal(
            self.reader.rows_np("weight", [3])[0], self.values[[3]]
        )

    def test_failed_page_read_drains_peers_before_close_and_reraises(self):
        entered, release, finished, closed = (threading.Event() for _ in range(4))
        real_pread = os.pread
        errors = []

        def pread(fd, size, offset):
            if offset == 0:
                if not entered.wait(2):
                    raise AssertionError("peer did not start")
                raise OSError("injected first page failure")
            entered.set()
            if not release.wait(2):
                raise AssertionError("peer not released")
            return real_pread(fd, size, offset)

        # Exercise two page requests even though the second is beyond this tiny
        # fixture's EOF; pread still owns the descriptor until it returns.
        original = self.reader._rows_np_owned

        def read_pages(*args):
            self.reader._prefetch_missing_pages(
                np.array([0, 1]), 0, NS["_PLE_PAGE_SIZE"]
            )
            return original(*args)

        self.reader._rows_np_owned = read_pages
        self.reader._seen_pages = bytearray(2)

        def read():
            try:
                self.reader.rows_np("weight", [0])
            except BaseException as exc:
                errors.append(exc)
            finally:
                finished.set()

        os.pread = pread
        worker = threading.Thread(target=read)
        closer = threading.Thread(target=lambda: (self.reader.close(), closed.set()))
        try:
            worker.start()
            self.assertTrue(entered.wait(1))
            closer.start()
            self.assertFalse(finished.wait(0.03))
            self.assertFalse(closed.is_set())
            release.set()
            worker.join(2)
            closer.join(2)
            self.assertTrue(finished.is_set())
            self.assertTrue(closed.is_set())
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], OSError)
            self.assertEqual(str(errors[0]), "injected first page failure")
        finally:
            release.set()
            worker.join(2)
            if closer.ident is not None:
                closer.join(2)
            os.pread = real_pread

    def test_prepare_uses_each_layers_live_history_without_mutation(self):
        calls = []
        histories = [np.array([[2, 3]]), np.array([[7, 8]])]
        caches = [[None, None, None, history] for history in histories]
        tokens = np.array([[4, 5]])
        tables = [self.embedding(), self.embedding()]
        def layer(index):
            embedding = types.SimpleNamespace(
                ngram_embedding=tables[index],
                prepare_index_reuse=lambda ids, history: calls.append((index, ids, history)),
            )
            return types.SimpleNamespace(ple=types.SimpleNamespace(ple_embedding=embedding))
        layers = [layer(0), layer(1)]
        try:
            NS["_ple_prepare_decode"](layers, tokens, caches)
            self.assertEqual([index for index, _, _ in calls], [0, 1])
            for index, ids, history in calls:
                self.assertIs(ids, tokens)
                self.assertIs(history, histories[index])
                self.assertIs(caches[index][3], histories[index])
            restored = np.array([[3, 4]])
            caches[0][3] = restored
            NS["_ple_prepare_decode"](layers, tokens[:, :1], caches)
            self.assertIs(calls[2][2], restored)
        finally:
            for table in tables:
                table.close()

    def test_prepare_skips_prefill_resident_and_missing_history(self):
        def fail(*args):
            self.fail("unexpected preparation")
        table = self.embedding()
        def layer(inner):
            return types.SimpleNamespace(ple=types.SimpleNamespace(
                ple_embedding=types.SimpleNamespace(ngram_embedding=inner, prepare_index_reuse=fail)
            ))
        prepare = NS["_ple_prepare_decode"]
        try:
            prepare([layer(table)], np.zeros((1, 1)), None)
            prepare([layer(table)], np.zeros((1, 65)), [[None, None, None, object()]])
            prepare([layer(table)], np.zeros((1, 1)), [[None] * 4])
            prepare([layer(table)], np.zeros((1, 1)), [None])
            prepare([layer(object())], np.zeros((1, 1)), [[None, None, None, object()]])
            prepare([object()], np.zeros((1, 1)), [object()])
        finally:
            table.close()

    def test_constructor_failure_closes_file(self):
        bad = Path(self.temp.name) / "bad.safetensors"
        bad.write_bytes(b"no")
        with self.assertRaises(struct.error):
            NS["_SafeTensorMMap"](bad)


if __name__ == "__main__":
    unittest.main()
