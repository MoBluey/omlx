"""Private diagnostics installed by the benchmark harness's trial_hooks.install.

No serving patch. Shapes and native-clock phase windows are cheap. Optional
expert-tensor capture is a separate diagnostic request, never timing evidence.
"""
import contextlib
import ctypes
import functools
import json
import os
from pathlib import Path
import threading
import time


def install():
    import mlx.core as mx
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear
    from mlx_vlm.models.switch_layers import QuantizedSwitchLinear as VLMQuantizedSwitchLinear
    from omlx.scheduler import Scheduler
    from omlx.models.vlm import VLMModelAdapter

    out = Path(os.environ["QWEN_PROBE_DIR"])
    lock = threading.Lock()
    signatures = set()
    held = []
    capture = os.environ.get("QWEN_CAPTURE_EXPERT_INPUTS") == "1"
    capture_bits = int(os.environ["QWEN_CAPTURE_EXPERT_BITS"]) if os.environ.get("QWEN_CAPTURE_EXPERT_BITS") else None
    captured = False
    active = False
    full_chunks = 0
    native = ctypes.CDLL(os.environ["QWEN_METAL_PROFILE_LIBRARY"])
    native.qwen_metal_profile_enable.argtypes = [ctypes.c_int]
    native.qwen_metal_profile_enable.restype = None
    adapter_call = VLMModelAdapter.__call__

    @functools.wraps(adapter_call)
    def adapter(self, input_ids, *args, **kwargs):
        nonlocal active, full_chunks
        if input_ids.ndim == 2 and input_ids.shape[1] == 4096:
            full_chunks += 1
            if full_chunks == 2:
                mx.synchronize()
                active = True
                native.qwen_metal_profile_enable(1)
                append("native-profile-window.jsonl", {"event": "start",
                    "clock": time.monotonic(), "chunk": full_chunks,
                    "tokens": 4096, "capture_tensors": capture})
        return adapter_call(self, input_ids, *args, **kwargs)

    VLMModelAdapter.__call__ = adapter

    def append(name, record):
        with lock, (out / name).open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    original_phase = Scheduler._phase_timer

    @contextlib.contextmanager
    def phase(self, name):
        started = time.monotonic()
        try:
            with original_phase(self, name):
                yield
        finally:
            append("native-phase-windows.jsonl", {"phase": name,
                "start": started, "end": time.monotonic()})

    Scheduler._phase_timer = phase
    def capture_linear(original_linear, self, x, indices, sorted_indices=False):
        nonlocal captured
        signature = (tuple(x.shape), tuple(indices.shape), tuple(self.weight.shape),
                     str(x.dtype), self.bits, self.group_size, self.mode, bool(sorted_indices))
        if indices.size >= 64 and signature not in signatures:
            signatures.add(signature)
            append("expert-shapes.jsonl", {"x": x.shape, "indices": indices.shape,
                "weight": self.weight.shape, "scales": self.scales.shape,
                "dtype": str(x.dtype), "bits": self.bits,
                "group_size": self.group_size, "mode": self.mode,
                "class_module": type(self).__module__,
                "sorted_indices": bool(sorted_indices)})
        if (capture and active and not captured and len(held) < 2 and indices.size >= 64
                and (capture_bits is None or self.bits == capture_bits)):
            tensors = {"x": x, "indices": indices, "weight": self.weight,
                       "scales": self.scales}
            if self.get("biases") is not None:
                tensors["biases"] = self.biases
            if self.get("global_scale") is not None:
                tensors["global_scale"] = self.global_scale
            held.append((tensors, {"bits": self.bits, "group_size": self.group_size,
                "mode": self.mode, "sorted_indices": bool(sorted_indices)}))
        return original_linear(self, x, indices, sorted_indices)

    def wrap_linear(cls):
        original_linear = cls.__call__

        @functools.wraps(original_linear)
        def linear(self, x, indices, sorted_indices=False):
            return capture_linear(original_linear, self, x, indices, sorted_indices)

        cls.__call__ = linear

    for cls in (QuantizedSwitchLinear, VLMQuantizedSwitchLinear):
        wrap_linear(cls)
    original_record = Scheduler._record_chunk_transient

    @functools.wraps(original_record)
    def record(self, *args, **kwargs):
        nonlocal captured, active
        result = original_record(self, *args, **kwargs)
        if active:
            mx.synchronize()
            native.qwen_metal_profile_enable(0)
            append("native-profile-window.jsonl", {"event": "end", "clock": time.monotonic(),
                "recorded_tokens": args[0] if args else kwargs.get("n_tokens")})
            active = False
        if held and not captured:
            captured = True
            started = time.monotonic()
            for index, (tensors, metadata) in enumerate(held):
                mx.save_safetensors(str(out / f"expert-{index}.safetensors"), tensors)
                (out / f"expert-{index}.json").write_text(json.dumps(metadata, indent=2))
            append("expert-capture.jsonl", {"tensors": len(held),
                "seconds": time.monotonic() - started, "diagnostic_only": True})
            held.clear()
        return result

    Scheduler._record_chunk_transient = record
