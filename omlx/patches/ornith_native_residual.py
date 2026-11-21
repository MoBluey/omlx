"""Fixed-factor private Ornith experiment. No adapter discovery or code loading."""
import hashlib
import json
import logging
import os
from pathlib import Path

FACTOR_SHA = "b9452a4691e511a12de1c30ee6756a93754e82bc4aa9291cae7778bf1ef4faef"
KEY = 'language_model.model.layers.39.self_attn.o_proj'
MODEL_NAME = "Huihui-Ornith-1.5-35B-A3B-abliterated-oQ5e-fp16-mtp"
PAYLOAD_SHA = {
    "weight": "ec59ec8be0eeaf16635b70567004ec4b43d69157f00f2a95ec25c6f17b830f06",
    "scales": "fc0dfe3ff108f39745fe3c10c0d671e9b991891b0bf1dd9d3abb9794c027875c",
    "biases": "262d267bbfeecd133ffcad115ec86b2caa499795476addef8b763cb8e17a8ee9",
}
LOG = logging.getLogger(__name__)


def install(model, model_path):
    mode = os.environ.get("OMLX_PRIVATE_ORNITH_RESIDUAL")
    if mode is None:
        return
    if mode not in {"zero", "half"}:
        raise RuntimeError("Invalid private residual mode")
    expected = os.environ.get("OMLX_PRIVATE_ORNITH_RESIDUAL_MODEL")
    path = Path(model_path).resolve(strict=True)
    if not expected or path != Path(expected).resolve(strict=True) or path.name != MODEL_NAME:
        raise RuntimeError("Private residual requires its exact owned Ornith instance")
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from mlx.utils import tree_flatten
    from . import qwen35_verify_qmm

    factor_path = Path(__file__).resolve().parents[2] / "private-ornith-residual.safetensors"
    if hashlib.sha256(factor_path.read_bytes()).hexdigest() != FACTOR_SHA:
        raise RuntimeError("Private residual factor identity failed")
    factors = mx.load(str(factor_path))
    if set(factors) != {KEY + ".lora_a", KEY + ".lora_b"}:
        raise RuntimeError("Unexpected private factor keys")
    a, b = factors[KEY + ".lora_a"], factors[KEY + ".lora_b"]
    mx.eval(a, b)
    if not bool(mx.all(mx.isfinite(a))) or not bool(mx.all(mx.isfinite(b))):
        raise RuntimeError("Nonfinite private factor")
    attention = model.language_model.model.layers[39].self_attn
    base = attention.o_proj
    physical = {name: hashlib.sha256(np.asarray(getattr(base, name)).tobytes()).hexdigest()
                for name in PAYLOAD_SHA}
    if physical != PAYLOAD_SHA:
        raise RuntimeError("Native base projection identity failed")
    before = dict(tree_flatten(model.parameters()))
    def report(event, wrapper):
        LOG.info("PRIVATE_ORNITH_RESIDUAL %s", json.dumps({
            "event": event, "mode": mode, "factor_sha256": FACTOR_SHA,
            "native_projection_sha256": physical,
            "hook_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "engine": "VLMBatchedEngine",
            "ordinary_calls": wrapper.ordinary_calls,
            "verify_calls": wrapper.verify_calls,
            "verify_two_row_calls": wrapper.verify_two_row_calls,
        }, sort_keys=True))
    def compare_zero_projection(x, y):
        # Same real armed two-row input, same original module, stock verifier
        # dispatch. This is projection fidelity, not a whole-backbone claim.
        from mlx_vlm.speculative.ops.linear import _target_verify_linear

        reference = _target_verify_linear(base, x)
        mx.eval(reference, y)
        actual, expected = np.asarray(y), np.asarray(reference)
        if (actual.shape != expected.shape or actual.dtype != expected.dtype
                or not np.isfinite(actual).all()
                or actual.tobytes() != expected.tobytes()):
            raise RuntimeError("Native zero projection differs from stock verifier dispatch")
        LOG.info("PRIVATE_ORNITH_RESIDUAL %s", json.dumps({
            "event": "zero_projection_equal", "mode": mode, "factor_sha256": FACTOR_SHA,
            "shape": list(actual.shape), "dtype": str(actual.dtype),
            "output_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
            "reference_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
            "scope": "first genuine armed two-row projection, same input and stock dispatcher",
            "whole_verifier_equivalence": False,
        }, sort_keys=True))

    wrapper = make_projection(base, a, b, active=mode == "half", mx=mx, nn=nn,
        verify_state=lambda: qwen35_verify_qmm._is_armed() or qwen35_verify_qmm.is_row_exact_armed(),
        report=report, compare_zero=compare_zero_projection if mode == "zero" else None)
    attention.o_proj = wrapper
    after = dict(tree_flatten(model.parameters()))
    normalised = {key.replace(KEY + ".linear.", KEY + "."): value for key, value in after.items()
                  if key not in {KEY + ".lora_a", KEY + ".lora_b"}}
    if set(normalised) != set(before) or any(normalised[key] is not value for key, value in before.items()):
        attention.o_proj = base
        raise RuntimeError("Private residual changed another parameter")
    report("loaded", wrapper)


def report_stop(model):
    if not os.environ.get("OMLX_PRIVATE_ORNITH_RESIDUAL") or model is None:
        return
    try:
        wrapper = model.language_model.model.layers[39].self_attn.o_proj
    except (AttributeError, IndexError):
        return
    if not hasattr(wrapper, "verify_two_row_calls"):
        return
    LOG.info("PRIVATE_ORNITH_RESIDUAL %s", json.dumps({
        "event": "stopped", "mode": os.environ["OMLX_PRIVATE_ORNITH_RESIDUAL"],
        "factor_sha256": FACTOR_SHA, "ordinary_calls": wrapper.ordinary_calls,
        "verify_calls": wrapper.verify_calls, "verify_two_row_calls": wrapper.verify_two_row_calls,
    }, sort_keys=True))


def make_projection(base, a, b, *, active, mx, nn, verify_state, report=lambda *args: None, compare_zero=None):
    assert type(active) is bool
    assert isinstance(base, nn.QuantizedLinear)
    assert base.bits == 8 and base.group_size == 64 and base.mode == 'affine'
    assert base.weight.shape == (2048, 1024)
    assert base.scales.shape == base.biases.shape == (2048, 64)
    assert a.shape == (4096, 4) and b.shape == (4, 2048)
    assert a.dtype == b.dtype == mx.float32
    class Residual(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = base
            self.lora_a, self.lora_b = a, b
            self.ordinary_calls = 0
            self.verify_calls = 0
            self.verify_two_row_calls = 0
        def __call__(self, x):
            if verify_state():
                self.verify_calls += 1
                if x.shape[-2] == 2:
                    self.verify_two_row_calls += 1
                if self.verify_calls == 1 or (x.shape[-2] == 2 and self.verify_two_row_calls == 1):
                    report("verify", self)
            else:
                self.ordinary_calls += 1
                if self.ordinary_calls == 1:
                    report("ordinary", self)
            y = self.linear(x)
            if not active:
                if compare_zero is not None and verify_state() and self.verify_two_row_calls == 1 and x.shape[-2] == 2:
                    compare_zero(x, y)
                return y  # Exact same native result object; no zero-add/rounding.
            delta = ((x.astype(mx.float32) @ self.lora_a) @ self.lora_b) * 0.5
            return y + delta.astype(y.dtype)
    result = Residual()
    result.freeze()
    assert result.linear is base
    return result
