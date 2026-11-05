"""Tiny real-MLX observer feasibility check; never loads model weights."""

import hashlib
import json
import os
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), '/Users/mohameddarwiche/work/p2pit/development/QwenLocal/state/runtime/omlx-0.7.3-m4-max-r2-sealed-site']

import mlx.core as mx
from omlx.scheduler import Scheduler
from omlx.request import SamplingParams
from idle_state_probe import install

out = ROOT / 'idle-observer-smoke'
out.mkdir(exist_ok=True)
os.environ['QWEN_PROBE_DIR'] = str(out)
os.environ['QWEN_TRIAL_IDLE_STATE'] = '1'
scheduler = SimpleNamespace(_xtc_special_tokens=[], _model_suppress_tokens=[])
params = SamplingParams(temperature=1.0, top_p=0.9, top_k=20, seed=20261106)
request = SimpleNamespace(request_id='observer-smoke', prompt_token_ids=[7, 11])
ordinary, ordinary_processors = Scheduler._build_sampler_and_processors(scheduler, params, request)
hooks = install()
value = hooks['value']

cases = [
    ('float32', mx.array([1.0, -2.0], dtype=mx.float32), struct.pack('<ff', 1, -2)),
    ('float16', mx.array([1.0, -2.0], dtype=mx.float16), struct.pack('<ee', 1, -2)),
    ('bfloat16', mx.array([1.0, -2.0], dtype=mx.bfloat16), struct.pack('<HH', 0x3f80, 0xc000)),
    ('int32', mx.array([1, -2], dtype=mx.int32), struct.pack('<ii', 1, -2)),
    ('uint32', mx.array([1, 2], dtype=mx.uint32), struct.pack('<II', 1, 2)),
    ('bool', mx.array([True, False]), bytes([1, 0])),
    ('scalar', mx.array(3, dtype=mx.int32), struct.pack('<i', 3)),
    ('strided', mx.array([1, 9, 2, 9], dtype=mx.int32)[::2], struct.pack('<ii', 1, 2)),
    ('empty', mx.array([], dtype=mx.float32), b''),
]
checked = []
for name, array, expected in cases:
    result = value(array)
    assert result['sha256'] == hashlib.sha256(expected).hexdigest(), (name, result)
    checked.append(name)
assert value({'x': [None, True, 2, 1.5, 'text', (3,)]}) == {'x': [None, True, 2, 1.5, 'text', [3]]}
assert len(value(list(mx.random.state))) == len(mx.random.state)
fake_cache = SimpleNamespace(state=[cases[0][1], None, cases[2][1], cases[3][1]],
                             meta_state=(None, 7, True), offset=2)
assert hooks['cache_state']([fake_cache])[0]['offset'] == 2

observed, observed_processors = Scheduler._build_sampler_and_processors(scheduler, params, request)
assert ordinary_processors == observed_processors
for attr in ('temp', 'top_p', 'min_p', 'top_k', 'min_tokens_to_keep', 'xtc_probability', '_mtp_batch_sampling_key'):
    assert getattr(ordinary, attr) == getattr(observed, attr), attr
assert callable(observed.sample_with_logprobs)
assert callable(observed._mtp_sampling_logits)
lp = mx.array([[i / 17.0 for i in range(128)]], dtype=mx.float32)
lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
mx.eval(lp)

def draw(sampler):
    mx.random.seed(params.seed)
    before = [key.tolist() for key in mx.random.state]
    tokens = []
    for _ in range(8):
        result = sampler(lp)
        mx.eval(result)
        tokens.append(result.tolist())
    return before, tokens, [key.tolist() for key in mx.random.state]

baseline = draw(ordinary)
instrumented = draw(observed)
assert baseline == instrumented, (baseline, instrumented)
records = json.loads((out / 'idle-state.json').read_text())['events']
sample = next(event for event in records if event['kind'] == 'first_sample')
loaded = mx.load(out / sample['logprobs_file'])
assert bool(mx.array_equal(loaded, lp).item())
assert sample['sampled_token_ids'] == baseline[1][0]
assert sample['rng_before'] == value([mx.array(key, dtype=mx.uint32) for key in baseline[0]])
summary = {'model_loaded': False, 'array_hash_checks': checked, 'nested_leaves': True,
           'random_state_sequence': True, 'sampler_metadata_preserved': True,
           'eight_tokens_exact': True, 'rng_before_after_exact': True,
           'saved_logprobs_exact': True, 'active_mlx_bytes': mx.get_active_memory()}
(out / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary))
