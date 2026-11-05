"""CPU/stdlib comparison of real production idle-prefix audit records."""

import argparse
from array import array
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from idle_state_probe import _different


def load_npy(path):
    with path.open('rb') as f:
        assert f.read(6) == b'\x93NUMPY'
        major, minor = f.read(2)
        size = struct.unpack('<H' if major == 1 else '<I', f.read(2 if major == 1 else 4))[0]
        header = ast.literal_eval(f.read(size).decode('latin1'))
        # MLX saves a (1,V) row with the Fortran flag; both orders have the
        # same bytes when at most one dimension has extent greater than one.
        assert header['descr'] == '<f4', header
        assert not header['fortran_order'] or sum(n > 1 for n in header['shape']) <= 1, header
        values = array('f')
        values.frombytes(f.read())
        if sys.byteorder != 'little':
            values.byteswap()
        assert len(values) == math.prod(header['shape'])
        return values


def logical_text_positions(state, n):
    """Normalise only byte-verified sequential text MRoPE broadcast planes."""
    result = deepcopy(state)
    raw = array('i', range(n))
    if sys.byteorder != 'little':
        raw.byteswap()
    one = hashlib.sha256(raw.tobytes()).hexdigest()
    three = hashlib.sha256(raw.tobytes()*3).hexdigest()
    normalised = []
    for i, layer in enumerate(result):
        if layer['class'] != 'QSAKVCache':
            continue
        leaf = layer['state'][3]
        if leaf['dtype'] == 'mlx.core.int32' and (
            (leaf['shape'] == [1, n] and leaf['sha256'] == one)
            or (leaf['shape'] == [3, 1, n] and leaf['sha256'] == three)
        ):
            layer['state'][3] = {'verified_text_positions': n}
            normalised.append(i)
    return result, normalised


def restore_summary(events):
    summary = []
    for event in events:
        if event['kind'] == 'trunk_restore':
            capture = next(x for x in events if x['kind'] in ('tail_capture', 'boundary_capture')
                and x['request_id'] == event['capture_request_id'] and x['tokens'] == event['tokens'])
            a, a_positions = logical_text_positions(capture['state'], event['tokens'])
            b, b_positions = logical_text_positions(event['state'], event['tokens'])
            differences = _different(a, b)
            summary.append({'kind': event['kind'], 'tokens': event['tokens'],
                'capture_kind': capture['kind'],
                'prefix_sha256': event['prefix_sha256'], 'raw_exact': event['exact'],
                'logical_exact': not differences, 'logical_differences': differences,
                'verified_position_layers': a_positions if a_positions == b_positions else []})
        elif event['kind'] == 'head_restore':
            summary.append({k: event.get(k) for k in ('kind', 'tokens', 'prefix_sha256',
                'sidecar_found', 'restored', 'exact', 'positions_exact', 'folded', 'expected_offset')})
    return summary


def compare(baseline, prepared):
    a_events = json.loads((baseline / 'idle-state.json').read_text())['events']
    b_events = json.loads((prepared / 'idle-state.json').read_text())['events']
    a_final = {e['prompt_sha256']: e for e in a_events if e['kind'] == 'final_prefill'}
    b_final = {e['prompt_sha256']: e for e in b_events if e['kind'] == 'final_prefill'}
    pairs = []
    for key in sorted(a_final.keys() & b_final.keys(), key=lambda k: a_final[k]['prompt_tokens']):
        a, b = a_final[key], b_final[key]
        differences = _different(a['state'], b['state'])
        head_differences = _different(a['head'], b['head'])
        changed_layers = sorted(set(int(re.match(r'\[(\d+)\]', p).group(1)) for p in differences))
        a_sample = next(e for e in a_events if e['kind'] == 'first_sample' and e['request_id'] == a['request_id'])
        b_sample = next(e for e in b_events if e['kind'] == 'first_sample' and e['request_id'] == b['request_id'])
        x = load_npy(baseline / a_sample['logprobs_file'])
        y = load_npy(prepared / b_sample['logprobs_file'])
        finite = [(u-v) for u, v in zip(x, y) if math.isfinite(u) and math.isfinite(v)]
        a_top = sorted(range(len(x)), key=x.__getitem__, reverse=True)[:10]
        b_top = sorted(range(len(y)), key=y.__getitem__, reverse=True)[:10]
        pairs.append({'prompt_sha256': key, 'prompt_tokens': a['prompt_tokens'],
            'baseline_cached': a['cached_tokens'], 'prepared_cached': b['cached_tokens'],
            'prefilled_tokens_equal': a['prefilled_tokens'] == b['prefilled_tokens'],
            'last_token_equal': a['last_tokens'] == b['last_tokens'],
            'trunk_exact': not differences, 'trunk_differences': differences,
            'changed_layers': changed_layers,
            'changed_layer_types': dict(Counter(a['state'][i]['class'] for i in changed_layers)),
            'head_exact': not head_differences, 'head_differences': head_differences,
            'baseline_head_positions': None if a['head'] is None else {k:a['head'][k] for k in ('folded','expected_offset','valid')},
            'prepared_head_positions': None if b['head'] is None else {k:b['head'][k] for k in ('folded','expected_offset','valid')},
            'sampling_logprobs_exact': a_sample['sampling_logprobs'] == b_sample['sampling_logprobs'],
            'logprobs': {'elements':len(x), 'equal':sum(u == v for u,v in zip(x,y)),
                'max_abs_delta':max(abs(d) for d in finite),
                'mean_abs_delta':sum(abs(d) for d in finite)/len(finite),
                'rmse':math.sqrt(sum(d*d for d in finite)/len(finite)),
                'nonfinite_mismatches':sum(u != v for u,v in zip(x,y) if not(math.isfinite(u) and math.isfinite(v))),
                'baseline_top10':[[i,x[i]] for i in a_top], 'prepared_top10':[[i,y[i]] for i in b_top]},
            'first_sample_token_baseline':a_sample['sampled_token_ids'],
            'first_sample_token_prepared':b_sample['sampled_token_ids'],
            'rng_before_equal':a_sample['rng_before'] == b_sample['rng_before'],
            'rng_after_equal':a_sample['rng_after'] == b_sample['rng_after'],
            'seed_equal':a_sample['seed'] == b_sample['seed']})
    return {'baseline':str(baseline), 'prepared':str(prepared),
            'baseline_event_counts':dict(Counter(e['kind'] for e in a_events)),
            'prepared_event_counts':dict(Counter(e['kind'] for e in b_events)),
            'baseline_restores':restore_summary(a_events), 'prepared_restores':restore_summary(b_events),
            'paired_real_prompts':pairs}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline', type=Path)
    parser.add_argument('prepared', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    assert args.baseline.is_absolute() and args.prepared.is_absolute() and args.output.is_absolute()
    result = compare(args.baseline, args.prepared)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps([{k:row[k] for k in ('prompt_tokens','trunk_exact','changed_layer_types',
        'head_exact','sampling_logprobs_exact','rng_before_equal','rng_after_equal')}
        for row in result['paired_real_prompts']], indent=2))
