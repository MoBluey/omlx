"""CPU-only checks against all four frozen normal TTFT conversations."""

import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from prepare_idle_prefix import _renderer
from omlx.idle_prefix import aligned_next_turn_prefix

WORKSPACE = Path('/Users/mohameddarwiche/work/p2pit/development/QwenLocal')
BENCH = WORKSPACE / 'state/benchmarks/qwen-idle-expert-20261105'


def corpus(chars):
    files = sorted((WORKSPACE / 'qwenlocal').glob('*.py')) + sorted((WORKSPACE / 'tests').glob('test_*.py'))
    out = []
    for path in files:
        out.append('# FILE ' + str(path.relative_to(WORKSPACE)) + '\n' + path.read_text(errors='replace'))
        text = '\n'.join(out)
        if len(text) >= chars:
            return text[:chars]
    return text[:chars]


engine = _renderer()
kwargs = {'enable_thinking': True, 'reasoning_effort': 'medium', 'preserve_thinking': False}
rows = []
for phase, chars, first_session in (('short', 8000, 0), ('long', 115000, 2)):
    for rep in (1, 2):
        session = first_session + rep - 1
        prompt = [{'role': 'system', 'content': f'Profiling session {chars}-{session:032x}. Reference code:\n\n' + corpus(chars)},
                  {'role': 'user', 'content': 'Explain the model startup and cleanup sequence in the reference code. Give four concise numbered points using the relevant function names.'}]
        fixture = json.loads((BENCH / f'fixed-followups/qwen-{phase}-{rep}.json').read_text())
        assert hashlib.sha256(json.dumps(prompt).encode()).hexdigest() == fixture['cold_prompt_sha256']
        history = prompt + [{'role': 'assistant', 'content': fixture['answer']}]
        text, ids, decision = aligned_next_turn_prefix(engine, history, chat_template_kwargs=kwargs)
        row = {'phase': phase, 'rep': rep, 'session': session, **decision}
        if phase == 'short':
            assert ids is None and decision['chosen_boundary'] == 0
            row['skipped'] = True
        else:
            assert decision['chosen_boundary'] == 28672
            assert list(engine.tokenizer.encode(text)) == ids
            sentinel = list(engine.tokenizer.encode('<|im_start|>'))
            assert len(sentinel) == 1
            assert list(engine.tokenizer.encode(text + '<|im_start|>')) == ids + sentinel
            # Actual next-user content is used only after selecting preparation.
            followup = history + [{'role': 'user', 'content': 'Which cleanup step matters most after a failed launch? Answer in one sentence.'}]
            _, real_ids = engine.prepare_chat_prompt(followup, chat_template_kwargs=kwargs)
            assert real_ids[:len(ids)] == ids
            row.update({'prefix_tokens': len(ids), 'warm_prompt_tokens': len(ids)+1,
                        'real_prompt_tokens': len(real_ids), 'remaining_tokens': len(real_ids)-len(ids),
                        'roundtrip_exact': True, 'sentinel_exact': True, 'real_prefix_exact': True})
        rows.append(row)
print(json.dumps({'model_loaded': False, 'rows': rows}, indent=2))
