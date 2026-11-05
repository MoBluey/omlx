"""CPU-only canonical Qwen template and exact-token-prefix feasibility check."""

import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SEAL = Path('/Users/mohameddarwiche/work/p2pit/development/QwenLocal/state/runtime/omlx-0.7.3-m4-max-r2-sealed-site')
sys.path[:0] = [str(ROOT), str(SEAL)]

from transformers import AutoTokenizer
from omlx.idle_prefix import canonical_next_turn_prefix


MODEL = Path('/Users/mohameddarwiche/work/p2pit/development/QwenLocal/models/candidates/omlx/jedisct1/Qwen3.8-Flash-Next-Uncensored-oQ4e-100K-MTP')
tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)


class Renderer:
    def __init__(self):
        self.tokenizer = tokenizer

    def prepare_chat_prompt(self, messages, *, tools=None, chat_template_kwargs=None,
                            is_partial=False, add_generation_prompt=True):
        text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False,
            add_generation_prompt=add_generation_prompt, **(chat_template_kwargs or {}))
        return text, tokenizer.encode(text)


engine = Renderer()
history = [
    {'role': 'system', 'content': 'You are a helpful coding assistant.'},
    {'role': 'user', 'content': 'Explain why this function returns None and show a correction.'},
    {'role': 'assistant', 'content': 'The function has no return statement. Add `return result` after calculating the value.'},
]
rows = []
for kwargs in (
    {'enable_thinking': True, 'reasoning_effort': 'medium', 'preserve_thinking': False},
    {'enable_thinking': True, 'reasoning_effort': 'low', 'preserve_thinking': False},
    {'enable_thinking': False, 'preserve_thinking': False},
):
    prefix_text, prefix_ids = canonical_next_turn_prefix(engine, history, chat_template_kwargs=kwargs)
    naive_text, naive_ids = engine.prepare_chat_prompt(history, chat_template_kwargs=kwargs, add_generation_prompt=False)
    for user in ('What about the async version?', ' x', 'A', '\nhello', '你好', '😀', '<tool_response>ok</tool_response>'):
        real_text, real_ids = engine.prepare_chat_prompt(history + [{'role': 'user', 'content': user}], chat_template_kwargs=kwargs)
        exact = real_ids[:len(prefix_ids)] == prefix_ids
        rows.append({'kwargs': kwargs, 'future_user': user, 'prefix_tokens': len(prefix_ids),
            'real_prompt_tokens': len(real_ids), 'unavoidable_tokens': len(real_ids)-len(prefix_ids),
            'exact': exact, 'naive_exact': real_ids[:len(naive_ids)] == naive_ids,
            'prefix_contains_think': '<think>' in prefix_text})
        assert exact == (not user.startswith('<tool_response>')), rows[-1]
        if exact:
            assert not rows[-1]['naive_exact'], rows[-1]
print(json.dumps({'model': str(MODEL), 'cpu_only': True, 'rows': rows}, indent=2))
