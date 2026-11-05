"""Private benchmark adapter. No model load and no future-user information.

Import with Python -BIS after sys.dont_write_bytecode=True. The caller owns
the private server, memory admission and swap guard. This helper is opt-in.
"""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request
from urllib.parse import urlparse

SOURCE = Path(__file__).resolve().parents[1]
SEAL = Path('/Users/mohameddarwiche/work/p2pit/development/QwenLocal/state/runtime/omlx-0.7.3-m4-max-r2-sealed-site')
MODEL = Path('/Users/mohameddarwiche/work/p2pit/development/QwenLocal/models/candidates/omlx/jedisct1/Qwen3.8-Flash-Next-Uncensored-oQ4e-100K-MTP')


def _digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


@lru_cache(maxsize=1)
def _renderer():
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(SOURCE), str(SEAL)]
    from transformers import AutoTokenizer

    class Renderer:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)

        def prepare_chat_prompt(self, messages, *, tools=None,
                                chat_template_kwargs=None, is_partial=False,
                                add_generation_prompt=True):
            text = self.tokenizer.apply_chat_template(
                messages, tools=tools, tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **(chat_template_kwargs or {}),
            )
            return text, list(self.tokenizer.encode(text))

    return Renderer()


def prepare_idle_prefix(history, api_base, key_path, *, chat_template_kwargs,
                        tools=None, timeout=120):
    """Prepare completed history only; return timing and cache-store evidence.

    Pass the SAME effective template kwargs the real server uses, including
    forced preserve_thinking=False and merged reasoning_effort. Raw completions
    accepts strings, not token IDs; verify its exact tokenisation locally.
    """
    if os.environ.get('QWEN_TRIAL_IDLE_PREFIX') != '1':
        return {'enabled': False}
    started = time.perf_counter()
    parsed = urlparse(api_base)
    if parsed.scheme != 'http' or parsed.hostname != '127.0.0.1':
        raise ValueError('idle experiment requires the private loopback server')
    if not Path(key_path).is_absolute():
        raise ValueError('private API key path must be absolute')
    renderer = _renderer()
    tokenizer_ready = time.perf_counter()
    from omlx.idle_prefix import canonical_next_turn_prefix

    text, ids = canonical_next_turn_prefix(
        renderer, history, tools=tools,
        chat_template_kwargs=chat_template_kwargs,
    )
    prompt = text + '<|im_start|>'
    warm_ids = list(renderer.tokenizer.encode(prompt))
    sentinel = list(renderer.tokenizer.encode('<|im_start|>'))
    if len(sentinel) != 1 or warm_ids != ids + sentinel:
        raise ValueError('raw warm prompt does not end exactly after canonical prefix')
    payload = {'model': MODEL.name, 'prompt': prompt, 'max_tokens': 1,
               'temperature': 0.0, 'stream': False}
    request = urllib.request.Request(
        api_base.rstrip('/') + '/completions',
        data=json.dumps(payload).encode(),
        headers={'Authorization': 'Bearer ' + Path(key_path).read_text().strip(),
                 'Content-Type': 'application/json'}, method='POST',
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request_started = time.perf_counter()
    with opener.open(request, timeout=timeout) as response:
        body = json.load(response)
    ended = time.perf_counter()
    usage = body.get('usage', {})
    if usage.get('prompt_tokens') != len(warm_ids):
        raise ValueError('server raw prompt token count differs from canonical warm request')
    return {'enabled': True, 'elapsed_s': ended-started,
            'tokenizer_s': tokenizer_ready-started,
            'render_s': request_started-tokenizer_ready,
            'warm_request_s': ended-request_started,
            'prefix_tokens': len(ids), 'prefix_sha256': _digest(ids),
            'warm_prompt_tokens': len(warm_ids), 'warm_prompt_sha256': _digest(warm_ids),
            'usage': usage, 'finish_reason': body['choices'][0].get('finish_reason')}


def verify_followup_prefix(messages, preparation, *, chat_template_kwargs, tools=None):
    """Observe actual normally templated followup; never rewrite its tokens."""
    renderer = _renderer()
    _, ids = renderer.prepare_chat_prompt(
        messages, tools=tools, chat_template_kwargs=chat_template_kwargs,
    )
    n = preparation['prefix_tokens']
    return {'exact_prefix': _digest(ids[:n]) == preparation['prefix_sha256'],
            'prompt_tokens': len(ids), 'unavoidable_tokens': len(ids)-n,
            'prompt_sha256': _digest(ids)}
