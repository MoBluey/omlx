"""Default-off experiments preparing canonical Qwen next-turn prefixes.

This module does not rewrite incoming prompts or reuse live decode state.
The next user is unknown: render an empty user only to select the template's
normal historical-assistant branch, then remove that entire user message.
"""

from copy import deepcopy


_EMPTY_USER_SUFFIX = "<|im_start|>user\n<|im_end|>\n"


def canonical_next_turn_prefix(engine, history, *, tools=None, chat_template_kwargs=None):
    """Return exact completed-history tokens for the supported Qwen template.

    Rendering history alone is insufficient: Qwen retains a final assistant's
    reasoning even with preserve_thinking=False until another user follows it.
    Refuse other suffixes instead of guessing about another template's state.
    """
    if not history or history[-1].get("role") != "assistant":
        raise ValueError("idle preparation requires completed assistant history")
    messages = deepcopy(history) + [{"role": "user", "content": ""}]
    text, _ = engine.prepare_chat_prompt(
        messages,
        tools=tools,
        chat_template_kwargs=chat_template_kwargs,
        is_partial=False,
        add_generation_prompt=False,
    )
    if not text.endswith(_EMPTY_USER_SUFFIX):
        raise ValueError("unsupported canonical next-user template suffix")
    prefix = text[:-len(_EMPTY_USER_SUFFIX)]
    return prefix, list(engine.tokenizer.encode(prefix))


def aligned_next_turn_prefix(engine, history, *, tools=None, chat_template_kwargs=None):
    """Select only a pre-existing 4096-token boundary beyond the old prompt.

    Return a disabled decision when the known answer reaches no new boundary
    or an arbitrary BPE boundary cannot round-trip through raw prompt text.
    """
    _, canonical_ids = canonical_next_turn_prefix(
        engine, history, tools=tools, chat_template_kwargs=chat_template_kwargs,
    )
    _, previous_ids = engine.prepare_chat_prompt(
        deepcopy(history[:-1]), tools=tools,
        chat_template_kwargs=chat_template_kwargs, is_partial=False,
        add_generation_prompt=True,
    )
    boundary = (len(canonical_ids) // 4096) * 4096
    decision = {'variant': 'aligned_boundary',
                'canonical_prefix_tokens': len(canonical_ids),
                'chosen_boundary': boundary,
                'previous_prompt_tokens': len(previous_ids)}
    if boundary <= len(previous_ids):
        return None, None, {**decision, 'skipped_reason': 'boundary_does_not_advance_previous_prompt'}
    ids = canonical_ids[:boundary]
    text = engine.tokenizer.decode(
        ids, skip_special_tokens=False, clean_up_tokenization_spaces=False,
    )
    if list(engine.tokenizer.encode(text)) != ids:
        return None, None, {**decision, 'skipped_reason': 'boundary_not_roundtrip_exact'}
    return text, ids, {**decision, 'roundtrip_exact': True}
