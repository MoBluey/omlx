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
