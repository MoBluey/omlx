# SPDX-License-Identifier: Apache-2.0
"""Regression coverage from upstream oMLX #5012 and #4999."""
import pytest

from omlx.api.anthropic_models import (
    AnthropicMessage, ContentBlockToolResult, ContentBlockToolUse, MessagesRequest,
)
from omlx.api.anthropic_utils import convert_anthropic_to_internal
from omlx.api.openai_models import Message
from omlx.api.utils import extract_text_content, extract_multimodal_content


class TestExtractTextContent:
    @pytest.mark.parametrize(
        "extract", [extract_text_content, extract_multimodal_content]
    )
    def test_native_tool_results_follow_tool_call_order(self, extract):
        """Tool results are rendered in the order of the assistant's tool_calls."""

        class NativeToolTokenizer:
            has_tool_calling = True

        calls = [
            {"id": tid, "function": {"name": "read", "arguments": "{}"}}
            for tid in ("call_a", "call_b")
        ]
        messages = [
            Message(role="assistant", content="", tool_calls=calls),
            Message(role="tool", tool_call_id="call_b", content="B"),
            Message(role="tool", tool_call_id="call_a", content="A"),
        ]

        result = extract(messages, tokenizer=NativeToolTokenizer())

        assert [(msg["tool_call_id"], msg["content"]) for msg in result[1:]] == [
            ("call_a", "A"),
            ("call_b", "B"),
        ]


class TestConvertAnthropicToInternal:
    def test_native_tool_results_follow_tool_call_order(self):
        """Each run of tool results follows its own turn's tool_calls order."""

        class NativeToolTokenizer:
            has_tool_calling = True

        def turn(call_ids, result_ids):
            return [
                AnthropicMessage(
                    role="assistant",
                    content=[
                        ContentBlockToolUse(id=tid, name="read", input={})
                        for tid in call_ids
                    ],
                ),
                AnthropicMessage(
                    role="user",
                    content=[
                        ContentBlockToolResult(tool_use_id=tid, content=tid.upper())
                        for tid in result_ids
                    ],
                ),
            ]

        request = MessagesRequest(
            model="claude-3",
            max_tokens=1024,
            messages=turn(["a", "b"], ["b", "a"]) + turn(["c", "d"], ["d", "x", "c"]),
        )

        result = convert_anthropic_to_internal(request, tokenizer=NativeToolTokenizer())

        assert [
            (msg["tool_call_id"], msg["content"])
            for msg in result
            if msg["role"] == "tool"
        ] == [("a", "A"), ("b", "B"), ("c", "C"), ("d", "D"), ("x", "X")]
