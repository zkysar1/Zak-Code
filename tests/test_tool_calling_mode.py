"""Tests for the facade's effective tool-calling-mode resolution.

``auto`` mode must route Ollama models to the text protocol (their native tool path
is unreliable via litellm — it returns empty responses for common local models),
while preserving ``auto`` for other providers and passing explicit ``native`` /
``text`` through unchanged.
"""

from __future__ import annotations

import pytest

from zakcode import _resolve_tool_calling_mode


@pytest.mark.parametrize(
    "model,expected",
    [
        ("ollama_chat/qwen2.5:3b", "text"),
        ("ollama/llama3.1", "text"),
        ("openai/gpt-4o", "auto"),
        # The OpenAI-compat shim to Ollama keeps the "openai" prefix; the downstream
        # supports-probe (False for an unknown openai model) already yields text, so
        # auto-resolution leaves it untouched here.
        ("openai/qwen2.5:3b", "auto"),
        ("gpt-4o", "auto"),  # bare model string, no provider prefix
    ],
)
def test_auto_routes_ollama_to_text(model: str, expected: str) -> None:
    assert _resolve_tool_calling_mode("auto", model) == expected


@pytest.mark.parametrize("mode", ["native", "text"])
@pytest.mark.parametrize("model", ["ollama_chat/qwen2.5:3b", "openai/gpt-4o"])
def test_explicit_modes_pass_through(mode: str, model: str) -> None:
    assert _resolve_tool_calling_mode(mode, model) == mode
