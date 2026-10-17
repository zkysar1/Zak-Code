"""Claude Sonnet 5.5 support: registry, temperature drop, thinking blocks, and pricing.

WHY THIS FILE EXISTS
Claude Sonnet 5.5 is an adaptive-thinking model that:
- uses output_config.effort (not budget_tokens) for reasoning depth
- rejects non-default temperature, top_p and top_k with a 400
- requires thinking blocks with opaque signature to be passed back unmodified
  in multi-turn conversations (a missing or altered signature yields a 400)

litellm 1.86.2 recognises it as an adaptive-thinking model and correctly maps
reasoning_effort to thinking: {type: "adaptive"} + output_config: {effort: level}.
What it does NOT strip is sampling parameters, so this runtime drops them at
_build_kwargs -- the same chokepoint and shape as the gpt-5 temperature rule
and the Gemini 3 sampling rule.

Thinking blocks are preserved through the message round-trip: _translate_messages
includes thinking_blocks on assistant dicts, _normalize captures them from the
response, and _assistant_message stores them as ThinkingBlocks with signatures on
the canonical Message.
"""

from __future__ import annotations

from typing import Any

import pytest

from zakcode.messages import Message, TextBlock, ThinkingBlock
from zakcode.providers.litellm_provider import (
    LiteLLMProvider,
    _is_anthropic_adaptive_thinking_model,
)
from zakcode.providers.registry import get_capabilities

MSGS = [{"role": "user", "content": "hi"}]


# -- the predicate -----------------------------------------------------------


def test_predicate_matches_claude_5_family() -> None:
    for model in (
        "claude-sonnet-5-5",
        "claude-opus-5-5",
        "claude-haiku-5-5",
        "anthropic/claude-sonnet-5-5",
        "anthropic/claude-opus-5-5",
        "claude-sonnet-5.6",
        "claude-opus-5.7-preview",
    ):
        assert _is_anthropic_adaptive_thinking_model(model), model


def test_predicate_excludes_older_claude_and_other_models() -> None:
    for model in (
        "claude-sonnet-4-6",
        "claude-opus-4-8",
        "anthropic/claude-haiku-4-5",
        "openai/gpt-5.6-luna",
        "ollama_chat/llama3",
        "gemini/gemini-3.5-flash",
    ):
        assert not _is_anthropic_adaptive_thinking_model(model), model


# -- the sampling-parameter drop at the build chokepoint ---------------------


def test_configured_temperature_is_not_sent_to_sonnet_55() -> None:
    provider = LiteLLMProvider(model="anthropic/claude-sonnet-5-5", temperature=0.7)
    kwargs = provider._build_kwargs(MSGS, None)
    assert "temperature" not in kwargs


def test_top_p_and_top_k_are_not_sent_to_sonnet_55() -> None:
    provider = LiteLLMProvider(model="anthropic/claude-sonnet-5-5")
    kwargs = provider._build_kwargs(MSGS, None, top_p=0.9, top_k=40)
    assert "top_p" not in kwargs
    assert "top_k" not in kwargs


def test_an_older_claude_keeps_its_temperature() -> None:
    """Positive control: Claude 4 is untouched."""
    provider = LiteLLMProvider(model="anthropic/claude-sonnet-4-6", temperature=0.7)
    kwargs = provider._build_kwargs(MSGS, None)
    assert kwargs["temperature"] == 0.7


def test_a_local_claude_5_named_model_keeps_its_temperature() -> None:
    """A self-hosted model named after Claude 5 has no such constraint."""
    provider = LiteLLMProvider(
        model="claude-sonnet-5-5",
        api_base="http://10.0.0.205:9090/v1",
        temperature=0.7,
        context_window=131072,
    )
    assert provider._build_kwargs(MSGS, None)["temperature"] == 0.7


# -- reasoning effort reaches the wire (litellm maps it) --------------------


def test_reasoning_effort_is_passed_to_litellm_for_sonnet_55() -> None:
    """zakcode passes reasoning_effort as a litellm kwarg; litellm maps it to
    output_config.effort for adaptive-thinking models."""
    provider = LiteLLMProvider(model="anthropic/claude-sonnet-5-5", reasoning_effort="medium")
    kwargs = provider._build_kwargs(MSGS, None)
    assert kwargs.get("reasoning_effort") == "medium"


# -- capability registry ---------------------------------------------------


@pytest.mark.parametrize(
    "model",
    ["claude-sonnet-5-5", "anthropic/claude-sonnet-5-5"],
)
def test_anthropic_sonnet_55_registry(model: str) -> None:
    caps = get_capabilities(model)
    assert caps.context_window == 1_000_000
    assert caps.max_output == 128_000
    assert caps.supports_tools is True
    assert caps.supports_vision is True
    assert caps.supports_caching is True


# -- thinking blocks round-trip ---------------------------------------------


def test_translate_messages_includes_thinking_blocks_with_signature() -> None:
    """An assistant message carrying ThinkingBlocks (with signatures) should
    produce thinking_blocks on the wire dict so litellm replays them."""
    messages = [
        Message.user("hello"),
        Message(
            role="assistant",
            blocks=[
                ThinkingBlock(text="reasoning here", signature="sig_abc123"),
                TextBlock(text="answer"),
            ],
        ),
        Message.user("follow-up"),
    ]
    wire = LiteLLMProvider._translate_messages(messages, system=None)
    # The assistant message (index 1) should carry thinking_blocks.
    assistant = wire[1]
    assert assistant["role"] == "assistant"
    assert "thinking_blocks" in assistant
    tb = assistant["thinking_blocks"]
    assert len(tb) == 1
    assert tb[0]["type"] == "thinking"
    assert tb[0]["thinking"] == "reasoning here"
    assert tb[0]["signature"] == "sig_abc123"


def test_translate_messages_omits_thinking_blocks_without_content() -> None:
    """ThinkingBlocks with no text and no signature are omitted."""
    messages = [
        Message.user("hello"),
        Message(
            role="assistant",
            blocks=[
                ThinkingBlock(text="", signature=None),
                TextBlock(text="answer"),
            ],
        ),
    ]
    wire = LiteLLMProvider._translate_messages(messages, system=None)
    assert "thinking_blocks" not in wire[1]


def test_translate_messages_without_thinking_blocks_has_no_key() -> None:
    """An assistant message without thinking blocks does not gain the key."""
    messages = [
        Message.user("hello"),
        Message(role="assistant", blocks=[TextBlock(text="plain answer")]),
    ]
    wire = LiteLLMProvider._translate_messages(messages, system=None)
    assert "thinking_blocks" not in wire[1]


# -- _normalize captures thinking_blocks -----------------------------------


class _Obj:
    """Attribute-style stand-in for a litellm response object."""

    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)

    def model_dump(self) -> dict[str, Any]:
        return dict(self.__dict__)


def test_normalize_captures_thinking_blocks_from_response() -> None:
    """_normalize reads thinking_blocks off the response message and stores them."""
    msg = _Obj(
        content="answer text",
        tool_calls=None,
        reasoning_content="thinking text",
        thinking_blocks=[
            {"type": "thinking", "thinking": "thinking text", "signature": "sig123"},
        ],
    )
    response = _Obj(
        choices=[_Obj(message=msg, finish_reason="end_turn")],
        usage=_Obj(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        _hidden_params={"response_cost": 0.001},
        model="claude-sonnet-5-5",
    )
    result = LiteLLMProvider._normalize(response)
    assert result.thinking == "thinking text"
    assert len(result.thinking_blocks) == 1
    assert result.thinking_blocks[0]["signature"] == "sig123"


def test_normalize_handles_missing_thinking_blocks() -> None:
    """_normalize does not break when thinking_blocks is absent."""
    msg = _Obj(content="answer", tool_calls=None, reasoning_content=None)
    response = _Obj(
        choices=[_Obj(message=msg, finish_reason="stop")],
        usage=_Obj(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        _hidden_params={"response_cost": 0.0},
        model="gpt-4o",
    )
    result = LiteLLMProvider._normalize(response)
    assert result.thinking_blocks == []


# -- pricing (litellm knows the model) ------------------------------------


def test_litellm_prices_sonnet_55() -> None:
    """litellm model cost map includes claude-sonnet-5-5 at its published rate."""
    # USD 2/M input, USD 10/M output: 100k input + 10k output
    cost = LiteLLMProvider._litellm_token_cost("claude-sonnet-5-5", 100_000, 10_000, 0, 0)
    assert cost is not None
    # 100k * USD 2/M + 10k * USD 10/M = USD 0.20 + USD 0.10 = USD 0.30
    assert cost == pytest.approx(0.30, rel=0.01)
