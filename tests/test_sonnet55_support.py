"""Claude Sonnet 5.5 support: registry, temperature drop, thinking blocks, and pricing.

WHY THIS FILE EXISTS
Claude Sonnet 5.5 is an adaptive-thinking model that:
- uses output_config.effort (not budget_tokens) for reasoning depth
- rejects non-default temperature, top_p and top_k with a 400
- binds each thinking block to the conversation that made it: replayed after the
  system prompt, the tools or an earlier message changed, the block is a 400 on
  organizations created on or after 2026-08-31 (7 of the credit pool's 9 accounts,
  measured 2026-10-17). Leaving a block out is accepted. This runtime edits history
  (compaction, the dynamic system section), so every request opts into drop_block
  (beta thinking-binding-controls-2026-08-01) and the API drops a stale block instead.

litellm 1.86.2 maps reasoning_effort to output_config: {effort: level} for it, but only
where its price map holds the model, and the map bundled in 1.86.2 does not: a process
that could not fetch the remote map read the model as not reasoning-capable and sent no
effort at all (measured). litellm_model_entries.json ships litellm's own entry for that
case (tests/test_model_entries.py). What litellm does NOT strip is sampling parameters,
so this runtime drops them at _build_kwargs -- the same chokepoint and shape as the
gpt-5 temperature rule and the Gemini 3 sampling rule.

Thinking blocks are preserved through the message round-trip: _translate_messages
replays the SIGNED blocks on assistant dicts, _normalize captures them from the
response, and _assistant_message stores them as ThinkingBlocks with signatures on
the canonical Message. On the stream, _parse_chunk reads one channel per chunk
(litellm sends each fragment on two) and the loop splits blocks at each signature.
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
        "claude-sonnet-5",
        "anthropic/claude-sonnet-5",
        "claude-fable-5-1",
    ):
        assert _is_anthropic_adaptive_thinking_model(model), model


def test_predicate_excludes_older_claude_and_other_models() -> None:
    for model in (
        "claude-sonnet-4-6",
        "claude-opus-4-8",
        "anthropic/claude-haiku-4-5",
        # The old name order puts the version first: "-5-sonnet" is NOT generation 5.
        "claude-3-5-sonnet-20241022",
        "anthropic/claude-3-5-haiku-20241022",
        "claude-3-7-sonnet-latest",
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


# -- preserved thinking: every request opts into drop_block -------------------

_DROP_BLOCK_THINKING = {
    "type": "adaptive",
    "display": "summarized",
    "block_binding": {"prefix_mismatch_behavior": "drop_block"},
}
_BINDING_BETA = "thinking-binding-controls-2026-08-01"


@pytest.mark.parametrize("model", ["anthropic/claude-sonnet-5-5", "anthropic/claude-haiku-5-5"])
def test_a_claude_5_request_opts_into_drop_block(model: str) -> None:
    """The field and the beta header that permits it travel together; without the
    header the field is a 400."""
    kwargs = LiteLLMProvider(model=model, reasoning_effort="medium")._build_kwargs(MSGS, None)
    assert kwargs["thinking"] == _DROP_BLOCK_THINKING
    assert kwargs["extra_headers"]["anthropic-beta"] == _BINDING_BETA
    assert kwargs["reasoning_effort"] == "medium"  # litellm maps it beside the explicit thinking


def test_the_binding_beta_joins_a_configured_beta_once() -> None:
    provider = LiteLLMProvider(
        model="anthropic/claude-sonnet-5-5",
        extra_headers={"anthropic-beta": "context-management-2025-06-27", "x-trace": "t"},
    )
    first = provider._build_kwargs(MSGS, None)["extra_headers"]
    second = provider._build_kwargs(MSGS, None)["extra_headers"]
    assert first["anthropic-beta"] == f"context-management-2025-06-27,{_BINDING_BETA}"
    assert second == first  # no growth across calls
    assert first["x-trace"] == "t"
    # The configured headers themselves are not mutated.
    assert provider.extra_headers["anthropic-beta"] == "context-management-2025-06-27"


def test_a_configured_binding_beta_is_not_repeated() -> None:
    provider = LiteLLMProvider(
        model="anthropic/claude-sonnet-5-5", extra_headers={"anthropic-beta": _BINDING_BETA}
    )
    assert provider._build_kwargs(MSGS, None)["extra_headers"]["anthropic-beta"] == _BINDING_BETA


def test_an_older_claude_gets_no_block_binding() -> None:
    """Positive control: Claude 4 requests are untouched."""
    kwargs = LiteLLMProvider(model="anthropic/claude-sonnet-4-6")._build_kwargs(MSGS, None)
    assert "thinking" not in kwargs
    assert "anthropic-beta" not in (kwargs.get("extra_headers") or {})


def test_a_local_claude_5_named_model_gets_no_block_binding() -> None:
    """A self-hosted server behind an api_base is not Anthropic's API."""
    provider = LiteLLMProvider(
        model="claude-sonnet-5-5", api_base="http://10.0.0.205:9090/v1", context_window=131072
    )
    kwargs = provider._build_kwargs(MSGS, None)
    assert "thinking" not in kwargs
    assert "anthropic-beta" not in (kwargs.get("extra_headers") or {})


# -- capability registry ---------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        "claude-sonnet-5-5",
        "anthropic/claude-sonnet-5-5",
        "claude-haiku-5-5",
        "anthropic/claude-haiku-5-5",
    ],
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


def test_translate_messages_replays_only_signed_blocks() -> None:
    """An unsigned block is another backend's reasoning text (a pod's reasoning_content):
    it stays in the transcript and is never sent back, as before this change."""
    messages = [
        Message.user("hello"),
        Message(
            role="assistant",
            blocks=[
                ThinkingBlock(text="pod reasoning", signature=None),
                ThinkingBlock(text="", signature=None),
                TextBlock(text="answer"),
            ],
        ),
        Message.user("again"),
        Message(
            role="assistant",
            blocks=[
                ThinkingBlock(text="unsigned", signature=None),
                ThinkingBlock(text="signed", signature="sig_1"),
                TextBlock(text="answer"),
            ],
        ),
    ]
    wire = LiteLLMProvider._translate_messages(messages, system=None)
    assert "thinking_blocks" not in wire[1]
    assert wire[3]["thinking_blocks"] == [
        {"type": "thinking", "thinking": "signed", "signature": "sig_1"}
    ]


# -- the stream: one channel per chunk ---------------------------------------


def _chunk(**delta: Any) -> dict[str, Any]:
    return {"choices": [{"delta": delta, "finish_reason": None}]}


def _thinking_events(chunk: dict[str, Any]) -> list[Any]:
    events, _ = LiteLLMProvider._parse_chunk(chunk)
    return [e for e in events if type(e).__name__ == "StreamThinkingDelta"]


def test_a_fragment_on_both_channels_is_read_once() -> None:
    """litellm 1.86.2 sends each Anthropic thinking fragment on reasoning_content AND on
    thinking_blocks (measured: 41 of 42 chunks); reading both doubled the thinking."""
    events = _thinking_events(
        _chunk(
            reasoning_content="weigh it",
            thinking_blocks=[{"type": "thinking", "thinking": "weigh it"}],
        )
    )
    assert [(e.text, e.signature) for e in events] == [("weigh it", None)]


def test_the_signature_only_chunk_carries_the_signature() -> None:
    events = _thinking_events(
        _chunk(
            reasoning_content="",
            thinking_blocks=[{"type": "thinking", "thinking": "", "signature": "S"}],
        )
    )
    assert [(e.text, e.signature) for e in events] == [("", "S")]


def test_reasoning_content_alone_still_streams() -> None:
    """Positive control: a pod's reasoning model has no thinking_blocks."""
    events = _thinking_events(_chunk(reasoning_content="pod thinking"))
    assert [(e.text, e.signature) for e in events] == [("pod thinking", None)]


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
