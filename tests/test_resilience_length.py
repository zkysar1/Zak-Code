"""finish_reason length-truncation continuation (parity #5).

A final answer cut off at the output cap is auto-continued (bounded) and the turn flagged
degraded, instead of being mis-reported as a clean completion. Scoped to the no-tool-calls
branch (a truncated tool-call response self-heals via the normal round-trip). Hermetic.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

from zakcode.agent.loop import _MAX_LENGTH_CONTINUATIONS, AgentLoop
from zakcode.config import load_settings
from zakcode.providers.base import (
    Capabilities,
    LLMResult,
    Provider,
    ProviderStreamEvent,
    StreamDone,
    StreamTextDelta,
    StreamToolCallDelta,
    ToolCall,
)
from zakcode.session.store import Session
from zakcode.tools.base import ToolRegistry


class ScriptedProvider(Provider):
    """Returns canned LLMResults in order; the last repeats once exhausted."""

    def __init__(self, script: Sequence[LLMResult]) -> None:
        self._script = list(script)
        self.calls = 0

    def _next(self) -> LLMResult:
        idx = min(self.calls, len(self._script) - 1)
        self.calls += 1
        return self._script[idx]

    async def acomplete(self, messages, *, system=None, tools=None, **kw) -> LLMResult:
        return self._next()

    async def astream(
        self, messages, *, system=None, tools=None, **kw
    ) -> AsyncIterator[ProviderStreamEvent]:
        r = self._next()
        if r.text:
            yield StreamTextDelta(text=r.text)
        for i, call in enumerate(r.tool_calls):
            yield StreamToolCallDelta(
                index=i,
                id=call.id,
                name=call.name,
                arguments_delta=json.dumps(call.arguments),
            )
        yield StreamDone(finish_reason=r.finish_reason)

    def count_tokens(self, messages, *, system=None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=8192)


def _loop(provider: Provider, **kw: Any) -> AgentLoop:
    settings = load_settings(workspace_root=Path.cwd())
    session = Session(cwd="/tmp/work", model="test/model")
    return AgentLoop(provider, ToolRegistry(), session, settings=settings, **kw)


def _truncated(text: str = "partial") -> LLMResult:
    return LLMResult(text=text, finish_reason="length")


def _complete(text: str = "the rest") -> LLMResult:
    return LLMResult(text=text, finish_reason="stop")


async def _collect(loop: AgentLoop, text: str) -> list[Any]:
    return [ev async for ev in loop.astream_turn(text)]


# ── buffered ─────────────────────────────────────────────────────────────────


def test_length_truncation_continues_then_completes() -> None:
    provider = ScriptedProvider([_truncated(), _complete()])
    loop = _loop(provider)
    result = asyncio.run(loop.arun_turn("write a long thing"))
    assert result.stop_reason == "completed"
    assert result.degraded  # the turn had to recover from truncation
    assert provider.calls == 2  # truncated, then continued
    # a continuation user message was injected into the conversation
    user_msgs = [m for m in loop.session.messages if m.role == "user"]
    assert any("cut off" in m.text for m in user_msgs)


def test_length_continuation_is_bounded() -> None:
    """After _MAX_LENGTH_CONTINUATIONS, a still-truncated answer is accepted as-is."""
    provider = ScriptedProvider([_truncated()])  # always truncated
    result = asyncio.run(_loop(provider).arun_turn("go"))
    assert result.stop_reason == "completed"
    assert result.degraded
    assert provider.calls == _MAX_LENGTH_CONTINUATIONS + 1  # initial + N continuations


def test_max_tokens_finish_reason_also_continues() -> None:
    """Anthropic-style 'max_tokens' finish reason is treated as truncation too."""
    provider = ScriptedProvider(
        [LLMResult(text="partial", finish_reason="max_tokens"), _complete()]
    )
    result = asyncio.run(_loop(provider).arun_turn("go"))
    assert result.stop_reason == "completed"
    assert provider.calls == 2


def test_empty_truncated_does_not_continue() -> None:
    """A length finish with NO text is not a continuable answer — it takes the
    empty-completion path (a "say something" nudge, 2026-08-26), never the
    "continue exactly where you left off" length continuation."""
    provider = ScriptedProvider(
        [LLMResult(text="", finish_reason="length"), LLMResult(text="answer")]
    )
    loop = _loop(provider)
    result = asyncio.run(loop.arun_turn("go"))
    assert result.stop_reason == "completed"
    assert result.assistant_messages[-1].text == "answer"
    assert not any(
        "cut off at the output limit" in m.text for m in loop.session.messages if m.role == "user"
    )  # the LENGTH continuation never fired


def test_length_with_tool_calls_does_not_continue() -> None:
    """A truncated response that still carries tool calls is NOT short-circuited — it goes
    through the normal tool round-trip (continuing would break tool_use/tool_result)."""
    truncated_with_tool = LLMResult(
        text="thinking",
        finish_reason="length",
        tool_calls=[ToolCall(id="c1", name="noop", arguments={})],
    )
    provider = ScriptedProvider([truncated_with_tool, _complete()])
    result = asyncio.run(_loop(provider).arun_turn("go"))
    # The tool call was attempted (noop is unregistered → error result), the model then
    # returned a clean completion. The point: no length-continuation user message preceded
    # the tool results, so pairing stayed valid and the turn completed normally.
    assert result.stop_reason == "completed"
    assert provider.calls == 2


def test_length_continuation_then_max_iterations_reports_max_iterations() -> None:
    """Outer-bound correctness: a perpetually-truncated turn capped by max_iterations
    reports max_iterations, never a clean completed."""
    provider = ScriptedProvider([_truncated()])
    result = asyncio.run(_loop(provider, max_iterations=2).arun_turn("go"))
    assert result.stop_reason == "max_iterations"
    assert result.degraded


def test_clean_completion_is_not_degraded() -> None:
    provider = ScriptedProvider([_complete()])
    result = asyncio.run(_loop(provider).arun_turn("go"))
    assert result.stop_reason == "completed"
    assert not result.degraded


# ── streaming ────────────────────────────────────────────────────────────────


def test_streaming_length_truncation_continues() -> None:
    provider = ScriptedProvider([_truncated(), _complete()])
    events = asyncio.run(_collect(_loop(provider), "go"))
    done = events[-1]
    assert done.stop_reason == "completed"
    assert done.degraded
    assert provider.calls == 2
    statuses = [ev.message for ev in events if type(ev).__name__ == "AgentStatus"]
    assert any("truncated" in s for s in statuses)


def test_streaming_bounded() -> None:
    """After _MAX_LENGTH_CONTINUATIONS, a still-truncated answer is accepted as-is."""
    provider = ScriptedProvider([_truncated()])  # always truncated
    loop = _loop(provider)
    events = asyncio.run(_collect(loop, "go"))
    done = events[-1]
    assert done.stop_reason == "completed"
    assert done.degraded
    assert provider.calls == _MAX_LENGTH_CONTINUATIONS + 1  # initial + N continuations


def test_streaming_max_tokens_finish_reason_also_continues() -> None:
    """Anthropic-style 'max_tokens' finish reason is treated as truncation too."""
    provider = ScriptedProvider(
        [LLMResult(text="partial", finish_reason="max_tokens"), _complete()]
    )
    events = asyncio.run(_collect(_loop(provider), "go"))
    done = events[-1]
    assert done.stop_reason == "completed"
    assert provider.calls == 2


def test_streaming_empty_truncated_does_not_continue() -> None:
    """A length finish with NO text is not a continuable answer — it takes the
    empty-completion path (a "say something" nudge, 2026-08-26), never the
    "continue exactly where you left off" length continuation."""
    provider = ScriptedProvider(
        [LLMResult(text="", finish_reason="length"), LLMResult(text="answer")]
    )
    loop = _loop(provider)
    events = asyncio.run(_collect(loop, "go"))
    done = events[-1]
    assert done.stop_reason == "completed"
    assert not any(
        "cut off at the output limit" in m.text for m in loop.session.messages if m.role == "user"
    )  # the LENGTH continuation never fired


def test_streaming_length_with_tool_calls_does_not_continue() -> None:
    """A truncated response that still carries tool calls is NOT short-circuited — it goes
    through the normal tool round-trip (continuing would break tool_use/tool_result)."""
    truncated_with_tool = LLMResult(
        text="thinking",
        finish_reason="length",
        tool_calls=[ToolCall(id="c1", name="noop", arguments={})],
    )
    provider = ScriptedProvider([truncated_with_tool, _complete()])
    loop = _loop(provider)
    events = asyncio.run(_collect(loop, "go"))
    done = events[-1]
    assert done.stop_reason == "completed"
    assert provider.calls == 2
    # The guard must route through the tool round-trip, not the continuation:
    # c1 is answered by a real tool_result, and no continuation rail was injected.
    result_ids = {
        b.tool_use_id
        for m in loop.session.messages
        if m.role == "tool"
        for b in m.blocks
        if getattr(b, "type", None) == "tool_result"
    }
    assert "c1" in result_ids
    assert not any(
        m.role == "user" and "cut off at the output limit" in (m.text or "")
        for m in loop.session.messages
    )


def test_streaming_length_continuation_then_max_iterations() -> None:
    """Outer-bound correctness: a perpetually-truncated turn capped by max_iterations
    reports max_iterations, never a clean completed. Each continuation consumes an
    iteration, so max_iterations still bounds the turn."""
    provider = ScriptedProvider([_truncated()])
    events = asyncio.run(_collect(_loop(provider, max_iterations=2), "go"))
    done = events[-1]
    assert done.stop_reason == "max_iterations"
    assert done.degraded
