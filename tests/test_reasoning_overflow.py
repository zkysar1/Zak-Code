"""Reasoning overflow is not silence (ADR-0056).

A completion with no visible text and no tool calls can be two different things: the
model went SILENT, or it REASONED and delivered nothing — a thinking channel arrived, or
the output cap cut it off mid-thought. Measured 2026-08-28 on the coach pod (Qwen3.8-27B
behind a reasoning parser): the fatal "empty" completion carried 8,192 completion tokens,
exactly the cap, with empty ``content`` and a ``reasoning_content`` still mid-sentence.
The loop nudged it with "your response was empty" (an instruction a template-enforced
thinking model cannot obey), and its per-turn CUMULATIVE budget gave up on the third
empty of the turn — eight successful tool calls after the second.

Hermetic: scripted providers that record per-call kwargs, an echo tool, a tmp workspace.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from zakcode.agent.loop import _MAX_EMPTY_RETRIES, AgentLoop
from zakcode.config import load_settings
from zakcode.messages import Message
from zakcode.providers.base import (
    Capabilities,
    LLMResult,
    Provider,
    StreamTextDelta,
    StreamThinkingDelta,
    ToolCall,
)
from zakcode.providers.routing import thinking_extra_body
from zakcode.session.store import Session
from zakcode.tools.base import Tool, ToolContext, ToolRegistry, ToolResult, ToolSpec

THINKING_OFF = thinking_extra_body(False)


class RecordingProvider(Provider):
    """Replays scripted results and records the kwargs of every ``acomplete`` call."""

    def __init__(self, results: list[LLMResult]) -> None:
        self._results = list(results)
        self.kwargs: list[dict[str, Any]] = []

    async def acomplete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        self.kwargs.append(dict(kwargs))
        if not self._results:
            raise AssertionError("provider ran out of scripted results")
        return self._results.pop(0)

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(context_window=8192)


class EchoTool(Tool):
    spec = ToolSpec(name="echo", description="Echo back the provided text.")

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(output=str(args.get("text", "")))


def _make_loop(provider: Provider, tmp_path: Path, *, max_iterations: int = 20) -> AgentLoop:
    registry = ToolRegistry()
    registry.register(EchoTool())
    return AgentLoop(
        provider,
        registry,
        Session(cwd=str(tmp_path), model="ollama_chat/llama3.1"),
        settings=load_settings(workspace_root=tmp_path),
        max_iterations=max_iterations,
    )


def _overflow(**over: Any) -> LLMResult:
    """The coach shape: reasoning arrived, nothing visible, the cap cut it off."""
    fields: dict[str, Any] = {
        "thinking": "Here's a thinking process: 1.",
        "finish_reason": "length",
    }
    fields.update(over)
    return LLMResult(**fields)


def _step(i: int) -> LLMResult:
    return LLMResult(tool_calls=[ToolCall(id=f"c{i}", name="echo", arguments={"text": f"s{i}"})])


def _rails(loop: AgentLoop) -> list[str]:
    return [m.text for m in loop.session.messages if m.role == "user"]


def _stop_reason(events: list[Any]) -> str | None:
    done = [ev for ev in events if getattr(ev, "stop_reason", None) is not None]
    return done[-1].stop_reason if done else None


# ── buffered twin ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_overflow_is_retried_once_with_thinking_off(tmp_path: Path) -> None:
    provider = RecordingProvider([_overflow(), LLMResult(text="the answer")])
    loop = _make_loop(provider, tmp_path)
    result = await loop.arun_turn("question")
    assert result.stop_reason == "completed"
    assert result.assistant_messages[-1].text == "the answer"
    assert result.degraded is True  # a truncation recovery, like a length continuation
    assert len(provider.kwargs) == 2
    assert "extra_body" not in provider.kwargs[0]
    assert provider.kwargs[1]["extra_body"] == THINKING_OFF
    rails = [r for r in _rails(loop) if "reasoning only" in r]
    assert rails and "Your response was empty" not in rails[-1]


@pytest.mark.asyncio
async def test_thought_then_stopped_without_answering_is_an_overflow(tmp_path: Path) -> None:
    # The 2,139-token shape: the thinking channel closed on its own, still no answer.
    provider = RecordingProvider([_overflow(finish_reason="stop"), LLMResult(text="the answer")])
    loop = _make_loop(provider, tmp_path)
    result = await loop.arun_turn("question")
    assert result.stop_reason == "completed"
    assert provider.kwargs[1]["extra_body"] == THINKING_OFF


@pytest.mark.asyncio
async def test_length_cut_with_no_reasoning_surfaced_is_an_overflow(tmp_path: Path) -> None:
    # A backend without a reasoning parser reports only the cap; still not silence.
    provider = RecordingProvider([LLMResult(finish_reason="length"), LLMResult(text="the answer")])
    loop = _make_loop(provider, tmp_path)
    result = await loop.arun_turn("question")
    assert result.stop_reason == "completed"
    assert provider.kwargs[1]["extra_body"] == THINKING_OFF


@pytest.mark.asyncio
async def test_thinking_off_is_one_shot(tmp_path: Path) -> None:
    provider = RecordingProvider([_overflow(), _step(1), LLMResult(text="the answer")])
    loop = _make_loop(provider, tmp_path)
    result = await loop.arun_turn("question")
    assert result.stop_reason == "completed"
    assert len(provider.kwargs) == 3
    assert provider.kwargs[1]["extra_body"] == THINKING_OFF
    assert "extra_body" not in provider.kwargs[2]


@pytest.mark.asyncio
async def test_plain_silence_keeps_the_generic_nudge(tmp_path: Path) -> None:
    provider = RecordingProvider([LLMResult(), LLMResult(text="the answer")])
    loop = _make_loop(provider, tmp_path)
    result = await loop.arun_turn("question")
    assert result.stop_reason == "completed"
    assert result.degraded is False
    assert "extra_body" not in provider.kwargs[1]
    assert any("Your response was empty" in r for r in _rails(loop))
    assert not any("reasoning only" in r for r in _rails(loop))


@pytest.mark.asyncio
async def test_overflow_after_prior_text_is_still_retried(tmp_path: Path) -> None:
    # A plain empty completion after text is a deliberate "nothing more to say" (clean
    # end); an overflow after text is a truncation, never deliberate.
    provider = RecordingProvider(
        [
            LLMResult(
                text="Looking.",
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "a"})],
            ),
            _overflow(),
            LLMResult(text="the answer"),
        ]
    )
    loop = _make_loop(provider, tmp_path)
    result = await loop.arun_turn("question")
    assert result.stop_reason == "completed"
    assert len(provider.kwargs) == 3
    assert result.assistant_messages[-1].text == "the answer"


@pytest.mark.asyncio
async def test_empty_budget_is_consecutive_not_cumulative(tmp_path: Path) -> None:
    # The coach death: empties separated by real work must never add up to gave_up.
    empties = 2 * (1 + _MAX_EMPTY_RETRIES)  # far past the bound if it were cumulative
    script: list[LLMResult] = []
    for i in range(empties):
        script += [LLMResult(), _step(i)]
    script.append(LLMResult(text="the answer"))
    provider = RecordingProvider(script)
    loop = _make_loop(provider, tmp_path, max_iterations=50)
    result = await loop.arun_turn("question")
    assert result.stop_reason == "completed"
    assert len(provider.kwargs) == len(script)


@pytest.mark.asyncio
async def test_consecutive_overflows_still_give_up(tmp_path: Path) -> None:
    # A model that overflows even with thinking off is stuck; the honest end survives.
    provider = RecordingProvider([_overflow()] * (1 + _MAX_EMPTY_RETRIES))
    loop = _make_loop(provider, tmp_path)
    result = await loop.arun_turn("question")
    assert result.stop_reason == "gave_up"
    assert len(provider.kwargs) == 1 + _MAX_EMPTY_RETRIES


# ── streaming twin ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_overflow_is_retried_with_thinking_off(tmp_path: Path) -> None:
    provider = RecordingProvider([_overflow(), LLMResult(text="the answer")])
    loop = _make_loop(provider, tmp_path)
    events = [ev async for ev in loop.astream_turn("question")]
    assert _stop_reason(events) == "completed"
    assert provider.kwargs[1]["extra_body"] == THINKING_OFF
    statuses = [getattr(ev, "message", "") for ev in events]
    assert any("thinking off" in s for s in statuses)
    assert not any("went silent" in s for s in statuses)


@pytest.mark.asyncio
async def test_stream_thought_then_stopped_is_an_overflow(tmp_path: Path) -> None:
    # No length signal — only the thinking channel (forwarded by the default astream).
    provider = RecordingProvider([_overflow(finish_reason="stop"), LLMResult(text="the answer")])
    loop = _make_loop(provider, tmp_path)
    events = [ev async for ev in loop.astream_turn("question")]
    assert _stop_reason(events) == "completed"
    assert provider.kwargs[1]["extra_body"] == THINKING_OFF


@pytest.mark.asyncio
async def test_stream_empty_budget_is_consecutive(tmp_path: Path) -> None:
    empties = 2 * (1 + _MAX_EMPTY_RETRIES)
    script: list[LLMResult] = []
    for i in range(empties):
        script += [LLMResult(), _step(i)]
    script.append(LLMResult(text="the answer"))
    provider = RecordingProvider(script)
    loop = _make_loop(provider, tmp_path, max_iterations=50)
    events = [ev async for ev in loop.astream_turn("question")]
    assert _stop_reason(events) == "completed"
    assert len(provider.kwargs) == len(script)


# ── the default astream forwards reasoning ────────────────────────────────────


@pytest.mark.asyncio
async def test_default_astream_forwards_thinking_as_its_own_event() -> None:
    provider = RecordingProvider([LLMResult(thinking="pondering", text="answer")])
    events = [ev async for ev in provider.astream([Message.user("q")])]
    assert isinstance(events[0], StreamThinkingDelta) and events[0].text == "pondering"
    assert isinstance(events[1], StreamTextDelta) and events[1].text == "answer"
