"""Degeneration guard (ADR-0018): the tail detector, both turn paths, provider defaults.

Field incident 2026-08-26: gemini-2.5-flash-lite, driven at the harness's old
temperature-0.0 default, collapsed into "I will now provide the information you
requested." repeated once a second toward a ~65k-token output cap; only the operator's
Ctrl-C ended the turn. These tests pin the three-piece fix: the pure tail detector, the
discard-retry-then-honest-stop flow in the buffered AND streaming paths (including
mid-stream cancellation of a runaway stream), and the provider-side defaults (no
temperature sent unless configured; a per-completion output cap on every call).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from zakcode.agent.degeneration import repeated_tail
from zakcode.agent.loop import _DEGENERATION_NUDGE, AgentLoop
from zakcode.events import AgentDone, AgentEvent, AgentStatus
from zakcode.messages import Message
from zakcode.providers.base import (
    Capabilities,
    LLMResult,
    Provider,
    ProviderStreamEvent,
    StreamDone,
    StreamTextDelta,
)
from zakcode.providers.litellm_provider import LiteLLMProvider
from zakcode.session.store import Session
from zakcode.tools.base import ToolRegistry

LOOP_LINE = "I will now provide the information you requested.\n\n"
FIELD_LOOP = (
    "My apologies! You are absolutely right.\n\n"
    + LOOP_LINE * 18
    + "I will now provide the information I requested.\n\n"  # token-level mutation
    + LOOP_LINE * 4
)
HEALTHY = "Here is the file list with type and status for each entry, as you asked."


# ── the pure detector ─────────────────────────────────────────────────────────────


def test_detects_the_field_incident_shape() -> None:
    # Mutation-tolerant: one token-level variant inside the tail must not acquit.
    assert repeated_tail(FIELD_LOOP) == "i will now provide the information you requested."


def test_detects_a_no_newline_loop() -> None:
    unit = repeated_tail("spam! " * 120)
    assert unit is not None and len(unit) == len("spam! ")


def test_detects_a_control_character_flood() -> None:
    assert repeated_tail("\b" * 500) is not None


def test_healthy_varied_prose_is_not_convicted() -> None:
    text = "\n".join(f"Line {i}: a distinct observation about file number {i}." for i in range(30))
    assert repeated_tail(text) is None


def test_only_the_tail_is_judged() -> None:
    # A loop that RECOVERED (varied lines after the repeats) is not convicted.
    text = LOOP_LINE * 20 + "\n".join(f"item {i}: unique detail {i * 7}" for i in range(20))
    assert repeated_tail(text) is None


def test_eleven_identical_trailing_lines_pass() -> None:
    # A legitimate fixture of identical lines sits under the 12-repeat conviction bar.
    prefix = "\n".join(f"prelude line {i} with distinct content padding it out" for i in range(12))
    assert repeated_tail(prefix + "\n" + '    "flag": 0,\n' * 11) is None


def test_short_text_gets_no_verdict() -> None:
    assert repeated_tail("yes. " * 10) is None


def test_long_repeated_paragraphs_are_not_convicted() -> None:
    # Restated prose (unit > 200 chars) is a quality problem, not a runaway loop.
    paragraph = ("This is a long restated paragraph about the design. " * 5).strip() + "\n\n"
    assert len(paragraph) > 200
    assert repeated_tail(paragraph * 10) is None


# ── buffered path ─────────────────────────────────────────────────────────────────


class _ByCallProvider(Provider):
    def __init__(self, factory: Any) -> None:
        self._factory = factory
        self.calls = 0

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:
        self.calls += 1
        return self._factory(self.calls)

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=8192)


def _loop(provider: Provider, tmp_path: Path) -> AgentLoop:
    return AgentLoop(
        provider,
        ToolRegistry(),
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        max_iterations=6,
    )


def test_buffered_discards_the_garbage_and_retries(tmp_path: Path) -> None:
    provider = _ByCallProvider(
        lambda n: LLMResult(text=FIELD_LOOP) if n == 1 else LLMResult(text=HEALTHY)
    )
    loop = _loop(provider, tmp_path)
    result = asyncio.run(loop.arun_turn("list the files with type and status"))
    assert result.stop_reason == "completed"
    assert result.degraded is True
    assert provider.calls == 2
    transcript = [m.text for m in loop.session.messages if m.text]
    assert not any("I will now provide" in t for t in transcript)  # garbage never landed
    assert any(_DEGENERATION_NUDGE in t for t in transcript)  # the corrective rail did
    assert transcript[-1] == HEALTHY


def test_buffered_second_strike_ends_honestly(tmp_path: Path) -> None:
    provider = _ByCallProvider(lambda n: LLMResult(text=FIELD_LOOP))
    loop = _loop(provider, tmp_path)
    result = asyncio.run(loop.arun_turn("list the files"))
    assert result.stop_reason == "degenerated"
    assert result.degraded is True
    assert provider.calls == 2  # one retry, then the honest stop
    assert not any(m.text and "I will now provide" in m.text for m in loop.session.messages)


def test_buffered_tool_batches_are_never_judged(tmp_path: Path) -> None:
    # A completion CALLING TOOLS is doing work — repetitive narration rides along.
    # (No tools registered: the unknown call errors, the model then answers.) The
    # guard must not have discarded the tool-calling message.
    from zakcode.providers.base import ToolCall

    provider = _ByCallProvider(
        lambda n: (
            LLMResult(
                text=LOOP_LINE * 20,
                tool_calls=[ToolCall(id="t1", name="nope", arguments={})],
            )
            if n == 1
            else LLMResult(text=HEALTHY)
        )
    )
    loop = _loop(provider, tmp_path)
    result = asyncio.run(loop.arun_turn("go"))
    assert result.stop_reason == "completed"
    assert any(m.text and "I will now provide" in m.text for m in loop.session.messages)


# ── streaming path ────────────────────────────────────────────────────────────────


class _StreamScriptProvider(Provider):
    """Replays canned event lists per astream call; an ``infinite`` script loops forever."""

    def __init__(self, scripts: list[list[ProviderStreamEvent] | str]) -> None:
        self._scripts = scripts
        self.calls = 0
        self.yielded = 0  # total deltas actually consumed by the loop

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:  # pragma: no cover — streaming tests never call this
        return LLMResult()

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=8192)

    async def astream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: Any = None,
        **kw: Any,
    ) -> Any:
        script = self._scripts[min(self.calls, len(self._scripts) - 1)]
        self.calls += 1
        if script == "infinite":
            while True:  # a runaway degenerate stream: never ends on its own
                self.yielded += 1
                yield StreamTextDelta(text=LOOP_LINE)
        else:
            assert not isinstance(script, str)
            for ev in script:
                self.yielded += 1
                yield ev


def _drain(loop: AgentLoop, text: str) -> list[AgentEvent]:
    async def run() -> list[AgentEvent]:
        return [ev async for ev in loop.astream_turn(text)]

    return asyncio.run(run())


def test_streaming_cancels_a_runaway_stream_and_retries(tmp_path: Path) -> None:
    provider = _StreamScriptProvider(
        ["infinite", [StreamTextDelta(text=HEALTHY), StreamDone(finish_reason="stop")]]
    )
    loop = _loop(provider, tmp_path)
    events = _drain(loop, "list the files")
    done = next(e for e in events if isinstance(e, AgentDone))
    assert done.stop_reason == "completed"
    assert done.degraded is True
    assert provider.calls == 2
    # The load-bearing assertion: the infinite stream was CUT within the probe cadence,
    # not consumed to some enormous bound.
    assert provider.yielded < 60
    statuses = [e.message for e in events if isinstance(e, AgentStatus)]
    assert any("degenerated into repetition" in s for s in statuses)
    transcript = [m.text for m in loop.session.messages if m.text]
    assert not any("I will now provide" in t for t in transcript)
    assert transcript[-1] == HEALTHY


def test_streaming_second_strike_ends_honestly(tmp_path: Path) -> None:
    provider = _StreamScriptProvider(["infinite"])
    loop = _loop(provider, tmp_path)
    events = _drain(loop, "list the files")
    done = next(e for e in events if isinstance(e, AgentDone))
    assert done.stop_reason == "degenerated"
    assert done.degraded is True
    assert provider.calls == 2
    statuses = [e.message for e in events if isinstance(e, AgentStatus)]
    assert any("keeps degenerating" in s for s in statuses)


def test_streaming_short_loop_that_ends_naturally_is_still_caught(tmp_path: Path) -> None:
    # A loop that finishes UNDER the probe cadence gets the full-text check at stream end.
    script: list[ProviderStreamEvent] = [StreamTextDelta(text=LOOP_LINE * 16)]
    script.append(StreamDone(finish_reason="stop"))
    provider = _StreamScriptProvider(
        [script, [StreamTextDelta(text=HEALTHY), StreamDone(finish_reason="stop")]]
    )
    loop = _loop(provider, tmp_path)
    events = _drain(loop, "list the files")
    done = next(e for e in events if isinstance(e, AgentDone))
    assert done.stop_reason == "completed"
    assert done.degraded is True
    assert not any(m.text and "I will now provide" in m.text for m in loop.session.messages)


# ── provider defaults (pieces 1 + 2) ──────────────────────────────────────────────


def test_no_temperature_sent_by_default() -> None:
    p = LiteLLMProvider(model="openai/gpt-4o")
    kwargs = p._build_kwargs([{"role": "user", "content": "hi"}], None)
    assert "temperature" not in kwargs  # the backend's own default applies
    assert kwargs["max_tokens"] == 8192  # the per-completion degeneration bound


def test_explicit_temperature_is_still_sent() -> None:
    p = LiteLLMProvider(model="openai/gpt-4o", temperature=0.3)
    kwargs = p._build_kwargs([{"role": "user", "content": "hi"}], None)
    assert kwargs["temperature"] == 0.3


def test_explicit_zero_temperature_is_honored() -> None:
    # 0.0 is a VALUE, not "unset" — a user who asks for it gets it.
    p = LiteLLMProvider(model="openai/gpt-4o", temperature=0.0)
    kwargs = p._build_kwargs([{"role": "user", "content": "hi"}], None)
    assert kwargs["temperature"] == 0.0


def test_per_call_max_tokens_override_wins() -> None:
    p = LiteLLMProvider(model="openai/gpt-4o")
    kwargs = p._build_kwargs([{"role": "user", "content": "hi"}], None, max_tokens=64)
    assert kwargs["max_tokens"] == 64
