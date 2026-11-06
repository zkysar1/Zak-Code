"""Restatement at the output cap (ADR-0368): the measure, the guard's scope, both turn paths.

Field incident 2026-11-05: a served model wrote eleven replies of 15k to 24k characters in 63
minutes, each stopped at the 8,192-token output cap and each restating the last; four were
byte-identical, and none called a tool. The degeneration guard (ADR-0018) leaves a repeating
unit longer than 200 characters alone, and the length continuation re-prompted every capped
reply, so only an operator restart ended the turn. These tests pin the fix: a capped,
text-only completion that restates itself, or repeats an earlier reply of the turn, is
discarded before it reaches the transcript and shares ADR-0018's one-retry bound, while a
long capped answer that does neither is still continued.
"""

from __future__ import annotations

import asyncio
import random
from pathlib import Path
from typing import Any

import pytest

import zakcode.agent.loop as loop_module
from zakcode.agent.degeneration import (
    RESTATEMENT_FLOOR,
    RESTATEMENT_MIN_CHARS,
    repeated_tail,
    self_repetition,
)
from zakcode.agent.loop import _RESTATEMENT_NUDGE, AgentLoop, _capped_restatement
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
from zakcode.session.store import Session
from zakcode.tools.base import ToolRegistry

_WORDS = [
    "parser",
    "record",
    "field",
    "checksum",
    "branch",
    "merge",
    "window",
    "budget",
    "router",
    "engine",
    "session",
    "cache",
    "ledger",
    "schema",
    "column",
    "index",
    "shard",
    "replica",
    "token",
    "stream",
    "buffer",
    "socket",
    "thread",
    "signal",
    "commit",
    "review",
    "module",
    "import",
    "export",
    "config",
    "option",
    "default",
    "fallback",
    "retry",
    "timeout",
    "limit",
    "cursor",
    "offset",
    "page",
    "batch",
    "queue",
    "worker",
    "reader",
    "writer",
    "header",
    "footer",
    "status",
]


def _prose(seed: int, chars: int) -> str:
    """Varied lines of seeded random words: no 64-byte window recurs by construction."""
    rng = random.Random(seed)
    lines: list[str] = []
    total = 0
    while total < chars:
        line = " ".join(rng.choice(_WORDS) for _ in range(rng.randint(8, 16)))
        line = f"{line.capitalize()} ({rng.randint(1000, 9999)})."
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines)


#: A long answer that does not repeat itself (well over the 12,000-char minimum).
CLEAN_LONG = _prose(1, 16_000)
#: One 1,800-char block written nine times over, a number changed in each copy: the
#: incident's shape (a long unit restated), which the tail detector does not convict.
_BLOCK = _prose(2, 1_800)
RESTATED = "\n\n".join(f"Attempt {i}.\n{_BLOCK}" for i in range(9))
HEALTHY = "Here is the file list with type and status for each entry, as you asked."


# ── the measure ───────────────────────────────────────────────────────────────────


def test_the_fixtures_have_the_shapes_they_claim() -> None:
    assert len(CLEAN_LONG) >= RESTATEMENT_MIN_CHARS
    assert len(RESTATED) >= RESTATEMENT_MIN_CHARS
    # The gap ADR-0368 closes: the ADR-0018 tail detector acquits the restated text.
    assert repeated_tail(RESTATED) is None


def test_a_restated_text_reads_above_the_floor() -> None:
    assert self_repetition(RESTATED) >= RESTATEMENT_FLOOR


def test_a_varied_long_text_reads_near_zero() -> None:
    assert self_repetition(CLEAN_LONG) < 0.05


def test_a_repeat_counts_wherever_it_starts() -> None:
    # Each copy is shifted by a prefix of a different length, so no copy starts on the same
    # offset modulo any stride: a strided measure misses this, a stride-1 one does not.
    shifted = "\n".join("x" * (i * 7 + 1) + _BLOCK for i in range(9))
    assert self_repetition(shifted) >= RESTATEMENT_FLOOR


def test_half_repeated_reads_near_one_half() -> None:
    half = _prose(3, 9_000) + "\n" + _prose(3, 9_000)
    assert 0.4 <= self_repetition(half) <= 0.55


def test_a_text_shorter_than_one_window_reads_zero() -> None:
    assert self_repetition("short") == 0.0


# ── the guard's scope ─────────────────────────────────────────────────────────────


def test_only_a_capped_completion_is_judged() -> None:
    assert _capped_restatement(RESTATED, "stop", set(), {}) is None
    assert _capped_restatement(RESTATED, None, set(), {}) is None
    assert _capped_restatement(RESTATED, "length", set(), {}) is not None
    assert _capped_restatement(RESTATED, "max_tokens", set(), {}) is not None


def test_a_long_capped_answer_that_does_not_repeat_is_left_alone() -> None:
    assert _capped_restatement(CLEAN_LONG, "length", set(), {}) is None


def test_a_short_capped_text_gets_no_verdict_from_the_floor() -> None:
    short = RESTATED[: RESTATEMENT_MIN_CHARS - 1]
    assert self_repetition(short) >= RESTATEMENT_FLOOR  # it IS repetitive...
    assert _capped_restatement(short, "length", set(), {}) is None  # ...but unmeasured there


def test_the_same_capped_text_twice_is_caught_even_when_it_does_not_repeat_itself() -> None:
    seen: set[str] = set()
    assert _capped_restatement(CLEAN_LONG, "length", seen, {}) is None
    why = _capped_restatement(CLEAN_LONG, "length", seen, {})
    assert why is not None and "earlier reply" in why


def test_a_capped_text_equal_to_an_earlier_uncapped_completion_is_caught() -> None:
    key = " ".join(CLEAN_LONG.split()).lower()
    counts = {key: 1}
    assert _capped_restatement(CLEAN_LONG, "length", set(), counts) is not None
    assert counts == {key: 1}  # the broken-record counts are read, never changed


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


def _texts(loop: AgentLoop) -> list[str]:
    return [m.text for m in loop.session.messages if m.text]


def _capped(text: str) -> LLMResult:
    return LLMResult(text=text, finish_reason="length")


def test_buffered_discards_a_capped_restatement_and_asks_for_a_tool_call(tmp_path: Path) -> None:
    provider = _ByCallProvider(lambda n: _capped(RESTATED) if n == 1 else LLMResult(text=HEALTHY))
    loop = _loop(provider, tmp_path)
    result = asyncio.run(loop.arun_turn("summarize the open work"))
    assert result.stop_reason == "completed"
    assert result.degraded is True
    assert provider.calls == 2
    texts = _texts(loop)
    assert not any(_BLOCK in t for t in texts)  # the restatement never landed
    assert any(_RESTATEMENT_NUDGE in t for t in texts)  # the tool-call rail did
    assert not any("cut off at the output limit" in t for t in texts)  # not continued
    assert texts[-1] == HEALTHY


def test_buffered_second_restatement_ends_the_turn_honestly(tmp_path: Path) -> None:
    provider = _ByCallProvider(lambda n: _capped(RESTATED))
    loop = _loop(provider, tmp_path)
    result = asyncio.run(loop.arun_turn("summarize the open work"))
    assert result.stop_reason == "degenerated"
    assert result.degraded is True
    assert provider.calls == 2  # one retry, then the honest stop: never to the iteration cap
    assert not any(_BLOCK in t for t in _texts(loop))


def test_buffered_a_clean_capped_answer_is_still_continued(tmp_path: Path) -> None:
    provider = _ByCallProvider(lambda n: _capped(CLEAN_LONG) if n == 1 else LLMResult(text=HEALTHY))
    loop = _loop(provider, tmp_path)
    result = asyncio.run(loop.arun_turn("write the long report"))
    assert result.stop_reason == "completed"
    assert provider.calls == 2
    texts = _texts(loop)
    assert CLEAN_LONG in texts
    assert any("cut off at the output limit" in t for t in texts)
    assert not any(_RESTATEMENT_NUDGE in t for t in texts)


def test_buffered_the_same_capped_reply_sent_again_is_discarded(tmp_path: Path) -> None:
    # The incident's byte-identical replies: the first copy is a fair capped answer and is
    # continued; the second copy is discarded although it does not repeat itself inside.
    replies = {1: _capped(CLEAN_LONG), 2: _capped(CLEAN_LONG)}
    provider = _ByCallProvider(lambda n: replies.get(n, LLMResult(text=HEALTHY)))
    loop = _loop(provider, tmp_path)
    result = asyncio.run(loop.arun_turn("write the long report"))
    assert result.stop_reason == "completed"
    assert result.degraded is True
    assert provider.calls == 3
    texts = _texts(loop)
    assert texts.count(CLEAN_LONG) == 1
    assert any(_RESTATEMENT_NUDGE in t for t in texts)


def test_buffered_an_uncapped_restatement_is_not_this_guard_s_call(tmp_path: Path) -> None:
    provider = _ByCallProvider(lambda n: LLMResult(text=RESTATED, finish_reason="stop"))
    loop = _loop(provider, tmp_path)
    asyncio.run(loop.arun_turn("summarize the open work"))
    assert not any(_RESTATEMENT_NUDGE in t for t in _texts(loop))


def test_mutation_proof_the_floor_is_what_catches_the_restatement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raise the floor out of reach: the restated reply is length-continued, as before the fix."""
    monkeypatch.setattr(loop_module, "RESTATEMENT_FLOOR", 1.01)
    provider = _ByCallProvider(lambda n: _capped(RESTATED) if n == 1 else LLMResult(text=HEALTHY))
    loop = _loop(provider, tmp_path)
    asyncio.run(loop.arun_turn("summarize the open work"))
    texts = _texts(loop)
    assert RESTATED in texts
    assert any("cut off at the output limit" in t for t in texts)
    assert not any(_RESTATEMENT_NUDGE in t for t in texts)


# ── streaming path ────────────────────────────────────────────────────────────────


class _StreamScriptProvider(Provider):
    """Replays one canned event list per astream call (the last one repeats)."""

    def __init__(self, scripts: list[list[ProviderStreamEvent]]) -> None:
        self._scripts = scripts
        self.calls = 0

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
        for ev in script:
            yield ev


def _stream(text: str, finish: str) -> list[ProviderStreamEvent]:
    return [StreamTextDelta(text=text), StreamDone(finish_reason=finish)]


def _drain(loop: AgentLoop, text: str) -> list[AgentEvent]:
    async def run() -> list[AgentEvent]:
        return [ev async for ev in loop.astream_turn(text)]

    return asyncio.run(run())


def test_streaming_discards_a_capped_restatement_and_retries(tmp_path: Path) -> None:
    provider = _StreamScriptProvider([_stream(RESTATED, "length"), _stream(HEALTHY, "stop")])
    loop = _loop(provider, tmp_path)
    events = _drain(loop, "summarize the open work")
    done = next(e for e in events if isinstance(e, AgentDone))
    assert done.stop_reason == "completed"
    assert done.degraded is True
    assert provider.calls == 2
    statuses = [e.message for e in events if isinstance(e, AgentStatus)]
    assert any("restated itself at the output limit" in s for s in statuses)
    texts = _texts(loop)
    assert not any(_BLOCK in t for t in texts)
    assert any(_RESTATEMENT_NUDGE in t for t in texts)
    assert texts[-1] == HEALTHY


def test_streaming_second_restatement_ends_the_turn_honestly(tmp_path: Path) -> None:
    provider = _StreamScriptProvider([_stream(RESTATED, "length")])
    loop = _loop(provider, tmp_path)
    events = _drain(loop, "summarize the open work")
    done = next(e for e in events if isinstance(e, AgentDone))
    assert done.stop_reason == "degenerated"
    assert provider.calls == 2


def test_streaming_a_clean_capped_answer_is_still_continued(tmp_path: Path) -> None:
    provider = _StreamScriptProvider([_stream(CLEAN_LONG, "length"), _stream(HEALTHY, "stop")])
    loop = _loop(provider, tmp_path)
    events = _drain(loop, "write the long report")
    done = next(e for e in events if isinstance(e, AgentDone))
    assert done.stop_reason == "completed"
    statuses = [e.message for e in events if isinstance(e, AgentStatus)]
    assert any("truncated; continuing" in s for s in statuses)
    assert CLEAN_LONG in _texts(loop)
