"""The turn-ended log line carries a parseable wall-clock duration, on BOTH turn paths.

Why this file exists (g-373-122). Sizing the framework-stop grace needs the
distribution of real turn LENGTHS from a served run's ``serve.log``. The loop has
logged an unconditional ``turn ended: stop_reason=... iterations=... tokens=...``
line since the first commit — measured 55 of them across a 52-log / 37,144-line
estate corpus on 2026-09-21 — but it carried no duration, so the corpus bounded
nothing. ``duration_s`` is that missing field.

Two properties are pinned here, and the second is the load-bearing one:

1. The field is present and parses as a float.
2. It is emitted on the **streamed** path as well as the buffered one.
   ``astream_turn`` is not a thin wrapper around ``arun_turn`` — it is a full
   twin with its own copy of the turn body and its own copy of this log call,
   and it is the path the server's ``_run_turn_for_say`` drives. An instrument
   added only to the buffered path would emit nothing at all on the runs this
   measurement is for, while reading as done.

The regex below is the analysis instrument itself, not a paraphrase of it: the
same pattern is what turns a corpus of ``serve.log`` files into a distribution.
Pinning it here means a reformat of the log call that breaks the grep breaks a
test instead of silently returning zero rows.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from zakcode.agent.loop import AgentLoop
from zakcode.config import load_settings
from zakcode.messages import Message
from zakcode.providers.base import (
    Capabilities,
    LLMResult,
    Provider,
    ProviderStreamEvent,
    StreamDone,
    StreamTextDelta,
    StreamUsage,
)
from zakcode.session.store import Session
from zakcode.tools.base import ToolRegistry
from zakcode.usage import Usage

#: The analysis instrument. ``grep -oE`` with this pattern over a serve.log corpus
#: yields one row per completed turn: its stop reason and its wall-clock seconds.
TURN_LINE_RE = re.compile(
    r"turn ended: stop_reason=(?P<stop_reason>\S+) "
    r"iterations=(?P<iterations>\d+) "
    r"tokens=(?P<tokens>\d+) "
    r"duration_s=(?P<duration_s>\d+\.\d+)"
)

LOGGER_NAME = "zakcode.agent.loop"


class _ScriptedProvider(Provider):
    """Buffered double: returns one canned result, then repeats it."""

    def __init__(self, script: Sequence[LLMResult]) -> None:
        self._script = list(script)
        self.calls = 0

    async def acomplete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        idx = min(self.calls, len(self._script) - 1)
        self.calls += 1
        return self._script[idx]

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=8192)


class _ScriptedStreamProvider(_ScriptedProvider):
    """Streamed double: replays canned stream events for the one iteration."""

    def __init__(self, events: list[ProviderStreamEvent]) -> None:
        super().__init__([LLMResult()])
        self._events = events

    async def astream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        self.calls += 1
        for event in self._events:
            yield event


def _loop(provider: Provider, tmp_path: Path) -> AgentLoop:
    settings = load_settings(workspace_root=tmp_path)
    session = Session(cwd=str(tmp_path), model="scripted/test")
    return AgentLoop(provider, ToolRegistry(), session, settings=settings, workspace_root=tmp_path)


def _turn_lines(caplog: Any) -> list[re.Match[str]]:
    """Every turn-ended line the loop logged, parsed by the analysis instrument."""
    matched = []
    for record in caplog.records:
        if record.name != LOGGER_NAME:
            continue
        m = TURN_LINE_RE.search(record.getMessage())
        if m is not None:
            matched.append(m)
    return matched


def test_buffered_turn_logs_a_parseable_duration(tmp_path: Path, caplog: Any) -> None:
    provider = _ScriptedProvider(
        [
            LLMResult(
                text="done",
                finish_reason="stop",
                usage=Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
            )
        ]
    )
    loop = _loop(provider, tmp_path)

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        result = asyncio.run(loop.arun_turn("hello"))

    assert result.stop_reason == "completed"
    lines = _turn_lines(caplog)
    assert len(lines) == 1, "the buffered path logs exactly one turn-ended line per turn"
    assert lines[0]["stop_reason"] == "completed"
    assert float(lines[0]["duration_s"]) >= 0.0


def test_streamed_turn_logs_a_parseable_duration(tmp_path: Path, caplog: Any) -> None:
    """The served path. Its absence here is the failure mode this file guards."""
    provider = _ScriptedStreamProvider(
        [
            StreamTextDelta(text="done"),
            StreamUsage(usage=Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5)),
            StreamDone(finish_reason="stop"),
        ]
    )
    loop = _loop(provider, tmp_path)

    async def _drive() -> None:
        async for _event in loop.astream_turn("hello"):
            pass

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        asyncio.run(_drive())

    lines = _turn_lines(caplog)
    assert len(lines) == 1, "the streamed path logs exactly one turn-ended line per turn"
    assert lines[0]["stop_reason"] == "completed"
    assert float(lines[0]["duration_s"]) >= 0.0


def test_instrument_rejects_the_pre_duration_line() -> None:
    """Positive control for the regex: the OLD line shape must NOT parse.

    Without this, a pattern that happened to match the duration-less line would
    make both tests above pass over an unchanged loop (guard-5501 class: prove
    the diagnostic can fail before trusting that it fired).
    """
    old = (
        "2026-09-19 01:57:32,426 INFO zakcode.agent.loop "
        "turn ended: stop_reason=veto_stall iterations=31 tokens=1772655"
    )
    new = old + " duration_s=412.500"
    assert TURN_LINE_RE.search(old) is None
    m = TURN_LINE_RE.search(new)
    assert m is not None
    assert m["stop_reason"] == "veto_stall"
    assert m["iterations"] == "31"
    assert float(m["duration_s"]) == 412.5
