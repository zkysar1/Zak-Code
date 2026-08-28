"""The two behavior changes ADR-0053 deferred, now landed (ADR-0058).

F9 — cross-gate cascade cap: the six evidence gates (claim, blocker, missing, identity,
figure, intent) each fire once per turn, so a model that answers every nudge in words
could be re-prompted six times in a row, each time in a different direction. Past
``_MAX_GATE_CASCADE`` consecutive text-only completions the gates stand down and the
answer stands, degraded and traced. A tool batch resets the count.

F11 — a glob is a search, and so is a read that actually returned content: the
missing-conclusion gate (ADR-0040) no longer asks a model that globbed for the name, or
read the real file, to go and grep. A FAILED read stays the one-path-tried miss.

Hermetic: scripted providers, in-memory tools, a tmp workspace.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from zakcode.agent.loop import (
    _IDENTITY_NUDGE,
    _INTENT_NUDGE,
    _MAX_GATE_CASCADE,
    _MISSING_NUDGE,
    AgentLoop,
)
from zakcode.events import AgentDone, AgentStatus
from zakcode.messages import Message
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.store import Session
from zakcode.tools.base import (
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)

INTENT = "Now I will update the config file to add the missing key."
MISSING = "The config file could not be found anywhere in the workspace."
IDENTITY = "settings.py is a python file, not a skill."


class _Echo(Tool):
    spec = ToolSpec(name="echo", description="Echo.", concurrency=ConcurrencyClass.READ_ONLY_SAFE)

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(str(args.get("text", "")))


class _Glob(Tool):
    spec = ToolSpec(
        name="glob", description="Path search.", concurrency=ConcurrencyClass.READ_ONLY_SAFE
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok("(no matches)")


class _ReadFile(Tool):
    spec = ToolSpec(
        name="read_file", description="Read.", concurrency=ConcurrencyClass.READ_ONLY_SAFE
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if args.get("path") == "missing.txt":
            return ToolResult.error("no such file: missing.txt")
        return ToolResult.ok("key = value")


class _Sequence(Provider):
    """Plays back scripted completions in order; repeats the last one forever."""

    def __init__(self, *results: LLMResult) -> None:
        self._results = list(results)
        self.calls = 0

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:
        self.calls += 1
        return self._results[min(self.calls, len(self._results)) - 1]

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=200_000)


def _text(text: str) -> LLMResult:
    return LLMResult(text=text, finish_reason="stop")


def _call(name: str, **args: object) -> LLMResult:
    return LLMResult(
        tool_calls=[ToolCall(id=f"c-{name}", name=name, arguments=dict(args))],
        finish_reason="tool_calls",
    )


def _loop(tmp_path: Path, provider: Provider) -> AgentLoop:
    registry = ToolRegistry()
    for tool in (_Echo(), _Glob(), _ReadFile()):
        registry.register(tool)
    return AgentLoop(
        provider,
        registry,
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        max_iterations=20,
    )


def _rails(loop: AgentLoop) -> list[str]:
    return [m.text for m in loop.session.messages if m.role == "user" and m.text]


def _count(loop: AgentLoop, nudge: str) -> int:
    return sum(nudge in r for r in _rails(loop))


# ── F9: the cascade cap ───────────────────────────────────────────────────────


def test_a_third_text_only_completion_is_not_re_prompted_in_a_third_direction(
    tmp_path: Path,
) -> None:
    assert _MAX_GATE_CASCADE == 2  # the shape below is written for two nudges, then a stand-down
    provider = _Sequence(_text(INTENT), _text(MISSING), _text(IDENTITY))
    loop = _loop(tmp_path, provider)
    result = asyncio.run(loop.arun_turn("fix the config"))
    assert provider.calls == 3  # intent nudge, missing nudge, then the answer stands
    assert result.stop_reason == "completed"
    assert result.degraded is True  # a capped cascade is an honest "struggled"
    assert _count(loop, _INTENT_NUDGE) == 1
    assert _count(loop, _MISSING_NUDGE) == 1
    assert _count(loop, _IDENTITY_NUDGE) == 0  # the third gate stood down
    assert any(e.data.get("kind") == "gate_cascade" for e in result.trace.events)


def test_a_tool_batch_resets_the_cascade_count(tmp_path: Path) -> None:
    provider = _Sequence(
        _text(INTENT),
        _call("echo", text="working"),
        _text(MISSING),
        _text(IDENTITY),
        _text("Done."),
    )
    loop = _loop(tmp_path, provider)
    result = asyncio.run(loop.arun_turn("fix the config"))
    assert provider.calls == 5
    assert result.stop_reason == "completed"
    assert _count(loop, _INTENT_NUDGE) == 1
    assert _count(loop, _MISSING_NUDGE) == 1
    assert _count(loop, _IDENTITY_NUDGE) == 1  # real work in between kept every gate armed


def test_two_nudges_then_a_clean_answer_is_not_degraded_by_the_cap(tmp_path: Path) -> None:
    # The cap only marks a turn degraded when it actually stands a gate down.
    provider = _Sequence(_text(INTENT), _call("echo", text="ok"), _text("All set."))
    loop = _loop(tmp_path, provider)
    result = asyncio.run(loop.arun_turn("fix the config"))
    assert result.stop_reason == "completed"
    assert not any(e.data.get("kind") == "gate_cascade" for e in result.trace.events)


def test_streaming_twin_announces_the_stand_down(tmp_path: Path) -> None:
    provider = _Sequence(_text(INTENT), _text(MISSING), _text(IDENTITY))
    loop = _loop(tmp_path, provider)

    async def run() -> list[Any]:
        return [ev async for ev in loop.astream_turn("fix the config")]

    events = asyncio.run(run())
    done = next(e for e in events if isinstance(e, AgentDone))
    assert done.stop_reason == "completed" and done.degraded is True
    assert provider.calls == 3
    statuses = [e.message for e in events if isinstance(e, AgentStatus)]
    assert any("evidence gates stand down" in s for s in statuses)
    assert _count(loop, _IDENTITY_NUDGE) == 0


# ── F11: what counts as a search ──────────────────────────────────────────────


def test_a_glob_this_turn_earns_a_not_found_conclusion(tmp_path: Path) -> None:
    provider = _Sequence(_call("glob", pattern="**/*config*"), _text(MISSING))
    loop = _loop(tmp_path, provider)
    asyncio.run(loop.arun_turn("find the config"))
    assert provider.calls == 2
    assert _count(loop, _MISSING_NUDGE) == 0


def test_a_read_that_returned_content_earns_the_conclusion(tmp_path: Path) -> None:
    provider = _Sequence(_call("read_file", path="settings.toml"), _text(MISSING))
    loop = _loop(tmp_path, provider)
    asyncio.run(loop.arun_turn("is the key in the settings"))
    assert provider.calls == 2
    assert _count(loop, _MISSING_NUDGE) == 0


def test_a_failed_read_is_still_the_one_path_tried_miss(tmp_path: Path) -> None:
    provider = _Sequence(
        _call("read_file", path="missing.txt"), _text(MISSING), _text("Still nothing.")
    )
    loop = _loop(tmp_path, provider)
    asyncio.run(loop.arun_turn("find the config"))
    assert provider.calls == 3  # nudged once for the grep, then the turn ended
    assert _count(loop, _MISSING_NUDGE) == 1
