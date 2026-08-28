"""The busy marker (ADR-0060): the turn in flight owns the say inbox.

Two consumers can share one workspace — a Mind runner whose whole night is one turn, and
a cockpit chat pane polling the inbox every 0.3 s between ITS turns — and the single
slot then goes to whoever reads first, which is always the idle one. Measured 2026-08-28
(coach on zc-03): every operator say of a morning reached the cockpit pane, none the
runner they were steering; one was a control command that flipped the runner's shared
mode file from under it.

A main-loop turn now claims ``<workspace>/.busy`` for its length; idle consumers (the
REPL mux between turns, the serve consumer beat) stand back while a FRESH marker names
another process; the holder's own mid-turn poll (ADR-0051) still takes the say. A
crashed holder's marker ages out. Hermetic: tmp workspaces, scripted providers.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from zakcode.agent.loop import AgentLoop
from zakcode.config import Settings
from zakcode.events import AgentDone, AgentEvent, AgentTextDelta
from zakcode.messages import Message
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.server.app import create_app
from zakcode.session.say_inbox import (
    BUSY_STALE_SECONDS,
    BusyLease,
    busy_elsewhere,
    busy_path,
    claim_busy,
    interrupt_path,
    refresh_busy,
    release_busy,
    say_path,
    say_pending,
    write_say,
)
from zakcode.session.store import Session, SessionStore
from zakcode.tools.base import Tool, ToolContext, ToolRegistry, ToolResult, ToolSpec
from zakcode.usage import Usage


def _foreign_marker(workspace: Path, *, age: float = 0.0) -> Path:
    """A fresh (or aged) marker written by some OTHER process."""
    path = busy_path(workspace)
    payload = {"pid": os.getpid() + 100_000, "session": "other", "since": time.time()}
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    if age:
        stamp = time.time() - age
        os.utime(path, (stamp, stamp))
    return path


# ── the marker primitives ─────────────────────────────────────────────────────


def test_claim_refresh_release_round_trip(tmp_path: Path) -> None:
    path = busy_path(tmp_path)
    assert not busy_elsewhere(path)
    assert claim_busy(path, "s1")
    assert not busy_elsewhere(path)  # our own marker is never "elsewhere"
    assert json.loads(path.read_text(encoding="utf-8"))["pid"] == os.getpid()
    old = time.time() - 90
    os.utime(path, (old, old))
    refresh_busy(path)
    assert time.time() - path.stat().st_mtime < 5  # touched back to now
    release_busy(path)
    assert not path.exists()


def test_a_fresh_foreign_marker_owns_the_inbox(tmp_path: Path) -> None:
    path = _foreign_marker(tmp_path)
    assert busy_elsewhere(path)
    assert not claim_busy(path, "s1")  # that turn owns it; this one runs without a claim
    refresh_busy(path)
    release_busy(path)  # never touch a marker that is not ours
    assert path.exists()


def test_a_stale_foreign_marker_names_nobody(tmp_path: Path) -> None:
    path = _foreign_marker(tmp_path, age=BUSY_STALE_SECONDS + 5)
    assert not busy_elsewhere(path)  # the holder crashed or hung; it aged out
    assert claim_busy(path, "s1")
    assert json.loads(path.read_text(encoding="utf-8"))["pid"] == os.getpid()


def test_a_garbage_marker_is_fail_open(tmp_path: Path) -> None:
    path = busy_path(tmp_path)
    path.write_text("not json", encoding="utf-8")
    assert not busy_elsewhere(path)


def test_lease_holds_for_its_scope_and_is_idempotent(tmp_path: Path) -> None:
    async def run() -> None:
        lease = BusyLease(busy_path(tmp_path), "s1")
        await lease.acquire()
        assert lease.held and busy_path(tmp_path).exists()
        await lease.release()
        assert not busy_path(tmp_path).exists()
        await lease.release()  # a second release is a no-op

    asyncio.run(run())


def test_lease_yields_to_a_foreign_holder(tmp_path: Path) -> None:
    _foreign_marker(tmp_path)

    async def run() -> None:
        lease = BusyLease(busy_path(tmp_path), "s1")
        await lease.acquire()
        assert not lease.held
        await lease.release()
        assert busy_path(tmp_path).exists()  # theirs, untouched

    asyncio.run(run())


# ── the loop holds it for a turn ──────────────────────────────────────────────


class _Peek(Tool):
    """Records who holds the busy marker at the moment the tool runs (mid-turn)."""

    spec = ToolSpec(name="peek", description="Peek at the busy marker.")

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.seen: list[int | None] = []

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = busy_path(self.workspace)
        holder = json.loads(path.read_text(encoding="utf-8"))["pid"] if path.exists() else None
        self.seen.append(holder)
        return ToolResult.ok("peeked")


class _SayWhileRunning(Tool):
    """Writes a say mid-turn — an operator steering the running agent."""

    spec = ToolSpec(name="steer", description="Write a say while the turn runs.")

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        assert write_say(say_path(self.workspace), "steer me")
        return ToolResult.ok("written")


class _Script(Provider):
    def __init__(self, results: list[LLMResult]) -> None:
        self._results = results
        self.calls = 0

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:
        i = min(self.calls, len(self._results) - 1)
        self.calls += 1
        return self._results[i]

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities()


def _loop(workspace: Path, provider: Provider, tool: Tool, *, consume: bool) -> AgentLoop:
    registry = ToolRegistry()
    registry.register(tool)
    return AgentLoop(
        provider,
        registry,
        Session(cwd=str(workspace), model="test"),
        workspace_root=workspace,
        max_iterations=6,
        consume_say_inbox=consume,
    )


def _peek_script(name: str = "peek") -> _Script:
    return _Script(
        [
            LLMResult(tool_calls=[ToolCall(id="c1", name=name, arguments={})]),
            LLMResult(text="done"),
        ]
    )


def test_main_loop_holds_the_marker_for_the_turn(tmp_path: Path) -> None:
    tool = _Peek(tmp_path)
    loop = _loop(tmp_path, _peek_script(), tool, consume=True)
    result = asyncio.run(loop.arun_turn("go"))
    assert result.stop_reason == "completed"
    assert tool.seen == [os.getpid()]  # held mid-turn, by this process
    assert not busy_path(tmp_path).exists()  # released at turn end


def test_bare_loop_never_claims(tmp_path: Path) -> None:
    # Sub-agents and bare loops do not consume the inbox, so they must not own it either.
    tool = _Peek(tmp_path)
    loop = _loop(tmp_path, _peek_script(), tool, consume=False)
    asyncio.run(loop.arun_turn("go"))
    assert tool.seen == [None]


def test_streaming_twin_holds_and_releases(tmp_path: Path) -> None:
    tool = _Peek(tmp_path)
    loop = _loop(tmp_path, _peek_script(), tool, consume=True)

    async def run() -> list[Any]:
        return [ev async for ev in loop.astream_turn("go")]

    events = asyncio.run(run())
    assert any(isinstance(ev, AgentDone) for ev in events)
    assert tool.seen == [os.getpid()]
    assert not busy_path(tmp_path).exists()


def test_the_holder_still_takes_a_say_written_mid_turn(tmp_path: Path) -> None:
    # ADR-0051 is untouched: the busy loop's own boundary poll delivers the say.
    tool = _SayWhileRunning(tmp_path)
    loop = _loop(tmp_path, _peek_script("steer"), tool, consume=True)
    asyncio.run(loop.arun_turn("go"))
    assert not say_pending(say_path(tmp_path))
    assert any("steer me" in (m.text or "") for m in loop.session.messages if m.role == "user")


# ── idle consumers stand back ─────────────────────────────────────────────────


def test_idle_repl_mux_stands_back_while_another_turn_runs(tmp_path: Path) -> None:
    from zakcode.cli import _InputMux

    mux = _InputMux(say_path(tmp_path), interrupt_path(tmp_path), keyboard=False)
    assert write_say(say_path(tmp_path), "for the runner")
    marker = _foreign_marker(tmp_path)
    assert mux.try_input() is None  # the say stays for the turn that owns the inbox
    assert say_pending(say_path(tmp_path))
    stale = time.time() - BUSY_STALE_SECONDS - 5
    os.utime(marker, (stale, stale))  # the holder died; the pane is the consumer again
    assert mux.try_input() == ("say", "for the runner")


class _FakeAgent:
    def __init__(self, session: Session) -> None:
        self.session = session

    async def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]:
        self.session.add_message(Message.user(user_text))
        self.session.add_message(Message.assistant_text("ok"))
        yield AgentTextDelta(text="ok")
        yield AgentDone(stop_reason="completed", iterations=1, usage=Usage())


def test_serve_consumer_beat_yields_to_another_process_turn(tmp_path: Path) -> None:
    settings = Settings(default_model="scripted/test", workspace_root=tmp_path)
    store = SessionStore(base_dir=tmp_path / "sessions")
    app = create_app(
        settings=settings,
        store=store,
        agent_factory=lambda session, model, prompter: _FakeAgent(session),
    )
    assert write_say(say_path(tmp_path), "hello runner")
    marker = _foreign_marker(tmp_path)
    assert asyncio.run(app.state.consume_one_say()) is False
    assert say_pending(say_path(tmp_path))  # left for the turn that owns the inbox
    marker.unlink()
    assert asyncio.run(app.state.consume_one_say()) is True
    assert not say_pending(say_path(tmp_path))
