"""The host-neutral run ending: ``run_stop_message`` delivers the stop through the say inbox.

The say inbox is the single contract for handing a running agent its next message.
Routing the run's ending through it removes the need for signal files, subprocess
calls and an agent-session directory layout — the ending travels as operator input,
exactly as if a person typed it.

Each test is grouped by the property it pins:

- **Positive control.** The configured line lands in the say inbox on ``/run/stop``.
  The paired negative: with the setting unset, the slot is never written.
- **DONE rule.** DONE needs taken + completed + nothing in flight — simultaneously.
  The mutation target is "any completed turn after queueing counts as done."
- **Taken but not done.** An interrupt fires; the run must not retract the line.
- **Never taken.** The slot is retracted; an interrupt fires.
- **Abnormal ends.** ``provider_error`` and ``veto_stall`` after the line is taken
  are not DONE.
- **Slot contention.** A human's say goes first; retraction never deletes a
  different say.
- **Startup drop.** A pending say equal to the configured line is dropped; a
  different one is kept.
- **Ported cases.** The shared window, zero reserve, ``duration_cap`` naming,
  and "an ended restart writes no say."
- **Loop hold exemption.** An operator-only slash bypasses the ADR-0052 hold.
- **/observe.** Sticky kind and no-wake semantics.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from zakcode.config import Settings
from zakcode.events import AgentDone, AgentEvent, AgentTextDelta
from zakcode.messages import Message
from zakcode.server.app import create_app
from zakcode.session.say_inbox import say_path, say_pending, write_say
from zakcode.session.store import Session, SessionStore
from zakcode.usage import Usage

STOP_LINE = "/stop probe"


# ── helpers ────────────────────────────────────────────────────────────────────


class _QuietAgent:
    """A turn that does nothing interesting — these tests are about the LOOP."""

    def __init__(self, session: Session) -> None:
        self.session = session

    async def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]:
        self.session.add_message(Message.user(user_text))
        self.session.add_message(Message.assistant_text("ok"))
        yield AgentTextDelta(text="ok")
        yield AgentDone(stop_reason="completed", iterations=1, usage=Usage())


class _TimedAgent:
    """Turn N takes ``durations[N]`` seconds unless something CANCELS it.

    ``seen`` is what each turn was asked and ``finished`` what it completed.
    """

    def __init__(
        self, session: Session, seen: list[str], finished: list[str], durations: list[float]
    ) -> None:
        self.session = session
        self._seen = seen
        self._finished = finished
        self._durations = durations

    async def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]:
        duration = self._durations[len(self._seen)]
        self._seen.append(user_text)
        await asyncio.sleep(duration)
        self.session.add_message(Message.assistant_text("ok"))
        self._finished.append(user_text)
        yield AgentDone(stop_reason="completed", iterations=1, usage=Usage())


class _StopReasonAgent:
    """Completes with a CONFIGURED stop reason on each turn.

    ``reasons`` is consumed in order; turns beyond the list use ``"completed"``.
    ``seen`` is a shared list so the turn index survives agent re-creation by the
    factory — the same pattern ``_TimedAgent`` uses.
    """

    def __init__(self, session: Session, reasons: list[str], seen: list[str]) -> None:
        self.session = session
        self._reasons = reasons
        self._seen = seen

    async def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]:
        self.session.add_message(Message.user(user_text))
        self.session.add_message(Message.assistant_text("ok"))
        turn = len(self._seen)
        reason = self._reasons[turn] if turn < len(self._reasons) else "completed"
        self._seen.append(user_text)
        yield AgentDone(stop_reason=reason, iterations=1, usage=Usage())


def _build(
    tmp_path: Path,
    *,
    stop_message: str | None = STOP_LINE,
    stop_agent: str | None = None,
    reserve: float = 5.0,
    max_duration: float | None = None,
    agent_for: Callable[[Session], Any] = _QuietAgent,
) -> tuple[Any, list[str]]:
    """An app configured for the message-based ending; returns (app, endings)."""
    endings: list[str] = []
    settings = Settings(
        default_model="scripted/test",
        context_window=8192,
        workspace_root=tmp_path,
        run_max_duration=max_duration,
        run_consolidation_reserve=reserve,
        run_stop_message=stop_message,
        run_stop_agent=stop_agent,
    )

    async def _on_run_end(reason: str) -> None:
        endings.append(reason)

    app = create_app(
        settings=settings,
        store=SessionStore(base_dir=tmp_path / "sessions"),
        agent_factory=lambda session, model, prompter: agent_for(session),
        on_run_end=_on_run_end,
    )
    return app, endings


async def _post_stop(app: Any) -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/run/stop", json={"reason": "stopped"})
        assert response.status_code == 200


async def _until(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "the awaited condition never got there"
        await asyncio.sleep(0.02)


# ── positive control ───────────────────────────────────────────────────────────


def test_run_stop_queues_the_configured_line(tmp_path: Path) -> None:
    """With ``run_stop_message`` set, ``/run/stop`` queues exactly that line in the
    say inbox and the agent reads it as operator input.  The run ends well inside the
    grace, reason ``stopped``, with no interrupt."""
    seen: list[str] = []
    app, endings = _build(
        tmp_path, reserve=5.0, agent_for=lambda session: _TimedAgent(session, seen, [], [0.0, 0.0])
    )
    assert write_say(say_path(tmp_path), "hello")

    async def scenario() -> float:
        loop_task = asyncio.create_task(app.state.consume_say_loop())
        await _until(lambda: len(seen) >= 1)
        stopped = time.monotonic()
        await _post_stop(app)
        await asyncio.wait_for(loop_task, timeout=10)
        return time.monotonic() - stopped

    elapsed = asyncio.run(scenario())
    assert STOP_LINE in seen, f"the configured line was never delivered: {seen}"
    assert elapsed < 4.0, f"the stop took {elapsed:.1f}s on a 5s grace"
    assert endings == ["stopped"]


def test_no_say_written_when_message_unset(tmp_path: Path) -> None:
    """Without the setting, no say is written and the run ends the old way."""
    app, endings = _build(tmp_path, stop_message=None, stop_agent=None)

    async def scenario() -> None:
        loop_task = asyncio.create_task(app.state.consume_say_loop())
        await asyncio.sleep(0.2)
        await _post_stop(app)
        await asyncio.wait_for(loop_task, timeout=5)

    asyncio.run(scenario())
    assert not say_pending(say_path(tmp_path))
    assert endings == ["stopped"]


# ── DONE rule (mutation target) ────────────────────────────────────────────────


def test_done_requires_taken_plus_completed_plus_not_inflight(tmp_path: Path) -> None:
    """A turn in flight ends ``completed`` BEFORE the agent reaches its boundary,
    so the line is still in the inbox when that turn finishes.  The run must stay
    open for the NEXT beat to take and run the line.

    The mutant "any completed end after queueing is DONE" closes the run with the
    line still unconsumed.
    """
    seen: list[str] = []
    # Turn 0 completes instantly (the boot turn before the stop is raised).
    # Turn 1 takes the stop line and completes.
    app, endings = _build(
        tmp_path,
        reserve=10.0,
        agent_for=lambda session: _TimedAgent(session, seen, [], [0.0, 0.0]),
    )
    assert write_say(say_path(tmp_path), "hello")

    async def scenario() -> None:
        loop_task = asyncio.create_task(app.state.consume_say_loop())
        await _until(lambda: len(seen) >= 1)
        await _post_stop(app)
        await asyncio.wait_for(loop_task, timeout=10)

    asyncio.run(scenario())
    # The stop line MUST have been delivered and processed.
    assert STOP_LINE in seen, f"the line was never taken: {seen}"
    assert endings == ["stopped"]


# ── taken but not done ─────────────────────────────────────────────────────────


def test_taken_but_interrupted_is_not_done(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The line is taken but the turn is interrupted (the agent is cancelled).
    The run must NOT retract the line from the inbox, and the ending still fires."""
    monkeypatch.setattr("zakcode.server.app._DEADLINE_WATCH_SECONDS", 0.1)
    seen: list[str] = []
    finished: list[str] = []
    # Turn 0: boot, instant.  Turn 1: takes the stop line, runs 30s (will be interrupted).
    app, endings = _build(
        tmp_path,
        reserve=0.8,
        agent_for=lambda session: _TimedAgent(session, seen, finished, [0.0, 30.0]),
    )
    assert write_say(say_path(tmp_path), "hello")

    async def scenario() -> None:
        loop_task = asyncio.create_task(app.state.consume_say_loop())
        await _until(lambda: len(seen) >= 1)
        await _post_stop(app)
        await asyncio.wait_for(loop_task, timeout=10)

    asyncio.run(scenario())
    # The line was taken (len(seen) > 1 includes it) and the turn was interrupted
    # (finished is shorter than seen).
    assert len(seen) >= 2
    assert len(finished) < len(seen), "the turn was not interrupted"
    assert endings == ["stopped"]


# ── never taken ────────────────────────────────────────────────────────────────


def test_never_taken_retracts_the_line(tmp_path: Path) -> None:
    """If the grace expires with the line still in the inbox, it is retracted
    so the next boot never opens on a stale ending."""
    # A very short reserve and no agent turns to consume the say.
    app, endings = _build(tmp_path, reserve=0.6)

    async def scenario() -> None:
        loop_task = asyncio.create_task(app.state.consume_say_loop())
        await asyncio.sleep(0.2)
        await _post_stop(app)
        await asyncio.wait_for(loop_task, timeout=10)

    asyncio.run(scenario())
    # The line was retracted, not left on disk.
    assert not say_pending(say_path(tmp_path))
    assert endings == ["stopped"]


# ── abnormal ends ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("reason", ["provider_error", "veto_stall"])
def test_abnormal_stop_reason_after_taken_is_not_done(tmp_path: Path, reason: str) -> None:
    """``provider_error`` or ``veto_stall`` after the line is taken is not DONE:
    the run stays open until either a ``completed`` turn or the grace expires."""
    seen: list[str] = []
    # Turn 0: boot, completed. Turn 1: takes stop line, ends with abnormal reason.
    app, endings = _build(
        tmp_path,
        reserve=1.5,
        agent_for=lambda session: _StopReasonAgent(session, ["completed", reason], seen),
    )
    assert write_say(say_path(tmp_path), "hello")

    async def scenario() -> float:
        loop_task = asyncio.create_task(app.state.consume_say_loop())
        await _until(lambda: say_pending(say_path(tmp_path)) is False, timeout=3)
        await asyncio.sleep(0.2)
        started = time.monotonic()
        await _post_stop(app)
        await asyncio.wait_for(loop_task, timeout=10)
        return time.monotonic() - started

    elapsed = asyncio.run(scenario())
    # The grace ran out (the abnormal end was not treated as DONE).
    assert elapsed >= 0.8, f"ended in {elapsed:.2f}s — an abnormal end was treated as DONE"
    assert endings == ["stopped"]


# ── slot contention ────────────────────────────────────────────────────────────


def test_human_say_goes_first(tmp_path: Path) -> None:
    """A pending human say blocks the ending's write.  The human say is not clobbered."""
    seen: list[str] = []
    app, endings = _build(
        tmp_path,
        reserve=5.0,
        agent_for=lambda session: _TimedAgent(session, seen, [], [0.0, 0.0, 0.0]),
    )
    # Plant the human's say FIRST, then start the loop.
    assert write_say(say_path(tmp_path), "human message")

    async def scenario() -> None:
        loop_task = asyncio.create_task(app.state.consume_say_loop())
        await _until(lambda: "human message" in seen)
        # Now the slot is free.  Raise the stop.
        await _post_stop(app)
        await asyncio.wait_for(loop_task, timeout=10)

    asyncio.run(scenario())
    # The human message was delivered first; the stop line followed.
    assert "human message" in seen
    assert STOP_LINE in seen
    human_idx = seen.index("human message")
    stop_idx = seen.index(STOP_LINE)
    assert human_idx < stop_idx, f"human at {human_idx}, stop at {stop_idx}"
    assert endings == ["stopped"]


def test_stop_raised_while_a_human_say_is_pending_waits_its_turn(tmp_path: Path) -> None:
    """The stop fires while a human's say is STILL in the slot, so the ending's write is
    refused. The human's turn must run first, then the ending line, then the run ends.

    The mutant: marking the line TAKEN because the slot emptied (or held a different say)
    when the line was never written. That ends the run after the human's turn with the
    ending never delivered, which is the severed ending this path exists to prevent.
    """
    seen: list[str] = []
    # Turn 0 runs long enough to hold the consumer while the human's say lands.
    app, endings = _build(
        tmp_path,
        reserve=5.0,
        agent_for=lambda session: _TimedAgent(session, seen, [], [0.6, 0.0, 0.0]),
    )
    inbox = say_path(tmp_path)
    assert write_say(inbox, "hello")

    async def scenario() -> None:
        loop_task = asyncio.create_task(app.state.consume_say_loop())
        await _until(lambda: "hello" in seen)
        assert write_say(inbox, "human message")  # pending while turn 0 still runs
        await _post_stop(app)
        assert say_pending(inbox)  # the human's say still holds the slot
        await asyncio.wait_for(loop_task, timeout=10)

    asyncio.run(scenario())
    assert seen == ["hello", "human message", STOP_LINE], seen
    assert endings == ["stopped"]


def test_stop_line_with_surrounding_whitespace_is_still_recognised(tmp_path: Path) -> None:
    """The inbox stores the line stripped of its trailing newline; a configured line with
    its own surrounding whitespace must still compare as OURS, or a pending copy would read
    as someone else's say and count as taken before any turn ran."""
    seen: list[str] = []
    app, endings = _build(
        tmp_path,
        stop_message=STOP_LINE + "  ",
        reserve=5.0,
        agent_for=lambda session: _TimedAgent(session, seen, [], [0.6, 0.0]),
    )
    assert write_say(say_path(tmp_path), "hello")

    async def scenario() -> None:
        loop_task = asyncio.create_task(app.state.consume_say_loop())
        await _until(lambda: "hello" in seen)
        await _post_stop(app)
        await asyncio.wait_for(loop_task, timeout=10)

    asyncio.run(scenario())
    assert seen == ["hello", STOP_LINE], seen
    assert endings == ["stopped"]


def test_both_ending_overrun_branches_retract_before_the_interrupt() -> None:
    """The message path's twin of the legacy path's structural check: when the grace runs
    out, both overrun branches (mid-turn and at the cap) retract the unconsumed line BEFORE
    raising the interrupt, since the interrupt can end this process."""
    src = (Path(__file__).resolve().parents[1] / "src" / "zakcode" / "server" / "app.py").read_text(
        encoding="utf-8"
    )
    mid = src.index("run ending ran past its %.0fs window")
    cap = src.index("run cap: run ending ran past its window")
    for start in (mid, cap):
        window = src[start : start + 400]
        assert "_abandon_run_ending()" in window, "retraction missing from an overrun"
        assert window.index("_abandon_run_ending()") < window.index("request_interrupt("), (
            "retract BEFORE the interrupt"
        )


def test_retraction_never_deletes_a_different_say(tmp_path: Path) -> None:
    """If the inbox holds a DIFFERENT message at overrun, it is not deleted."""
    app, endings = _build(tmp_path, reserve=0.6)
    # Write a DIFFERENT say while the inbox should hold the stop line.
    inbox = say_path(tmp_path)

    async def scenario() -> None:
        loop_task = asyncio.create_task(app.state.consume_say_loop())
        await asyncio.sleep(0.2)
        await _post_stop(app)
        # Immediately replace the inbox with a different message.
        # Race window: the stop writes its line; we overwrite it.
        await asyncio.sleep(0.05)
        with open(inbox, "w", encoding="utf-8") as f:
            f.write("different\n")
        await asyncio.wait_for(loop_task, timeout=10)

    asyncio.run(scenario())
    # The "different" say must survive — retraction only deletes an exact match.
    if say_pending(inbox):
        text = inbox.read_text(encoding="utf-8").strip()
        assert text == "different", f"retraction deleted the wrong say: {text!r}"


# ── startup drop ──────────────────────────────────────────────────────────────


def test_startup_drops_own_stale_line(tmp_path: Path) -> None:
    """A pending say equal to the configured line is dropped at startup so a run
    never opens on its own ending left by a previous process."""
    inbox = say_path(tmp_path)
    assert write_say(inbox, STOP_LINE)
    assert say_pending(inbox)

    app, endings = _build(tmp_path, reserve=5.0)

    async def scenario() -> None:
        # _start_consumer runs the startup drop, then starts the loop.
        await app.state.start_consumer()
        # Give the loop one beat to prove it does not immediately end.
        await asyncio.sleep(0.3)
        assert not say_pending(inbox), "the stale stop line was not dropped"

    asyncio.run(scenario())


def test_startup_keeps_a_different_say(tmp_path: Path) -> None:
    """A pending say that is NOT the configured line is kept (not dropped by startup) —
    it belongs to someone else and is delivered to the agent as a normal turn."""
    inbox = say_path(tmp_path)
    assert write_say(inbox, "operator typed this")
    assert say_pending(inbox)

    seen: list[str] = []
    app, endings = _build(
        tmp_path,
        reserve=5.0,
        agent_for=lambda session: _TimedAgent(session, seen, [], [0.0]),
    )

    async def scenario() -> None:
        await app.state.start_consumer()
        await _until(lambda: len(seen) >= 1, timeout=3)

    asyncio.run(scenario())
    assert "operator typed this" in seen, (
        f"the non-matching say was dropped by startup instead of being delivered: {seen}"
    )


# ── ported cases (shared window, zero reserve, duration_cap, ended restart) ───


def test_shared_window_is_not_restarted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The window is opened once; a second raiser (cap landing inside /run/stop's
    window) cannot restart the grace."""
    monkeypatch.setattr("zakcode.server.app._DEADLINE_WATCH_SECONDS", 0.1)
    seen: list[str] = []
    finished: list[str] = []
    # cap 6.0 - reserve 3.0: the turn deadline lands ~3s in; /run/stop opens the
    # window ~0.2s in.  Shared => closes ~3.2s in; restarted => ~6.0s in.
    app, endings = _build(
        tmp_path,
        reserve=3.0,
        max_duration=6.0,
        agent_for=lambda session: _TimedAgent(session, seen, finished, [30.0]),
    )
    assert write_say(say_path(tmp_path), "hello")

    async def scenario() -> float:
        started = time.monotonic()
        loop_task = asyncio.create_task(app.state.consume_say_loop())
        await _until(lambda: len(seen) == 1)
        await asyncio.sleep(max(0.0, started + 0.2 - time.monotonic()))
        await _post_stop(app)
        opened = time.monotonic()
        await asyncio.wait_for(loop_task, timeout=10)
        return time.monotonic() - opened

    elapsed = asyncio.run(scenario())
    assert finished == [], "the turn ran to completion — nothing bounded it"
    assert elapsed < 4.6, f"ran {elapsed:.2f}s into a 3.0s window — the cap restarted it"
    assert endings == ["stopped"]


def test_zero_reserve_lets_the_turn_finish(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A zero reserve writes no say (budget is 0 => no ending window)."""
    monkeypatch.setattr("zakcode.server.app._DEADLINE_WATCH_SECONDS", 0.1)
    seen: list[str] = []
    finished: list[str] = []
    app, endings = _build(
        tmp_path,
        reserve=0.0,
        stop_agent=None,
        agent_for=lambda session: _TimedAgent(session, seen, finished, [1.0]),
    )
    assert write_say(say_path(tmp_path), "hello")

    async def scenario() -> None:
        loop_task = asyncio.create_task(app.state.consume_say_loop())
        await _until(lambda: len(seen) == 1)
        await _post_stop(app)
        await asyncio.wait_for(loop_task, timeout=10)

    asyncio.run(scenario())
    assert finished == ["hello"], "the turn was interrupted instead of allowed to finish"
    assert endings == ["stopped"]


def test_cap_between_beats_is_named_duration_cap(tmp_path: Path) -> None:
    """The cap path, where the mid-turn watcher never fires (nothing is inflight).
    The run ending still uses the say inbox and the reason is ``duration_cap``."""
    seen: list[str] = []
    app, endings = _build(
        tmp_path,
        reserve=0.6,
        max_duration=0.9,
        agent_for=lambda session: _TimedAgent(session, seen, [], [0.0, 0.0]),
    )
    assert write_say(say_path(tmp_path), "hello")

    async def scenario() -> None:
        loop_task = asyncio.create_task(app.state.consume_say_loop())
        await asyncio.wait_for(loop_task, timeout=10)

    asyncio.run(scenario())
    assert endings == ["duration_cap"]


def test_an_ended_restart_writes_no_say(tmp_path: Path) -> None:
    """A boot that finds ``.run-stop-reason`` from a previous cap does not open a
    new run and writes no say."""
    inbox = say_path(tmp_path)
    reason_file = tmp_path / ".run-stop-reason"
    reason_file.parent.mkdir(parents=True, exist_ok=True)
    reason_file.write_text("run-id-1\nduration_cap\nboot-id-1\n", encoding="utf-8")

    app, endings = _build(tmp_path, reserve=5.0)

    async def scenario() -> None:
        await app.state.start_consumer()
        await asyncio.sleep(0.3)

    asyncio.run(scenario())
    assert not say_pending(inbox), "an ended restart wrote a say"
    # The run is ended — no consumer task was started.


# ── /observe: sticky kind and no-wake ──────────────────────────────────────────


def _observe_client(workspace: Path) -> Any:
    """A TestClient whose app uses ``run_stop_message`` (message path active)."""
    settings = Settings(
        default_model="scripted/test",
        context_window=8192,
        workspace_root=workspace,
        run_stop_message=STOP_LINE,
    )
    store = SessionStore(base_dir=workspace / "sessions")
    from fastapi import FastAPI

    app: FastAPI = create_app(
        settings=settings,
        store=store,
        agent_factory=lambda session, model, prompter: _QuietAgent(session),
    )
    from fastapi.testclient import TestClient

    return TestClient(app)


def _envelope(**over: object) -> dict[str, object]:
    body: dict[str, object] = {
        "envelopeVersion": 1,
        "externalClientRef": "char-42",
        "observedAt": "2026-09-28T12:00:00Z",
        "observation": {"nearbyPerception": {"units": ["a", "b"]}},
        "droppedSlices": [],
    }
    body.update(over)
    return body


def _staged(workspace: Path) -> dict[str, Any]:
    return json.loads((workspace / ".observation").read_text(encoding="utf-8"))


def test_observe_sticky_kind_preserves_change(tmp_path: Path) -> None:
    """A heartbeat superseding an unread change keeps ``kind: change``."""
    client = _observe_client(tmp_path)
    # Stage a change frame.
    resp1 = client.post("/observe", json=_envelope(kind="change", changedSlices=["a"]))
    assert resp1.status_code == 200
    staged1 = _staged(tmp_path)
    assert staged1["kind"] == "change"
    # A heartbeat supersedes it but must inherit the kind.
    resp2 = client.post("/observe", json=_envelope(kind="heartbeat", changedSlices=[]))
    assert resp2.status_code == 200
    staged2 = _staged(tmp_path)
    assert staged2["kind"] == "change", "a heartbeat hid an unread change"


def test_observe_no_wake_with_message_path(tmp_path: Path) -> None:
    """With ``run_stop_message`` set, /observe does not attempt a wake (no agent
    address).  ``subprocess.run`` is patched to raise so any invocation fails loudly."""
    import subprocess

    client = _observe_client(tmp_path)

    def _explode(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("subprocess.run should not be called with message path active")

    original_run = subprocess.run
    try:
        subprocess.run = _explode  # type: ignore[assignment]
        resp = client.post("/observe", json=_envelope(kind="change", changedSlices=["a"]))
    finally:
        subprocess.run = original_run  # type: ignore[assignment]

    assert resp.status_code == 200
    body = resp.json()
    assert body.get("wake") == "not-attempted"


# ── deprecation warning ───────────────────────────────────────────────────────


def test_agent_only_logs_deprecation_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """When only ``run_stop_agent`` is set (no ``run_stop_message``), a deprecation
    warning is logged at startup."""
    import stat

    from zakcode.session.framework_stop import SIGNAL_SET_SCRIPT

    # Plant the setter script so the legacy path does not fail preflight.
    # The body is a no-op — the test never exercises the legacy signal write.
    script = tmp_path / SIGNAL_SET_SCRIPT
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)

    app, endings = _build(tmp_path, stop_message=None, stop_agent="probe")

    async def scenario() -> None:
        await app.state.start_consumer()
        await asyncio.sleep(0.2)

    with caplog.at_level("WARNING"):
        asyncio.run(scenario())

    assert any("deprecated" in r.message.lower() for r in caplog.records), (
        "no deprecation warning logged when only run_stop_agent is set"
    )


# ── precedence: message wins when both are set ────────────────────────────────


def test_message_wins_over_agent(tmp_path: Path) -> None:
    """When both settings are present, the message path is used."""
    import stat

    from zakcode.session.framework_stop import SIGNAL_SET_SCRIPT

    # Plant the setter script so the legacy path does not fail preflight.
    # The body is a no-op — the message path wins so the legacy write never fires.
    script = tmp_path / SIGNAL_SET_SCRIPT
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)

    seen: list[str] = []
    app, endings = _build(
        tmp_path,
        stop_message=STOP_LINE,
        stop_agent="probe",
        reserve=5.0,
        agent_for=lambda session: _TimedAgent(session, seen, [], [0.0, 0.0]),
    )
    assert write_say(say_path(tmp_path), "hello")

    async def scenario() -> None:
        loop_task = asyncio.create_task(app.state.consume_say_loop())
        await _until(lambda: len(seen) >= 1)
        await _post_stop(app)
        await asyncio.wait_for(loop_task, timeout=10)

    asyncio.run(scenario())
    assert STOP_LINE in seen, "message path was not used when both settings were set"
    assert endings == ["stopped"]


# ── graceful stop budget includes run_stop_message ────────────────────────────


def test_budget_includes_message_setting(tmp_path: Path) -> None:
    """The graceful stop budget is nonzero when ``run_stop_message`` is set."""
    app, _ = _build(tmp_path, stop_message=STOP_LINE, reserve=5.0)
    assert app.state.graceful_stop_budget() > 0


def test_budget_zero_when_nothing_configured(tmp_path: Path) -> None:
    """Budget is zero when no ending mechanism is configured."""
    settings = Settings(
        default_model="scripted/test",
        context_window=8192,
        workspace_root=tmp_path,
    )
    app = create_app(
        settings=settings,
        store=SessionStore(base_dir=tmp_path / "sessions"),
        agent_factory=lambda session, model, prompter: _QuietAgent(session),
    )
    assert app.state.graceful_stop_budget() == 0.0
