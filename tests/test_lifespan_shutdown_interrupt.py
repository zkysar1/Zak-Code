"""The lifespan shutdown ends a turn that nothing else will end (ADR-0322).

A SIGTERM (a member's hard Stop, a box shutdown) reaches ``_stop_consumer`` and nothing else.
``run_stopping`` only stops NEW beats, so a mind's perpetual turn kept calling the model, with no
meter behind it, for the whole wait until the supervisor's SIGKILL: 23, 25 and 21 post-stop calls
over about 120 s on three DEV runs, with ``Application shutdown complete`` never logged. The wait
was the digest reserve (350 s on a bounded run), which is longer than the unit's 120 s.

These drive the shutdown through the ``start_consumer`` / ``stop_consumer`` seams, with a turn that
only a cancellation can end (the stand-in for the mind's perpetual loop) and real, small durations.
``interrupts`` records every ``request_interrupt`` the app makes, so a test can tell the shutdown's
own interrupt from the window watcher's.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

import zakcode.server.app as app_module
from zakcode.config import Settings
from zakcode.events import AgentDone, AgentEvent, AgentTextDelta
from zakcode.messages import Message
from zakcode.server.app import create_app
from zakcode.session.say_inbox import interrupt_path, say_path, write_say
from zakcode.session.store import Session, SessionStore
from zakcode.usage import Usage

BEGIN = "begin the perpetual loop"
DIGEST = "digest: what did we do?"


class _PerpetualAgent:
    """A turn that ends only when something ends it, for the texts in ``hang_on``."""

    def __init__(self, session: Session, turns: list[str], hang_on: frozenset[str]) -> None:
        self.session = session
        self._turns = turns
        self._hang_on = hang_on

    async def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]:
        self._turns.append(user_text)
        self.session.add_message(Message.user(user_text))
        if user_text in self._hang_on:
            await asyncio.Event().wait()  # nothing in this turn ever sets it
        self.session.add_message(Message.assistant_text("ok"))
        yield AgentTextDelta(text="ok")
        yield AgentDone(stop_reason="completed", iterations=1, usage=Usage())


def _build(
    tmp_path: Path,
    *,
    hang_on: frozenset[str],
    reserve: float = 0.0,
    max_duration: float | None = None,
    message: str | None = None,
    stop_message: str | None = None,
    configured: bool = True,
) -> tuple[Any, list[str], list[str]]:
    """An app; returns (app, turns the agent was given, endings reported).

    ``configured=False`` is the default server: no digest, no stop line, no ending callback,
    so the graceful-stop budget is zero.
    """
    turns: list[str] = []
    endings: list[str] = []
    settings = Settings(
        default_model="scripted/test",
        context_window=8192,
        workspace_root=tmp_path,
        run_consolidation_reserve=reserve,
        run_max_duration=max_duration,
        run_consolidation_message=message,
        run_stop_message=stop_message,
    )

    async def _on_run_end(reason: str) -> None:
        endings.append(reason)

    app = create_app(
        settings=settings,
        store=SessionStore(base_dir=tmp_path / "sessions"),
        agent_factory=lambda session, model, prompter: _PerpetualAgent(session, turns, hang_on),
        on_run_end=_on_run_end if configured else None,
    )
    return app, turns, endings


def _record_interrupts(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Wrap ``request_interrupt`` so every call the app makes is kept (and still made)."""
    calls: list[Path] = []
    real = app_module.request_interrupt

    def _recording(path: Path) -> None:
        calls.append(path)
        real(path)

    monkeypatch.setattr(app_module, "request_interrupt", _recording)
    return calls


async def _until(predicate: Callable[[], bool], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition never became true"
        await asyncio.sleep(0.02)


# ── the fix: a turn in flight with no window open is interrupted ─────────────


def test_shutdown_interrupts_a_turn_nothing_else_will_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mind's perpetual turn is in flight when the lifespan stops the consumer.

    Before the fix the wait ran the whole 10 s budget (350 s on a bounded run) and then
    cancelled: the turn kept running, and the run never reported its ending. Now the shutdown
    interrupts the turn, the loop winds down, the digest runs, and the ending is reported.

    THE MARGIN IS SIZED IN BUDGETS, NOT BEATS. Interrupt, unwind, one beat and a short digest
    take about a second; the broken path takes the full 10.5 s. The threshold sits between.
    """
    interrupts = _record_interrupts(monkeypatch)
    app, turns, endings = _build(tmp_path, hang_on=frozenset({BEGIN}), reserve=10.0, message=DIGEST)
    assert write_say(say_path(tmp_path), BEGIN)

    async def scenario() -> float:
        # The real lifespan, which is what uvicorn runs on SIGTERM: entering it starts the
        # consumer, and leaving it is the shutdown under test.
        async with app.router.lifespan_context(app):
            await _until(lambda: BEGIN in turns)  # the perpetual turn is now in flight
            started = time.monotonic()
        return time.monotonic() - started

    elapsed = asyncio.run(scenario())

    assert elapsed < 5.0, f"shutdown waited {elapsed:.1f}s on a turn nothing was ending"
    assert interrupts == [interrupt_path(tmp_path)]
    assert turns == [BEGIN, DIGEST]  # the digest still ran, after the interrupted turn
    assert endings == ["stopped"]


def test_shutdown_wait_is_capped_below_the_supervisor_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A digest that hangs cannot hold the shutdown past the ceiling.

    The reserve is 600 s, so the budget alone would wait ten minutes: past the unit's 120 s,
    which is a SIGKILL instead of ``Application shutdown complete``. The ceiling is patched to
    one second so the test stays fast. This test cannot run on the old code, which has no such
    constant, so its red there is an AttributeError, not a behaviour.
    """
    monkeypatch.setattr(app_module, "_HARD_STOP_WAIT_SECONDS", 1.0)
    app, turns, _endings = _build(
        tmp_path, hang_on=frozenset({BEGIN, DIGEST}), reserve=600.0, message=DIGEST
    )
    assert write_say(say_path(tmp_path), BEGIN)

    async def scenario() -> float:
        await app.state.start_consumer()
        await _until(lambda: BEGIN in turns)
        started = time.monotonic()
        await app.state.stop_consumer()
        return time.monotonic() - started

    elapsed = asyncio.run(scenario())

    assert DIGEST in turns, "the digest never started, so the ceiling was not what ended the wait"
    assert elapsed < 5.0, f"shutdown waited {elapsed:.1f}s against a 1s ceiling"


# ── what the fix must leave alone ────────────────────────────────────────────


def test_shutdown_leaves_a_graceful_ending_in_flight_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``/run/stop`` already open the mind is ending itself: the shutdown adds no interrupt.

    The window is 30 s, so its own watcher cannot fire inside the second this test watches;
    an interrupt recorded in that second is the shutdown's.
    """
    interrupts = _record_interrupts(monkeypatch)
    app, turns, _endings = _build(
        tmp_path, hang_on=frozenset({BEGIN}), reserve=30.0, stop_message="please wrap up"
    )
    assert write_say(say_path(tmp_path), BEGIN)

    async def scenario() -> None:
        await app.state.start_consumer()
        await _until(lambda: BEGIN in turns)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/run/stop", json={})
        assert response.json()["stopping"] is True
        stopper = asyncio.create_task(app.state.stop_consumer())
        await asyncio.sleep(1.0)
        assert not stopper.done(), "the shutdown stopped waiting inside an open ending window"
        assert interrupts == []
        stopper.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await stopper

    asyncio.run(scenario())


def test_shutdown_does_not_cut_a_run_that_is_already_ending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run whose cap already fired is mid-digest: that turn is the receipt, not a stray.

    The cap raised ``duration_cap`` and the digest turn is in flight (and hangs, so it is still
    there when the shutdown arrives). Interrupting it would sever the one thing the reserve
    exists to deliver; the shutdown waits the budget instead, and the ending is still reported.
    """
    interrupts = _record_interrupts(monkeypatch)
    app, turns, endings = _build(
        tmp_path, hang_on=frozenset({DIGEST}), reserve=2.0, max_duration=3.0, message=DIGEST
    )

    async def scenario() -> None:
        await app.state.start_consumer()
        await _until(lambda: DIGEST in turns)  # the cap has fired; the digest is in flight
        await app.state.stop_consumer()

    asyncio.run(scenario())

    assert interrupts == []
    assert endings == ["duration_cap"]


def test_shutdown_with_no_turn_in_flight_interrupts_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An idle consumer has no turn to end: the loop winds down by itself and digests.

    An interrupt file written with nothing running would sit on disk for the next reader, so
    it is not written. This is the ordinary graceful shutdown and it must read as before.
    """
    interrupts = _record_interrupts(monkeypatch)
    app, turns, endings = _build(tmp_path, hang_on=frozenset(), reserve=2.0, message=DIGEST)

    async def scenario() -> None:
        await app.state.start_consumer()
        await asyncio.sleep(0.05)
        await app.state.stop_consumer()

    asyncio.run(scenario())

    assert interrupts == []
    assert turns == [DIGEST]
    assert endings == ["stopped"]


def test_a_default_server_still_shuts_down_at_once_without_an_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No digest, no stop line, no ending callback: the budget is zero and the cancel is at once.

    The positive control for the cases above: a fix that interrupted unconditionally would
    write the interrupt file here too.
    """
    interrupts = _record_interrupts(monkeypatch)
    app, turns, _endings = _build(tmp_path, hang_on=frozenset({BEGIN}), configured=False)
    assert app.state.graceful_stop_budget() == 0.0
    assert write_say(say_path(tmp_path), BEGIN)

    async def scenario() -> float:
        await app.state.start_consumer()
        await _until(lambda: BEGIN in turns)
        started = time.monotonic()
        await app.state.stop_consumer()
        return time.monotonic() - started

    elapsed = asyncio.run(scenario())

    assert elapsed < 2.0, f"a default server took {elapsed:.1f}s to shut down"
    assert interrupts == []
