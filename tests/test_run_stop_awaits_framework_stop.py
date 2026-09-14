"""The run must keep BEATING while the mind runs its own graceful stop (g-373-16).

Raising a framework stop writes signals the mind reads at the TOP of its next beat.
Setting ``run_stopping`` in the same breath ended the loop before that beat could
happen, so the mind never read the signal it had just been sent. Measured on two live
dev vessels: ``stop-requested`` landed, ``stop-loop`` and ``handoff.yaml`` never
appeared, and the vessel was killed at grace expiry with no consolidation. The served
process IS the mind's only reader, so stopping the beat removes the reader.

Each test below fails against the pre-fix loop, and the first one fails LOUDLY: the
run ends while the mind is still mid-stop.

The window must also BOUND every turn in it — the second g-373-16 fix, measured on a
live dev vessel: a turn that started inside the window ran on after it closed, and
nothing ended the run. Those tests fail against the loop before THAT fix, except the
one pinning a zero-reserve stop, which keeps that loop's behaviour on purpose.
"""

from __future__ import annotations

import asyncio
import stat
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
from zakcode.session.framework_stop import (
    SIGNAL_SET_SCRIPT,
    STOP_LOOP_SIGNAL,
    framework_session_dir,
    framework_stop_complete,
)
from zakcode.session.say_inbox import say_path, write_say
from zakcode.session.store import Session, SessionStore
from zakcode.usage import Usage

AGENT = "probe"


def _plant_setter(root: Path) -> None:
    """Plant a faithful stand-in for the framework's ``session-signal-set.sh``."""
    script = root / SIGNAL_SET_SCRIPT
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        'dir="agents/${AYOAI_AGENT}/session"\n'
        'mkdir -p "$dir"\n'
        'touch "$dir/$1"\n',
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)


def _sign_off(root: Path, agent: str = AGENT) -> Path:
    """What the framework's Phase -1.4 does when its obligations are complete."""
    marker = framework_session_dir(root, agent) / STOP_LOOP_SIGNAL
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    return marker


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

    ``seen`` is what each turn was asked and ``finished`` what it completed. The second
    list is the discriminator: a run that ends with the right reason proves nothing if
    the turn it should have bounded ran to completion first.
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


def _build(
    tmp_path: Path,
    *,
    agent: str | None = AGENT,
    reserve: float = 5.0,
    max_duration: float | None = None,
    agent_for: Callable[[Session], Any] = _QuietAgent,
) -> tuple[Any, list[str]]:
    """An app whose run can be stopped; returns (app, endings)."""
    if agent is not None:
        _plant_setter(tmp_path)
    endings: list[str] = []
    settings = Settings(
        default_model="scripted/test",
        context_window=8192,
        workspace_root=tmp_path,
        run_max_duration=max_duration,
        run_consolidation_reserve=reserve,
        run_stop_agent=agent,
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
        assert time.monotonic() < deadline, "the awaited turn never got there"
        await asyncio.sleep(0.02)


# ── the defect itself ────────────────────────────────────────────────────────────


def test_run_stop_keeps_beating_until_the_mind_signs_off(tmp_path: Path) -> None:
    """THE regression. The run must still be alive after the stop is raised.

    Pre-fix, `run_stopping.set()` ended the loop on the very next guard evaluation, so
    the assertion below that the loop is STILL RUNNING is what fails — which is exactly
    the live-vessel finding, reproduced without a vessel.
    """
    app, endings = _build(tmp_path, reserve=5.0)

    async def scenario() -> None:
        loop_task = asyncio.create_task(app.state.consume_say_loop())
        await asyncio.sleep(0.2)
        await _post_stop(app)
        await asyncio.sleep(0.3)

        # The mind has been ASKED but has not signed off: the door must still be open.
        assert not loop_task.done(), "the run ended before the mind could run its stop"
        assert (framework_session_dir(tmp_path, AGENT) / "stop-requested").exists()

        _sign_off(tmp_path)  # Phase -1.4 completes
        await asyncio.wait_for(loop_task, timeout=5)

    asyncio.run(scenario())
    assert endings == ["stopped"]


def test_the_sign_off_ends_the_run_promptly_not_at_grace_expiry(tmp_path: Path) -> None:
    """The wait ends EARLY on the mind's own permit — the grace is a bound, not a sleep."""
    app, endings = _build(tmp_path, reserve=30.0)  # a grace far longer than the test

    async def scenario() -> float:
        loop_task = asyncio.create_task(app.state.consume_say_loop())
        await asyncio.sleep(0.2)
        await _post_stop(app)
        _sign_off(tmp_path)
        started = time.monotonic()
        await asyncio.wait_for(loop_task, timeout=10)
        return time.monotonic() - started

    elapsed = asyncio.run(scenario())
    assert elapsed < 5.0, f"waited {elapsed:.1f}s on a 30s grace — the permit was not read"
    assert endings == ["stopped"]


def test_a_mind_that_never_signs_off_cannot_hold_the_vessel_open(tmp_path: Path) -> None:
    """The other bound. A wedged mind costs the grace and not one second more."""
    app, endings = _build(tmp_path, reserve=0.6)

    async def scenario() -> float:
        loop_task = asyncio.create_task(app.state.consume_say_loop())
        await asyncio.sleep(0.2)
        started = time.monotonic()
        await _post_stop(app)
        await asyncio.wait_for(loop_task, timeout=10)  # never signed off
        return time.monotonic() - started

    elapsed = asyncio.run(scenario())
    assert elapsed >= 0.4, f"ended in {elapsed:.2f}s — the door was never held"
    assert elapsed < 6.0, f"took {elapsed:.2f}s on a 0.6s grace — the bound did not hold"
    assert endings == ["stopped"]


def test_a_non_seed_workspace_ends_exactly_as_before(tmp_path: Path) -> None:
    """No `run_stop_agent` => no raise, no window, and the old ending is untouched."""
    app, endings = _build(tmp_path, agent=None)

    async def scenario() -> float:
        loop_task = asyncio.create_task(app.state.consume_say_loop())
        await asyncio.sleep(0.2)
        started = time.monotonic()
        await _post_stop(app)
        await asyncio.wait_for(loop_task, timeout=5)
        return time.monotonic() - started

    elapsed = asyncio.run(scenario())
    assert elapsed < 1.0, f"a non-seed run waited {elapsed:.2f}s for a stop it never raised"
    assert endings == ["stopped"]


def test_the_cap_landing_between_beats_still_raises_the_mind_s_own_stop(tmp_path: Path) -> None:
    """The cap path, where the mid-turn watcher never fires (nothing is `inflight`).

    The watcher skips whenever no turn is running, so an idle run that reaches its cap
    was ending with no graceful stop raised at all — a second starvation path, on a
    different trigger from the `/run/stop` one above.
    """
    app, endings = _build(tmp_path, reserve=0.6, max_duration=0.9)

    asyncio.run(asyncio.wait_for(app.state.consume_say_loop(), timeout=10))

    assert (framework_session_dir(tmp_path, AGENT) / "stop-requested").exists(), (
        "the cap ended the run without ever asking the mind to stop"
    )
    assert endings == ["duration_cap"]


# ── one stop window bounds EVERY turn in it ──────────────────────────────────────


def test_a_turn_started_inside_the_cap_s_stop_window_is_still_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE measured shape. The watcher used to follow only the turn in flight at the cap.

    Live dev vessel: the cap raised the mind's stop inside its boot turn, that turn ENDED
    inside the window, the watcher returned, and a re-issued ``/start`` began a second
    turn nothing watched — still running 123s after the window closed, no run end ever
    reported. Pre-fix this test times out waiting for that second turn.
    """
    monkeypatch.setattr("zakcode.server.app._DEADLINE_WATCH_SECONDS", 0.1)
    seen: list[str] = []
    finished: list[str] = []
    # cap 4.0 - reserve 3.0: the cap raises the stop ~1.0s in and its window closes ~4.0s
    # in, so the 1.8s boot turn ends inside the window even behind a slow raise.
    app, endings = _build(
        tmp_path,
        reserve=3.0,
        max_duration=4.0,
        agent_for=lambda session: _TimedAgent(session, seen, finished, [1.8, 30.0]),
    )
    assert write_say(say_path(tmp_path), "/start probe")

    async def scenario() -> float:
        started = time.monotonic()
        loop_task = asyncio.create_task(app.state.consume_say_loop())
        await _until(lambda: len(finished) == 1)  # the boot turn ended INSIDE the window
        await asyncio.sleep(0.4)  # idle beats: exactly where the old watcher returned
        assert write_say(say_path(tmp_path), "/start probe")  # the env-server's re-issue
        await asyncio.wait_for(loop_task, timeout=10)
        return time.monotonic() - started

    elapsed = asyncio.run(scenario())
    assert (framework_session_dir(tmp_path, AGENT) / "stop-requested").exists()
    assert seen == ["/start probe", "/start probe"], seen
    assert finished == ["/start probe"], "the second turn ran to completion — nothing bounded it"
    assert elapsed < 7.0, f"run took {elapsed:.2f}s against a 4s cap"
    assert endings == ["duration_cap"]
    marker = (tmp_path / ".run-stop-reason").read_text(encoding="utf-8").splitlines()
    assert marker[-1] == "duration_cap", marker


def test_a_run_stop_window_bounds_the_turn_in_flight_on_a_capless_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``/run/stop`` opens the same window, and on a capless run nothing watched it.

    The watcher skipped every tick with no ``turn_deadline``, so the grace bounded only
    the BEATS: a turn in flight at the stop ran until it ended on its own.
    """
    monkeypatch.setattr("zakcode.server.app._DEADLINE_WATCH_SECONDS", 0.1)
    seen: list[str] = []
    finished: list[str] = []
    app, endings = _build(
        tmp_path,
        reserve=0.8,
        agent_for=lambda session: _TimedAgent(session, seen, finished, [30.0]),
    )
    assert write_say(say_path(tmp_path), "/start probe")

    async def scenario() -> float:
        loop_task = asyncio.create_task(app.state.consume_say_loop())
        await _until(lambda: len(seen) == 1)
        stopped = time.monotonic()
        await _post_stop(app)
        await asyncio.wait_for(loop_task, timeout=10)
        return time.monotonic() - stopped

    elapsed = asyncio.run(scenario())
    assert finished == [], "the turn ran to completion — the window never bounded it"
    assert elapsed < 5.0, f"took {elapsed:.2f}s on a 0.8s window"
    assert endings == ["stopped"]


def test_a_zero_reserve_run_stop_still_lets_the_turn_in_flight_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ZERO window bounds nothing: the turn in flight finishes, then the run ends.

    That is ADR-0047's ending, and a non-seed run's. Pinned so the bound above cannot
    quietly widen into "interrupt on every ``/run/stop``" — this passes before the fix
    too, and fails only if a zero-length window starts cutting turns.
    """
    monkeypatch.setattr("zakcode.server.app._DEADLINE_WATCH_SECONDS", 0.1)
    seen: list[str] = []
    finished: list[str] = []
    app, endings = _build(
        tmp_path,
        reserve=0.0,
        agent_for=lambda session: _TimedAgent(session, seen, finished, [1.0]),
    )
    assert write_say(say_path(tmp_path), "/start probe")

    async def scenario() -> None:
        loop_task = asyncio.create_task(app.state.consume_say_loop())
        await _until(lambda: len(seen) == 1)
        await _post_stop(app)
        await asyncio.wait_for(loop_task, timeout=10)

    asyncio.run(scenario())
    assert (framework_session_dir(tmp_path, AGENT) / "stop-requested").exists()
    assert finished == ["/start probe"], "a zero-reserve stop interrupted the turn in flight"
    assert endings == ["stopped"]


def test_a_cap_landing_inside_a_run_stop_window_does_not_restart_the_grace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The window is SHARED: a cap landing inside it cannot buy the turn a second grace.

    `_begin_framework_stop` already made every caller share ONE window, but the watcher
    then timed its own grace from the turn deadline, so the turn ran a full reserve past
    it — out to the hard cap.
    """
    monkeypatch.setattr("zakcode.server.app._DEADLINE_WATCH_SECONDS", 0.1)
    seen: list[str] = []
    finished: list[str] = []
    # cap 6.0 - reserve 3.0: the turn deadline lands 3.0s in, inside a window the stop
    # opens ~0.2s in. Shared, that window closes ~3.2s in; restarted, ~6.0s in.
    app, endings = _build(
        tmp_path,
        reserve=3.0,
        max_duration=6.0,
        agent_for=lambda session: _TimedAgent(session, seen, finished, [30.0]),
    )
    assert write_say(say_path(tmp_path), "/start probe")

    async def scenario() -> float:
        started = time.monotonic()
        loop_task = asyncio.create_task(app.state.consume_say_loop())
        await _until(lambda: len(seen) == 1)
        await asyncio.sleep(max(0.0, started + 0.2 - time.monotonic()))
        await _post_stop(app)
        opened = time.monotonic()  # the route returns once the window is open
        await asyncio.wait_for(loop_task, timeout=10)
        return time.monotonic() - opened

    elapsed = asyncio.run(scenario())
    assert finished == [], "the turn ran to completion — nothing bounded it"
    assert elapsed < 4.6, f"ran {elapsed:.2f}s into a 3.0s stop window — the cap restarted it"
    assert endings == ["stopped"]


# ── the permit reader ────────────────────────────────────────────────────────────


def test_framework_stop_complete_reads_the_permit(tmp_path: Path) -> None:
    assert framework_stop_complete(tmp_path, AGENT) is False  # absent = "not yet"
    _sign_off(tmp_path)
    assert framework_stop_complete(tmp_path, AGENT) is True


def test_framework_stop_complete_without_an_agent_is_false(tmp_path: Path) -> None:
    """No agent = no seed = nothing to wait for; never a wait on an empty path."""
    assert framework_stop_complete(tmp_path, "") is False
