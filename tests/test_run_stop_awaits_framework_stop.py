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

The wait must end when the mind's stop has FINISHED — the third g-373-16 fix, found by
reading the framework, not on a vessel: the key read ``stop-loop``, which the graceful
stop sets at D2, before it consolidates, and removes again at D6. Against that key every
test that replays a finished stop fails, the older ones too: their sign-off used to TOUCH
``stop-loop``, a stop the framework never performs.
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
    AGENT_MODE_FILENAME,
    SIGNAL_SET_SCRIPT,
    STOP_CHECKPOINT_FILENAME,
    STOP_REQUESTED_SIGNAL,
    STOP_TARGET_MODE_FILENAME,
    framework_session_dir,
    framework_stop_complete,
)
from zakcode.session.say_inbox import say_path, write_say
from zakcode.session.store import Session, SessionStore
from zakcode.usage import Usage
from zakcode.wakeup import LOOP_SENTINEL, fired_line

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


#: The framework graceful stop's file effects, in its own order (``aspirations-graceful-stop``
#: GS-0, then D2-D7.1): (step, file, content, or None to remove). Only files whose TIMING
#: matters are replayed: the ones the key reads, ``stop-loop`` (the key it used to read),
#: and the handoff (proof that consolidation ran).
_GRACEFUL_STOP: tuple[tuple[str, str, str | None], ...] = (
    ("GS-0 checkpoint", STOP_CHECKPOINT_FILENAME, "{}"),
    ("D2 stop-loop", "stop-loop", ""),
    ("D3 consume the ask", STOP_REQUESTED_SIGNAL, None),
    ("D4 handoff", "handoff.yaml", "handoff: written\n"),
    ("D6 cleanup", "stop-loop", None),
    ("D7 target mode", AGENT_MODE_FILENAME, "assistant\n"),
    ("D7 target mode consumed", STOP_TARGET_MODE_FILENAME, None),
    ("D7.1 checkpoint clear", STOP_CHECKPOINT_FILENAME, None),
)


def _boot(root: Path, agent: str = AGENT) -> Path:
    """A mind whose loop is running, as ``/start`` leaves it. Returns its session dir."""
    session_dir = framework_session_dir(root, agent)
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / AGENT_MODE_FILENAME).write_text("autonomous\n", encoding="utf-8")
    return session_dir


def _remove(path: Path, *, patience: float = 2.0) -> None:
    """Remove a signal file as a mind's script does, waiting out a Windows sharing violation.

    POSIX removes a name another process still has open. Windows refuses: ``unlink`` raises
    ``PermissionError`` (WinError 32) until the other handle closes. The sidecar raises the
    stop on a worker thread and holds ``stop-requested`` open for the instant it signs it,
    and these stand-in minds consume the ask within a millisecond of seeing it, which no
    model-driven mind can do. Measured 2026-09-19 on windows-latest (run 35410055874, a
    commit with no product code): the unlink raised inside the turn, the turn failed, the
    run never ended, and the test read a bare ``TimeoutError``. The wait below is the
    ordering a real mind gets for free; a handle that never closes still fails, loudly.
    """
    deadline = time.monotonic() + patience
    while True:
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


def _sign_off(
    root: Path,
    agent: str = AGENT,
    *,
    checkpoint: bool = True,
    after_each: Callable[[str], None] | None = None,
) -> None:
    """Replay the framework's graceful stop to its end, step by step, in its order.

    ``checkpoint=False`` is a GS-0 whose checkpoint write failed, which the framework
    tolerates. ``after_each`` is called with each step's name once that step has landed.
    """
    session_dir = framework_session_dir(root, agent)
    session_dir.mkdir(parents=True, exist_ok=True)
    for step, name, content in _GRACEFUL_STOP:
        if name == STOP_CHECKPOINT_FILENAME and not checkpoint:
            continue
        if content is None:
            _remove(session_dir / name)
        else:
            (session_dir / name).write_text(content, encoding="utf-8")
        if after_each is not None:
            after_each(step)


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


class _StoppingMind:
    """A running loop that, once asked, runs its OWN graceful stop inside the turn in flight.

    That is where Phase -1.4 runs it, so by the time the say loop next looks the stop is
    already FINISHED — past D6, where ``stop-loop`` is gone again.
    """

    def __init__(self, session: Session, root: Path) -> None:
        self.session = session
        self._root = root

    async def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]:
        self.session.add_message(Message.user(user_text))
        session_dir = _boot(self._root)
        while not (session_dir / STOP_REQUESTED_SIGNAL).exists():
            await asyncio.sleep(0.02)
        _sign_off(self._root)
        self.session.add_message(Message.assistant_text("stopped"))
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


def test_the_stand_in_mind_waits_out_a_sharing_violation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Windows ordering, reproduced on any platform: the first two removals meet the
    sidecar's open handle, the third lands. Without the wait the first one ends the turn."""
    marker = tmp_path / STOP_REQUESTED_SIGNAL
    marker.write_text("", encoding="utf-8")
    real_unlink = Path.unlink
    refusals: list[Path] = []

    def held_open_twice(self: Path, missing_ok: bool = False) -> None:
        if self == marker and len(refusals) < 2:
            refusals.append(self)
            raise PermissionError(13, "The process cannot access the file", str(self))
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", held_open_twice)
    _remove(marker)
    assert len(refusals) == 2
    assert not marker.exists()


def test_a_handle_that_never_closes_still_fails_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / STOP_REQUESTED_SIGNAL
    marker.write_text("", encoding="utf-8")

    def always_held(self: Path, missing_ok: bool = False) -> None:
        raise PermissionError(13, "The process cannot access the file", str(self))

    monkeypatch.setattr(Path, "unlink", always_held)
    with pytest.raises(PermissionError):
        _remove(marker, patience=0.05)


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


def test_a_stop_finished_inside_the_turn_ends_the_run_when_that_turn_ends(tmp_path: Path) -> None:
    """R4 end to end (g-373-16): the run ends when the mind's own stop does.

    The mind runs its whole stop inside the turn in flight, so the loop only ever looks at
    a FINISHED stop. Keyed on ``stop-loop``, which D6 had already removed, the loop read
    "not yet" and beat on until the grace ran out.
    """
    app, endings = _build(
        tmp_path, reserve=30.0, agent_for=lambda session: _StoppingMind(session, tmp_path)
    )
    assert write_say(say_path(tmp_path), "/start probe")
    mode_file = framework_session_dir(tmp_path, AGENT) / AGENT_MODE_FILENAME

    async def scenario() -> float:
        loop_task = asyncio.create_task(app.state.consume_say_loop())
        await _until(mode_file.exists)  # the loop is running
        stopped = time.monotonic()
        await _post_stop(app)
        await asyncio.wait_for(loop_task, timeout=10)
        return time.monotonic() - stopped

    elapsed = asyncio.run(scenario())
    assert elapsed < 5.0, f"the stop had finished, yet the run waited {elapsed:.1f}s"
    assert (framework_session_dir(tmp_path, AGENT) / "handoff.yaml").exists()
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
    signal = framework_session_dir(tmp_path, AGENT) / "stop-requested"

    async def scenario() -> None:
        loop_task = asyncio.create_task(app.state.consume_say_loop())
        # Observed WHILE the window is open: once it closes on a loop at rest the pair is
        # retired (ADR-0189), so its presence after the run is no longer the evidence.
        await _until(signal.exists, timeout=10)
        await asyncio.wait_for(loop_task, timeout=10)

    asyncio.run(scenario())
    assert not signal.exists(), "the unread stop was left on disk for the next boot"
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
    # The stop WAS raised and its window then overran -- so the pair is now retired
    # rather than left on disk (g-373-92). This assertion read `.exists()` until that
    # fix: it was incidental corroboration that the raise happened, and what it actually
    # pinned was the orphan whose survival poisons the NEXT vessel boot. The subject of
    # this test is unchanged and is asserted below: the second turn is BOUNDED.
    assert not (framework_session_dir(tmp_path, AGENT) / "stop-requested").exists()
    assert not (framework_session_dir(tmp_path, AGENT) / "stop-target-mode").exists()
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

    signal = framework_session_dir(tmp_path, AGENT) / "stop-requested"

    async def scenario() -> None:
        loop_task = asyncio.create_task(app.state.consume_say_loop())
        await _until(lambda: len(seen) == 1)
        await _post_stop(app)
        await _until(signal.exists)  # raised, while the turn is still in flight
        await asyncio.wait_for(loop_task, timeout=10)

    asyncio.run(scenario())
    assert not signal.exists()  # a zero window is spent at once: the unread pair is retired
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


# ── the completion key ───────────────────────────────────────────────────────────


def test_framework_stop_complete_reads_the_finished_stop(tmp_path: Path) -> None:
    assert framework_stop_complete(tmp_path, AGENT) is False  # absent = "not yet"
    _sign_off(tmp_path)
    assert framework_stop_complete(tmp_path, AGENT) is True


@pytest.mark.parametrize("checkpoint", [True, False], ids=["checkpoint", "gs0-write-failed"])
def test_the_key_reads_done_only_after_the_stop_s_last_step(
    tmp_path: Path, checkpoint: bool
) -> None:
    """R4 (g-373-16), read at EVERY step of a real stop's order.

    Keyed on ``stop-loop``, the key said "done" from D2, before the handoff existed, until
    D6, and "not yet" from D6 on, which is where every finished stop leaves it.
    """
    session_dir = _boot(tmp_path)
    (session_dir / STOP_TARGET_MODE_FILENAME).write_text("assistant", encoding="utf-8")
    (session_dir / STOP_REQUESTED_SIGNAL).touch()  # the ask, as the sidecar raises it
    readings = [("asked", framework_stop_complete(tmp_path, AGENT))]

    def read(step: str) -> None:
        readings.append((step, framework_stop_complete(tmp_path, AGENT)))

    _sign_off(tmp_path, checkpoint=checkpoint, after_each=read)

    last_step = readings[-1][0]
    assert last_step == ("D7.1 checkpoint clear" if checkpoint else "D7 target mode consumed")
    assert [step for step, done in readings if done] == [last_step], readings


@pytest.mark.parametrize(
    "marker", [STOP_REQUESTED_SIGNAL, STOP_TARGET_MODE_FILENAME, STOP_CHECKPOINT_FILENAME]
)
def test_any_file_a_stop_removes_on_its_way_out_holds_the_key_open(
    tmp_path: Path, marker: str
) -> None:
    _sign_off(tmp_path)
    assert framework_stop_complete(tmp_path, AGENT) is True
    (framework_session_dir(tmp_path, AGENT) / marker).write_text("", encoding="utf-8")
    assert framework_stop_complete(tmp_path, AGENT) is False


@pytest.mark.parametrize(
    ("mode", "done"),
    [
        ("assistant\n", True),
        ("reader\n", True),
        ("autonomous\n", False),  # the loop is running again
        ("\xff\xfe", False),  # unreadable as a mode
        (None, False),  # never written: the framework's disk default is not a stop's write
    ],
    ids=["assistant", "reader", "autonomous", "garbage", "absent"],
)
def test_only_a_mode_a_stop_lands_in_reads_done(
    tmp_path: Path, mode: str | None, done: bool
) -> None:
    """The positive half: with no stop ever raised, every in-flight file is absent too."""
    _sign_off(tmp_path)
    mode_file = framework_session_dir(tmp_path, AGENT) / AGENT_MODE_FILENAME
    if mode is None:
        mode_file.unlink()
    else:
        mode_file.write_bytes(mode.encode("latin-1"))
    assert framework_stop_complete(tmp_path, AGENT) is done


def test_framework_stop_complete_without_an_agent_is_false(tmp_path: Path) -> None:
    """No agent = no seed = nothing to wait for; never a wait on an empty path."""
    assert framework_stop_complete(tmp_path, "") is False


# ── a stop raised between turns starts the turn that reads it (ADR-0189) ──


class _LoopMindAtRest:
    """A mind whose loop ran once (its hook-named re-entry is known) and whose turn then
    ENDED — the prod shape of 2026-09-18: the loop turn died ``veto_stall`` 22 minutes
    before the cap, so the signed stop landed with nothing in flight to read it. Its
    second turn — the harness's re-entry — runs the graceful stop where Phase -1.4 would.
    """

    def __init__(self, session: Session, root: Path, seen: list[str]) -> None:
        self.session = session
        self._root = root
        self._seen = seen

    async def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]:
        self._seen.append(user_text)
        self.session.add_message(Message.user(user_text))
        if len(self._seen) == 1:
            _boot(self._root)
            self.session.loop_skill = "aspirations loop"  # what a Stop-hook re-entry records
        else:
            session_dir = framework_session_dir(self._root, AGENT)
            assert (session_dir / STOP_REQUESTED_SIGNAL).exists(), "kicked with no stop to read"
            _sign_off(self._root)
        self.session.add_message(Message.assistant_text("ok"))
        yield AgentDone(stop_reason="completed", iterations=1, usage=Usage())


def test_a_stop_raised_between_turns_starts_the_loop_reentry_that_reads_it(
    tmp_path: Path,
) -> None:
    """THE prod gap. Pre-fix the raise wrote the signal and the loop kept beating with
    nothing to beat FOR: no say, no nudge, no turn — the signal was retired unconsumed at
    grace expiry and the run ended as it would have without the raise."""
    seen: list[str] = []
    app, endings = _build(
        tmp_path, reserve=30.0, agent_for=lambda session: _LoopMindAtRest(session, tmp_path, seen)
    )
    assert write_say(say_path(tmp_path), "/start probe")

    async def scenario() -> float:
        loop_task = asyncio.create_task(app.state.consume_say_loop())
        await _until(lambda: len(seen) == 1)
        await asyncio.sleep(0.3)  # at rest: the (instant) turn is over, nothing is queued
        stopped = time.monotonic()
        await _post_stop(app)
        await asyncio.wait_for(loop_task, timeout=10)
        return time.monotonic() - stopped

    elapsed = asyncio.run(scenario())
    assert seen == ["/start probe", fired_line(LOOP_SENTINEL)]
    assert (framework_session_dir(tmp_path, AGENT) / "handoff.yaml").exists()
    assert elapsed < 5.0, f"the re-entry signed off, yet the run waited {elapsed:.1f}s"
    assert endings == ["stopped"]


def test_a_mind_with_no_known_reentry_is_not_kicked(tmp_path: Path) -> None:
    """No hook-named loop skill on the session => nothing composable to run: the raise
    behaves exactly as before (the window opens, the grace bounds it)."""
    seen: list[str] = []

    class _Quiet(_QuietAgent):
        async def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]:
            seen.append(user_text)
            async for event in super().astream_turn(user_text):
                yield event

    app, endings = _build(tmp_path, reserve=0.6, agent_for=_Quiet)
    assert write_say(say_path(tmp_path), "hello")

    async def scenario() -> None:
        loop_task = asyncio.create_task(app.state.consume_say_loop())
        await _until(lambda: len(seen) == 1)
        await asyncio.sleep(0.3)  # at rest
        await _post_stop(app)
        await asyncio.wait_for(loop_task, timeout=10)

    asyncio.run(scenario())
    assert seen == ["hello"]
    assert endings == ["stopped"]


def test_a_stop_nobody_read_is_retired_when_its_window_closes_with_no_turn(tmp_path: Path) -> None:
    """The third orphan path (g-373-92). The watcher retires the pair when a TURN overruns
    the window; a window that closes on a loop at rest used to leave the signed pair on
    EFS for the next boot to read — measured on prod 2026-09-18 after teardown."""
    app, endings = _build(tmp_path, reserve=0.6)
    session_dir = framework_session_dir(tmp_path, AGENT)

    async def scenario() -> None:
        loop_task = asyncio.create_task(app.state.consume_say_loop())
        await asyncio.sleep(0.2)
        await _post_stop(app)
        await _until(lambda: (session_dir / STOP_REQUESTED_SIGNAL).exists())
        await asyncio.wait_for(loop_task, timeout=10)  # never signed off; grace spent

    asyncio.run(scenario())
    assert not (session_dir / STOP_REQUESTED_SIGNAL).exists()
    assert not (session_dir / STOP_TARGET_MODE_FILENAME).exists()
    assert endings == ["stopped"]
