"""The serve-side reactive say consumer — the web convergence's turn-runner.

The web page (and every other surface) is a pure viewer + say-writer: input
reaches the agent ONLY through the workspace say inbox, and the SERVER runs the
turn. These tests exercise the consumer beat through the ``app.state.consume_one_say``
seam (httpx's ASGITransport does not run lifespan events, so the background loop
never starts here — each test drives exactly the beats it means to).

Covers: a say becomes a turn on the workspace's current session (marker written,
session saved, inbox consumed exactly once); an idle inbox is a no-op beat; a beat
yields while a turn is in flight; ``POST /interrupt`` writes the contract's
interrupt file; the ``GET /sessions/current`` join flow the web page boots through;
and — against a REAL loopback uvicorn server, where the lifespan actually starts
the background loop — the full path the web page rides: POST /say → the consumer
runs the turn → a late-joining ``?full=1`` watcher replays ``user_message`` +
AgentEvents. (The live server is mandatory for any watch-stream read: the SSE
parks after replay and in-process transports buffer the whole body — see
test_server_watch.py.)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import sys
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import uvicorn

from zakcode.config import Settings
from zakcode.events import AgentDone, AgentEvent, AgentTextDelta
from zakcode.messages import Message
from zakcode.server.app import create_app
from zakcode.session.say_inbox import interrupt_path, say_path, write_say
from zakcode.session.store import Session, SessionStore
from zakcode.usage import Usage


class _FakeAgent:
    """Instant scripted turn: records the user text, emits one delta + done."""

    def __init__(self, session: Session, gate: asyncio.Event | None = None) -> None:
        self.session = session
        self._gate = gate

    async def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]:
        self.session.add_message(Message.user(user_text))
        if self._gate is not None:
            await self._gate.wait()
        self.session.add_message(Message.assistant_text("ok"))
        yield AgentTextDelta(text="ok")
        yield AgentDone(stop_reason="completed", iterations=1, usage=Usage())


def _build(tmp_path: Path, gate: asyncio.Event | None = None) -> tuple[Any, SessionStore]:
    settings = Settings(default_model="scripted/test", workspace_root=tmp_path)
    store = SessionStore(base_dir=tmp_path / "sessions")
    app = create_app(
        settings=settings,
        store=store,
        agent_factory=lambda session, model, prompter: _FakeAgent(session, gate),
    )
    return app, store


def test_say_becomes_a_turn_on_the_current_session(tmp_path: Path) -> None:
    """One say → one server-run turn: marker written, session saved, slot consumed."""
    app, store = _build(tmp_path)
    assert write_say(say_path(tmp_path), "hello there")

    ran = asyncio.run(app.state.consume_one_say())
    assert ran is True
    assert not say_path(tmp_path).exists()  # consumed exactly once

    marker = (tmp_path / ".current-session").read_text(encoding="utf-8").strip()
    session = store.load(marker)
    assert [m.role for m in session.messages] == ["user", "assistant"]


def test_idle_inbox_is_a_noop_beat(tmp_path: Path) -> None:
    app, _ = _build(tmp_path)
    assert asyncio.run(app.state.consume_one_say()) is False
    assert not (tmp_path / ".current-session").exists()  # no phantom session


def test_beat_yields_while_a_turn_is_in_flight(tmp_path: Path) -> None:
    """The single-turn discipline: a say arriving mid-turn WAITS (stays queued in
    the slot) instead of racing a second concurrent turn."""
    gate = asyncio.Event()
    app, _ = _build(tmp_path, gate)

    async def go() -> tuple[bool, bool]:
        assert write_say(say_path(tmp_path), "first")
        first = asyncio.create_task(app.state.consume_one_say())
        await asyncio.sleep(0.05)  # first turn is now blocked on the gate
        assert write_say(say_path(tmp_path), "second")
        skipped = await app.state.consume_one_say()
        gate.set()
        return await first, skipped

    ran_first, skipped = asyncio.run(go())
    assert ran_first is True
    assert skipped is False
    assert say_path(tmp_path).read_text(encoding="utf-8").strip() == "second"  # still queued


def test_interrupt_route_writes_the_contract_file(tmp_path: Path) -> None:
    app, _ = _build(tmp_path)

    async def go() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/interrupt")

    resp = asyncio.run(go())
    assert resp.status_code == 200
    assert resp.json() == {"requested": True}
    assert interrupt_path(tmp_path).exists()


def test_sessions_current_join_flow(tmp_path: Path) -> None:
    """The web page's boot: 404 with no marker → create (writes the marker) →
    ``/sessions/current`` returns the same session thereafter."""
    app, _ = _build(tmp_path)

    async def go() -> tuple[int, str, str]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            missing = await client.get("/sessions/current")
            created = await client.post("/sessions", json={})
            current = await client.get("/sessions/current")
            assert current.status_code == 200
            return missing.status_code, created.json()["id"], current.json()["id"]

    missing_status, created_id, current_id = asyncio.run(go())
    assert missing_status == 404
    assert created_id == current_id
    marker = (tmp_path / ".current-session").read_text(encoding="utf-8").strip()
    assert marker == created_id


# ── the real thing: live server, background loop ON, a watcher on ?full=1 ────────


class _LiveServer:
    """Loopback uvicorn in a daemon thread (copied per test_server_watch.py precedent)."""

    def __init__(self, app: Any) -> None:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        self.port = int(s.getsockname()[1])
        s.close()
        config = uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def __enter__(self) -> str:
        self._thread.start()
        deadline = time.monotonic() + 10.0
        while not self._server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("uvicorn did not start within 10s")
            time.sleep(0.02)
        return f"http://127.0.0.1:{self.port}"

    def __exit__(self, *exc: object) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10.0)


async def test_web_page_path_say_to_full_watch(tmp_path: Path) -> None:
    """The exact path the served web page rides, end to end: POST /say → the
    lifespan-started consumer runs the turn → GET /sessions/current resolves →
    a late-joining ``?full=1`` watcher replays the ``user_message`` marker and
    the turn's AgentEvents in order."""
    app, _ = _build(tmp_path)  # under a real server the lifespan starts the consumer
    with _LiveServer(app) as base_url:
        timeout = httpx.Timeout(10.0, read=5.0)
        async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as ac:
            resp = await ac.post("/say", json={"text": "hello there"})
            assert resp.status_code in (200, 202)

            sid = None
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                current = await ac.get("/sessions/current")
                if current.status_code == 200:
                    sid = current.json()["id"]
                    break
                await asyncio.sleep(0.05)
            assert sid is not None, "consumer never minted the current session"

            frames: list[dict[str, Any]] = []
            async with ac.stream("GET", f"/watch/{sid}?full=1") as watch:
                assert watch.status_code == 200
                async for line in watch.aiter_lines():
                    if line.startswith("data:"):
                        frames.append(json.loads(line[len("data:") :].strip()))
                        if any(f.get("event") == "done" for f in frames):
                            break

    assert frames[0]["event"] == "user_message"
    assert frames[0]["text"] == "hello there"
    # Exactly once: the say produces ONE user row on the bus (the double-publish
    # class the retired driver-era e2e guarded — same claim, new turn-runner).
    assert sum(1 for f in frames if f.get("event") == "user_message") == 1
    assert {"event": "text", "text": "ok"} in frames
    assert frames[-1]["event"] == "done"


def test_current_session_heals_a_dangling_marker(tmp_path: Path) -> None:
    """A marker naming a deleted session must be removed on the 404 (fresh-eyes F-2):
    POST /sessions refuses to overwrite an existing marker, so without the heal the
    page binds to a session the consumer never runs."""
    app, _ = _build(tmp_path)
    (tmp_path / ".current-session").write_text("deadbeef" * 4 + "\n", encoding="utf-8")

    async def go() -> tuple[int, str]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            missing = await client.get("/sessions/current")
            assert not (tmp_path / ".current-session").exists()  # marker healed
            created = await client.post("/sessions", json={})
            return missing.status_code, created.json()["id"]

    missing_status, created_id = asyncio.run(go())
    assert missing_status == 404
    # The follow-up create now adopts the marker, reconverging page and consumer.
    assert (tmp_path / ".current-session").read_text(encoding="utf-8").strip() == created_id


async def test_prestream_failure_publishes_terminal_done(tmp_path: Path) -> None:
    """A turn that dies before streaming (agent factory raise) must still end with a
    terminal frame on the bus, or every watcher sticks on 'thinking…' (fresh-eyes F-3)."""
    settings = Settings(default_model="scripted/test", workspace_root=tmp_path)
    store = SessionStore(base_dir=tmp_path / "sessions")

    def raising_factory(session: Session, model: object, prompter: object) -> _FakeAgent:
        raise RuntimeError("provider misconfigured")

    app = create_app(settings=settings, store=store, agent_factory=raising_factory)
    with _LiveServer(app) as base_url:
        timeout = httpx.Timeout(10.0, read=5.0)
        async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as ac:
            resp = await ac.post("/say", json={"text": "hello?"})
            assert resp.status_code in (200, 202)
            sid = None
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                current = await ac.get("/sessions/current")
                if current.status_code == 200:
                    sid = current.json()["id"]
                    break
                await asyncio.sleep(0.05)
            assert sid is not None

            frames: list[dict[str, Any]] = []
            async with ac.stream("GET", f"/watch/{sid}?full=1") as watch:
                assert watch.status_code == 200
                async for line in watch.aiter_lines():
                    if line.startswith("data:"):
                        frames.append(json.loads(line[len("data:") :].strip()))
                        if frames[-1].get("event") == "done":
                            break

    assert frames[0]["event"] == "user_message"
    done = frames[-1]
    assert done["event"] == "done"
    assert done["stop_reason"] == "provider_error"
    assert done["degraded"] is True


# ── bounded runs (ADR-0039) ──────────────────────────────────────────────────
# A run is one served process. The cap stops the turn loop; the reserve is carved
# OUT of the cap so the digest turn still has clock left; `on_run_end` reports how
# the run ended so the caller can bring the vessel down. These drive the loop
# directly through the `consume_say_loop` / `start_consumer` / `stop_consumer`
# seams — no server, no lifespan, real (small) durations.


class _TimingAgent:
    """Scripted turn that records what it was asked and how far into the run."""

    def __init__(self, session: Session, seen: list[tuple[str, float]], t0: float) -> None:
        self.session = session
        self._seen = seen
        self._t0 = t0

    async def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]:
        self._seen.append((user_text, time.monotonic() - self._t0))
        self.session.add_message(Message.user(user_text))
        self.session.add_message(Message.assistant_text("ok"))
        yield AgentTextDelta(text="ok")
        yield AgentDone(stop_reason="completed", iterations=1, usage=Usage())


def _build_bounded(
    tmp_path: Path,
    *,
    max_duration: float | None = None,
    reserve: float = 0.0,
    message: str | None = None,
    run_end_command: str | None = None,
) -> tuple[Any, list[tuple[str, float]], list[str], float]:
    """An app whose run is bounded; returns (app, turns_seen, endings, t0)."""
    t0 = time.monotonic()
    seen: list[tuple[str, float]] = []
    endings: list[str] = []
    settings = Settings(
        default_model="scripted/test",
        workspace_root=tmp_path,
        run_max_duration=max_duration,
        run_consolidation_reserve=reserve,
        run_consolidation_message=message,
        run_end_command=run_end_command,
    )

    async def _on_run_end(reason: str) -> None:
        endings.append(reason)

    app = create_app(
        settings=settings,
        store=SessionStore(base_dir=tmp_path / "sessions"),
        agent_factory=lambda session, model, prompter: _TimingAgent(session, seen, t0),
        on_run_end=_on_run_end,
    )
    return app, seen, endings, t0


def test_duration_cap_ends_the_run_and_names_the_reason(tmp_path: Path) -> None:
    """The cap fires on its own — no say, no stop, nobody watching."""
    app, _seen, endings, _t0 = _build_bounded(tmp_path, max_duration=0.3)

    asyncio.run(app.state.consume_say_loop())  # returns only because the cap fired

    assert endings == ["duration_cap"]


def test_cap_hit_still_consolidates(tmp_path: Path) -> None:
    """THE cap-hit path: a run that runs out the clock still delivers its digest.

    The failure this pins is a run that ends by simply stopping — the customer pays
    for the whole window and gets a severed stream instead of a receipt.
    """
    app, seen, endings, _t0 = _build_bounded(
        tmp_path, max_duration=0.4, reserve=0.3, message="wrap up: what did we do?"
    )

    asyncio.run(app.state.consume_say_loop())

    assert [text for text, _ in seen] == ["wrap up: what did we do?"]
    assert endings == ["duration_cap"]


def test_reserve_is_carved_out_of_the_cap_not_added_to_it(tmp_path: Path) -> None:
    """The digest starts BEFORE the cap, because the reserve came out of it.

    Discriminates the three ways this goes wrong: a reserve ADDED to the cap (digest
    starts after `cap`), a reserve IGNORED (digest starts at ~`cap`), and the correct
    carve-out (digest starts at ~`cap - reserve`).
    """
    cap, reserve = 1.2, 0.9
    app, seen, _endings, _t0 = _build_bounded(
        tmp_path, max_duration=cap, reserve=reserve, message="digest"
    )

    asyncio.run(app.state.consume_say_loop())

    assert len(seen) == 1
    _text, started_at = seen[0]
    assert started_at < cap * 0.75, f"digest started at {started_at:.2f}s of a {cap}s cap"


def test_reserve_larger_than_the_cap_still_consolidates(tmp_path: Path) -> None:
    """A reserve >= the cap takes zero turns rather than a NEGATIVE deadline.

    The turn loop must not run (its deadline is the run start), and the digest must
    still fire — otherwise the run bills for the vessel and returns nothing at all.
    """
    app, seen, endings, _t0 = _build_bounded(
        tmp_path, max_duration=0.2, reserve=5.0, message="digest"
    )
    assert write_say(say_path(tmp_path), "this should never become a turn")

    asyncio.run(app.state.consume_say_loop())

    assert [text for text, _ in seen] == ["digest"]
    assert endings == ["duration_cap"]


def test_explicit_stop_consolidates_and_reads_as_stopped(tmp_path: Path) -> None:
    """A human ending the run gets the same receipt as the clock running out — and a
    DIFFERENT reason, because those are different stories to tell."""
    app, seen, endings, _t0 = _build_bounded(tmp_path, reserve=2.0, message="digest")

    async def go() -> None:
        await app.state.start_consumer()
        await asyncio.sleep(0.05)
        await app.state.stop_consumer()

    asyncio.run(go())

    assert [text for text, _ in seen] == ["digest"]
    assert endings == ["stopped"]


def test_an_unbounded_run_never_ends_itself(tmp_path: Path) -> None:
    """The default is unchanged: no cap, no digest, no ending — the loop just runs.

    The positive control for every test above. Without it a bug that ended EVERY run
    immediately would leave them all green.
    """
    app, seen, endings, _t0 = _build_bounded(tmp_path)

    async def go() -> bool:
        task = asyncio.create_task(app.state.consume_say_loop())
        await asyncio.sleep(0.9)  # well past the cap the other tests use
        still_running = not task.done()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return still_running

    assert asyncio.run(go()) is True
    assert seen == []
    assert endings == []


class _SlowAgent:
    """A turn that takes real time — needed to test a BUDGET, which is a ceiling."""

    def __init__(self, session: Session, seen: list[str], delay: float) -> None:
        self.session = session
        self._seen = seen
        self._delay = delay

    async def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]:
        self._seen.append(user_text)
        await asyncio.sleep(self._delay)
        self.session.add_message(Message.assistant_text("ok"))
        yield AgentDone(stop_reason="completed", iterations=1, usage=Usage())


def test_a_reserve_larger_than_the_cap_cannot_overrun_the_cap(tmp_path: Path) -> None:
    """The clamp: `run_consolidation_reserve` is a FLOOR on the digest budget, so an
    unclamped reserve larger than the cap would push the run past its own ceiling —
    and a cap a misconfiguration can exceed is not a hard cap.

    An instant agent cannot detect this (a budget is a ceiling, not a sleep), so the
    digest here deliberately takes longer than the cap: clamped, it is cut off at the
    cap; unclamped, it would run the full 1.5s.
    """
    cap, raw_reserve, digest_time = 0.3, 5.0, 1.5
    seen: list[str] = []
    settings = Settings(
        default_model="scripted/test",
        workspace_root=tmp_path,
        run_max_duration=cap,
        run_consolidation_reserve=raw_reserve,
        run_consolidation_message="digest",
    )
    app = create_app(
        settings=settings,
        store=SessionStore(base_dir=tmp_path / "sessions"),
        agent_factory=lambda session, model, prompter: _SlowAgent(session, seen, digest_time),
    )

    started = time.monotonic()
    asyncio.run(app.state.consume_say_loop())
    elapsed = time.monotonic() - started

    assert seen == ["digest"]  # the digest was still attempted
    assert elapsed < digest_time, f"run took {elapsed:.2f}s — the reserve escaped the cap"


# ── run_end_command: the ending leaves the process (ADR-0046) ─────────────────


def _sink_command(tmp_path: Path) -> tuple[str, Path]:
    """A command that writes its stdin to a file; returns (command, that file)."""
    sink = tmp_path / "sink.py"
    sink.write_text(
        "import pathlib, sys\npathlib.Path(sys.argv[1]).write_bytes(sys.stdin.buffer.read())\n",
        encoding="utf-8",
    )
    out = tmp_path / "ending.json"
    return f"{sys.executable} {sink} {out}", out


def test_run_end_command_receives_reason_and_digest_before_the_vessel_goes_down(
    tmp_path: Path,
) -> None:
    """THE delivery seam: after the digest turn, the ending — reason + digest text — is
    handed to the operator's command on stdin, and only then does on_run_end fire."""
    command, out = _sink_command(tmp_path)
    app, seen, endings, _t0 = _build_bounded(
        tmp_path, max_duration=0.4, reserve=0.2, message="wrap up", run_end_command=command
    )

    asyncio.run(app.state.consume_say_loop())

    ending = json.loads(out.read_text(encoding="utf-8"))
    assert ending["event"] == "run_end"
    assert ending["reason"] == "duration_cap"
    assert ending["digest"] == "ok"  # what _TimingAgent answered the digest prompt with
    assert ending["cwd"] == str(tmp_path)
    assert seen[-1][0] == "wrap up"  # the digest turn ran first ...
    assert endings == ["duration_cap"]  # ... and the caller was told last


def test_run_end_command_failures_are_fail_open(tmp_path: Path) -> None:
    """A command that exits non-zero, or cannot start at all, never strands the ending."""
    boom = tmp_path / "boom.py"
    boom.write_text("import sys\nsys.stdin.read()\nraise SystemExit(3)\n", encoding="utf-8")
    for command in (f"{sys.executable} {boom}", str(tmp_path / "no-such-command")):
        app, _seen, endings, _t0 = _build_bounded(
            tmp_path, max_duration=0.3, message="wrap up", run_end_command=command
        )
        asyncio.run(app.state.consume_say_loop())
        assert endings == ["duration_cap"]


def test_run_end_command_runs_without_a_digest_turn(tmp_path: Path) -> None:
    """No consolidation message = no digest turn, but the ENDING is still reported."""
    command, out = _sink_command(tmp_path)
    app, seen, endings, _t0 = _build_bounded(tmp_path, max_duration=0.3, run_end_command=command)

    asyncio.run(app.state.consume_say_loop())

    ending = json.loads(out.read_text(encoding="utf-8"))
    assert ending["reason"] == "duration_cap" and ending["digest"] == ""
    assert seen == []
    assert endings == ["duration_cap"]


# ── POST /run/stop: a bound outside the process ends the run WITH its receipt (ADR-0047) ──


def test_run_stop_route_ends_the_run_with_its_digest(tmp_path: Path) -> None:
    """A money cap (or an operator) that lives outside the process ends the run the way a
    cap-hit does: digest turn, then run_end_command, then on_run_end — reason preserved."""
    command, out = _sink_command(tmp_path)
    app, seen, endings, _t0 = _build_bounded(tmp_path, message="wrap up", run_end_command=command)

    async def scenario() -> None:
        loop_task = asyncio.create_task(app.state.consume_say_loop())
        await asyncio.sleep(0.2)  # idling, unbounded — nothing would ever end this run
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            bad = await client.post("/run/stop", json={"reason": "Not A Token!"})
            assert bad.status_code == 400
            first = await client.post("/run/stop", json={"reason": "budget_exhausted"})
            assert first.status_code == 200
            assert first.json() == {"stopping": True, "reason": "budget_exhausted"}
            again = await client.post("/run/stop", json={"reason": "stopped"})
            assert again.json() == {"stopping": True, "reason": "budget_exhausted"}  # first wins
            await asyncio.wait_for(loop_task, timeout=5)
            after = await client.post("/run/stop")
            assert after.json() == {"stopping": False, "ended": True, "reason": "budget_exhausted"}

    asyncio.run(scenario())
    assert seen[-1][0] == "wrap up"  # the digest turn ran
    assert json.loads(out.read_text(encoding="utf-8"))["reason"] == "budget_exhausted"
    assert endings == ["budget_exhausted"]
