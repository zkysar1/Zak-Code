"""The served marker (ADR-0320): a served workspace's say slot and interrupt file are the server's.

``zakcode webapp`` runs every turn on the chat its page has open (ADR-0319), and the page names
that chat on every say it sends. The slot is one file the whole workspace shares, and a terminal
REPL left idle on the same workspace polls it every 0.3 s against the server's 0.5 s beat. Measured
2026-10-16: the REPL took the page's say ten times in ten and ran it in its own conversation, and
in one trial of eight it cleared the page's Stop as a stale signal, so the page's turn ran on.

The server now marks the workspace ``.served`` while its say consumer runs. While a fresh marker
names another process, a REPL reads no say and clears, reports or obeys no stop: idle, prompting,
mid-turn and at its start. The server's own consumers are unchanged. Hermetic: tmp workspaces and
scripted turns, plus one child process for the REPL, since the marker tells processes apart.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from rich.console import Console

import zakcode
from tests.test_busy_marker import _loop, _peek_script, _SayWhileRunning
from tests.test_cli_chat import FakeAgent, runner
from tests.test_webapp_sessions import _Served
from zakcode.cli import _InputMux, app
from zakcode.events import AgentDone, AgentEvent
from zakcode.session.say_inbox import (
    BUSY_STALE_SECONDS,
    busy_path,
    claim_busy,
    interrupt_path,
    request_interrupt,
    say_path,
    say_pending,
    served_path,
    write_say,
)
from zakcode.usage import Usage


def _foreign_served(workspace: Path) -> Path:
    """A fresh served marker written by some OTHER process: a web app serving the workspace."""
    path = served_path(workspace)
    payload = {"pid": os.getpid() + 100_000, "session": "served", "since": time.time()}
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def _age(path: Path) -> None:
    stamp = time.time() - BUSY_STALE_SECONDS - 5
    os.utime(path, (stamp, stamp))


def _stopped_after(seconds: float) -> threading.Event:
    stop = threading.Event()
    threading.Timer(seconds, stop.set).start()
    return stop


# ── the server holds the marker ───────────────────────────────────────────────


def test_the_server_marks_the_workspace_for_as_long_as_it_serves(tmp_path: Path) -> None:
    served = _Served(tmp_path)
    marker = served_path(tmp_path)
    assert not marker.exists()
    with TestClient(served.app):
        # Written before the server takes a request, naming this process, and fresh.
        assert json.loads(marker.read_text(encoding="utf-8"))["pid"] == os.getpid()
        assert time.time() - marker.stat().st_mtime < BUSY_STALE_SECONDS
    assert not marker.exists()


def test_a_starting_server_takes_over_the_marker_a_dead_one_left(tmp_path: Path) -> None:
    # A server killed mid-run leaves its marker fresh for two minutes, and the service unit
    # restarts it after every crash: the new one must not run unmarked for its whole life.
    _foreign_served(tmp_path)
    with TestClient(_Served(tmp_path).app):
        marker = json.loads(served_path(tmp_path).read_text(encoding="utf-8"))
        assert marker["pid"] == os.getpid()


def test_a_server_that_starts_ended_does_not_mark_the_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A restart that keeps this boot's duration cap starts ENDED (ADR-0256): no say consumer runs,
    # so nothing serves the slot, and a REPL on the workspace must not stand back for nobody.
    monkeypatch.setattr("zakcode.server.app._boot_id", lambda: "boot-A")
    served = _Served(tmp_path)
    earlier = served.saved(("an earlier ask", "a reply"))
    served.mark(earlier)
    (tmp_path / ".run-stop-reason").write_text(
        f"{earlier.id}\nduration_cap\nboot-A", encoding="utf-8"
    )
    with TestClient(served.app) as client:
        assert client.get("/sidecar/health").json()["last_run_stop_reason"] == "duration_cap"
        assert not served_path(tmp_path).exists()


# ── a REPL on a served workspace leaves the slot and the stop to the server ───


def test_an_idle_repl_leaves_a_served_workspaces_say_and_stop(tmp_path: Path) -> None:
    mux = _InputMux(say_path(tmp_path), interrupt_path(tmp_path), keyboard=False)
    assert write_say(say_path(tmp_path), "for the chat on the page")
    request_interrupt(interrupt_path(tmp_path))
    marker = _foreign_served(tmp_path)
    assert mux.try_input() is None
    assert mux.next_input(idle=True, stop=_stopped_after(0.8)) == ("cancelled", None)
    assert say_pending(say_path(tmp_path))  # still the server's to take
    assert interrupt_path(tmp_path).exists()  # still the server's turn's to obey
    _age(marker)  # the server died: the REPL is the workspace's consumer again
    assert mux.try_input() == ("say", "for the chat on the page")
    assert not interrupt_path(tmp_path).exists()  # cleared as a stale signal, as before


def test_a_prompting_repl_leaves_a_served_workspaces_say_and_stop(tmp_path: Path) -> None:
    # A permission prompt is answered from the keyboard: a "y" the page sent for its own chat
    # must not approve a tool call in the terminal, and the page's Stop must not end its turn.
    mux = _InputMux(say_path(tmp_path), interrupt_path(tmp_path), keyboard=False)
    assert write_say(say_path(tmp_path), "y")
    request_interrupt(interrupt_path(tmp_path))
    marker = _foreign_served(tmp_path)
    assert mux.next_input(idle=False, stop=_stopped_after(0.8)) == ("cancelled", None)
    assert say_pending(say_path(tmp_path))
    assert interrupt_path(tmp_path).exists()
    marker.unlink()  # control: the same prompt on an unserved workspace
    assert mux.next_input(idle=False) == ("stop", None)
    interrupt_path(tmp_path).unlink()
    assert mux.next_input(idle=False) == ("say", "y")


def test_a_repl_turn_leaves_a_served_workspaces_say_in_the_slot(tmp_path: Path) -> None:
    _foreign_served(tmp_path)
    loop = _loop(tmp_path, _peek_script("steer"), _SayWhileRunning(tmp_path), consume=True)
    asyncio.run(loop.arun_turn("go"))
    assert say_pending(say_path(tmp_path))
    assert not any("steer me" in (m.text or "") for m in loop.session.messages)


def test_the_servers_own_turn_still_takes_a_say_mid_turn(tmp_path: Path) -> None:
    # The marker names THIS process: the server's own loop delivers at its boundary (ADR-0051).
    assert claim_busy(served_path(tmp_path), "served", take_over=True)
    loop = _loop(tmp_path, _peek_script("steer"), _SayWhileRunning(tmp_path), consume=True)
    asyncio.run(loop.arun_turn("go"))
    assert not say_pending(say_path(tmp_path))
    assert any("steer me" in (m.text or "") for m in loop.session.messages if m.role == "user")


def _run_turn_with_stop_file(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[bool, str]:
    import zakcode.cli as cli_mod
    from zakcode.cli.render import StreamRenderer

    loop = asyncio.new_event_loop()
    monkeypatch.setattr(cli_mod, "_SESSION_LOOP", loop)
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, force_terminal=False)

    async def stream() -> AsyncIterator[AgentEvent]:
        await asyncio.sleep(0.8)  # three of the file watcher's 0.25 s polls
        yield AgentDone(stop_reason="completed", iterations=1, usage=Usage())

    try:
        completed = cli_mod._run_streamed_turn(
            console,
            stream,
            StreamRenderer(console=console),
            interrupt_file=interrupt_path(workspace),
        )
    finally:
        loop.close()
    return completed, buffer.getvalue()


def test_a_repl_turn_leaves_a_served_workspaces_stop_to_the_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_interrupt(interrupt_path(tmp_path))
    marker = _foreign_served(tmp_path)
    completed, _ = _run_turn_with_stop_file(tmp_path, monkeypatch)
    assert completed is True  # the page's Stop was not this turn's
    assert interrupt_path(tmp_path).exists()
    marker.unlink()  # control: unserved, the same file stops the same turn
    completed, out = _run_turn_with_stop_file(tmp_path, monkeypatch)
    assert completed is False
    assert "interrupted" in out
    assert not interrupt_path(tmp_path).exists()


@pytest.mark.parametrize("serving", [True, False])
def test_a_repl_starting_on_a_served_workspace_keeps_the_pending_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, serving: bool
) -> None:
    monkeypatch.setattr(zakcode, "Agent", FakeAgent)
    monkeypatch.setenv("ZAKCODE_WORKSPACE_ROOT", str(tmp_path))
    request_interrupt(interrupt_path(tmp_path))
    if serving:
        _foreign_served(tmp_path)
    result = runner.invoke(app, ["cli"], input="/exit\n")
    assert result.exit_code == 0
    # Served, the stop may be for the server's running turn; unserved, it predates the session.
    assert interrupt_path(tmp_path).exists() is serving


def test_a_switch_refused_for_another_sessions_turn_says_so(tmp_path: Path) -> None:
    served = _Served(tmp_path)
    current = served.saved(("an ask", "a reply"))
    other = served.saved(("another ask", "another reply"))
    served.mark(current)
    busy_path(tmp_path).write_text(
        json.dumps({"pid": os.getpid() + 100_000, "session": "terminal", "since": time.time()}),
        encoding="utf-8",
    )
    resp = served.put_current({"session_id": other.id})
    assert resp.status_code == 409
    assert resp.json()["detail"] == (
        "a turn is running in another session on this workspace; let it finish, then switch"
    )
    assert served.marker() == current.id


# ── the field shape: the page's say, with a REPL idle in another process ──────

_IDLE_REPL = """
import sys, threading
from pathlib import Path
import zakcode
from zakcode.cli import _InputMux
from zakcode.session.say_inbox import interrupt_path, say_path
workspace = Path(sys.argv[1])
mux = _InputMux(say_path(workspace), interrupt_path(workspace), keyboard=False)
stop = threading.Event()
threading.Timer(float(sys.argv[2]), stop.set).start()
print("listening", zakcode.__file__, flush=True)
kind, text = mux.next_input(idle=True, stop=stop)
print(kind, text or "", flush=True)
"""


def test_the_pages_say_runs_on_its_chat_with_a_repl_idle_on_the_workspace(
    tmp_path: Path,
) -> None:
    # The child must import the code under test, not whatever install its interpreter finds
    # first: pytest puts src/ on THIS process's path only (`pythonpath` in pyproject.toml).
    src = str(Path(zakcode.__file__).resolve().parents[1])
    child_path = os.pathsep.join(p for p in (src, os.environ.get("PYTHONPATH", "")) if p)
    served = _Served(tmp_path)
    served.saved(("first chat", "ok"), sid="chat-a")
    chat_b = served.saved(("second chat", "ok"), sid="chat-b")
    with TestClient(served.app) as client:
        assert client.put("/sessions/current", json={"session_id": chat_b.id}).status_code == 200
        repl = subprocess.Popen(
            [sys.executable, "-c", _IDLE_REPL, str(tmp_path), "6"],
            stdout=subprocess.PIPE,
            text=True,
            env={**os.environ, "PYTHONPATH": child_path},
        )
        try:
            assert repl.stdout is not None
            ready = repl.stdout.readline().strip().split(" ", 1)
            assert ready[0] == "listening", ready
            assert Path(ready[1]).resolve() == Path(zakcode.__file__).resolve(), ready
            time.sleep(0.6)  # both doors polling: the REPL every 0.3 s, the server every 0.5 s
            body = {"text": "typed on the page for chat B", "session_id": chat_b.id}
            assert client.post("/say", json=body).json() == {"queued": True}
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not served.seen:
                time.sleep(0.05)
            out, _ = repl.communicate(timeout=60)
        finally:
            if repl.poll() is None:
                repl.kill()
    assert out.strip() == "cancelled"  # the REPL stood back for its whole wait
    assert [sid for sid, _ in served.seen] == [chat_b.id]  # the page's chat ran it
