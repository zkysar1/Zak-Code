"""Self-restart on update (ADR-0034): the install probe, the idle-only mux door, the exec.

Field incident 2026-08-26 (serene): `zakcode update` landed while a chat sat idle, printed
"running chat sessions keep the old build until restarted", and the chat kept running the
old build for the rest of the evening — the next turn collapsed on code that had already
been fixed. These tests pin the three pieces: the install-marker comparison (keyed on the
reinstall, never on a moving dev HEAD), the mux's idle-only ``restart`` kind, and the
handoff that stamps the session for the build that will read it before exec.
"""

from __future__ import annotations

import io
import os
import queue
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

import zakcode.build_info as bi
import zakcode.cli as cli
from zakcode.cli import _InputMux, _restart_args
from zakcode.session.store import Session, SessionStore

# ── _restart_args (pure) ─────────────────────────────────────────────────────


def test_restart_args_pins_the_session_and_names_the_cli_command() -> None:
    # The inserted name must be a command Typer HAS: the REPL command is ``cli`` (renamed
    # from ``chat`` in #204). This test pinned the old name for a while, so a bare-``zakcode``
    # REPL self-restarted into ``zakcode chat --session …`` — a usage error, not a resume.
    assert _restart_args([], "abc") == ["cli", "--session", "abc"]
    assert _restart_args(["--model", "m"], "abc") == ["cli", "--model", "m", "--session", "abc"]
    assert _restart_args(["cli", "-s", "old", "--model", "m"], "abc") == [
        "cli",
        "--model",
        "m",
        "--session",
        "abc",
    ]
    assert _restart_args(["cli", "--session=old"], "abc") == ["cli", "--session", "abc"]
    assert _restart_args(["cli", "--session", "old", "-w", "/w"], "abc") == [
        "cli",
        "-w",
        "/w",
        "--session",
        "abc",
    ]


def test_restart_args_names_a_command_the_cli_actually_registers() -> None:
    # Positive control for the rename class: whatever ``_restart_args`` inserts must be a
    # registered command name, or the exec'd process dies at argument parsing.
    from typer.main import get_command

    registered = set(get_command(cli.app).commands)  # the click group: names as Typer builds them
    assert "cli" in registered and "chat" not in registered
    assert _restart_args([], "abc")[0] in registered


# ── install_changed (the probe) ──────────────────────────────────────────────


def test_install_changed_keys_on_the_reinstall_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bi, "_RUNNING_IDENTITY", ("aaa", 100.0))
    monkeypatch.setattr(bi, "install_identity", lambda: ("aaa", 100.0))
    assert bi.install_changed() is None
    # A reinstall rewrites direct_url.json: new marker → changed, labels reported.
    monkeypatch.setattr(bi, "install_identity", lambda: ("bbb", 200.0))
    assert bi.install_changed() == ("aaa", "bbb")
    # A dev checkout whose HEAD moved WITHOUT a reinstall is not a new install.
    monkeypatch.setattr(bi, "install_identity", lambda: ("ccc", 100.0))
    assert bi.install_changed() is None


def test_install_changed_is_inert_without_an_installed_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bi, "_RUNNING_IDENTITY", ("", None))
    monkeypatch.setattr(bi, "install_identity", lambda: ("x", 5.0))
    assert bi.install_changed() is None
    monkeypatch.setattr(bi, "_RUNNING_IDENTITY", ("aaa", 100.0))
    monkeypatch.setattr(bi, "install_identity", lambda: ("", None))
    assert bi.install_changed() is None


def test_running_build_is_frozen_at_import(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bi, "_RUNNING_IDENTITY", ("frozen", 1.0))
    monkeypatch.setattr(bi, "install_identity", lambda: ("moved", 2.0))
    assert bi.running_build() == "frozen"


def test_install_identity_labels_a_git_install_by_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        bi, "_read_direct_url", lambda: {"vcs_info": {"vcs": "git", "commit_id": "b" * 40}}
    )
    monkeypatch.setattr(bi, "_install_marker", lambda: 42.0)
    assert bi.install_identity() == ("b" * 12, 42.0)


def test_install_identity_labels_a_local_checkout_by_its_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bi, "_read_direct_url", lambda: {"dir_info": {}, "url": "file:///opt/somewhere/Zak-Code"}
    )
    monkeypatch.setattr(bi, "_checkout_head", lambda directory: "c" * 12)
    monkeypatch.setattr(bi, "_install_marker", lambda: 43.0)
    assert bi.install_identity() == ("c" * 12, 43.0)


# ── the mux door ─────────────────────────────────────────────────────────────


def test_idle_mux_reports_a_restart_when_the_probe_fires(tmp_path: Path) -> None:
    mux = _InputMux(tmp_path / "say", tmp_path / "stop", keyboard=False, idle_probe=lambda: True)
    mux._idle_probe_every = 0.0
    assert mux.next_input(idle=True) == ("restart", None)


def test_mid_turn_wait_never_consults_the_probe(tmp_path: Path) -> None:
    # A permission prompt is not an idle boundary: the probe must not fire there.
    asked: list[int] = []

    def probe() -> bool:
        asked.append(1)
        return True

    mux = _InputMux(tmp_path / "say", tmp_path / "stop", keyboard=False, idle_probe=probe)
    mux._idle_probe_every = 0.0
    stop = threading.Event()
    threading.Timer(0.5, stop.set).start()
    assert mux.next_input(idle=False, stop=stop) == ("cancelled", None)
    assert asked == []


def test_mux_without_a_probe_keeps_waiting(tmp_path: Path) -> None:
    mux = _InputMux(tmp_path / "say", tmp_path / "stop", keyboard=False)
    stop = threading.Event()
    threading.Timer(0.5, stop.set).start()
    assert mux.next_input(idle=True, stop=stop) == ("cancelled", None)


# ── the handoff ──────────────────────────────────────────────────────────────


def _console() -> Console:
    return Console(theme=cli.ZAK_THEME, file=io.StringIO(), force_terminal=False, width=100)


def _agent(tmp_path: Path) -> tuple[SimpleNamespace, SessionStore]:
    store = SessionStore(tmp_path / "sessions")
    session = Session(cwd=str(tmp_path), model="m", build="old-build")
    store.save(session)
    return SimpleNamespace(session=session, loop=SimpleNamespace(store=store)), store


def test_restart_stamps_the_session_then_execs_with_it_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, store = _agent(tmp_path)
    monkeypatch.setattr(cli, "install_changed", lambda: ("old-build", "new-build"))
    execs: list[list[str]] = []
    monkeypatch.setattr(cli.os, "execv", lambda path, argv: execs.append([path, *argv]))
    monkeypatch.setattr(cli.sys, "argv", ["zakcode", "chat", "-s", "stale-id", "--model", "m"])
    cli._restart_into_new_build(_console(), agent)
    assert execs == [
        [
            sys.executable,
            sys.executable,
            "-m",
            "zakcode",
            "chat",
            "--model",
            "m",
            "--session",
            agent.session.id,
        ]
    ]
    # Stamped for the build that will READ it: the resumed session is an upgrade, not a
    # cross-build collapse, so ADR-0033's compaction has nothing to say.
    loaded = store.load(agent.session.id)
    assert loaded.build == "new-build"
    assert loaded.resume_notice(running_build="new-build") is None


def test_restart_is_a_no_op_when_nothing_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, store = _agent(tmp_path)
    monkeypatch.setattr(cli, "install_changed", lambda: None)
    execs: list[str] = []
    monkeypatch.setattr(cli.os, "execv", lambda path, argv: execs.append(path))
    cli._restart_into_new_build(_console(), agent)
    assert execs == []
    assert store.load(agent.session.id).build == "old-build"


def test_a_failed_exec_keeps_serving_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, _store = _agent(tmp_path)
    monkeypatch.setattr(cli, "install_changed", lambda: ("old-build", "new-build"))

    def boom(path: str, argv: list[str]) -> None:
        raise OSError("exec denied")

    monkeypatch.setattr(cli.os, "execv", boom)
    out = io.StringIO()
    console = Console(theme=cli.ZAK_THEME, file=out, force_terminal=False, width=100)
    cli._restart_into_new_build(console, agent)  # must not raise
    assert "restart failed" in out.getvalue()


# ── an unattended session continues at the prompt (ADR-0090) ─────────────────


def _unattended_agent(
    tmp_path: Path,
    *,
    unattended: bool = True,
    statuses: tuple[str, ...] = ("done", "pending", "pending"),
) -> SimpleNamespace:
    from zakcode.tasks import Task

    session = Session(cwd=str(tmp_path), model="m", build="new-build")
    session.task_network.tasks = [
        Task(title=f"step {i}", status=s)
        for i, s in enumerate(statuses, 1)  # type: ignore[arg-type]
    ]
    session.task_network.normalize()
    return SimpleNamespace(session=session, loop=SimpleNamespace(unattended=lambda: unattended))


def test_restart_marks_the_new_process_as_a_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, _store = _agent(tmp_path)
    monkeypatch.delenv("ZAKCODE_RESTARTED_INTO", raising=False)
    monkeypatch.setattr(cli, "install_changed", lambda: ("old-build", "new-build"))
    monkeypatch.setattr(cli.os, "execv", lambda path, argv: None)
    monkeypatch.setattr(cli.sys, "argv", ["zakcode", "cli", "-s", "stale-id"])
    cli._restart_into_new_build(_console(), agent)
    assert os.environ["ZAKCODE_RESTARTED_INTO"] == "new-build"


def test_an_unattended_restart_with_open_steps_continues_the_plan(tmp_path: Path) -> None:
    """coach-w3 (2026-08-29): a doom-loop end, the restart into the next build, then 46
    minutes at the prompt with 20 of 23 steps open — nobody types at a worker Body."""
    line = cli._unattended_continuation(
        _unattended_agent(tmp_path), restarted="new-build", stop_reason=None
    )
    assert line is not None
    assert line.startswith("[harness] this session was restarted into build new-build")
    assert "2 of 3 plan steps are still open" in line and "Do not stop to wait" in line


def test_a_collapsed_turn_continues_and_any_other_end_does_not(tmp_path: Path) -> None:
    agent = _unattended_agent(tmp_path)
    for reason in ("doom_loop", "gave_up", "degenerated", "stuck"):
        line = cli._unattended_continuation(agent, restarted=None, stop_reason=reason)
        assert line is not None and f"the previous turn ended {reason!r}" in line
    for reason in ("completed", "interrupted", "budget_exhausted", None):
        assert cli._unattended_continuation(agent, restarted=None, stop_reason=reason) is None


def test_no_continuation_when_attended_or_nothing_is_open(tmp_path: Path) -> None:
    attended = _unattended_agent(tmp_path, unattended=False)
    assert cli._unattended_continuation(attended, restarted="new-build", stop_reason=None) is None
    finished = _unattended_agent(tmp_path, statuses=("done", "done", "cancelled"))
    assert cli._unattended_continuation(finished, restarted="new-build", stop_reason=None) is None
    empty = _unattended_agent(tmp_path, statuses=())
    assert cli._unattended_continuation(empty, restarted="b", stop_reason="doom_loop") is None
    bare = SimpleNamespace(session=finished.session)  # a thin remote agent: no loop at all
    assert cli._unattended_continuation(bare, restarted="new-build", stop_reason=None) is None


def test_after_a_collapse_the_continuation_is_queued_once_in_a_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _unattended_agent(tmp_path)
    agent.session.last_stop_reason = "doom_loop"
    compacted: list[str] = []
    monkeypatch.setattr(
        cli, "_announce_resume", lambda console, a: compacted.append(a.session.last_stop_reason)
    )
    mux = SimpleNamespace(queue=queue.Queue())
    collapsed = SimpleNamespace(stop_reason="doom_loop")
    kicks = cli._continue_after_collapse(_console(), agent, mux, collapsed, 0)
    assert kicks == 1 and compacted == ["doom_loop"]
    kind, line = mux.queue.get_nowait()
    assert kind == "harness" and "the previous turn ended 'doom_loop'" in line
    # A second collapse in a row ends at the prompt, for the operator to see.
    assert cli._continue_after_collapse(_console(), agent, mux, collapsed, kicks) == 1
    assert mux.queue.empty()
    # A clean turn resets the run; an attended session is never continued.
    clean = SimpleNamespace(stop_reason="completed")
    assert cli._continue_after_collapse(_console(), agent, mux, clean, 1) == 0
    attended = _unattended_agent(tmp_path, unattended=False)
    assert cli._continue_after_collapse(_console(), attended, mux, collapsed, 0) == 0
    assert mux.queue.empty()


@pytest.fixture(autouse=True)
def _no_restart_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_restart_into_new_build`` sets the ADR-0090 marker for the process it execs into;
    with the exec monkeypatched the marker would outlive the test."""
    monkeypatch.delenv("ZAKCODE_RESTARTED_INTO", raising=False)
