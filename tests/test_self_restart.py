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


def test_restart_args_pins_the_session_and_names_the_chat_command() -> None:
    assert _restart_args([], "abc") == ["chat", "--session", "abc"]
    assert _restart_args(["--model", "m"], "abc") == ["chat", "--model", "m", "--session", "abc"]
    assert _restart_args(["chat", "-s", "old", "--model", "m"], "abc") == [
        "chat",
        "--model",
        "m",
        "--session",
        "abc",
    ]
    assert _restart_args(["chat", "--session=old"], "abc") == ["chat", "--session", "abc"]
    assert _restart_args(["chat", "--session", "old", "-w", "/w"], "abc") == [
        "chat",
        "-w",
        "/w",
        "--session",
        "abc",
    ]


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
