"""Tests for ``zakcode cockpit`` / ``zakcode say`` (tmux fully faked — no server needed)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import typer
from rich.console import Console

from zakcode.cli import cockpit
from zakcode.cli._theme import ZAK_THEME


def _rec_console() -> Console:
    return Console(theme=ZAK_THEME, highlight=False, record=True, width=100, force_terminal=False)


class _FakeTmux:
    """Records every subprocess argv; answers has-session / capture-pane per config."""

    def __init__(self, *, has_session_rc: int = 1, capture_stdout: bytes = b"") -> None:
        self.calls: list[list[str]] = []
        self.has_session_rc = has_session_rc
        self.capture_stdout = capture_stdout

    def __call__(self, argv, **kwargs):  # noqa: ANN001, ANN003, ANN204
        self.calls.append([str(a) for a in argv])
        sub = argv[1] if len(argv) > 1 else ""
        if sub == "has-session":
            return subprocess.CompletedProcess(argv, self.has_session_rc)
        if sub == "capture-pane":
            return subprocess.CompletedProcess(argv, 0, stdout=self.capture_stdout, stderr=b"")
        return subprocess.CompletedProcess(argv, 0, stdout=b"", stderr=b"")

    def subcommands(self) -> list[str]:
        return [c[1] for c in self.calls if len(c) > 1]


@pytest.fixture
def fake_tmux(monkeypatch: pytest.MonkeyPatch) -> _FakeTmux:
    fake = _FakeTmux()
    monkeypatch.setattr(
        cockpit.shutil,
        "which",
        lambda name: {"tmux": "/usr/bin/tmux", "zakcode": "/usr/bin/zakcode"}.get(name),
    )
    monkeypatch.setattr(cockpit.subprocess, "run", fake)
    monkeypatch.setattr(cockpit, "console", _rec_console())
    return fake


def test_cockpit_refuses_without_tmux(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cockpit.shutil, "which", lambda name: None)
    monkeypatch.setattr(cockpit, "console", _rec_console())
    with pytest.raises(typer.Exit) as excinfo:
        cockpit.cockpit(session="x", workspace=tmp_path, ledger=None, attach=True)
    assert excinfo.value.exit_code == 1
    assert "tmux" in cockpit.console.export_text()


def test_cockpit_creates_session_in_inheritance_safe_order(
    fake_tmux: _FakeTmux, tmp_path: Path
) -> None:
    cockpit.cockpit(session="agentbox", workspace=tmp_path, ledger=None, attach=True)
    subs = fake_tmux.subcommands()
    # Say box first, then session options, THEN the chat split — history-limit is
    # applied at pane creation, so the order is load-bearing.
    assert subs.index("new-session") < subs.index("set-option")
    assert max(i for i, s in enumerate(subs) if s == "set-option") < subs.index("split-window")
    split = next(c for c in fake_tmux.calls if c[1] == "split-window")
    assert "-b" in split and "-v" in split
    history = next(c for c in fake_tmux.calls if "history-limit" in c)
    assert str(cockpit._HISTORY_LIMIT) in history
    resize = next(c for c in fake_tmux.calls if c[1] == "resize-pane")
    assert resize[-1] == str(cockpit._SAY_BOX_HEIGHT) and "agentbox:0.1" in resize
    # stdin is not a tty under pytest, so no attach happened.
    assert "attach-session" not in subs


def test_cockpit_existing_session_attaches_only(
    fake_tmux: _FakeTmux, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_tmux.has_session_rc = 0
    monkeypatch.setattr(cockpit.sys.stdin, "isatty", lambda: True)
    cockpit.cockpit(session="x", workspace=tmp_path, ledger=None, attach=True)
    assert fake_tmux.subcommands() == ["has-session", "attach-session"]


def test_say_refuses_without_session(fake_tmux: _FakeTmux, tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    with pytest.raises(typer.Exit) as excinfo:
        cockpit.say(text="hi", session="ghost", ledger=ledger, operator=None)
    assert excinfo.value.exit_code == 1
    assert not ledger.exists()
    assert "not running" in cockpit.console.export_text()


def test_say_ledgers_and_sends(fake_tmux: _FakeTmux, tmp_path: Path) -> None:
    fake_tmux.has_session_rc = 0
    ledger = tmp_path / "deep" / "ledger.jsonl"
    cockpit.say(text="hello there", session="agentbox", ledger=ledger, operator="alpha@cc-14")
    row = json.loads(ledger.read_text(encoding="utf-8").strip())
    assert row["text"] == "hello there"
    assert row["via"] == "zakcode-say"
    assert row["operator"] == "alpha@cc-14"
    sends = [c for c in fake_tmux.calls if c[1] == "send-keys"]
    assert len(sends) == 2
    assert sends[0][-1] == "hello there" and "-l" in sends[0] and "agentbox:0.0" in sends[0]
    assert sends[1][-1] == "Enter"
    assert "sent" in cockpit.console.export_text()


def test_say_reports_queued_mid_turn(fake_tmux: _FakeTmux, tmp_path: Path) -> None:
    fake_tmux.has_session_rc = 0
    fake_tmux.capture_stdout = b"thinking (ctrl-c to interrupt . 12s)"
    cockpit.say(text="hi", session="x", ledger=tmp_path / "l.jsonl", operator=None)
    assert "queued" in cockpit.console.export_text()


def test_append_ledger_default_operator(tmp_path: Path) -> None:
    ledger = tmp_path / "a" / "b.jsonl"
    cockpit._append_ledger(ledger, "line one", via="cockpit-say-box", operator=None)
    cockpit._append_ledger(ledger, "line two", via="cockpit-say-box", operator=None)
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert [r["text"] for r in rows] == ["line one", "line two"]
    assert "@" in rows[0]["operator"]
    assert rows[0]["ts"]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX execute bit")
def test_banner_hook_requires_execute_bit(tmp_path: Path) -> None:
    hook = tmp_path / ".zakcode" / "banner"
    hook.parent.mkdir()
    hook.write_text("#!/bin/sh\necho hi\n")
    assert cockpit._banner_hook(tmp_path) is None
    hook.chmod(0o755)
    assert cockpit._banner_hook(tmp_path) == hook


def test_self_invocation_falls_back_to_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cockpit.shutil, "which", lambda name: None)
    assert cockpit._self_invocation() == [sys.executable, "-m", "zakcode"]


def test_cockpit_main_runs_chat_frameless_then_exits_on_eof(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runs: list[tuple[list[str], dict[str, str]]] = []

    def _record(argv, **kwargs):  # noqa: ANN001, ANN003, ANN204
        runs.append(([str(a) for a in argv], dict(kwargs.get("env") or {})))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(cockpit.subprocess, "run", _record)
    monkeypatch.setattr(
        cockpit.shutil, "which", lambda name: "/usr/bin/zakcode" if name == "zakcode" else None
    )
    monkeypatch.setattr(cockpit, "console", _rec_console())
    monkeypatch.setenv("HOME", str(tmp_path))  # empty session store for the banner
    monkeypatch.setattr("builtins.input", _raise_eof)
    hook = tmp_path / ".zakcode" / "banner"
    hook.parent.mkdir()
    hook.write_text("#!/bin/sh\necho deployment line\n")
    if os.name == "posix":
        hook.chmod(0o755)
    cockpit.cockpit_main(workspace=tmp_path)
    chat_runs = [(argv, env) for argv, env in runs if argv[-1] == "chat"]
    assert len(chat_runs) == 1
    assert chat_runs[0][1].get("ZAKCODE_INPUT_FRAME") == "off"
    text = cockpit.console.export_text()
    assert "zakcode cockpit" in text
    assert "press Enter to relaunch" in text


def _raise_eof() -> str:
    raise EOFError


def test_say_box_ledgers_sends_then_exits_on_ctrl_c(
    fake_tmux: _FakeTmux, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_tmux.has_session_rc = 0
    ledger = tmp_path / "ledger.jsonl"
    lines = iter(["  ", "do the thing"])

    def _next_line(prompt: str = "") -> str:
        try:
            return next(lines)
        except StopIteration:
            raise KeyboardInterrupt from None

    monkeypatch.setattr("builtins.input", _next_line)
    with pytest.raises(typer.Exit) as excinfo:
        cockpit.cockpit_say_box(session="agentbox", ledger=ledger, operator="human@box")
    assert excinfo.value.exit_code == 0
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    # The blank line was ignored; only the real one was ledgered and sent.
    assert [r["text"] for r in rows] == ["do the thing"]
    assert rows[0]["via"] == "cockpit-say-box"
    sends = [c for c in fake_tmux.calls if c[1] == "send-keys"]
    assert len(sends) == 2 and sends[0][-1] == "do the thing"


def test_multiline_send_uses_one_bracketed_paste(fake_tmux: _FakeTmux, tmp_path: Path) -> None:
    """A multi-line message must reach the chat pane as ONE prompt, never line-by-line."""
    fake_tmux.has_session_rc = 0
    text = "first line\nsecond line\nthird line"
    cockpit.say(text=text, session="agentbox", ledger=tmp_path / "l.jsonl", operator=None)
    subs = fake_tmux.subcommands()
    assert "load-buffer" in subs
    paste = next(c for c in fake_tmux.calls if c[1] == "paste-buffer")
    assert "-p" in paste and "agentbox:0.0" in paste
    # Exactly one Enter submits the whole block; no per-line send-keys of content.
    sends = [c for c in fake_tmux.calls if c[1] == "send-keys"]
    assert [c[-1] for c in sends] == ["Enter"]
    # The ledger keeps the message whole, as one row.
    row = json.loads((tmp_path / "l.jsonl").read_text(encoding="utf-8").strip())
    assert row["text"] == text


def test_say_box_multiline_reports_line_count(
    fake_tmux: _FakeTmux, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_tmux.has_session_rc = 0
    lines = iter(["alpha\nbeta"])

    def _next_line(prompt: str = "") -> str:
        try:
            return next(lines)
        except StopIteration:
            raise KeyboardInterrupt from None

    monkeypatch.setattr("builtins.input", _next_line)
    with pytest.raises(typer.Exit):
        cockpit.cockpit_say_box(session="x", ledger=tmp_path / "l.jsonl", operator="h@b")
    assert "2 lines" in cockpit.console.export_text()
    subs = fake_tmux.subcommands()
    assert "load-buffer" in subs and "paste-buffer" in subs
