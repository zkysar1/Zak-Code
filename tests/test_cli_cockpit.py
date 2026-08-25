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
        self.display_stdout = b"%0"
        self.display_rc = 0

    def __call__(self, argv, **kwargs):  # noqa: ANN001, ANN003, ANN204
        self.calls.append([str(a) for a in argv])
        sub = argv[1] if len(argv) > 1 else ""
        if sub == "has-session":
            return subprocess.CompletedProcess(argv, self.has_session_rc)
        if sub == "capture-pane":
            return subprocess.CompletedProcess(argv, 0, stdout=self.capture_stdout, stderr=b"")
        if sub == "display-message":
            return subprocess.CompletedProcess(
                argv, self.display_rc, stdout=self.display_stdout, stderr=b""
            )
        return subprocess.CompletedProcess(argv, 0, stdout=b"", stderr=b"")

    def subcommands(self) -> list[str]:
        return [c[1] for c in self.calls if len(c) > 1]


@pytest.fixture
def fake_tmux(monkeypatch: pytest.MonkeyPatch) -> _FakeTmux:
    fake = _FakeTmux()
    # Deterministic regardless of whether the test process itself runs inside tmux.
    monkeypatch.delenv("TMUX_PANE", raising=False)
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


def test_cockpit_main_shields_relaunch_loop_from_ctrl_c(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SIGINT is ignored in the wrapper while chat runs (and restored after), and the
    chat child gets default disposition back via preexec — so Ctrl-C interrupts chat
    without killing the pane's relaunch loop."""
    sig_calls: list[tuple[object, object]] = []
    real_signal = cockpit.signal

    class _SigStub:
        SIGINT = real_signal.SIGINT
        SIG_IGN = real_signal.SIG_IGN
        SIG_DFL = real_signal.SIG_DFL

        @staticmethod
        def signal(num: object, handler: object) -> object:
            sig_calls.append((num, handler))
            return "prev-handler"

    monkeypatch.setattr(cockpit, "signal", _SigStub)
    runs: list[dict[str, object]] = []

    def _record(argv, **kwargs):  # noqa: ANN001, ANN003, ANN204
        runs.append({"argv": [str(a) for a in argv], "preexec_fn": kwargs.get("preexec_fn")})
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(cockpit.subprocess, "run", _record)
    monkeypatch.setattr(
        cockpit.shutil, "which", lambda name: "/usr/bin/zakcode" if name == "zakcode" else None
    )
    monkeypatch.setattr(cockpit, "console", _rec_console())
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("builtins.input", _raise_eof)
    cockpit.cockpit_main(workspace=tmp_path)
    chat = next(r for r in runs if r["argv"][-1] == "chat")  # type: ignore[index]
    if os.name == "posix":
        assert chat["preexec_fn"] is not None
    # Ignored while chat ran, then restored to whatever was there before.
    assert (_SigStub.SIGINT, _SigStub.SIG_IGN) in sig_calls
    assert (_SigStub.SIGINT, "prev-handler") in sig_calls


# ── say-inbox transport (the converged path: tmux is never the wire) ──────────────


def test_say_writes_inbox_and_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cockpit, "console", _rec_console())
    ledger = tmp_path / "ledger.jsonl"
    cockpit.say(
        text="first line\nsecond line", workspace=tmp_path, url=None, ledger=ledger, operator="a@b"
    )
    inbox = tmp_path / ".say"
    assert inbox.read_text(encoding="utf-8") == "first line\nsecond line\n"
    row = json.loads(ledger.read_text(encoding="utf-8").strip())
    assert row["text"] == "first line\nsecond line"
    assert row["via"] == "zakcode-say"
    assert "sent" in cockpit.console.export_text()


def test_say_refuses_while_pending(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cockpit, "console", _rec_console())
    (tmp_path / ".say").write_text("earlier\n", encoding="utf-8")
    ledger = tmp_path / "ledger.jsonl"
    with pytest.raises(typer.Exit) as excinfo:
        cockpit.say(text="hi", workspace=tmp_path, url=None, ledger=ledger, operator=None)
    assert excinfo.value.exit_code == 1
    assert not ledger.exists()
    assert (tmp_path / ".say").read_text(encoding="utf-8") == "earlier\n"
    assert "already pending" in cockpit.console.export_text()


def test_say_url_posts_to_daemon_with_bearer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx

    monkeypatch.setattr(cockpit, "console", _rec_console())
    monkeypatch.setenv("ZAKCODE_AUTH_TOKEN", "sekret")
    posts: list[tuple[str, dict, dict | None]] = []

    def _fake_post(url, json=None, headers=None, timeout=None):  # noqa: ANN001, ANN204
        posts.append((url, json, headers))
        return httpx.Response(200, json={"queued": True}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", _fake_post)
    ledger = tmp_path / "ledger.jsonl"
    cockpit.say(
        text="over the wire",
        workspace=tmp_path,
        url="http://127.0.0.1:8000/",
        ledger=ledger,
        operator=None,
    )
    assert posts == [
        (
            "http://127.0.0.1:8000/say",
            {"text": "over the wire"},
            {"Authorization": "Bearer sekret"},
        )
    ]
    assert not (tmp_path / ".say").exists()  # remote transport — no local file
    assert json.loads(ledger.read_text(encoding="utf-8").strip())["text"] == "over the wire"


def test_say_url_maps_429_to_pending_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx

    monkeypatch.setattr(cockpit, "console", _rec_console())
    monkeypatch.delenv("ZAKCODE_AUTH_TOKEN", raising=False)

    def _fake_post(url, json=None, headers=None, timeout=None):  # noqa: ANN001, ANN204
        return httpx.Response(429, text="pending", request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", _fake_post)
    ledger = tmp_path / "ledger.jsonl"
    with pytest.raises(typer.Exit) as excinfo:
        cockpit.say(text="hi", workspace=tmp_path, url="http://x:1", ledger=ledger, operator=None)
    assert excinfo.value.exit_code == 1
    assert not ledger.exists()
    assert "already pending" in cockpit.console.export_text()


def _say_box_input(monkeypatch: pytest.MonkeyPatch, lines: list[str]) -> None:
    seq = iter(lines)

    def _next(prompt: str = "") -> str:
        try:
            return next(seq)
        except StopIteration:
            raise KeyboardInterrupt from None

    monkeypatch.setattr("builtins.input", _next)


def test_say_box_delivers_multiline_to_inbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cockpit, "console", _rec_console())
    _say_box_input(monkeypatch, ["  ", "alpha\nbeta"])
    ledger = tmp_path / "ledger.jsonl"
    with pytest.raises(typer.Exit) as excinfo:
        cockpit.cockpit_say_box(workspace=tmp_path, ledger=ledger, operator="h@b")
    assert excinfo.value.exit_code == 0
    assert (tmp_path / ".say").read_text(encoding="utf-8") == "alpha\nbeta\n"
    rows = [json.loads(x) for x in ledger.read_text(encoding="utf-8").splitlines()]
    assert [r["text"] for r in rows] == ["alpha\nbeta"]
    assert rows[0]["via"] == "cockpit-say-box"
    assert "2 lines" in cockpit.console.export_text()


def test_say_box_busy_notice_while_message_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cockpit, "console", _rec_console())
    (tmp_path / ".say").write_text("unconsumed\n", encoding="utf-8")
    _say_box_input(monkeypatch, ["hello"])
    ledger = tmp_path / "ledger.jsonl"
    with pytest.raises(typer.Exit):
        cockpit.cockpit_say_box(workspace=tmp_path, ledger=ledger, operator=None)
    assert not ledger.exists()
    assert (tmp_path / ".say").read_text(encoding="utf-8") == "unconsumed\n"
    assert "busy" in cockpit.console.export_text()
