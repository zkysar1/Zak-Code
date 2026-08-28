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
    monkeypatch.delenv("TMUX", raising=False)
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


def test_cockpit_sets_focus_follows_color_borders(fake_tmux: _FakeTmux, tmp_path: Path) -> None:
    """The focused pane's border glows orange, unfocused recede to gray — set by the
    cockpit itself, never left to a host tmux.conf (2026-08-25: zc-03 had a hand
    conf, serene did not, and the two boxes looked different)."""
    cockpit.cockpit(session="agentbox", workspace=tmp_path, ledger=None, attach=True)
    active = next(c for c in fake_tmux.calls if "pane-active-border-style" in c)
    assert active[-1] == "fg=colour214"
    inactive = next(
        c for c in fake_tmux.calls if "pane-border-style" in c and "active" not in " ".join(c)
    )
    assert inactive[-1] == "fg=colour240"


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
    chat_runs = [(argv, env) for argv, env in runs if argv[-1] == "cli"]
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
    chat = next(r for r in runs if r["argv"][-1] == "cli")  # type: ignore[index]
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


def test_interrupt_command_writes_signal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cockpit, "console", _rec_console())
    cockpit.interrupt(workspace=tmp_path)
    assert (tmp_path / ".interrupt").exists()
    assert "stop signal sent" in cockpit.console.export_text()


def test_say_box_prompt_falls_back_to_input_without_tty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("builtins.input", lambda prompt="": "typed line")
    kind, line = cockpit._say_box_prompt(tmp_path / ".say", tmp_path / ".interrupt")
    assert (kind, line) == ("line", "typed line")


# ── one interface: chat elevates itself into the cockpit ──────────────────────────
# 2026-08-25 operator directive: "when I start zakcode, I expect zakcode to start
# the cockpit if it needs it … one canonical input". These pin the elevation rules
# and the derived per-workspace session identity.


def test_cockpit_session_name_is_stable_sanitized_and_per_workspace(tmp_path: Path) -> None:
    a = tmp_path / "serene.mind"
    b = tmp_path / "other" / "serene.mind"
    name_a = cockpit._cockpit_session_name(a)
    assert name_a == cockpit._cockpit_session_name(a)  # stable
    assert name_a != cockpit._cockpit_session_name(b)  # same basename, different workspace
    assert "." not in name_a and " " not in name_a  # tmux-safe
    assert name_a.startswith("zakcode-serene-mind-")


def test_launch_cockpit_derives_session_name(fake_tmux: _FakeTmux, tmp_path: Path) -> None:
    cockpit.launch_cockpit(tmp_path, attach=False)
    expected = cockpit._cockpit_session_name(tmp_path.resolve())
    new_session = next(c for c in fake_tmux.calls if c[1] == "new-session")
    assert expected in new_session


def test_launch_cockpit_inside_tmux_switches_client(
    fake_tmux: _FakeTmux, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_tmux.has_session_rc = 0
    monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,123,0")
    cockpit.launch_cockpit(tmp_path, session="x", attach=True)
    assert fake_tmux.subcommands() == ["has-session", "switch-client"]


def test_cockpit_main_marks_pane_env_for_no_recursion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chat child must carry ZAKCODE_COCKPIT_PANE=1 (stops self-elevation
    recursing) and ZAKCODE_INPUT_FRAME=off (the say box is the one input)."""
    envs: list[dict] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001, ANN003, ANN202
        if "cli" in cmd:
            envs.append(kwargs.get("env") or {})
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(cockpit.subprocess, "run", fake_run)
    monkeypatch.setattr(cockpit, "console", _rec_console())
    monkeypatch.setattr(cockpit, "_print_cockpit_banner", lambda ws: None)
    # One chat round, then the relaunch prompt ends the loop.
    monkeypatch.setattr("builtins.input", lambda *a: (_ for _ in ()).throw(EOFError()))
    cockpit.cockpit_main(workspace=tmp_path)
    assert envs and envs[0]["ZAKCODE_COCKPIT_PANE"] == "1"
    assert envs[0]["ZAKCODE_INPUT_FRAME"] == "off"


def _tty_env(monkeypatch: pytest.MonkeyPatch, *, tmux: bool = True) -> None:
    import zakcode.cli as cli_mod

    monkeypatch.delenv("ZAKCODE_COCKPIT_PANE", raising=False)
    monkeypatch.setattr(cli_mod.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(cli_mod.sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(cli_mod.os, "name", "posix")
    monkeypatch.setattr(
        cli_mod.shutil, "which", lambda name: "/usr/bin/tmux" if (tmux and name == "tmux") else None
    )


def test_elevation_yes_on_plain_interactive_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    from zakcode.cli import _cockpit_elevation

    _tty_env(monkeypatch)
    kwargs = dict(
        prompt=None,
        server=None,
        session=None,
        model=None,
        no_rules=False,
        skill_dir=None,
        extra_root=None,
        trace=False,
    )
    assert _cockpit_elevation(**kwargs) == (True, None)
    # Expert flags run the inline one-off engine instead.
    assert _cockpit_elevation(**{**kwargs, "session": "abc"}) == (False, "expert-flags")
    assert _cockpit_elevation(**{**kwargs, "prompt": "do it"}) == (False, "non-interactive")
    # Inside a cockpit pane the marker stops recursion.
    monkeypatch.setenv("ZAKCODE_COCKPIT_PANE", "1")
    assert _cockpit_elevation(**kwargs) == (False, "inside-pane")
    monkeypatch.delenv("ZAKCODE_COCKPIT_PANE")
    # tmux missing is the one obstacle worth hinting about.
    _tty_env(monkeypatch, tmux=False)
    assert _cockpit_elevation(**kwargs) == (False, "no-tmux")


def test_chat_elevates_into_cockpit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A plain interactive `zakcode chat` becomes the cockpit — no REPL banner,
    no second way to chat; launch_cockpit gets the resolved workspace."""
    from typer.testing import CliRunner

    import zakcode.cli as cli_mod
    from zakcode.cli import app

    monkeypatch.setenv("ZAKCODE_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setattr(cli_mod, "_cockpit_elevation", lambda **kw: (True, None))
    launched: list[Path] = []
    monkeypatch.setattr(cockpit, "launch_cockpit", lambda ws, **kw: launched.append(ws))
    result = CliRunner().invoke(app, ["cli"])
    assert result.exit_code == 0
    assert launched == [tmp_path]
    assert "tip:" not in result.output


class _NoTurnAgent:
    """Just enough Agent for a chat that only ever sees /exit (no turns run)."""

    def __init__(self, **overrides: object) -> None:
        from zakcode.config import load_settings
        from zakcode.hooks import HookManager
        from zakcode.permissions import PermissionPolicy
        from zakcode.session.store import Session

        self.overrides = overrides
        self.settings = load_settings()
        self.session = Session(cwd=".", model=self.settings.default_model)
        self.permission_policy = PermissionPolicy(self.settings.permission_mode)
        self.hook_manager = HookManager()


def test_chat_hints_when_only_tmux_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    import zakcode
    import zakcode.cli as cli_mod
    from zakcode.cli import app

    monkeypatch.setattr(zakcode, "Agent", _NoTurnAgent)
    monkeypatch.setattr(cli_mod, "_cockpit_elevation", lambda **kw: (False, "no-tmux"))
    result = CliRunner().invoke(app, ["cli"], input="/exit\n")
    assert result.exit_code == 0
    assert "install tmux" in result.output


def test_bare_zakcode_starts_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    """`zakcode` with no subcommand IS the chat — no launch ceremony."""
    from typer.testing import CliRunner

    import zakcode
    from zakcode.cli import app

    monkeypatch.setattr(zakcode, "Agent", _NoTurnAgent)
    result = CliRunner().invoke(app, [], input="/exit\n")
    assert result.exit_code == 0
    assert "goodbye" in result.output


def test_root_dispatch_covers_every_chat_option() -> None:
    """Bare `zakcode` dispatches as a DIRECT call `chat(**defaults)` — any chat option
    missing from _root's defaults dict binds to its raw typer.Option sentinel, which is
    TRUTHY. Measured 2026-08-28: --dangerously-skip-permissions was added to chat but not
    to the dict, so every bare launch silently exported
    ZAKCODE_PERMISSION_MODE=bypassPermissions (caught 30 files later as env pollution)."""
    import inspect

    from zakcode.cli import _root, chat

    src = inspect.getsource(_root)
    for name in inspect.signature(chat).parameters:
        assert f'"{name}"' in src, (
            f"_root's defaults dict is missing chat option {name!r} — a direct "
            "chat(**defaults) call would bind it to its truthy typer.Option sentinel"
        )


# ── first touch: focus the box; never eat the operator's boot-time message ────────
# 2026-08-25 serene report, reproduced live on zc-03: focus landed on the screen
# pane ("couldn't type at first") and the first message typed while the agent
# booted was discarded by the stale-say guard ("Enter did nothing").


def test_launch_cockpit_focuses_the_message_box(fake_tmux: _FakeTmux, tmp_path: Path) -> None:
    cockpit.launch_cockpit(tmp_path, session="x", attach=False)
    selects = [c for c in fake_tmux.calls if c[1] == "select-pane" and "-T" not in c]
    assert selects and selects[-1][-1].endswith(":0.1")  # the box, not the screen


def test_launch_cockpit_clears_stale_say_at_creation(fake_tmux: _FakeTmux, tmp_path: Path) -> None:
    """A .say that predates the cockpit (dead session, repo-shipped) dies at the
    cockpit boundary — before any pane the operator could have typed into."""
    from zakcode.session.say_inbox import say_path, write_say

    assert write_say(say_path(tmp_path), "repo-shipped or leftover")
    cockpit.launch_cockpit(tmp_path, session="x", attach=False)
    assert not say_path(tmp_path).exists()


def test_launch_cockpit_existing_session_keeps_pending_say(
    fake_tmux: _FakeTmux, tmp_path: Path
) -> None:
    """Re-joining a LIVE cockpit must not clear the inbox — a message can be
    legitimately in flight between the box and the chat."""
    from zakcode.session.say_inbox import say_path, write_say

    fake_tmux.has_session_rc = 0
    assert write_say(say_path(tmp_path), "in flight")
    cockpit.launch_cockpit(tmp_path, session="x", attach=False)
    assert say_path(tmp_path).exists()


def test_chat_delivers_boot_time_say(monkeypatch, tmp_path: Path) -> None:
    """The box is live from cockpit creation and chat never second-guesses the
    inbox: a say already queued when chat starts is the operator's boot-time
    message — delivered as the first input, not discarded."""
    from typer.testing import CliRunner

    import zakcode
    from zakcode.cli import app
    from zakcode.session.say_inbox import say_path, write_say

    class _RecordingAgent(_NoTurnAgent):
        turns: list[str] = []

        def astream_turn(self, text: str):  # noqa: ANN202
            _RecordingAgent.turns.append(text)
            from zakcode.events import AgentDone, AgentTextDelta
            from zakcode.usage import Usage

            async def _gen():  # noqa: ANN202
                yield AgentTextDelta(text="ok\n")
                yield AgentDone(
                    stop_reason="completed",
                    iterations=1,
                    usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                )

            return _gen()

    _RecordingAgent.turns = []
    monkeypatch.setattr(zakcode, "Agent", _RecordingAgent)
    monkeypatch.setenv("ZAKCODE_WORKSPACE_ROOT", str(tmp_path))
    assert write_say(say_path(tmp_path), "typed while the agent was booting")
    result = CliRunner().invoke(app, ["cli"], input="/exit\n")
    assert result.exit_code == 0
    assert "discarded" not in result.output
    assert _RecordingAgent.turns == ["typed while the agent was booting"]
    assert "(say) typed while the agent was booting" in result.output


# ── one door: inside a cockpit pane the pane keyboard is not read at all ──────────
# Operator ruling (2026-08-25): the say-inbox FILE is the input contract — it is
# what lets outside programs inject into a session — and it must never coexist
# with a parallel keystroke door inside the cockpit.


def test_mux_without_keyboard_never_touches_stdin(monkeypatch, tmp_path: Path) -> None:
    import zakcode.cli as cli_mod
    from zakcode.cli import _InputMux
    from zakcode.session.say_inbox import interrupt_path, say_path, write_say

    touched: list[bool] = []
    monkeypatch.setattr(cli_mod, "_read_stdin_line", lambda: touched.append(True) or "")
    mux = _InputMux(say_path(tmp_path), interrupt_path(tmp_path), keyboard=False)
    assert write_say(say_path(tmp_path), "via the contract")
    assert mux.next_input(idle=True) == ("say", "via the contract")
    assert not touched


def test_chat_in_cockpit_pane_ignores_pane_keyboard(monkeypatch, tmp_path: Path) -> None:
    """With the pane marker set, chat consumes ONLY the say inbox: stdin is never
    read, and the session is driven (and ended) entirely through the contract."""
    from typer.testing import CliRunner

    import zakcode
    import zakcode.cli as cli_mod
    from zakcode.cli import app
    from zakcode.session.say_inbox import say_path, write_say

    touched: list[bool] = []
    monkeypatch.setattr(cli_mod, "_read_stdin_line", lambda: touched.append(True) or "")
    monkeypatch.setattr(zakcode, "Agent", _NoTurnAgent)
    monkeypatch.setenv("ZAKCODE_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("ZAKCODE_COCKPIT_PANE", "1")
    assert write_say(say_path(tmp_path), "/exit")
    result = CliRunner().invoke(app, ["cli"], input="this text must never reach anything\n")
    assert result.exit_code == 0
    assert "goodbye" in result.output
    assert not touched


def test_chat_exit_tears_down_the_whole_cockpit(fake_tmux: _FakeTmux, tmp_path: Path) -> None:
    """Leaving the chat (double ctrl-c / EOF / crash) must close the WHOLE cockpit — the
    chat pane's command chains a kill-session, so a dead chat never strands a headless
    say box (operator report 2026-08-25)."""
    cockpit.cockpit(session="agentbox", workspace=tmp_path, ledger=None, attach=True)
    split = next(c for c in fake_tmux.calls if c[1] == "split-window")
    pane_cmd = split[-1]
    assert "cockpit-main" in pane_cmd
    assert "kill-session" in pane_cmd and "agentbox" in pane_cmd
