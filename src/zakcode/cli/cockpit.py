"""``zakcode cockpit`` — a supervised two-pane tmux workstation for a long-running agent.

Top pane: a session header (generic lines from zakcode, deployment lines from an
optional ``.zakcode/banner`` hook in the workspace), then ``zakcode chat``, then an
Enter-to-relaunch loop so a finished or crashed chat never leaves a dead pane.

Bottom pane: the *say box* — a persistent input that appends every message to a
JSONL ledger (operator provenance) and delivers it through the workspace **say
inbox** (``<workspace>/.say`` — ``zakcode.session.say_inbox``): the SAME single-slot
contract the server's ``POST /say`` writes and the serve driver consumes. Every chat
always listens to its workspace's say inbox (messages arrive exactly like typed
lines, slash commands included), and the pane runs with ``ZAKCODE_INPUT_FRAME=off``
so there is exactly one place to type. tmux is only the window manager here — it is never the
message transport.

Everything here is session-scoped tmux configuration — the operator's global
``.tmux.conf`` is never touched.
"""

from __future__ import annotations

import getpass
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from zakcode import __version__
from zakcode.build_info import version_line
from zakcode.cli._layout import kv_table, notice_error, notice_info, panel
from zakcode.cli._theme import ZAK_THEME
from zakcode.session.say_inbox import say_path, write_say

console = Console(theme=ZAK_THEME, highlight=False)

#: Say-box pane height in rows: one input line plus the last-send status line.
_SAY_BOX_HEIGHT = 5
#: Scrollback for the chat pane — a long-running agent's day fits comfortably.
_HISTORY_LIMIT = 50000
#: Detached-create size; tmux resizes to the client on attach.
_CREATE_SIZE = ("220", "50")
#: Module-level singleton so a call never appears in an argument default (B008).
_DOT = Path(".")


def _tmux_bin() -> str:
    """Path to tmux, or a clean refusal — the cockpit is a tmux feature by design."""
    exe = shutil.which("tmux")
    if exe is None:
        notice_error(console, "tmux not found", "the cockpit needs tmux (Linux/macOS/WSL)")
        raise typer.Exit(code=1)
    return exe


def _tmux(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run([_tmux_bin(), *args], check=check)


def _has_session(name: str) -> bool:
    return (
        subprocess.run(
            [_tmux_bin(), "has-session", "-t", name],
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


def _self_invocation() -> list[str]:
    """How a tmux pane re-invokes this very install of zakcode."""
    exe = shutil.which("zakcode")
    if exe is not None:
        return [exe]
    return [sys.executable, "-m", "zakcode"]


def _default_ledger() -> Path:
    return Path.home() / ".zakcode" / "say-ledger.jsonl"


def _append_ledger(ledger: Path, text: str, *, via: str, operator: str | None) -> None:
    """One JSONL row per human line: who said what, when, through which door."""
    ledger.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S"),
        "via": via,
        "operator": operator or f"{getpass.getuser()}@{socket.gethostname()}",
        "text": text,
    }
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def _restore_sigint() -> None:
    """preexec for the chat child: undo the parent's SIG_IGN so Ctrl-C reaches chat.

    Python only installs its KeyboardInterrupt handler when SIGINT is NOT inherited
    as ignored, so without this the child could never be interrupted at all.
    """
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def _banner_hook(workspace: Path) -> Path | None:
    """The workspace's deployment banner: ``.zakcode/banner``, if executable."""
    hook = workspace / ".zakcode" / "banner"
    if hook.is_file() and os.access(hook, os.X_OK):
        return hook
    return None


def _print_cockpit_banner(workspace: Path) -> None:
    """The generic header every cockpit gets; the hook adds deployment lines after."""
    rows = [
        ("Workspace", str(workspace)),
        ("Build", version_line(__version__)),
    ]
    try:
        from zakcode.cli import _age_str
        from zakcode.session.store import SessionStore

        recent = SessionStore().list_recent()
        if recent:
            sid, mtime = recent[0]
            age = _age_str(time.time() - mtime)
            rows.append(("Last session", f"{sid}  ({age}) — /resume {sid[:8]}"))
    except Exception:  # noqa: BLE001 — the banner must never block the chat
        pass
    rows.append(("Input", "type in the box below · wheel scrolls (q snaps back)"))
    console.print(panel(console, "zakcode cockpit", kv_table(rows), border_style="banner.border"))


def cockpit(
    session: Annotated[str, typer.Option("--session", "-s", help="tmux session name.")] = "zakcode",
    workspace: Annotated[
        Path, typer.Option("--workspace", "-w", help="Directory the agent works in.")
    ] = _DOT,
    ledger: Annotated[
        Path | None,
        typer.Option("--ledger", help="Say-box ledger JSONL (default ~/.zakcode/say-ledger.jsonl)"),
    ] = None,
    attach: Annotated[
        bool, typer.Option("--attach/--no-attach", help="Attach after creating.")
    ] = True,
) -> None:
    """Open (or re-join) the two-pane cockpit: chat on top, the say box below."""
    tmux = _tmux_bin()
    workspace = workspace.resolve()
    if not _has_session(session):
        ledger_path = ledger if ledger is not None else _default_ledger()
        zakcode = _self_invocation()
        say_cmd = shlex.join(
            [
                *zakcode,
                "cockpit-say-box",
                "--workspace",
                str(workspace),
                "--ledger",
                str(ledger_path),
            ]
        )
        main_cmd = shlex.join([*zakcode, "cockpit-main", "--workspace", str(workspace)])
        # The say box is created FIRST so the session-scoped history-limit is already
        # set when the chat pane (split -b, placed above → index 0) is created — tmux
        # applies history-limit at pane creation, never retroactively.
        _tmux(
            "new-session",
            "-d",
            "-s",
            session,
            "-c",
            str(workspace),
            "-x",
            _CREATE_SIZE[0],
            "-y",
            _CREATE_SIZE[1],
            say_cmd,
        )
        _tmux("set-option", "-t", session, "history-limit", str(_HISTORY_LIMIT))
        _tmux("set-option", "-t", session, "mouse", "on")
        _tmux("set-option", "-w", "-t", f"{session}:0", "pane-border-status", "top")
        _tmux("set-option", "-w", "-t", f"{session}:0", "pane-border-format", " #{pane_title} ")
        _tmux("split-window", "-b", "-v", "-t", f"{session}:0.0", "-c", str(workspace), main_cmd)
        _tmux("resize-pane", "-t", f"{session}:0.1", "-y", str(_SAY_BOX_HEIGHT))
        screen_title = (
            f"{workspace.name.upper()} — screen · type in the box below"
            " · wheel scrolls (q snaps back)"
        )
        _tmux("select-pane", "-t", f"{session}:0.0", "-T", screen_title)
        _tmux("select-pane", "-t", f"{session}:0.1", "-T", "YOUR MESSAGE — click here, type, Enter")
        _tmux("select-pane", "-t", f"{session}:0.0")
        notice_info(console, f"cockpit session '{session}' created")
    if attach and sys.stdin.isatty():
        subprocess.run([tmux, "attach-session", "-t", session], check=False)
    elif not attach:
        notice_info(
            console,
            f"cockpit session '{session}' running — attach with: tmux attach -t {session}",
        )


def cockpit_main(
    workspace: Annotated[
        Path, typer.Option("--workspace", help="Directory the agent works in.")
    ] = _DOT,
) -> None:
    """(internal) Top-pane loop: banner, chat, Enter-to-relaunch."""
    workspace = workspace.resolve()
    # The say box below is the single input: hide chat's own input frame. Chat
    # always listens to the workspace say inbox (the shared POST /say contract),
    # so the box's messages arrive as real input — no keystroke injection.
    env = {**os.environ, "ZAKCODE_INPUT_FRAME": "off"}
    while True:
        console.clear()
        _print_cockpit_banner(workspace)
        hook = _banner_hook(workspace)
        if hook is not None:
            subprocess.run([str(hook)], cwd=workspace, check=False)
        # Ctrl-C in this pane must reach ONLY chat (which handles it: interrupt a
        # running reply, double-press to exit). Without this, the same SIGINT also
        # raises KeyboardInterrupt here, the relaunch loop dies, the pane closes,
        # and the say box below becomes the sole pane — its own send-keys target
        # (observed live on zc-03, 2026-08-25).
        prev = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            subprocess.run(
                [*_self_invocation(), "chat"],
                cwd=workspace,
                env=env,
                check=False,
                preexec_fn=_restore_sigint if os.name == "posix" else None,
            )
        finally:
            signal.signal(signal.SIGINT, prev)
        console.print(
            "── chat exited · press Enter to relaunch · Ctrl-C to close this pane ──",
            style="notice.dim",
        )
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            break


def cockpit_say_box(
    workspace: Annotated[
        Path, typer.Option("--workspace", help="Workspace whose say inbox to feed.")
    ] = _DOT,
    ledger: Annotated[Path | None, typer.Option("--ledger", help="Ledger JSONL path.")] = None,
    operator: Annotated[
        str | None, typer.Option("--operator", help="Provenance label (default user@host).")
    ] = None,
) -> None:
    """(internal) Bottom-pane loop: read a message, ledger it, drop it in the say inbox.

    Bracketed paste is enabled on the box's own readline, so a multi-line paste
    arrives as ONE editable block returned by a single ``input()`` — and the inbox
    delivers it as ONE message. Single-slot semantics: while the agent is mid-turn
    with a message already waiting, a new one is refused with a clear notice
    (identical to POST /say's 429) instead of silently stacking.
    """
    from zakcode.cli import _prepare_interactive_terminal

    _prepare_interactive_terminal()
    inbox = say_path(workspace.resolve())
    ledger_path = ledger if ledger is not None else _default_ledger()
    last = ""
    while True:
        console.clear()
        if last:
            console.print(last, style="notice.dim")
        try:
            line = input("▸ ")
        except EOFError:
            time.sleep(1)
            continue
        except KeyboardInterrupt:
            raise typer.Exit(code=0) from None
        if not line.strip():
            continue
        stamp = time.strftime("%H:%M")
        if not write_say(inbox, line):
            last = f"⏳ agent is busy — previous message still pending; try again shortly {stamp}"
            continue
        _append_ledger(ledger_path, line, via="cockpit-say-box", operator=operator)
        lines = line.count("\n") + 1
        last = f"✓ sent ({lines} lines) {stamp}" if lines > 1 else f"✓ sent {stamp}"


def say(
    text: Annotated[str, typer.Argument(help="The message to deliver to the agent.")],
    workspace: Annotated[
        Path, typer.Option("--workspace", "-w", help="Workspace whose say inbox to write.")
    ] = _DOT,
    url: Annotated[
        str | None,
        typer.Option(
            "--url", help="POST to a zakcode serve daemon's /say instead of the local inbox."
        ),
    ] = None,
    ledger: Annotated[Path | None, typer.Option("--ledger", help="Ledger JSONL path.")] = None,
    operator: Annotated[
        str | None, typer.Option("--operator", help="Provenance label (default user@host).")
    ] = None,
) -> None:
    """Send one ledgered message to a running agent through the say inbox.

    ONE contract everywhere: locally this writes ``<workspace>/.say``; with ``--url``
    it POSTs the daemon's ``/say`` (bearer token from ``ZAKCODE_AUTH_TOKEN`` if set),
    which writes the same file on the daemon's side. Whatever consumes the inbox —
    an interactive chat in say-inbox mode (the cockpit's pane), or the serve
    driver — receives it as its next message. Single slot: a message already
    pending refuses this one; send again when the agent has picked it up.
    """
    if url:
        import httpx

        token = os.environ.get("ZAKCODE_AUTH_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else None
        try:
            resp = httpx.post(
                f"{url.rstrip('/')}/say", json={"text": text}, headers=headers, timeout=10.0
            )
        except httpx.HTTPError as exc:
            notice_error(console, "daemon unreachable", str(exc))
            raise typer.Exit(code=1) from None
        if resp.status_code == 429:
            notice_error(
                console, "a message is already pending", "wait for the agent to pick it up"
            )
            raise typer.Exit(code=1)
        if resp.status_code != 200:
            notice_error(console, f"daemon refused ({resp.status_code})", resp.text[:200])
            raise typer.Exit(code=1)
    else:
        if not write_say(say_path(workspace.resolve()), text):
            notice_error(
                console, "a message is already pending", "wait for the agent to pick it up"
            )
            raise typer.Exit(code=1)
    ledger_path = ledger if ledger is not None else _default_ledger()
    _append_ledger(ledger_path, text, via="zakcode-say", operator=operator)
    notice_info(console, "sent")


def register_cockpit_commands(app: typer.Typer) -> None:
    app.command()(cockpit)
    app.command()(say)
    app.command(name="cockpit-main", hidden=True)(cockpit_main)
    app.command(name="cockpit-say-box", hidden=True)(cockpit_say_box)
