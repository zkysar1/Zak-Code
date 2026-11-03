"""The cockpit in a real tmux, read off its own screen (ADR-0356).

Every other cockpit test fakes tmux: it records the commands the cockpit would send and never runs
them. Nothing showed that a real tmux accepts them, that the panes start with what the cockpit gives
them, or that a line typed in the box reaches the chat and its reply comes back where a person
looks. ADR-0346 and ADR-0352 both recorded "a real tmux cockpit" as not verified, because the
machine that built them has no tmux, and the cockpit is where a person typed a line and saw it only
on the web page.

This drives the real thing. ``zakcode cockpit --no-attach`` builds the cockpit in a tmux server of
the test's own, ``tmux send-keys`` types into the box, ``tmux capture-pane`` reads what the screen
shows, and a stand-in model on loopback answers every call with a reply nothing else prints. Each
caller holds its own key, so the stand-in can say which process asked. It needs tmux and a POSIX
terminal, so it is skipped unless ``ZAKCODE_COCKPIT_E2E=1``; the ``cockpit-e2e`` CI job installs
tmux and sets that.

Every wait is a deadline, never a sleep, and a deadline that passes prints both panes as they
stood. A marker the test waits for is never one it typed, and never text a banner or a help line
also shows.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from zakcode.session.say_inbox import served_url

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("ZAKCODE_COCKPIT_E2E") != "1",
        reason="set ZAKCODE_COCKPIT_E2E=1 to run (needs tmux and a POSIX terminal)",
    ),
    pytest.mark.skipif(sys.platform == "win32", reason="the cockpit is a tmux feature"),
    pytest.mark.timeout(600),
]

#: The stand-in answers as a model the bundled price map knows, so the chat sizes its context and
#: prices its turns as it would for a real one, while every call still lands on loopback.
MODEL = "openai/gpt-4o"
#: The chat's screen is the top pane; the box a person types in is the one below it.
SCREEN, BOX = "0.0", "0.1"
#: How long one step may take. Generous because a CI runner can be slow to start a process; a
#: deadline that holds costs nothing, and one that passes is reported with both panes.
DEADLINE = 90.0
_USAGE = {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16}


# ── the stand-in model ──────────────────────────────────────────────────────────────────────────


@dataclass
class StandIn:
    """A loopback OpenAI-compatible endpoint that answers every completion with ``reply``."""

    url: str
    reply: str
    #: Every completion asked for: the key it carried and the request body.
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def asked(self, key: str, text: str) -> int:
        """How many completions the caller holding ``key`` asked with ``text`` in a user message."""
        return sum(1 for held, body in list(self.calls) if held == key and text in _user_text(body))


def _user_text(body: dict[str, Any]) -> str:
    parts: list[str] = []
    for message in body.get("messages", []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return "\n".join(parts)


def _chunk(model: str, delta: dict[str, Any], finish: str | None) -> dict[str, Any]:
    return {
        "id": "stand-in",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


@pytest.fixture
def stand_in() -> Iterator[StandIn]:
    reply = f"stand-in reply {uuid.uuid4().hex[:10]}"
    calls: list[tuple[str, dict[str, Any]]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - a model listing, for any reachability probe
            listing = {
                "object": "list",
                "data": [{"id": MODEL.split("/", 1)[1], "object": "model"}],
            }
            self._send("application/json", json.dumps(listing).encode())

        def do_POST(self) -> None:  # noqa: N802 - the http.server hook name
            length = int(self.headers.get("content-length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            key = self.headers.get("authorization", "").removeprefix("Bearer ").strip()
            calls.append((key, body))
            model = str(body.get("model", "stand-in"))
            if body.get("stream"):
                usage = {**_chunk(model, {}, None), "choices": [], "usage": _USAGE}
                events = [
                    _chunk(model, {"role": "assistant", "content": reply}, None),
                    _chunk(model, {}, "stop"),
                    usage,
                ]
                stream = b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events)
                self._send("text/event-stream", stream + b"data: [DONE]\n\n")
                return
            whole = {
                "id": "stand-in",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": "stop",
                    }
                ],
                "usage": _USAGE,
            }
            self._send("application/json", json.dumps(whole).encode())

        def _send(self, kind: str, payload: bytes) -> None:
            self.send_response(200)
            self.send_header("content-type", kind)
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield StandIn(f"http://127.0.0.1:{server.server_address[1]}/v1", reply, calls)
    finally:
        server.shutdown()
        server.server_close()


# ── a workspace, an environment, a tmux server of the test's own ──────────────────────────────


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A workspace that runs the framework, with one agent a new chat can start (ADR-0351)."""
    ws = tmp_path / "ws"
    skill = ws / ".claude" / "skills" / "start"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: start\n---\n# /start\n", encoding="utf-8")
    agent = ws / "agents" / "scout"
    agent.mkdir(parents=True)
    (agent / "self.md").write_text("# scout\n", encoding="utf-8")
    return ws


@pytest.fixture
def tmux_dir() -> Iterator[Path]:
    """Where the test's own tmux server keeps its socket, and that server's end.

    Short and outside the test's tmp dir on purpose: a unix socket path is limited to about 100
    bytes, and a pytest tmp path with this file's test names runs past that.
    """
    root = Path(tempfile.mkdtemp(prefix="zct-", dir="/tmp" if os.path.isdir("/tmp") else None))
    try:
        yield root
    finally:
        subprocess.run(
            ["tmux", "kill-server"],
            env={**os.environ, "TMUX_TMPDIR": str(root)},
            capture_output=True,
            check=False,
        )
        shutil.rmtree(root, ignore_errors=True)


def _env(tmp_path: Path, tmux_dir: Path, stand_in: StandIn, key: str) -> dict[str, str]:
    """What a person's shell would hand the cockpit, pointed at the stand-in and nothing else."""
    env = {k: v for k, v in os.environ.items() if k not in ("TMUX", "TMUX_PANE")}
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env.update(
        {
            "TMUX_TMPDIR": str(tmux_dir),
            "HOME": str(home),
            "ZAKCODE_HOME": str(home / ".zakcode"),
            "ZAKCODE_DEFAULT_MODEL": MODEL,
            "ZAKCODE_API_BASE": stand_in.url,
            "OPENAI_API_KEY": key,
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "TERM": "xterm-256color",
            "PYTHONUTF8": "1",
        }
    )
    return env


def _tmux_run(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["tmux", *args],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )


def _tmux(env: dict[str, str], *args: str) -> str:
    proc = _tmux_run(env, *args)
    assert proc.returncode == 0, f"tmux {' '.join(args)} -> rc {proc.returncode}: {proc.stderr}"
    return proc.stdout


@dataclass
class Cockpit:
    """One cockpit in the test's tmux server, read and typed into the way a person would."""

    env: dict[str, str]
    session: str
    #: Each pane's tmux id, by the index it had when the cockpit opened. When a pane closes, tmux
    #: renumbers the others, so an index can come to name the wrong pane; an id never does.
    panes: dict[str, str]

    def screen(self, pane: str = SCREEN) -> str:
        """Everything the pane has shown, its scrollback included, wrapped lines joined."""
        return _tmux(self.env, "capture-pane", "-p", "-J", "-S", "-", "-t", self.panes[pane])

    def wait_for(self, what: str, holds: Callable[[str], bool], pane: str = SCREEN) -> str:
        """Poll ``pane`` until ``holds`` is true of it, or fail showing both panes."""
        end = time.monotonic() + DEADLINE
        while True:
            seen = self.screen(pane)
            if holds(seen):
                return seen
            if time.monotonic() > end:
                pytest.fail(f"{what}: not on the {pane} pane after {DEADLINE:.0f}s\n{self.both()}")
            time.sleep(0.25)

    def type_in_box(self, text: str) -> None:
        """Type ``text`` into the box and press Enter, as a person at the keyboard does."""
        _tmux(self.env, "send-keys", "-t", self.panes[BOX], "-l", text)
        _tmux(self.env, "send-keys", "-t", self.panes[BOX], "Enter")

    def alive(self) -> bool:
        return _tmux_run(self.env, "has-session", "-t", self.session).returncode == 0

    def both(self) -> str:
        """Both panes as they stand, for a failure report; a pane that has closed says so."""
        if not self.alive():
            return "(the cockpit's tmux session is gone)"
        shown = []
        for name, pane in (("screen", SCREEN), ("box", BOX)):
            proc = _tmux_run(
                self.env, "capture-pane", "-p", "-J", "-S", "-", "-t", self.panes[pane]
            )
            body = proc.stdout if proc.returncode == 0 else f"(closed: {proc.stderr.strip()})"
            shown.append(f"--- {name} ({pane}) ---\n{body}")
        return "\n".join(shown)


def _open(env: dict[str, str], workspace: Path, tmp_path: Path) -> Cockpit:
    """Build the workspace's cockpit without attaching, the way provisioning does."""
    session = f"e2e-{uuid.uuid4().hex[:6]}"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "zakcode",
            "cockpit",
            "--no-attach",
            "--workspace",
            str(workspace),
            "--session",
            session,
            "--ledger",
            str(tmp_path / "say-ledger.jsonl"),
        ],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, f"cockpit -> rc {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    listing = _tmux(
        env, "list-panes", "-s", "-t", session, "-F", "#{window_index}.#{pane_index} #{pane_id}"
    )
    panes = dict(line.split() for line in listing.splitlines() if line.strip())
    assert SCREEN in panes and BOX in panes, f"the cockpit opened panes {panes}\n{proc.stdout}"
    return Cockpit(env, session, panes)


def _last_line(screen: str) -> str:
    lines = [line for line in screen.splitlines() if line.strip()]
    return lines[-1] if lines else ""


# ── what a person sees ──────────────────────────────────────────────────────────────────────────


def test_a_new_chat_in_the_cockpit_offers_the_workspace_s_agent(
    tmp_path: Path, workspace: Path, tmux_dir: Path, stand_in: StandIn
) -> None:
    """The chat pane prints the start offer under its welcome box (ADR-0351, item 4)."""
    cockpit = _open(_env(tmp_path, tmux_dir, stand_in, "key-cockpit"), workspace, tmp_path)
    screen = cockpit.wait_for("the start offer", lambda s: "/start scout --mode assistant" in s)
    assert "Start an agent before you ask anything" in screen, cockpit.both()


def test_a_line_typed_in_the_box_is_answered_on_the_screen(
    tmp_path: Path, workspace: Path, tmux_dir: Path, stand_in: StandIn
) -> None:
    """A line typed in the box reaches the model through the chat, and its reply is on screen."""
    cockpit = _open(_env(tmp_path, tmux_dir, stand_in, "key-cockpit"), workspace, tmp_path)
    cockpit.wait_for("the chat starting", lambda s: "/start scout --mode assistant" in s)
    line = f"typed in the box {uuid.uuid4().hex[:6]}"
    cockpit.type_in_box(line)
    screen = cockpit.wait_for("the model's reply", lambda s: stand_in.reply in s)
    assert line in screen, cockpit.both()
    assert stand_in.asked("key-cockpit", line) >= 1, "the model was never asked the typed line"


def test_on_a_served_workspace_the_screen_shows_the_web_app_s_chat(
    tmp_path: Path, workspace: Path, tmux_dir: Path, stand_in: StandIn
) -> None:
    """While a web app serves the workspace, a line typed in the box runs in the web app's chat,
    and the cockpit's screen shows that chat: the line and its reply (ADR-0346, ADR-0352). The
    pane itself asks the model nothing."""
    port = _free_port()
    web_env = _env(tmp_path, tmux_dir, stand_in, "key-web-app")
    log = tmp_path / "webapp.log"
    with log.open("wb") as out:
        web = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "zakcode",
                "webapp",
                "--workspace",
                str(workspace),
                "--port",
                str(port),
            ],
            cwd=workspace,
            env=web_env,
            stdout=out,
            stderr=subprocess.STDOUT,
        )
    try:
        url = f"http://127.0.0.1:{port}"
        _wait_until_served(workspace, url, web, log)
        cockpit = _open(_env(tmp_path, tmux_dir, stand_in, "key-cockpit"), workspace, tmp_path)
        cockpit.wait_for(
            "the served notice", lambda s: f"showing the web app's chat from {url}" in s
        )
        line = f"typed while served {uuid.uuid4().hex[:6]}"
        cockpit.type_in_box(line)
        cockpit.wait_for("the box's receipt", lambda s: "the web app's chat" in s, pane=BOX)
        screen = cockpit.wait_for("the web app's reply", lambda s: stand_in.reply in s)
        assert line in screen, cockpit.both()
        assert stand_in.asked("key-web-app", line) >= 1, "the web app never asked the typed line"
        assert stand_in.asked("key-cockpit", line) == 0, "the cockpit's own chat ran the line too"
    finally:
        web.terminate()
        try:
            web.wait(timeout=20)
        except subprocess.TimeoutExpired:
            web.kill()
            web.wait(timeout=20)


def test_exit_in_the_box_ends_the_chat_and_ctrl_c_closes_the_cockpit(
    tmp_path: Path, workspace: Path, tmux_dir: Path, stand_in: StandIn
) -> None:
    """``/exit`` ends the chat cleanly, the pane offers to relaunch it, and Ctrl-C there closes
    the whole cockpit, so a dead chat never strands the box."""
    cockpit = _open(_env(tmp_path, tmux_dir, stand_in, "key-cockpit"), workspace, tmp_path)
    cockpit.wait_for("the chat starting", lambda s: "/start scout --mode assistant" in s)
    cockpit.type_in_box("/exit")
    screen = cockpit.wait_for("the chat ending", lambda s: "chat exited" in _last_line(s))
    assert "Traceback" not in screen, cockpit.both()
    _tmux(cockpit.env, "send-keys", "-t", cockpit.panes[SCREEN], "C-c")
    end = time.monotonic() + DEADLINE
    while cockpit.alive():
        if time.monotonic() > end:
            pytest.fail(
                f"the cockpit was still open {DEADLINE:.0f}s after Ctrl-C\n{cockpit.both()}"
            )
        time.sleep(0.25)


# ── helpers for the served case ─────────────────────────────────────────────────────────────────


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_until_served(workspace: Path, url: str, web: subprocess.Popen[bytes], log: Path) -> None:
    """Until the web app answers and its served marker names where it listens (ADR-0352)."""
    end = time.monotonic() + DEADLINE
    while True:
        if web.poll() is not None:
            pytest.fail(
                f"the web app exited rc {web.returncode}\n{log.read_text(errors='replace')}"
            )
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as response:
                answered = response.status == 200
        except OSError:
            answered = False
        if answered and served_url(workspace) == url:
            return
        if time.monotonic() > end:
            pytest.fail(
                f"the web app did not serve {url} in time\n{log.read_text(errors='replace')}"
            )
        time.sleep(0.25)
