"""A cockpit pane shows the chat a web app serves on its workspace (ADR-0352).

On a served workspace the say box feeds the web app's chat (ADR-0320), and a pane that only stood
back showed nothing of it: a line sent through the box went to a chat the pane could not see. The
web app now names its address in the served marker, and the pane follows its current chat through
the transcript and the watch stream, as the page does: the saved conversation once, then every
line from any door with its reply. It writes nothing: the server stays the workspace's one reader.

A pane whose permission prompt meets a web app serving its workspace can no longer be answered,
since the box feeds the web app, and its turn held the busy marker the web app waits on. The prompt
denies and the turn stops, so the web app goes on (ADR-0352).

The web apps here are real servers on loopback ports (the view reads a live watch stream), with a
scripted agent. They run in this process, so a pane test writes the marker as another process's.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import re
import signal
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from rich.console import Console
from sse_starlette.sse import AppStatus

import zakcode
import zakcode.cli as cli_mod
from tests.test_cli_chat import CANNED_TEXT, FakeAgent, runner
from tests.test_served_marker import _age, _stopped_after
from tests.test_server_watch import _free_port
from zakcode.cli import _InputMux, _served_url, app, served_mirror
from zakcode.cli._theme import ZAK_THEME
from zakcode.cli.render import StreamRenderer
from zakcode.cli.served_mirror import ServedChatMirror
from zakcode.config import PermissionTier, Settings
from zakcode.events import AgentDone, AgentEvent, AgentTextDelta
from zakcode.messages import Message
from zakcode.permissions import PermissionOutcome, PermissionRequest
from zakcode.server import app as server_app
from zakcode.server.app import create_app
from zakcode.server.client import ServerClient
from zakcode.server.event_bus import EventBusRegistry
from zakcode.session.say_inbox import (
    claim_busy,
    interrupt_path,
    request_interrupt,
    say_path,
    say_pending,
    served_path,
    served_url,
    write_say,
)
from zakcode.session.store import Session, SessionStore
from zakcode.usage import Usage


@pytest.fixture(autouse=True)
def _streams_serve(monkeypatch: pytest.MonkeyPatch) -> None:
    # A web app stopped as Ctrl-C stops it sets sse-starlette's process-wide exit flag, which ends
    # every watch stream; each test starts with it clear, and leaves it so.
    monkeypatch.setattr(AppStatus, "should_exit", False)


class _Echo:
    """The web app's agent, scripted: each line gets a saved reply. A line asking to run a command
    first asks the web app's prompter, as a tool call needing approval does."""

    def __init__(self, session: Session, prompter: Any) -> None:
        self.session = session
        self._prompter = prompter

    async def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]:
        self.session.add_message(Message.user(user_text))
        reply = f"reply to {user_text}"
        if user_text.startswith("run "):
            command = user_text[len("run ") :]
            asked = PermissionRequest(
                tool_name="bash",
                tier=PermissionTier.DANGER_FULL_ACCESS,
                arguments={"command": command},
                reason="it deletes files",
            )
            outcome = await self._prompter.confirm(asked)
            reply = f"{outcome.value}: {command}"
        self.session.add_message(Message.assistant_text(reply))
        yield AgentTextDelta(text=reply)
        yield AgentDone(stop_reason="completed", iterations=1, usage=Usage())


def _web_app(workspace: Path, **kwargs: Any) -> FastAPI:
    token = kwargs.pop("token", None)
    settings = Settings(
        default_model="scripted/test",
        context_window=8192,
        workspace_root=workspace,
        auth_token=token,
    )
    return create_app(
        settings=settings,
        store=SessionStore(base_dir=workspace / "sessions"),
        agent_factory=lambda session, model, prompter: _Echo(session, prompter),
        **kwargs,
    )


class _Web:
    """A web app serving ``workspace`` on a loopback port, as ``zakcode webapp`` serves it: it marks
    the workspace served, naming where it listens, for as long as it serves."""

    def __init__(
        self,
        workspace: Path,
        *,
        port: int = 0,
        token: str | None = None,
        new_session: bool = True,
    ) -> None:
        self.port = port or _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self._headers = {"Authorization": f"Bearer {token}"} if token else None
        fastapi_app = _web_app(workspace, token=token, new_session=new_session, served_url=self.url)
        config = uvicorn.Config(fastapi_app, host="127.0.0.1", port=self.port, log_level="error")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def __enter__(self) -> _Web:
        self._thread.start()
        _until(lambda: self._server.started, what="the web app to listen")
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def stop(self) -> None:
        """Stop as Ctrl-C stops it: sse-starlette ends the watch streams, a pane's among them, so
        the server can close its connections and run its shutdown."""
        if self._thread.is_alive():
            self._server.handle_exit(signal.SIGINT, None)
            self._thread.join(timeout=20)
            assert not self._thread.is_alive(), "the web app did not stop"
        AppStatus.should_exit = False  # the next web app's streams serve

    def _http(self) -> httpx.Client:
        return httpx.Client(base_url=self.url, timeout=10.0, headers=self._headers)

    def transcript(self) -> dict[str, Any]:
        with self._http() as http:
            resp = http.get("/sessions/current/transcript")
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return data

    def page_say(self, text: str) -> None:
        """A line sent from the page: ``POST /say`` naming the chat it has open."""
        with self._http() as http:
            current = http.get("/sessions/current/transcript").json()["session_id"]
            resp = http.post("/say", json={"text": text, "session_id": current})
        assert resp.status_code == 200, resp.text

    def open_chat(self, session_id: str | None) -> None:
        """The page opening another chat. Refused (409) while a turn runs or a say waits."""
        deadline = time.monotonic() + 20
        with self._http() as http:
            while True:
                resp = http.put("/sessions/current", json={"session_id": session_id})
                if resp.status_code != 409 or time.monotonic() > deadline:
                    break
                time.sleep(0.1)
        assert resp.status_code == 200, resp.text

    def answered(self, text: str) -> bool:
        transcript = self.transcript()
        replies = [m["text"] for m in transcript["messages"] if m["role"] == "assistant"]
        return not transcript["running"] and f"reply to {text}" in replies


class _Screen(io.StringIO):
    """The pane's terminal, read while the view's thread writes to it."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()

    def write(self, s: str) -> int:
        with self._lock:
            return super().write(s)

    def text(self) -> str:
        with self._lock:
            return " ".join(self.getvalue().split())  # lines a notice wrapped, joined


def _view(url: str, *, token: str | None = None) -> tuple[_Screen, ServedChatMirror]:
    screen = _Screen()
    console = Console(
        file=screen, width=120, force_terminal=False, theme=ZAK_THEME, highlight=False
    )
    return screen, ServedChatMirror(console, url, token=token)


def _until(check: Callable[[], bool], *, what: str, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while not check():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.05)


def _say(workspace: Path, text: str) -> None:
    """A line from the say box (or ``zakcode say``): the say file."""
    _until(lambda: write_say(say_path(workspace), text), what="the say slot to be free")


@contextlib.contextmanager
def _showing(mirror: ServedChatMirror) -> Iterator[ServedChatMirror]:
    mirror.resume()
    try:
        yield mirror
    finally:
        mirror.pause()


def _foreign_marker(workspace: Path, url: str | None) -> Path:
    """A fresh served marker another process wrote: a web app serving ``workspace``."""
    marker: dict[str, Any] = {
        "pid": os.getpid() + 100_000,
        "session": "served",
        "since": time.time(),
    }
    if url is not None:
        marker["url"] = url
    path = served_path(workspace)
    path.write_text(json.dumps(marker) + "\n", encoding="utf-8")
    return path


# ── the served marker names where the web app listens ─────────────────────────


def test_a_pane_reads_the_web_apps_address_from_a_fresh_marker_another_process_wrote(
    tmp_path: Path,
) -> None:
    assert served_url(tmp_path) is None  # nothing serves the workspace
    marker = _foreign_marker(tmp_path, "http://127.0.0.1:8123")
    assert served_url(tmp_path) == "http://127.0.0.1:8123"
    _age(marker)
    assert served_url(tmp_path) is None  # a web app that died names nobody
    _foreign_marker(tmp_path, None)
    assert served_url(tmp_path) is None  # an older build names no address
    _foreign_marker(tmp_path, "file:///etc/passwd")
    assert served_url(tmp_path) is None  # only a web address is followed
    assert claim_busy(served_path(tmp_path), "served", take_over=True, url="http://127.0.0.1:8123")
    assert json.loads(served_path(tmp_path).read_text(encoding="utf-8"))["url"] == (
        "http://127.0.0.1:8123"
    )
    assert served_url(tmp_path) is None  # this process's own web app is not another's


def test_a_web_app_names_its_address_in_the_served_marker(tmp_path: Path) -> None:
    with TestClient(_web_app(tmp_path, served_url="http://127.0.0.1:8123")):
        marker = json.loads(served_path(tmp_path).read_text(encoding="utf-8"))
    assert marker["url"] == "http://127.0.0.1:8123"
    # Control: one given no address names none, and a pane on the workspace keeps to its own chat.
    with TestClient(_web_app(tmp_path)):
        assert "url" not in json.loads(served_path(tmp_path).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("host", "port", "url"),
    [
        ("127.0.0.1", 8000, "http://127.0.0.1:8000"),
        ("localhost", 8000, "http://localhost:8000"),
        ("0.0.0.0", 8000, "http://127.0.0.1:8000"),
        ("", 8000, "http://127.0.0.1:8000"),
        ("::", 8000, "http://[::1]:8000"),
        ("::1", 8000, "http://[::1]:8000"),
        ("[::1]", 8000, "http://[::1]:8000"),
        ("fe80::1", 9000, "http://[fe80::1]:9000"),
        ("127.0.0.1", 0, None),
    ],
)
def test_the_address_a_pane_reaches_a_web_app_at(host: str, port: int, url: str | None) -> None:
    # A bind to every interface is reached over loopback; port 0 is not known until the bind.
    assert _served_url(host, port) == url


# ── the client reads a chat as the page does ──────────────────────────────────


async def test_a_watch_stream_gives_each_frame_with_its_cursor() -> None:
    body = (
        'id: 1\r\ndata: {"event": "text", "text": "hi"}\r\n\r\n'
        ": ping\r\n\r\n"
        "id: 2\r\ndata: not json\r\n\r\n"
        "id: 3\r\ndata: [1, 2]\r\n\r\n"
        'data: {"a":\r\ndata:  1}\r\n\r\n'
    )
    asked: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append(request.url)
        headers = {"X-Bus-Epoch": "bus-2", "X-Bus-Oldest": "1"}
        return httpx.Response(200, headers=headers, content=body.encode())

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://web") as http:
        client = ServerClient("http://web", http_client=http)
        async with client.watch("s1", since=4, epoch="bus-1") as stream:
            frames = [frame async for frame in stream.frames()]
    assert (stream.epoch, stream.oldest) == ("bus-2", 1)
    # A frame that is not a JSON object comes through as None, for the view to say so.
    assert frames == [(1, {"event": "text", "text": "hi"}), (2, None), (3, None), (None, {"a": 1})]
    assert asked[0].path == "/watch/s1"
    assert dict(asked[0].params) == {"full": "1", "since": "4", "epoch": "bus-1"}


# ── the view of the web app's chat ────────────────────────────────────────────


def test_a_pane_shows_every_line_said_in_the_web_apps_chat_with_its_reply(tmp_path: Path) -> None:
    with _Web(tmp_path) as web:
        screen, mirror = _view(web.url)
        with _showing(mirror):
            _say(tmp_path, "from the say box")
            _until(lambda: "reply to from the say box" in screen.text(), what="the box's reply")
            web.page_say("from the page")
            _until(lambda: "reply to from the page" in screen.text(), what="the page's reply")
    text = screen.text()
    order = [
        "from the say box (web app",
        "reply to from the say box",
        "from the page (web app",
        "reply to from the page",
    ]
    assert [text.count(part) for part in order] == [1, 1, 1, 1]
    assert [text.index(part) for part in order] == sorted(text.index(part) for part in order)
    # The view only read: the web app took the say, and nothing was asked to stop.
    assert not say_pending(say_path(tmp_path))
    assert not interrupt_path(tmp_path).exists()


def test_a_pane_joining_draws_the_saved_chat_once_then_what_the_bus_holds(tmp_path: Path) -> None:
    store = SessionStore(base_dir=tmp_path / "sessions")
    saved = Session(cwd=str(tmp_path), model="scripted/test")
    saved.add_message(Message.user("an old question"))
    saved.add_message(Message.assistant_text("an old answer"))
    store.save(saved)
    (tmp_path / ".current-session").write_text(saved.id + "\n", encoding="utf-8")
    with _Web(tmp_path, new_session=False) as web:  # the web app continues the saved chat
        first, mirror = _view(web.url)
        with _showing(mirror):
            _until(lambda: "an old answer" in first.text(), what="the saved chat")
            _say(tmp_path, "a new question")
            _until(lambda: "reply to a new question" in first.text(), what="the new turn")
        # A pane opened now: the old turn is only saved, and the new one is on the bus as well.
        second, later = _view(web.url)
        with _showing(later):
            _until(lambda: "reply to a new question" in second.text(), what="the bus's turn")
    for screen in (first, second):
        text = screen.text()
        assert text.count("replaying 2 messages") == 1
        assert text.count("an old question") == 1
        assert text.count("an old answer") == 1
        assert text.count("a new question (web app") == 1  # drawn from the bus, not the save
        assert text.count("reply to a new question") == 1


def test_a_pane_follows_the_web_app_to_the_chat_its_page_opens(tmp_path: Path) -> None:
    with _Web(tmp_path) as web:
        store = SessionStore(base_dir=tmp_path / "sessions")
        other = Session(cwd=str(tmp_path), model="scripted/test")
        other.add_message(Message.user("yesterday's question"))
        other.add_message(Message.assistant_text("yesterday's answer"))
        store.save(other)
        screen, mirror = _view(web.url)
        with _showing(mirror):
            _say(tmp_path, "today")
            _until(lambda: "reply to today" in screen.text(), what="the first chat's turn")
            web.open_chat(other.id)
            _until(lambda: "yesterday's answer" in screen.text(), what="the other chat")
            _say(tmp_path, "and now")
            _until(lambda: "reply to and now" in screen.text(), what="the other chat's turn")
    text = screen.text()
    order = ["reply to today", "the web app moved to another chat", "yesterday's answer"]
    order.append("reply to and now")
    assert [text.index(part) for part in order] == sorted(text.index(part) for part in order)
    assert text.count("reply to today") == 1


def test_a_paused_pane_draws_nothing_twice_and_misses_nothing(tmp_path: Path) -> None:
    with _Web(tmp_path) as web:
        screen, mirror = _view(web.url)
        with _showing(mirror):
            _say(tmp_path, "one")
            _until(lambda: "reply to one" in screen.text(), what="the first turn")
        # The pane's own turn runs now: the view draws nothing of what the web app does meanwhile.
        _say(tmp_path, "two")
        _until(lambda: web.answered("two"), what="the web app to answer while the view pauses")
        assert "reply to two" not in screen.text()
        with _showing(mirror):  # the REPL waits again: the view goes on from its place
            _until(lambda: "reply to two" in screen.text(), what="the turn it missed")
            _say(tmp_path, "three")
            _until(lambda: "reply to three" in screen.text(), what="the next turn")
    text = screen.text()
    assert [text.count(f"reply to {n}") for n in ("one", "two", "three")] == [1, 1, 1]


def test_a_pane_says_what_went_by_when_the_bus_let_it_go_while_paused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A bus that holds four frames: a pane paused for three turns finds its place gone.
    monkeypatch.setattr(server_app, "EventBusRegistry", lambda: EventBusRegistry(maxlen=4))
    with _Web(tmp_path) as web:
        screen, mirror = _view(web.url)
        with _showing(mirror):
            _say(tmp_path, "one")
            _until(lambda: "reply to one" in screen.text(), what="the first turn")
        for line in ("two", "three", "four"):
            _say(tmp_path, line)
            _until(lambda line=line: web.answered(line), what=f"the web app to answer {line}")
        with _showing(mirror):
            _until(lambda: "reply to four" in screen.text(), what="the last turn")
    text = screen.text()
    assert text.count("part of the chat went by while this pane was not drawing") == 1
    assert "reply to two" not in text  # gone from the bus: the page has it
    # The turn the bus kept the end of is drawn with no length: one measured from its middle would
    # read as a short turn. The footer goes from its cost straight to its clock.
    assert len(re.findall(r"\$0\.00 \S \d\d:\d\d", text)) == 1
    assert len(re.findall(r"\$0\.00 \S \d+(\.\d+)?s \S \d\d:\d\d", text)) == 2


@pytest.mark.parametrize(
    "new_session",
    [False, True],
    ids=["a-restart-into-a-new-build-keeps-the-chat", "a-new-start-opens-a-new-chat"],
)
def test_a_pane_follows_the_web_app_through_a_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, new_session: bool
) -> None:
    monkeypatch.setattr(served_mirror, "_RETRY_SECONDS", 0.2)
    with _Web(tmp_path) as first:
        screen, mirror = _view(first.url)
        mirror.resume()
        try:
            _say(tmp_path, "before")
            _until(lambda: "reply to before" in screen.text(), what="the turn before the restart")
            first.stop()
            # The same address; the bus is made again, and the chat is the one the start opens.
            with _Web(tmp_path, port=first.port, new_session=new_session) as second:
                _say(tmp_path, "after")
                _until(lambda: "reply to after" in screen.text(), what="the turn after it")
                assert second.answered("after")
        finally:
            mirror.pause()
    text = screen.text()
    assert text.count("reply to before") == 1
    assert "replaying" not in text  # the chat on screen is not drawn again, and a new one is empty


def test_a_pane_the_web_app_refuses_says_why_and_stops(tmp_path: Path) -> None:
    with _Web(tmp_path, token="sekrit") as web:
        screen, mirror = _view(web.url)  # the pane's environment has no token
        mirror.resume()
        _until(lambda: mirror.stopped, what="the view to stop")
        mirror.pause()
        assert "refused this pane (401)" in screen.text()
        assert "ZAKCODE_AUTH_TOKEN" in screen.text()
        # Control: with the web app's token, the same pane shows the chat.
        allowed, view = _view(web.url, token="sekrit")
        with _showing(view):
            web.page_say("with the token")
            _until(lambda: "reply to with the token" in allowed.text(), what="the reply")
    assert "refused" not in allowed.text()


def test_a_pane_says_once_when_it_cannot_reach_the_web_app_and_again_when_it_can(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(served_mirror, "_UNREACHABLE_GRACE_SECONDS", 0.3)
    monkeypatch.setattr(served_mirror, "_RETRY_SECONDS", 0.1)
    monkeypatch.setattr(served_mirror, "_RETRY_CAP_SECONDS", 0.2)
    tries: list[str] = []
    failing = ServedChatMirror._failing

    def counted(self: ServedChatMirror, key: str, message: str) -> None:
        tries.append(key)
        failing(self, key, message)

    monkeypatch.setattr(ServedChatMirror, "_failing", counted)
    port = _free_port()
    screen, mirror = _view(f"http://127.0.0.1:{port}")  # nothing listens there yet
    with _showing(mirror):
        _until(lambda: "cannot reach the web app" in screen.text(), what="the view to say so")
        # Counted in tries, not seconds: a refused connection takes about 2 s on Windows.
        said_at = len(tries)
        _until(lambda: len(tries) >= said_at + 2, what="two more tries")
        assert screen.text().count("cannot reach the web app") == 1
        with _Web(tmp_path, port=port):
            _until(lambda: "showing its chat again" in screen.text(), what="the view to connect")
            _say(tmp_path, "now")
            _until(lambda: "reply to now" in screen.text(), what="the reply")


def test_a_pane_shows_what_the_web_apps_turn_asks_to_run_and_where_to_answer(
    tmp_path: Path,
) -> None:
    with _Web(tmp_path) as web:
        screen, mirror = _view(web.url)
        with _showing(mirror):
            _say(tmp_path, "run rm -rf build")
            _until(lambda: "asks to run" in screen.text(), what="the approval to be shown")
            assert "rm -rf build" in screen.text().split("asks to run", 1)[1]
            assert "in the say box, or on the page" in screen.text()
            _say(tmp_path, "y")  # answered from the say box, as the cockpit's box answers
            _until(lambda: "allow_once: rm -rf build" in screen.text(), what="the turn to go on")
        assert web.transcript()["running"] is False


def test_a_pane_goes_on_in_the_view_it_had_when_the_web_app_comes_back_at_the_same_address(
    tmp_path: Path,
) -> None:
    settings = Settings(default_model="scripted/test", context_window=8192, workspace_root=tmp_path)
    screen = _Screen()
    console = Console(file=screen, width=200, force_terminal=False, theme=ZAK_THEME)
    _foreign_marker(tmp_path, "http://127.0.0.1:8123")
    first = cli_mod._served_chat_mirror(console, settings)
    assert first is not None and first.url == "http://127.0.0.1:8123"
    assert "showing the web app's chat from http://127.0.0.1:8123" in screen.text()
    assert cli_mod._served_chat_mirror(console, settings, kept=first) is first
    _foreign_marker(tmp_path, "http://127.0.0.1:8124")
    moved = cli_mod._served_chat_mirror(console, settings, kept=first)
    assert moved is not None and moved is not first and moved.url == "http://127.0.0.1:8124"
    moved.stopped = True  # refused: a view the web app refused is not gone on in
    assert cli_mod._served_chat_mirror(console, settings, kept=moved) is not moved
    _foreign_marker(tmp_path, None)
    assert cli_mod._served_chat_mirror(console, settings, kept=moved) is None
    assert "the web app does not say where it listens" in screen.text()


# ── a cockpit pane on a served workspace ──────────────────────────────────────


def _run_pane(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    while_served: Callable[[], None],
    *,
    agent: type[FakeAgent] = FakeAgent,
    first: Callable[[threading.Event], None] | None = None,
) -> str:
    """A cockpit pane on ``workspace``, run as the cockpit runs it: frameless, its keyboard unread,
    its one door the say box. ``first`` runs first, handed an event set once a permission prompt
    waits; once the pane's idle wait says the workspace is served, ``while_served`` runs; then the
    marker goes and the say box ends the pane. What the pane printed, wrapped lines joined."""
    monkeypatch.setattr(zakcode, "Agent", agent)
    monkeypatch.setenv("ZAKCODE_WORKSPACE_ROOT", str(workspace))
    monkeypatch.setenv("ZAKCODE_COCKPIT_PANE", "1")
    monkeypatch.setenv("ZAKCODE_INPUT_FRAME", "off")
    monkeypatch.delenv("ZAKCODE_AUTH_TOKEN", raising=False)
    answered: list[str] = []  # what the idle waits answered
    prompting = threading.Event()
    wait = cli_mod._InputMux.next_input
    # A pane that never gets its /exit fails below, not by hanging: past the deadline an idle wait
    # ends, and the pane is handed the /exit.
    expired = threading.Event()
    deadline = threading.Timer(60, expired.set)
    deadline.daemon = True

    def recorded(self: _InputMux, **kwargs: Any) -> tuple[str, str | None]:
        if expired.is_set():
            return ("say", "/exit")
        if not kwargs["idle"]:
            prompting.set()
        kwargs.setdefault("stop", expired)
        got = wait(self, **kwargs)
        if got == ("cancelled", None) and expired.is_set():
            return ("say", "/exit")
        if kwargs["idle"]:
            answered.append(got[0])
        return got

    monkeypatch.setattr(cli_mod._InputMux, "next_input", recorded)
    failed: list[BaseException] = []

    def drive() -> None:
        try:
            if first is not None:
                first(prompting)
            _until(lambda: "served" in answered, what="the pane to say the workspace is served")
            while_served()
        except BaseException as exc:  # noqa: BLE001 — raised in the test's thread below
            failed.append(exc)
        finally:
            # Windows refuses to delete a file another handle has open, and the pane opens the
            # marker at every poll, so the delete is tried again until it lands.
            marker = served_path(workspace)
            give_up = time.monotonic() + 10
            while marker.exists() and time.monotonic() < give_up:
                with contextlib.suppress(PermissionError):
                    marker.unlink()
                time.sleep(0.01)
            with contextlib.suppress(AssertionError):
                _until(lambda: "unserved" in answered, what="the pane to say it is unserved")
            write_say(say_path(workspace), "/exit")

    threading.Thread(target=drive, daemon=True).start()
    deadline.start()
    try:
        result = runner.invoke(app, ["cli"])
    finally:
        deadline.cancel()
    if failed:
        raise failed[0]
    assert result.exit_code == 0, result.output
    return " ".join(result.output.split())


def test_a_cockpit_pane_shows_the_chat_a_web_app_serves_on_its_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    drawn: list[Any] = []
    draw = ServedChatMirror._draw_frame

    async def recorded_draw(self: ServedChatMirror, frame: dict[str, Any] | None) -> None:
        await draw(self, frame)
        drawn.append((frame or {}).get("event"))

    monkeypatch.setattr(ServedChatMirror, "_draw_frame", recorded_draw)
    with _Web(tmp_path) as web:
        # The web app runs in this test's process, so its own marker would name the pane's process
        # too: it is written as another process's, naming the web app's address.
        _foreign_marker(tmp_path, web.url)

        def through_the_say_box() -> None:
            try:
                _say(tmp_path, "from the say box")
                _until(lambda: "done" in drawn, what="the pane to draw the web app's reply")
            finally:
                web.stop()  # before the box's /exit, which the web app would take

        out = _run_pane(tmp_path, monkeypatch, through_the_say_box)
    dash = cli_mod.GLYPHS["dash"]
    served = cli_mod._SERVED_NOTICE["served"].format(dash=dash)
    unserved = cli_mod._SERVED_NOTICE["unserved"].format(dash=dash)
    order = [
        served,
        f"showing the web app's chat from {web.url}",
        "from the say box (web app",
        "reply to from the say box",
        unserved,
    ]
    assert [out.count(part) for part in order] == [1, 1, 1, 1, 1]
    assert [out.index(part) for part in order] == sorted(out.index(part) for part in order)
    assert CANNED_TEXT not in out  # the pane's own chat never ran the line
    # The view stopped when the pane stopped showing it.
    assert not any(t.name == "served-chat-view" and t.is_alive() for t in threading.enumerate())


def test_a_cockpit_pane_says_why_it_cannot_show_a_web_app_that_names_no_address(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _foreign_marker(tmp_path, None)  # a web app on an older build
    out = _run_pane(tmp_path, monkeypatch, lambda: None)
    assert out.count("this pane cannot show the web app's chat") == 1
    assert "the web app does not say where it listens" in out
    assert "showing the web app's chat" not in out


# ── a pane's permission prompt gives way to the web app ───────────────────────


def test_a_panes_prompt_on_a_served_workspace_answers_served_and_leaves_the_say_and_stop(
    tmp_path: Path,
) -> None:
    mux = _InputMux(
        say_path(tmp_path), interrupt_path(tmp_path), keyboard=False, report_served=True
    )
    assert write_say(say_path(tmp_path), "y")
    request_interrupt(interrupt_path(tmp_path))
    _foreign_marker(tmp_path, None)
    # Every wait carries a stop, so a wait that never answers fails the test instead of hanging.
    assert mux.next_input(idle=False, stop=_stopped_after(5)) == ("served", None)
    console = Console(file=io.StringIO(), width=120, force_terminal=False, theme=ZAK_THEME)
    with pytest.raises(KeyboardInterrupt):  # the prompt denies, as at Ctrl-C ...
        mux.answer_line(console, stop=_stopped_after(5))
    assert mux.served_stop.is_set()  # ... and its turn is to stop
    assert say_pending(say_path(tmp_path))  # the "y" and the Stop are the web app's
    assert interrupt_path(tmp_path).exists()


def _prompting_turn(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[bool, str, list[PermissionOutcome]]:
    """A pane's turn whose tool call asks permission, run as the REPL runs it: the prompter's
    answer comes through the pane's input mux, and the turn goes on after it."""
    loop = asyncio.new_event_loop()
    monkeypatch.setattr(cli_mod, "_SESSION_LOOP", loop)
    buffer = io.StringIO()
    console = Console(file=buffer, width=160, force_terminal=False, theme=ZAK_THEME)
    mux = _InputMux(
        say_path(workspace), interrupt_path(workspace), keyboard=False, report_served=True
    )
    mux.served_stop.set()  # left set by an earlier turn: nothing to this one
    prompter = cli_mod.ConsolePermissionPrompter(
        console,
        line_source=lambda stop: mux.answer_line(
            console, stop=stop, accept=cli_mod._parse_permission_answer
        ),
    )
    outcomes: list[PermissionOutcome] = []

    async def stream() -> AsyncIterator[AgentEvent]:
        asked = PermissionRequest(
            tool_name="bash",
            tier=PermissionTier.DANGER_FULL_ACCESS,
            arguments={"command": "rm -rf build"},
        )
        outcomes.append(await prompter.confirm(asked))
        await asyncio.sleep(1.5)  # the model goes on after the answer
        yield AgentDone(stop_reason="completed", iterations=1, usage=Usage())

    try:
        completed = cli_mod._run_streamed_turn(
            console,
            stream,
            StreamRenderer(console=console),
            interrupt_file=interrupt_path(workspace),
            served_stop=mux.served_stop,
        )
    finally:
        loop.close()
    return completed, " ".join(buffer.getvalue().split()), outcomes


def test_a_panes_turn_stops_when_its_prompt_meets_a_web_app_serving_the_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert write_say(say_path(tmp_path), "y")
    request_interrupt(interrupt_path(tmp_path))
    marker = _foreign_marker(tmp_path, None)
    completed, out, outcomes = _prompting_turn(tmp_path, monkeypatch)
    assert "rm -rf build" in out  # the prompt was shown: the turn stopped at it, not before
    # The prompt denies and the turn's watcher stops the turn, and which lands first is a race, as
    # at the stop file: the tool call sees the deny or is cancelled before it does. Never an allow.
    assert outcomes in ([], [PermissionOutcome.DENY_ONCE])
    assert completed is False  # stopped: its hold on the busy marker ends, and the web app goes on
    assert "nothing could answer this turn's permission prompt; turn stopped" in out
    assert say_pending(say_path(tmp_path))  # the web app reads them
    assert interrupt_path(tmp_path).exists()
    # Control: unserved, the box's "y" answers the same prompt and the turn runs to its end.
    marker.unlink()
    interrupt_path(tmp_path).unlink()
    completed, out, outcomes = _prompting_turn(tmp_path, monkeypatch)
    assert outcomes == [PermissionOutcome.ALLOW_ONCE]
    assert completed is True
    assert "turn stopped" not in out


def test_a_panes_turn_gives_way_when_a_web_app_starts_serving_during_its_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deadlock as a pane met it: its turn asks permission, a web app starts serving the
    workspace, and the say box that could answer now feeds the web app, which waits on the turn's
    busy marker. The turn stops, and the pane goes on to say the workspace is served."""
    asked: list[PermissionOutcome] = []

    class _AsksFirst(FakeAgent):
        def astream_turn(self, text: str) -> AsyncIterator[AgentEvent]:
            self.turns.append(text)
            return self._asks()

        async def _asks(self) -> AsyncIterator[AgentEvent]:
            prompter = self.overrides["prompter"]
            request = PermissionRequest(
                tool_name="bash",
                tier=PermissionTier.DANGER_FULL_ACCESS,
                arguments={"command": "rm -rf build"},
            )
            asked.append(await prompter.confirm(request))  # type: ignore[attr-defined]
            await asyncio.sleep(1.5)  # the model goes on after the answer
            yield AgentTextDelta(text="went on after the answer\n")
            yield AgentDone(stop_reason="completed", iterations=1, usage=Usage())

    def a_web_app_starts_serving_during_the_prompt(prompting: threading.Event) -> None:
        _say(tmp_path, "go")
        _until(prompting.is_set, what="the pane's turn to ask permission")
        _foreign_marker(tmp_path, None)

    out = _run_pane(
        tmp_path,
        monkeypatch,
        lambda: None,
        agent=_AsksFirst,
        first=a_web_app_starts_serving_during_the_prompt,
    )
    assert "rm -rf build" in out  # the prompt was shown
    assert asked in ([], [PermissionOutcome.DENY_ONCE])  # denied, or cancelled first (as above)
    assert "went on after the answer" not in out
    dash = cli_mod.GLYPHS["dash"]
    order = [
        "nothing could answer this turn's permission prompt; turn stopped",
        cli_mod._SERVED_NOTICE["served"].format(dash=dash),
        cli_mod._SERVED_NOTICE["unserved"].format(dash=dash),
    ]
    assert [out.count(part) for part in order] == [1, 1, 1]
    assert [out.index(part) for part in order] == sorted(out.index(part) for part in order)
