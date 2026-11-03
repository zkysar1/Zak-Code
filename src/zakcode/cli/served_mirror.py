"""A cockpit pane's view of the chat a web app serves on its workspace (ADR-0352).

On a served workspace the say box feeds the web app's chat (ADR-0320), and the pane's own chat
receives nothing. So the pane shows the web app's chat: the end of its saved conversation when it
joins, then every line said in it (through the say box, the page or ``zakcode say``) with the
replies, drawn as the terminal draws its own turns. It is a view and nothing more. It reads the
server's transcript and watch stream and writes neither the say slot, nor the interrupt file, nor
any route, so the server stays the workspace's one reader (ADR-0320).

It joins as the page joins (ADR-0289, ADR-0325): the saved conversation older than the oldest turn
the watch bus holds whole, then the bus from where that turn begins. A dropped stream resumes after
the last frame drawn. A bus made again (the web app restarted) joins the chat that is current now,
drawing its saved conversation only when it is another chat; a ``session_rotated`` frame, sent when
the page opens another chat, does the same.

It runs on a thread of its own, with an event loop of its own, and the REPL lets it draw only
while the REPL waits idle: it pauses the view around everything else, so the pane's own output
never lands inside the web app's. A pause keeps the place, so the resume draws nothing twice.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
from rich.console import Console, Group
from rich.text import Text

from zakcode.cli._glyphs import resolve_glyphs
from zakcode.cli._layout import block, notice_info, user_line
from zakcode.cli.render import StreamRenderer, display_call
from zakcode.cli.saybox import fold_lines
from zakcode.events import AgentDone, AgentEvent, AgentStatus
from zakcode.server.client import ServerClient, WatchStream
from zakcode.server.wire import event_from_dict

#: How a line of the web app's chat is shown to have come in: through the web app, whichever of its
#: doors (the page, the say box, ``zakcode say``) it was sent through.
VIA = "web app"
#: A watch stream sends a keepalive every 15 s, so a read silent this long is a dead connection, and
#: the view connects again instead of waiting on it for good.
_READ_TIMEOUT_SECONDS = 45.0
#: After a dropped or refused connection the view asks again after this long, doubling to the cap.
_RETRY_SECONDS = 1.5
_RETRY_CAP_SECONDS = 10.0
#: A web app marks the workspace before it listens, and a restart leaves a gap, so the view says it
#: cannot reach the web app only once it has failed for this long.
_UNREACHABLE_GRACE_SECONDS = 5.0
#: How long a pause waits for the view's thread to end.
_STOP_SECONDS = 5.0


class _Rejoin(Exception):
    """The chat on view, or its bus, changed: join the web app's current chat."""


class _TurnDrawer:
    """Draws one turn of the web app's chat through a :class:`StreamRenderer`, event by event.

    The renderer reads its events from a queue, and :meth:`draw` returns once the event is drawn,
    so whatever the view draws next lands after it.
    """

    def __init__(self, renderer: StreamRenderer, *, timed: bool) -> None:
        self._queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
        self._task = asyncio.ensure_future(renderer.render(self._events(), timed=timed))

    async def _events(self) -> AsyncIterator[AgentEvent]:
        while True:
            event = await self._queue.get()
            try:
                if event is None:
                    return
                yield event
            finally:
                # Reached when the renderer asks for the next event: this one is drawn.
                self._queue.task_done()

    async def draw(self, event: AgentEvent) -> None:
        await self._queue.put(event)
        if isinstance(event, AgentDone):
            await self._task  # the renderer ends the turn on its done frame, footer and all
            return
        drawn = asyncio.ensure_future(self._queue.join())
        try:
            await asyncio.wait({drawn, self._task}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            drawn.cancel()
        if self._task.done():
            self._task.result()  # the renderer stopped early: say what broke it

    async def end(self) -> None:
        """End the turn as drawn so far: buffered prose is drawn, and no footer (no done frame)."""
        if not self._task.done():
            await self._queue.put(None)
            await self._task


class ServedChatMirror:
    """The web app's chat, drawn in a cockpit pane while the web app serves the workspace."""

    def __init__(self, console: Console, url: str, *, token: str | None = None) -> None:
        self.url = url
        self._console = console
        self._token = token
        self._dash = resolve_glyphs(console)["dash"]
        #: The chat being watched; None until the view joins the web app's current chat.
        self._session: str | None = None
        #: The chat whose saved conversation is on screen: joining it again draws none of it.
        self._shown: str | None = None
        #: The bus the cursor belongs to, and the last frame drawn from it. At a join it is where
        #: the bus's replay takes over from the saved conversation (ADR-0325).
        self._epoch: str | None = None
        self._cursor: int | None = None
        self._turn: _TurnDrawer | None = None
        #: When the frame being drawn was published: the clock the turns are timed and stamped
        #: by, so a turn drawn from a replay shows when it happened, not when the pane drew it.
        self._at = time.time()
        #: Problems said, each said once until the view connects again.
        self._said: set[str] = set()
        self._failing_since: float | None = None
        #: Set when the web app refused the pane: the view has stopped for good.
        self.stopped = False
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[None] | None = None

    # ── the REPL's two controls ───────────────────────────────────────────────────

    def resume(self) -> None:
        """Draw the web app's chat, on the view's own thread, until :meth:`pause`."""
        if self._thread is not None or self.stopped:
            return
        started = threading.Event()

        def run() -> None:
            loop = asyncio.new_event_loop()
            self._loop = loop
            self._task = loop.create_task(self._follow())
            started.set()
            try:
                with contextlib.suppress(asyncio.CancelledError):
                    loop.run_until_complete(self._task)
            finally:
                _close(loop)

        self._thread = threading.Thread(target=run, name="served-chat-view", daemon=True)
        self._thread.start()
        started.wait(_STOP_SECONDS)

    def pause(self) -> None:
        """Stop drawing, keeping the place: the next :meth:`resume` goes on after the last frame
        drawn. Safe to repeat, so an interrupted pause can be run again."""
        thread = self._thread
        if thread is None:
            return
        loop, task = self._loop, self._task
        if loop is not None and task is not None:
            with contextlib.suppress(RuntimeError):  # the loop closed: the view had ended
                loop.call_soon_threadsafe(task.cancel)
        thread.join(_STOP_SECONDS)
        self._thread = None

    # ── following the web app's chat ─────────────────────────────────────────────

    async def _follow(self) -> None:
        timeout = httpx.Timeout(10.0, read=_READ_TIMEOUT_SECONDS)
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else None
        async with httpx.AsyncClient(base_url=self.url, timeout=timeout, headers=headers) as http:
            client = ServerClient(self.url, http_client=http)
            wait = _RETRY_SECONDS
            while True:
                try:
                    try:
                        if self._session is None:
                            await self._join(client)
                        await self._tail(client)
                    except asyncio.CancelledError:
                        await self._end_turn()  # a pause keeps what the turn has drawn
                        raise
                    wait = _RETRY_SECONDS  # the web app ended the stream: ask again
                except _Rejoin:
                    self._session = None
                    continue
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    if status in (401, 403):
                        self._note(
                            f"the web app at {self.url} refused this pane ({status}) {self._dash} "
                            "it needs the web app's ZAKCODE_AUTH_TOKEN in the pane's environment; "
                            "its chat is on its page"
                        )
                        self.stopped = True
                        return
                    if status == 404 and self._session is not None:
                        self._session = None  # the chat is gone: join the current one
                        continue
                    self._failing(
                        "no-chat" if status == 404 else f"status-{status}",
                        f"the web app has no chat open yet {self._dash} waiting for one"
                        if status == 404
                        else f"the web app at {self.url} answered {status} {self._dash} "
                        "asking again",
                    )
                except (httpx.TransportError, OSError) as exc:
                    self._failing(
                        "unreachable",
                        f"cannot reach the web app at {self.url} "
                        f"({type(exc).__name__}) {self._dash} asking again",
                    )
                except Exception as exc:  # noqa: BLE001 — a view says what broke it, never goes blank
                    self._note(f"stopped showing the web app's chat {self._dash} {exc!r}")
                    self.stopped = True
                    return
                await asyncio.sleep(wait)
                wait = min(wait * 2, _RETRY_CAP_SECONDS)

    async def _join(self, client: ServerClient) -> None:
        """Join the web app's current chat: its saved conversation older than what the watch bus
        replays, and the place to watch it from (ADR-0325)."""
        transcript = await client.transcript("current")
        self._connected()
        session = str(transcript.get("session_id") or "")
        since = transcript.get("bus_since")
        epoch = transcript.get("bus_epoch")
        self._epoch = epoch if isinstance(epoch, str) else None
        self._cursor = since if isinstance(since, int) else None
        if session != self._shown:
            # A chat not on screen yet: the first one, or another the page opened. One the web
            # app restarted on is on screen already, and its saved conversation is not drawn again.
            self._draw_saved(_saved_before_the_bus(transcript))
            self._shown = session
        self._session = session

    async def _tail(self, client: ServerClient) -> None:
        """Draw the chat being watched from its watch stream, until the stream ends."""
        assert self._session is not None
        async with client.watch(self._session, since=self._cursor, epoch=self._epoch) as stream:
            self._connected()
            await self._check_bus(stream)
            async for cursor, frame in stream.frames():
                await self._draw_frame(frame)
                if cursor is not None:
                    self._cursor = cursor

    async def _check_bus(self, stream: WatchStream) -> None:
        """Go on from the cursor, or join again when the bus cannot (ADR-0289, ADR-0325)."""
        if stream.epoch is None:
            return  # a server that names no bus never differs
        if self._epoch is not None and stream.epoch != self._epoch:
            # The bus was made again, most often by a restart of the web app: the cursor means
            # nothing to it, and the chat the web app opened on may be another one.
            if self._turn is not None:
                await self._end_turn()
                self._note(
                    f"the web app restarted during a turn {self._dash} "
                    "what this pane missed of it is on the page"
                )
            raise _Rejoin
        evicted = (
            self._cursor is not None
            and stream.oldest is not None
            and stream.oldest > self._cursor + 1
        )
        if evicted:
            await self._end_turn()
            self._note(
                f"part of the chat went by while this pane was not drawing {self._dash} "
                "the page has all of it"
            )
        self._epoch = stream.epoch

    async def _draw_frame(self, frame: dict[str, Any] | None) -> None:
        if frame is None:
            self._note("[bad frame]")
            return
        at = frame.get("at")
        if isinstance(at, (int, float)):
            self._at = float(at)
        kind = frame.get("event")
        if kind == "user_message":
            await self._end_turn()
            self._user_line(str(frame.get("text") or ""))
            self._turn = self._new_turn(timed=True)
            return
        if kind == "session_rotated":
            await self._end_turn()
            reason = str(frame.get("reason") or "")
            self._note(
                "the web app moved to another chat" + (f" {self._dash} {reason}" if reason else "")
            )
            raise _Rejoin
        if isinstance(kind, str):
            event = _agent_event(frame)
            if event is None:
                self._note(f"[unknown event: {kind}]")
                return
            if self._turn is None:
                # A turn whose start this view never saw: drawn from here, and untimed, since a
                # length measured from the middle would read as a short turn.
                self._turn = self._new_turn(timed=False)
            await self._turn.draw(event)
            if isinstance(event, AgentDone) or (
                # The server ends a stopped turn with this status and no done frame.
                isinstance(event, AgentStatus) and event.message == "interrupted"
            ):
                await self._end_turn()
            return
        if frame.get("type") == "action_required":
            self._approval(frame)
            return
        # Another control frame (an error, a status): shown, never silently dropped.
        detail = frame.get("message") or frame.get("detail") or frame
        self._note(f"[{frame.get('type') or 'frame'}] {detail}")

    # ── drawing ───────────────────────────────────────────────────────────────────

    def _new_turn(self, *, timed: bool) -> _TurnDrawer:
        renderer = StreamRenderer(self._console, clock=lambda: self._at, wall=lambda: self._at)
        return _TurnDrawer(renderer, timed=timed)

    async def _end_turn(self) -> None:
        turn, self._turn = self._turn, None
        if turn is not None:
            await turn.end()

    def _user_line(self, text: str) -> None:
        stamp = time.strftime("%H:%M", time.localtime(self._at))
        user_line(self._console, fold_lines(text), via=VIA, stamp=stamp)

    def _approval(self, frame: dict[str, Any]) -> None:
        """The web app's turn waits on a permission: what it asks, and where to answer it."""
        g = resolve_glyphs(self._console)
        arguments = frame.get("arguments")
        rows: list[Text] = [
            Text("the web app's agent asks to run:", style="warn"),
            display_call(
                str(frame.get("tool_name") or "a tool"),
                arguments if isinstance(arguments, dict) else {},
                glyphs=g,
            ),
        ]
        if frame.get("reason"):
            rows.append(Text(str(frame["reason"]), style="notice.dim"))
        rows.append(
            Text(
                "answer y (allow once), a (allow for this session) or n in the say box, "
                "or on the page",
                style="notice.dim",
            )
        )
        self._console.print()
        self._console.print(
            block(self._console, Group(*rows), marker=g["bang"], marker_style="warn")
        )

    def _draw_saved(self, messages: list[dict[str, Any]]) -> None:
        # The replay /resume draws. Imported here: the package imports this module lazily.
        from zakcode.cli import _render_replay

        rows: list[tuple[str, str, int]] = []
        for message in messages:
            hook = message.get("hook")
            role = "hook" if isinstance(hook, str) else str(message.get("role") or "")
            text = (hook if isinstance(hook, str) else str(message.get("text") or "")).strip()
            if text and role in ("hook", "user", "assistant"):
                rows.append((role, text, 0))
        _render_replay(self._console, rows)

    def _note(self, message: str) -> None:
        notice_info(self._console, message)

    def _failing(self, key: str, message: str) -> None:
        """A connection that did not work: said once it has not worked for a while."""
        now = time.monotonic()
        if self._failing_since is None:
            self._failing_since = now
        if now - self._failing_since >= _UNREACHABLE_GRACE_SECONDS and key not in self._said:
            self._said.add(key)
            self._note(message)

    def _connected(self) -> None:
        if self._said:
            self._note(f"reached the web app at {self.url} {self._dash} showing its chat again")
        self._said.clear()
        self._failing_since = None


def _saved_before_the_bus(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    """The saved messages a joining view draws; the watch replay draws the rest (ADR-0325).

    At a turn start (``bus_since``) they end at that turn's own user line, the first one at or
    after the moment the bus starts (``bus_from``). With no turn start, a message at or after it
    is the bus's. The page cuts the same way (ADR-0327).
    """
    messages = [m for m in transcript.get("messages") or [] if isinstance(m, dict)]
    bus_from = transcript.get("bus_from")
    if not isinstance(bus_from, (int, float)):
        return messages
    if isinstance(transcript.get("bus_since"), int):
        for index, message in enumerate(messages):
            at = message.get("at")
            if message.get("role") == "user" and isinstance(at, (int, float)) and at >= bus_from:
                return messages[:index]
        return messages
    return [m for m in messages if not isinstance(m.get("at"), (int, float)) or m["at"] < bus_from]


def _agent_event(frame: dict[str, Any]) -> AgentEvent | None:
    """The agent event a frame carries, or None for one this build does not know."""
    try:
        return event_from_dict({key: value for key, value in frame.items() if key != "at"})
    except ValueError:  # pydantic's ValidationError is a ValueError
        return None


def _close(loop: asyncio.AbstractEventLoop) -> None:
    """Cancel what the view left running on its loop, let it unwind, and close the loop."""
    pending = asyncio.all_tasks(loop)
    for task in pending:
        task.cancel()
    with contextlib.suppress(Exception):
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    with contextlib.suppress(Exception):
        loop.run_until_complete(loop.shutdown_asyncgens())
    loop.close()


__all__ = ["ServedChatMirror", "VIA"]
