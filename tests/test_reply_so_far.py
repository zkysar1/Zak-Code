"""A page or pane opened in the middle of a long reply draws the reply whole (ADR-0363).

The watch bus keeps its last N events, and a reply streams as one event per part, so a long reply
pushes its own start out of the bus while it is still being written. A page opened then drew the
reply from the bus's oldest part on: its start was in no saved message yet, and in no event the bus
still held. The transcript route now asks the running turn's loop instead, on the event loop where
the turn runs: the chat as the loop holds it, what the model call in progress has written so far
(``partial``), and the bus's cursor, read in one step, so the replay after that cursor is exactly
the reply's rest.

That holds only if :attr:`AgentLoop.reply_so_far` has every part the call has streamed once its
event is out, and none once the reply is saved or thrown away: a part in neither is a gap on the
page, a part in both is drawn twice. The loop tests walk a turn event by event and read it at each
one; the transcript tests check the cut against every event the bus was given. The page and the
pane are tested in test_webclient_reload_history.py and test_served_mirror.py.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncGenerator, AsyncIterator
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

import zakcode.agent.loop as loop_module
from tests.test_conclusion_rails import APOLOGY
from tests.test_degeneration import HEALTHY, LOOP_LINE
from tests.test_loop_retry import FlakyStreamProvider, _AlwaysFailsMidStream
from tests.test_loop_stream import EchoTool, ScriptedStreamProvider, _build_loop
from tests.test_server_consumer import _LiveServer
from zakcode.agent.loop import AgentLoop
from zakcode.config import Settings
from zakcode.events import (
    AgentDone,
    AgentEvent,
    AgentStatus,
    AgentTextDelta,
    AgentToolCall,
    AgentToolResult,
)
from zakcode.messages import Message
from zakcode.providers.base import RateLimited, StreamDone, StreamTextDelta, StreamToolCallDelta
from zakcode.server import app as server_app
from zakcode.server import event_bus
from zakcode.server.app import create_app
from zakcode.session.store import Session, SessionStore
from zakcode.tools.base import ToolRegistry
from zakcode.usage import Usage

# ── the loop: what the call in progress has written ──────────────────────────────


def _walk(loop: AgentLoop, text: str) -> list[tuple[AgentEvent, str, list[str]]]:
    """The turn's events, each with what the loop held as it came out: the reply so far, and the
    replies the chat held."""

    async def run() -> list[tuple[AgentEvent, str, list[str]]]:
        walked = []
        async for event in loop.astream_turn(text):
            said = [m.text for m in loop.session.messages if m.role == "assistant" and m.text]
            walked.append((event, loop.reply_so_far, said))
        return walked

    return asyncio.run(run())


def _after_the_last_part(
    walked: list[tuple[AgentEvent, str, list[str]]],
) -> list[tuple[AgentEvent, str, list[str]]]:
    last = max(i for i, (event, _, _) in enumerate(walked) if isinstance(event, AgentTextDelta))
    return walked[last + 1 :]


async def _no_wait(delay: float) -> None:
    return None


def test_each_part_is_held_once_its_event_is_out_and_none_once_the_reply_is_saved(
    tmp_path: Path,
) -> None:
    provider = ScriptedStreamProvider(
        [
            [
                StreamTextDelta(text="one "),
                StreamTextDelta(text="two "),
                StreamTextDelta(text="three"),
                StreamDone(finish_reason="stop"),
            ]
        ]
    )
    loop = _build_loop(tmp_path, provider)
    assert loop.reply_so_far == ""
    walked = _walk(loop, "count")
    parts = [(held, said) for event, held, said in walked if isinstance(event, AgentTextDelta)]
    assert parts == [("one ", []), ("one two ", []), ("one two three", [])]
    after = _after_the_last_part(walked)
    assert after, "the turn ended on its last part"
    assert all(held == "" and said == ["one two three"] for _, held, said in after)
    assert loop.reply_so_far == ""


def test_a_reply_saved_before_its_tool_runs_is_the_chats_and_the_next_call_starts_empty(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    provider = ScriptedStreamProvider(
        [
            [
                StreamTextDelta(text="Let me look. "),
                StreamToolCallDelta(index=0, id="call-1", name="echo"),
                StreamToolCallDelta(index=0, arguments_delta='{"value": "ping"}'),
                StreamDone(finish_reason="tool_calls"),
            ],
            [
                StreamTextDelta(text="It says "),
                StreamTextDelta(text="ping."),
                StreamDone(finish_reason="stop"),
            ],
        ]
    )
    loop = _build_loop(tmp_path, provider, registry=registry)
    walked = _walk(loop, "echo ping")
    steps = (AgentTextDelta, AgentToolCall, AgentToolResult)
    seen = [
        (type(event).__name__, held, said)
        for event, held, said in walked
        if isinstance(event, steps)
    ]
    assert seen == [
        ("AgentTextDelta", "Let me look. ", []),
        ("AgentToolCall", "", ["Let me look. "]),
        ("AgentToolResult", "", ["Let me look. "]),
        ("AgentTextDelta", "It says ", ["Let me look. "]),
        ("AgentTextDelta", "It says ping.", ["Let me look. "]),
    ]
    assert walked[-1][1:] == ("", ["Let me look. ", "It says ping."])


def test_a_call_retried_mid_reply_holds_none_of_what_it_threw_away(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loop_module.asyncio, "sleep", _no_wait)
    loop = _build_loop(tmp_path, FlakyStreamProvider([], fail_midstream=RateLimited("429")))
    walked = _walk(loop, "hi")
    seen = [
        (type(event).__name__, held, said)
        for event, held, said in walked
        if isinstance(event, AgentTextDelta)
        or (isinstance(event, AgentStatus) and "discarded" in event.message)
    ]
    # Thrown away, the first attempt is in neither: the retry writes the reply again.
    assert seen == [
        ("AgentTextDelta", "hello", []),
        ("AgentStatus", "", []),
        ("AgentTextDelta", "hello", []),
    ]
    assert walked[-1][1:] == ("", ["hello"])


def test_a_call_that_fails_mid_reply_saves_what_it_wrote_and_holds_none_of_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loop_module.asyncio, "sleep", _no_wait)
    monkeypatch.setattr(loop_module, "_RATE_LIMIT_RETRY_HORIZON", 0.0)
    loop = _build_loop(tmp_path, _AlwaysFailsMidStream([]))
    walked = _walk(loop, "hi")
    done = walked[-1][0]
    assert isinstance(done, AgentDone) and done.stop_reason == "provider_error"
    after = _after_the_last_part(walked)
    assert any(isinstance(event, AgentStatus) for event, _, _ in after)
    assert all(held == "" and said == ["hello"] for _, held, said in after)


def test_a_reply_thrown_away_as_degenerate_is_in_neither(tmp_path: Path) -> None:
    provider = ScriptedStreamProvider(
        [
            [StreamTextDelta(text=LOOP_LINE * 16), StreamDone(finish_reason="stop")],
            [StreamTextDelta(text=HEALTHY), StreamDone(finish_reason="stop")],
        ]
    )
    loop = _build_loop(tmp_path, provider)
    walked = _walk(loop, "list the files")
    notice = [
        (held, said)
        for event, held, said in walked
        if isinstance(event, AgentStatus) and "degenerated" in event.message
    ]
    assert notice == [("", [])]
    assert walked[-1][1:] == ("", [HEALTHY])


def test_a_reply_thrown_away_as_an_apology_spiral_is_in_neither(tmp_path: Path) -> None:
    provider = ScriptedStreamProvider(
        [
            [StreamTextDelta(text=APOLOGY), StreamDone(finish_reason="stop")],
            [StreamTextDelta(text="Here is what I found."), StreamDone(finish_reason="stop")],
        ]
    )
    loop = _build_loop(tmp_path, provider)
    walked = _walk(loop, "how big is the tree?")
    notice = [
        (held, said)
        for event, held, said in walked
        if isinstance(event, AgentStatus) and "apology spiral" in event.message
    ]
    assert notice == [("", [])]
    assert walked[-1][1:] == ("", ["Here is what I found."])


def test_a_turn_stopped_mid_reply_holds_none_of_it(tmp_path: Path) -> None:
    provider = ScriptedStreamProvider(
        [
            [
                StreamTextDelta(text="a"),
                StreamTextDelta(text="b"),
                StreamTextDelta(text="c"),
                StreamDone(finish_reason="stop"),
            ]
        ]
    )
    loop = _build_loop(tmp_path, provider)

    async def run() -> tuple[str, str]:
        turn = cast(AsyncGenerator[AgentEvent, None], loop.astream_turn("go"))
        async for event in turn:
            if isinstance(event, AgentTextDelta) and event.text == "b":
                break
        during = loop.reply_so_far
        await turn.aclose()  # a turn stopped from outside: its stream is closed where it is
        return during, loop.reply_so_far

    assert asyncio.run(run()) == ("ab", "")


# ── the transcript: a running turn longer than the bus ───────────────────────────


class _Loop:
    """What the transcript asks a running turn's loop: the chat as the loop holds it, and what the
    model call in progress has written that the chat does not hold yet."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.parts: list[str] = []

    @property
    def reply_so_far(self) -> str:
        return "".join(self.parts)


class _LongReply:
    """A turn whose reply is longer than a ring of four, with a loop to ask: it streams six parts,
    waits until ``gate`` is set, streams the rest, and saves the reply after the stream, as the
    agent loop does."""

    def __init__(self, session: Session, parts: list[str], gate: threading.Event) -> None:
        self.session = session
        self.loop = _Loop(session)
        self._parts = parts
        self._gate = gate

    async def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]:
        self.session.add_message(Message.user(user_text))
        for i, part in enumerate(self._parts):
            if i == 6:
                deadline = time.monotonic() + 20
                while not self._gate.is_set() and time.monotonic() < deadline:
                    await asyncio.sleep(0.01)
            self.loop.parts.append(part)
            yield AgentTextDelta(text=part)
        self.session.add_message(Message.assistant_text(self.loop.reply_so_far))
        self.loop.parts = []
        yield AgentDone(stop_reason="completed", iterations=1, usage=Usage())


def _mid_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, parts: list[str], door: str = "say"
) -> tuple[dict[str, Any], list[tuple[int, Any]], dict[str, Any]]:
    """One turn whose reply is ``parts``, on a ring of four, started through ``door``: the say
    inbox (``say``) or POST /chat/stream (``stream``). Returns the transcript read while the turn
    waits after its sixth part, every event the bus was given with its cursor, and the transcript
    once the turn has ended."""
    monkeypatch.setattr(
        server_app, "EventBusRegistry", lambda: event_bus.EventBusRegistry(maxlen=4)
    )
    published: list[tuple[int, Any]] = []
    publish = event_bus.SessionEventBus.publish

    def recorded(bus: event_bus.SessionEventBus, event: Any) -> int:
        cursor = publish(bus, event)
        published.append((cursor, event))
        return cursor

    monkeypatch.setattr(event_bus.SessionEventBus, "publish", recorded)
    gate = threading.Event()
    settings = Settings(default_model="scripted/test", context_window=8192, workspace_root=tmp_path)
    app = create_app(
        settings=settings,
        store=SessionStore(base_dir=tmp_path / "sessions"),
        agent_factory=lambda session, model, prompter: _LongReply(session, parts, gate),
    )
    with _LiveServer(app) as base, httpx.Client(base_url=base, timeout=10) as http:
        streamed: threading.Thread | None = None
        if door == "say":
            assert http.post("/say", json={"text": "a long one"}).status_code in (200, 202)
            path = "/sessions/current/transcript"
        else:
            sid = http.post("/sessions").json()["id"]
            path = f"/sessions/{sid}/transcript"
            ask = {"session_id": sid, "message": "a long one"}

            def stream() -> None:
                with httpx.stream("POST", base + "/chat/stream", json=ask, timeout=30) as resp:
                    for _ in resp.iter_lines():
                        pass

            streamed = threading.Thread(target=stream)
            streamed.start()
        held: dict[str, Any] = {}
        deadline = time.monotonic() + 15
        while not held and time.monotonic() < deadline:
            resp = http.get(path)
            body = resp.json() if resp.status_code == 200 else {}
            if (body.get("partial") or "").endswith(parts[5]):
                held = body
            else:
                time.sleep(0.02)
        gate.set()
        if streamed is not None:
            streamed.join(30)
        ended = http.get(path).json()
        deadline = time.monotonic() + 15
        while ended["running"] and time.monotonic() < deadline:
            time.sleep(0.05)
            ended = http.get(path).json()
    assert held, "the transcript never held the reply being written"
    return held, published, ended


@pytest.mark.parametrize("door", ["say", "stream"])
def test_a_running_turn_longer_than_the_bus_is_read_from_its_loop_and_drawn_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, door: str
) -> None:
    """Once the turn's line and six parts are out, a ring of four holds parts 2 to 5 and not where
    the turn began. The transcript reads the chat from the loop, the six parts as ``partial`` and
    the cursor with them: the parts the bus is given after that cursor are the reply's rest. So for
    a say turn and for a turn a terminal streams through POST /chat/stream."""
    parts = [f"part{i} " for i in range(8)]
    held, published, ended = _mid_reply(tmp_path, monkeypatch, parts, door)

    assert held["running"] is True
    assert held["bus_from"] is None  # every message is drawn from the transcript
    assert held["bus_since"] == 7  # the turn's line and six parts: the cursor read with them
    assert [m["text"] for m in held["messages"]] == ["a long one"]  # the chat the loop holds
    assert held["partial"] == "".join(parts[:6])
    rest = "".join(
        event.text
        for cursor, event in published
        if cursor > held["bus_since"] and isinstance(event, AgentTextDelta)
    )
    assert held["partial"] + rest == "".join(parts)  # every part drawn, each once

    assert ended["running"] is False
    assert ended["partial"] is None
    assert [m["text"] for m in ended["messages"]] == ["a long one", "".join(parts)]


def test_the_reply_being_written_is_redacted_like_the_chat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider key the model writes mid-reply never leaves the box in clear."""
    key = "sk-live-1234567890abcdefghijklmnop"
    monkeypatch.setenv("OPENAI_API_KEY", key)
    parts = [f"my key is {key} ", *(f"part{i} " for i in range(1, 8))]
    held, _, _ = _mid_reply(tmp_path, monkeypatch, parts)
    assert key not in held["partial"]
    assert held["partial"].startswith("my key is ")
    assert held["partial"].endswith(" part1 part2 part3 part4 part5 ")
