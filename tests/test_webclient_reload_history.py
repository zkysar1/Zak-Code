"""A reopened web page shows the whole saved conversation, each message drawn once.

The watch bus (``zakcode.server.event_bus``) is in memory: it starts empty on every server start
and keeps only its last N events. The page used to build the conversation from the bus alone, so
a browser reopened after a server restart showed the empty start screen, and one reopened against
a bus that had already evicted the early turns showed only the newest. The saved session, which
the transcript route serves, still held all of it.

The page now draws the saved messages older than the bus's oldest retained event
(``bus_from``) and lets the watch replay draw the rest. These run the page's OWN script, cut
from the shipped page, under node against the DOM the turn-clock tests use, but with a fetch that
talks to a REAL loopback server (the one test_server_consumer.py runs), so the transcript route,
the watch replay and the page's join of the two are exercised together. Not a browser; see
test_webclient_turn_clock.py for why.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.test_server_consumer import _LiveServer
from tests.test_webclient_turn_clock import _NODE, _PRELUDE, _page_script
from zakcode.config import Settings
from zakcode.events import AgentDone, AgentEvent, AgentTextDelta
from zakcode.messages import Message
from zakcode.server import app as server_app
from zakcode.server.app import create_app
from zakcode.server.event_bus import EventBusRegistry
from zakcode.session.store import Session, SessionStore
from zakcode.usage import Usage

pytestmark = pytest.mark.skipif(_NODE is None, reason="needs node on PATH (CI runners carry it)")


class _EchoAgent:
    """A scripted turn that answers with its input, so a duplicated or missing row shows."""

    def __init__(self, session: Session) -> None:
        self.session = session

    async def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]:
        self.session.add_message(Message.user(user_text))
        reply = f"echo {user_text}"
        self.session.add_message(Message.assistant_text(reply))
        yield AgentTextDelta(text=reply)
        yield AgentDone(stop_reason="completed", iterations=1, usage=Usage())


class _PartsAgent:
    """Streams its reply in six parts and saves it after the stream, as the agent loop does: a
    reply is a saved message only once its last part is out."""

    def __init__(self, session: Session) -> None:
        self.session = session

    async def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]:
        self.session.add_message(Message.user(user_text))
        for part in _parts(user_text):
            yield AgentTextDelta(text=part)
        self.session.add_message(Message.assistant_text("".join(_parts(user_text))))
        yield AgentDone(stop_reason="completed", iterations=1, usage=Usage())


def _parts(text: str) -> list[str]:
    return [f"{text}-{i} " for i in range(6)]


def _app(workspace: Path, agent: Callable[[Session], Any] = _EchoAgent) -> Any:
    settings = Settings(
        default_model="scripted/test", context_window=8192, workspace_root=workspace
    )
    store = SessionStore(base_dir=workspace / "sessions")
    return create_app(
        settings=settings,
        store=store,
        agent_factory=lambda session, model, prompter: agent(session),
    )


def _transcript(base: str) -> dict[str, Any]:
    resp = httpx.get(base + "/sessions/current/transcript", timeout=5)
    resp.raise_for_status()
    body: dict[str, Any] = resp.json()
    return body


def _converse(base: str, texts: list[str]) -> None:
    """Say each text and wait until its turn is saved, so the turns land one after another."""
    resp = httpx.get(base + "/sessions/current/transcript", timeout=5)
    have = len(resp.json()["messages"]) if resp.status_code == 200 else 0
    for text in texts:
        posted = httpx.post(base + "/say", json={"text": text}, timeout=5)
        assert posted.status_code in (200, 202), posted.text
        have += 2
        deadline = time.monotonic() + 15
        while True:
            resp = httpx.get(base + "/sessions/current/transcript", timeout=5)
            if resp.status_code == 200 and len(resp.json()["messages"]) >= have:
                break
            assert time.monotonic() < deadline, f"the turn for {text!r} was not saved in 15s"
            time.sleep(0.05)


#: Runs after the page's DOM and before its script. `fetch` becomes the real one, pointed at the
#: live server; the tail exits once `__WANT` user and assistant rows are drawn and nothing has
#: changed for a beat (so a late duplicate would still show), or at a hard deadline. With `__SAY`
#: it sends that message after `__BEFORE` rows are drawn, as a user pressing Enter would.
_TAIL = """
const __BASE = process.argv[2];
const __WANT = Number(process.argv[3]);
const __SAY = process.argv[4];
const __BEFORE = Number(process.argv[5]);
const __realFetch = globalThis.fetch;
globalThis.fetch = (url, opts) => __realFetch(new URL(url, __BASE), opts);
const __log = document.getElementById("log");
const __t0 = Date.now();
let __seen = "", __changed = Date.now(), __said = false;
setInterval(() => {
  const rows = __log.children.map((r) => ({
    cls: [...r.classes].filter((c) => c !== "row").join(" "),
    text: r.textContent,
  }));
  const key = JSON.stringify(rows);
  if (key !== __seen) { __seen = key; __changed = Date.now(); }
  const spoken = rows.filter((r) => r.cls === "user" || r.cls === "assistant").length;
  if (__SAY && !__said && spoken >= __BEFORE && Date.now() - __changed > 300) {
    __said = true;
    __realFetch(new URL("/say", __BASE), {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text: __SAY }),
    });
  }
  if ((spoken >= __WANT && Date.now() - __changed > 400) || Date.now() - __t0 > 15000) {
    const errs = __created
      .filter((n) => n.parent !== null && n.classes.has("errcard"))
      .map((n) => n.textContent);
    process.stdout.write(JSON.stringify({ rows, errs }));
    process.exit(0);
  }
}, 50);
"""

_CLOCK = re.compile(r"\s+\(\d\d:\d\d\)$")


def _open_page(
    base: str, tmp_path: Path, want: int, say: str = "", before: int = 0
) -> list[tuple[str, str]]:
    """Open the page against the server; return the user and assistant rows it drew, in order."""
    assert _NODE is not None
    script = tmp_path / f"page-{time.monotonic_ns()}.js"
    dom = _PRELUDE.split("const __STREAMS = ")[0]
    script.write_text(dom + _TAIL + _page_script(), encoding="utf-8")
    proc = subprocess.run(
        [_NODE, str(script), base, str(want), say, str(before)],
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    drawn = json.loads(proc.stdout.decode("utf-8"))
    # A page error draws an error card, and so does a DOM this fake lacks: either one means the
    # rows below are not the page's whole story.
    assert drawn["errs"] == [], drawn["errs"]
    rows: list[tuple[str, str]] = []
    for row in drawn["rows"]:
        if row["cls"] == "user":
            rows.append(("user", _CLOCK.sub("", row["text"]).removeprefix("›")))
        elif row["cls"] == "assistant":
            rows.append(("assistant", row["text"]))
    return rows


_ONE_AND_TWO = [
    ("user", "one"),
    ("assistant", "echo one"),
    ("user", "two"),
    ("assistant", "echo two"),
]


def test_a_page_reopened_after_a_server_restart_draws_the_saved_conversation(
    tmp_path: Path,
) -> None:
    with _LiveServer(_app(tmp_path)) as base:
        _converse(base, ["one", "two"])
    # The server stops and starts again: its in-memory bus is gone, the conversation is on disk.
    with _LiveServer(_app(tmp_path)) as base:
        assert _open_page(base, tmp_path, want=4) == _ONE_AND_TWO


def test_a_page_reopened_while_the_server_still_holds_the_conversation_draws_each_message_once(
    tmp_path: Path,
) -> None:
    with _LiveServer(_app(tmp_path)) as base:
        _converse(base, ["one", "two"])
        assert _open_page(base, tmp_path, want=4) == _ONE_AND_TWO


def test_a_page_reopened_when_the_bus_kept_only_the_newest_turn_draws_the_older_ones_from_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A turn publishes three events (the user's line, the reply, done), so a ring of three holds
    # the newest turn alone and the two before it exist only in the saved session.
    monkeypatch.setattr(server_app, "EventBusRegistry", lambda: EventBusRegistry(maxlen=3))
    with _LiveServer(_app(tmp_path)) as base:
        _converse(base, ["one", "two", "three"])
        assert _open_page(base, tmp_path, want=6) == [
            *_ONE_AND_TWO,
            ("user", "three"),
            ("assistant", "echo three"),
        ]


def test_a_page_reopened_on_a_turn_longer_than_the_bus_draws_its_reply_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The turn publishes eight events (its line, six parts, done), so a ring of four holds its last
    # three parts and done: where the turn began is gone from the bus. The page hid the saved reply
    # (newer than the bus's oldest event) and drew the bus's three parts as the reply (ADR-0325).
    monkeypatch.setattr(server_app, "EventBusRegistry", lambda: EventBusRegistry(maxlen=4))
    with _LiveServer(_app(tmp_path, _PartsAgent)) as base:
        _converse(base, ["one"])
        assert _open_page(base, tmp_path, want=2) == [
            ("user", "one"),
            ("assistant", "".join(_parts("one"))),
        ]


def test_a_page_reopened_on_a_bus_that_starts_inside_a_turn_joins_the_bus_at_the_next_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two turns of eight events in a ring of ten: the bus starts on the first turn's last part. That
    # turn is drawn from the saved session and the second from the bus, each once and whole.
    monkeypatch.setattr(server_app, "EventBusRegistry", lambda: EventBusRegistry(maxlen=10))
    with _LiveServer(_app(tmp_path, _PartsAgent)) as base:
        _converse(base, ["one", "two"])
        assert _open_page(base, tmp_path, want=4) == [
            ("user", "one"),
            ("assistant", "".join(_parts("one"))),
            ("user", "two"),
            ("assistant", "".join(_parts("two"))),
        ]


def test_a_message_sent_from_a_reopened_page_is_drawn_once_after_the_history(
    tmp_path: Path,
) -> None:
    with _LiveServer(_app(tmp_path)) as base:
        _converse(base, ["one"])
    with _LiveServer(_app(tmp_path)) as base:
        rows = _open_page(base, tmp_path, want=4, say="three", before=2)
    assert rows == [
        ("user", "one"),
        ("assistant", "echo one"),
        ("user", "three"),
        ("assistant", "echo three"),
    ]


def test_the_transcript_stamps_each_message_and_says_where_the_bus_starts(tmp_path: Path) -> None:
    with _LiveServer(_app(tmp_path)) as base:
        _converse(base, ["one"])
        transcript = _transcript(base)
        # The first frame the watch replays carries the stamp the transcript reports as the
        # start of the bus: the page's two sources are on one clock.
        with httpx.stream(
            "GET",
            f"{base}/watch/{transcript['session_id']}?full=1",
            timeout=httpx.Timeout(10.0, read=5.0),
        ) as resp:
            first = next(line for line in resp.iter_lines() if line.startswith("data:"))
        first_at = json.loads(first[len("data:") :])["at"]
        assert transcript["bus_from"] == pytest.approx(first_at, abs=0.001)
        stamps = [m["at"] for m in transcript["messages"]]
        assert all(isinstance(s, float) for s in stamps) and stamps == sorted(stamps)
        assert stamps[0] >= transcript["bus_from"]  # the turn the bus holds starts inside it


def test_an_empty_bus_reports_no_start(tmp_path: Path) -> None:
    with _LiveServer(_app(tmp_path)) as base:
        _converse(base, ["one"])
    with _LiveServer(_app(tmp_path)) as base:
        transcript = _transcript(base)
    assert transcript["bus_from"] is None
    assert [m["text"] for m in transcript["messages"]] == ["one", "echo one"]
