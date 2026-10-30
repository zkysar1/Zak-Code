"""Switching chats and resuming the watch, on the page (ADR-0325, ADR-0319).

The page shows one chat at a time and is pointed at another at any moment: a click in the chat
list, Make active, a rotation it follows. In between it may lose its watch connection and resume
it. These run the page's OWN script, cut from the shipped page, under node, on the DOM the
turn-clock tests use (test_webclient_turn_clock.py says why not a browser), against a scripted
server holding several chats. Each chat has its transcripts (served in order, the last again once
they are spent), its watch connections keyed by the ``since`` each one asks with ("null" on the
first; dropped after their frames or held open, each naming the oldest cursor its bus holds), and
an answer delay for its GET /sessions/{id}. Timed steps click a chat in the list or look at what
the page shows.

One test runs a real server instead, for a rotation the page used to replay from the bus.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.test_server_consumer import _LiveServer
from tests.test_webclient_reload_history import _TAIL, _app, _converse
from tests.test_webclient_turn_clock import _NODE, _PRELUDE, _page_script

pytestmark = pytest.mark.skipif(_NODE is None, reason="needs node on PATH (CI runners carry it)")

_T0 = 1_790_000_000.0  # any fixed epoch: nothing here reads the clock
_USAGE = {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10, "cost_usd": 0.0}
A, X, Y = "aaaaaaaa-0001", "xxxxxxxx-0002", "yyyyyyyy-0003"

_DOM = _PRELUDE.split("const __STREAMS = ")[0]

#: The scripted server and the timed steps. Runs after the DOM and before the page's script.
_SERVER = r"""
const __SCN = JSON.parse(process.argv[2]);
const __later = setTimeout;  // the real clock, for this script's own timing
// The reconnect backoff is a wait, not behaviour under test: the page's timers fire at once.
globalThis.setTimeout = (fn, _ms, ...args) => __later(fn, 0, ...args);
const __asked = {};  // transcripts asked for, per chat
const __watched = {};  // watch connections asked for, per chat
const __posted = [];  // what the page posted that the script has no answer of its own for
const __open = [];  // ends each watch connection held open, for a server that restarts
function __sse(frames) {
  return frames.map(([id, f]) => `id: ${id}\r\ndata: ${JSON.stringify(f)}\r\n\r\n`).join("");
}
function __answer(body, status, delay) {
  const resp = { ok: status >= 200 && status < 300, status, json: async () => body,
                 headers: { get: () => null } };
  return delay ? new Promise((r) => __later(() => r(resp), delay)) : Promise.resolve(resp);
}
globalThis.fetch = async (url, opts) => {
  opts = opts || {};
  const u = new URL(url, "http://page.test");
  const chats = __SCN.chats;
  if (u.pathname === "/sessions/current") {
    if ((opts.method || "GET") === "PUT") {
      // New chat asks for none by name: the server makes the chat the scenario keeps for it.
      const asked = JSON.parse(opts.body).session_id;
      __SCN.current = asked === null ? Object.keys(chats).find((id) => chats[id].fresh) : asked;
    }
    return __answer(chats[__SCN.current].info, 200, 0);
  }
  if (u.pathname === "/sessions/recent") {
    if (__SCN.listFails) return __answer({}, 500, 0);
    const list = Object.values(chats).map((c) => ({
      id: c.info.id, title: c.title, message_count: 2,
      updated_at: "2026-10-27T10:00:00+00:00", current: c.info.id === __SCN.current,
    }));
    return __answer(list, 200, __SCN.listDelays.shift() || 0);  // the list as it was when asked
  }
  let m = u.pathname.match(/^\/sessions\/([^/]+)\/transcript$/);
  if (m) {
    // A transcript may answer with a `status` other than 200, and after a `delay`.
    const chat = chats[m[1]];
    const n = (__asked[m[1]] = (__asked[m[1]] || 0) + 1);
    const t = chat.transcripts[Math.min(n, chat.transcripts.length) - 1];
    return __answer({ session_id: m[1], messages: [], message_count: 0, bus_from: null,
                      bus_since: null, bus_epoch: chat.epoch, running: false, ...t },
                    t.status || 200, t.delay || 0);
  }
  m = u.pathname.match(/^\/sessions\/([^/]+)$/);
  if (m) return __answer(chats[m[1]].info, chats[m[1]].fails ? 500 : 200, chats[m[1]].delay);
  m = u.pathname.match(/^\/watch\/([^/]+)$/);
  if (m) {
    const chat = chats[m[1]];
    __watched[m[1]] = (__watched[m[1]] || 0) + 1;
    const conn = chat.watch[String(u.searchParams.get("since"))];
    if (conn && conn.status) return __answer({ detail: "scripted" }, conn.status, 0);  // every time
    if (!conn || conn.used) return new Promise(() => {});  // the script is spent: never answers
    conn.used = true;
    if (conn.delay) await new Promise((r) => __later(r, conn.delay));  // a replay slow to come
    const bytes = new TextEncoder().encode(__sse(conn.frames));
    let sent = false;
    const read = () => {
      if (!sent) { sent = true; return Promise.resolve({ done: false, value: bytes }); }
      if (conn.drop) return Promise.resolve({ done: true });  // the connection drops
      // Held open, as a live watch is; an abort rejects the read, as a browser's does, and a
      // server that restarts ends it.
      return new Promise((resolve, reject) => {
        __open.push(() => resolve({ done: true }));
        if (opts.signal) opts.signal.addEventListener("abort", () => reject(new Error("aborted")));
      });
    };
    const named = { "x-bus-epoch": chat.epoch,
                    "x-bus-oldest": conn.oldest === undefined ? null : String(conn.oldest) };
    const headers = { get: (name) => named[name.toLowerCase()] ?? null };
    return { ok: true, status: 200, headers, body: { getReader: () => ({ read }) } };
  }
  if (opts.method === "POST") {
    __posted.push({ path: u.pathname, body: opts.body ? JSON.parse(opts.body) : null });
  }
  return __answer({ queued: true }, 200, 0);  // /say, /interrupt
};
const __seen = [];
function __look(label) {
  const rows = document.getElementById("log").children.map((r) => ({
    cls: [...r.classes].filter((c) => c !== "row").join(" "), text: r.textContent }));
  const statusOn = document.getElementById("stream-status").classes.has("show");
  return {
    label, rows,
    send: document.getElementById("send").textContent,
    approval_hidden: document.getElementById("approval").hidden,
    status: statusOn ? document.getElementById("status-text").textContent : "",
    connection: document.getElementById("conn-label").textContent,
    chip: document.getElementById("chip-session").textContent,
    banner_hidden: document.getElementById("view-banner").hidden,
    placeholder: document.getElementById("input").placeholder,
    asked: { ...__asked },
    watched: { ...__watched },
    posted: [...__posted],
  };
}
for (const step of __SCN.steps) {
  __later(() => {
    if (step.click) {
      const item = document.getElementById("chats").children.find(
        (c) => c.children[0] && c.children[0].textContent === step.click);
      if (!item) throw new Error("no chat in the list titled " + step.click);
      item.onclick();
    } else if (step.press) {
      document.getElementById(step.press).onclick();
    } else if (step.type !== undefined) {
      document.getElementById("input").value = step.type;
    } else if (step.restart !== undefined) {
      // The server restarts: every bus is made again, the chat it opens (if any) is current, and
      // every open watch connection ends.
      for (const chat of Object.values(__SCN.chats)) chat.epoch += "-again";
      if (step.restart !== null) __SCN.current = step.restart;
      for (const end of __open.splice(0)) end();
    } else if (step.set) {
      Object.assign(__SCN, step.set);
    } else {
      __seen.push(__look(step.look));
    }
  }, step.at);
}
__later(() => {
  const errors = __created.filter((n) => n.parent !== null && n.classes.has("errcard"))
    .map((n) => n.textContent);
  process.stdout.write(JSON.stringify({ seen: __seen, errors }));
  process.exit(0);
}, Math.max(...__SCN.steps.map((s) => s.at)) + 100);
"""

_CLOCK = re.compile(r"\s+\(\d\d:\d\d\)$")


def _chat(
    sid: str,
    title: str,
    *,
    watch: dict[str, Any] | None = None,
    transcripts: list[dict[str, Any]] | None = None,
    delay: int = 0,
    fails: bool = False,
    fresh: bool = False,
) -> dict[str, Any]:
    """One chat the scripted server holds. ``watch`` maps the ``since`` a watch connection asks
    with to its ``frames``, whether it ``drop``s after them, the ``oldest`` cursor it names and a
    ``delay`` before it answers, or to a ``status`` it answers every time instead; a chat that
    ``fails`` answers its GET /sessions/{id} with a 500; a ``fresh`` one is the chat New chat
    makes."""
    return {
        "info": {"id": sid, "model": "fake-model"},
        "title": title,
        "epoch": "bus-" + sid[:4],
        "delay": delay,
        "fails": fails,
        "fresh": fresh,
        "watch": watch or {"null": {"frames": []}},
        "transcripts": transcripts or [{}],
    }


def _page(
    tmp_path: Path,
    chats: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    errors: list[str] | None = None,
    list_delays: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Run the page against the scripted chats, the first one current; what it showed at each
    ``look`` step. The page must draw exactly the error cards in ``errors`` (none by default).
    ``list_delays`` holds back the answers to the page's chat-list requests, in the order asked."""
    assert _NODE is not None
    script = tmp_path / f"page-{time.monotonic_ns()}.js"
    script.write_text(_DOM + _SERVER + _page_script(), encoding="utf-8")
    scenario = {
        "current": chats[0]["info"]["id"],
        "chats": {chat["info"]["id"]: chat for chat in chats},
        "steps": steps,
        "listDelays": list_delays or [],
    }
    proc = subprocess.run(
        [_NODE, str(script), json.dumps(scenario)], capture_output=True, timeout=60, check=False
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    shown = json.loads(proc.stdout.decode("utf-8"))
    # A page error draws an error card, and so does a DOM this fake lacks: either one means the
    # rows below are not the page's whole story.
    assert shown["errors"] == (errors or []), shown["errors"]
    seen: list[dict[str, Any]] = shown["seen"]
    return seen


def _frame(cursor: int, **frame: Any) -> list[Any]:
    return [str(cursor), frame]


def _spoken(seen: dict[str, Any]) -> list[tuple[str, str]]:
    """The user and assistant rows, in order, a user row without its arrow or its clock."""
    rows: list[tuple[str, str]] = []
    for row in seen["rows"]:
        if row["cls"] == "user":
            rows.append(("user", _CLOCK.sub("", row["text"]).removeprefix("›")))
        elif row["cls"] == "assistant":
            rows.append(("assistant", row["text"]))
    return rows


def _tool_card(seen: dict[str, Any]) -> str:
    [card] = [row["text"] for row in seen["rows"] if row["cls"] == "tool"]
    return card


_TURN = [
    _frame(1, event="user_message", text="deploy it", at=_T0),
    _frame(
        2, event="tool_call", id="t1", name="run", arguments={"command": "make deploy"}, at=_T0 + 1
    ),
]


# ── a dropped watch connection (ADR-0325) ────────────────────────────────────────────────────


def test_a_dropped_watch_keeps_the_turn_on_screen_and_the_approval_it_waits_on(
    tmp_path: Path,
) -> None:
    """A dropped connection is not a turn end: the resume replays this chat from its cursor, so
    nothing the turn put on screen comes off. The page reset the turn's view on every drop, and
    only what follows the cursor comes back: the approval card was hidden for good, so nobody was
    asked and the server's fail-closed deny ran out the clock, the running tool read
    "interrupted", and Stop turned back into Send."""
    asks = _frame(
        3,
        type="action_required",
        tool_name="run",
        arguments={"command": "make deploy"},
        reason="needs your approval",
        at=_T0 + 2,
    )
    chat = _chat(
        A, "chat A", watch={"null": {"frames": [*_TURN, asks], "drop": True}, "3": {"frames": []}}
    )
    [seen] = _page(tmp_path, [chat], [{"at": 400, "look": "resumed"}])
    assert seen["connection"] == "connected"  # the resume was answered
    assert seen["approval_hidden"] is False
    assert seen["status"] == "waiting on you"
    assert _tool_card(seen).endswith("…")  # still running
    assert seen["send"] == "■"  # Stop


def test_a_tool_result_that_lands_after_a_resume_resolves_its_own_card(tmp_path: Path) -> None:
    """The resume carries on the turn the drop interrupted: the result of a call made before the
    drop resolves that call's card. It used to find the card abandoned as "interrupted" and draw
    a row of its own."""
    result = _frame(
        3, event="tool_result", tool_use_id="t1", output="deployed", is_error=False, at=_T0 + 40
    )
    chat = _chat(
        A, "chat A", watch={"null": {"frames": _TURN, "drop": True}, "2": {"frames": [result]}}
    )
    [seen] = _page(tmp_path, [chat], [{"at": 400, "look": "resumed"}])
    card = _tool_card(seen)
    assert "Ran · 1 line" in card
    assert "interrupted" not in card
    assert [row["cls"] for row in seen["rows"]] == ["user", "tool"]  # no row of its own


def test_a_resume_after_a_cursor_the_bus_has_evicted_starts_the_chat_over(tmp_path: Path) -> None:
    """The page was away while the bus evicted what came after its cursor, so resuming there
    would skip it. The watch names the oldest cursor its bus holds; one past the page's cursor
    sends the page to start over, from the saved conversation and the replay after its cut."""
    first = [
        _frame(1, event="user_message", text="one", at=_T0),
        _frame(2, event="text", text="reply one", at=_T0 + 1),
        _frame(3, event="done", stop_reason="completed", iterations=1, usage=_USAGE, at=_T0 + 2),
    ]
    after = [
        _frame(9, event="user_message", text="three", at=_T0 + 30),
        _frame(10, event="text", text="reply three", at=_T0 + 31),
    ]
    saved = {
        "messages": [
            {"role": "user", "text": "one", "at": _T0},
            {"role": "assistant", "text": "reply one", "at": _T0 + 1.5},
            {"role": "user", "text": "two", "at": _T0 + 10},
            {"role": "assistant", "text": "reply two", "at": _T0 + 11},
        ],
        "bus_from": _T0 + 30,
        "bus_since": 8,
    }
    chat = _chat(
        A,
        "chat A",
        watch={"null": {"frames": first, "drop": True}, "3": {"frames": after, "oldest": 9}},
        transcripts=[{}, saved],
    )
    [seen] = _page(tmp_path, [chat], [{"at": 400, "look": "resumed"}])
    assert seen["asked"][A] == 2  # it asked for the saved conversation again
    assert _spoken(seen) == [
        ("user", "one"),
        ("assistant", "reply one"),
        ("user", "two"),
        ("assistant", "reply two"),
        ("user", "three"),
        ("assistant", "reply three"),
    ]


# ── a chat that no longer exists (ADR-0327) ──────────────────────────────────────────────────


def test_a_chat_that_no_longer_exists_is_not_asked_for_again_and_again(tmp_path: Path) -> None:
    """A watch that answers 404 names a chat that is gone (deleted, or its file removed), and no
    reconnect brings it back. The page asked again every 1.5 s for as long as it stayed open, with
    the turn it showed running still running. It now says the chat is gone, ends that turn on
    screen, and stops asking."""
    chat = _chat(A, "chat A", watch={"null": {"frames": _TURN, "drop": True}, "2": {"status": 404}})
    [seen] = _page(
        tmp_path,
        [chat],
        [{"at": 400, "look": "gone"}],
        errors=["this chat no longer exists — open another from the list"],
    )
    assert seen["watched"][A] == 2  # the first connection, and the one reconnect it answered
    assert seen["connection"] == "chat not found"
    assert seen["send"] == "↑"  # the turn shown running ended with it


# ── Stop (ADR-0327) ──────────────────────────────────────────────────────────────────────────


def test_stop_names_the_chat_on_screen(tmp_path: Path) -> None:
    """Stop asked the workspace's say turn to stop, the one turn that signal reaches, so a turn a
    terminal ran on this chat through POST /chat or /chat/stream ran on. It now names the chat,
    and the server stops whichever turn that chat runs."""
    chat = _chat(A, "chat A", watch={"null": {"frames": _TURN}})
    running, stopped = _page(
        tmp_path,
        [chat],
        [
            {"at": 300, "look": "running"},
            {"at": 320, "press": "send"},
            {"at": 400, "look": "stopped"},
        ],
    )
    assert running["send"] == "■"  # the turn runs, so the button is Stop
    assert running["posted"] == []
    assert stopped["posted"] == [{"path": "/interrupt", "body": {"session_id": A}}]


# ── opening a chat (ADR-0325) ────────────────────────────────────────────────────────────────


def test_a_reply_stamped_in_the_tick_the_next_turn_starts_in_is_drawn_once(tmp_path: Path) -> None:
    """On a clock as coarse as Windows' (15.6 ms) the reply that ends one turn and the user line
    that starts the next can carry one stamp. The cut falls on the second turn's user line, and
    the page drew from the saved conversation only what was stamped before it: the reply was
    hidden there, and the replay, which starts at the second turn, did not draw it either
    (ADR-0327)."""
    tick = _T0 + 60
    saved = {
        "messages": [
            {"role": "user", "text": "one", "at": _T0},
            {"role": "assistant", "text": "reply one", "at": tick},
            {"role": "user", "text": "two", "at": tick},
            {"role": "assistant", "text": "reply two", "at": tick + 5},
        ],
        "bus_from": tick,
        "bus_since": 0,
    }
    turn = [
        _frame(1, event="user_message", text="two", at=tick),
        _frame(2, event="text", text="reply two", at=tick + 5),
        _frame(3, event="done", stop_reason="completed", iterations=1, usage=_USAGE, at=tick + 6),
    ]
    chat = _chat(A, "chat A", watch={"0": {"frames": turn}}, transcripts=[saved])
    [seen] = _page(tmp_path, [chat], [{"at": 300, "look": "opened"}])
    assert _spoken(seen) == [
        ("user", "one"),
        ("assistant", "reply one"),
        ("user", "two"),
        ("assistant", "reply two"),
    ]


@pytest.mark.parametrize(("running", "button"), [(True, "■"), (False, "↑")])
def test_a_chat_with_a_turn_running_shows_it_running_before_the_replay_shows_it_start(
    tmp_path: Path, running: bool, button: str
) -> None:
    """A long turn's user line leaves the bus first, so a page opening the chat midway can
    replay no turn start at all, and it drew the running turn with Send where Stop belongs. The
    transcript says when a turn is running, and the page shows it running from the start."""
    tail = [_frame(5, event="text", text="still going", at=_T0 + 50)]
    saved = {
        "messages": [{"role": "user", "text": "a long one", "at": _T0}],
        "bus_from": _T0 + 40,
        "running": running,
    }
    chat = _chat(A, "chat A", watch={"null": {"frames": tail}}, transcripts=[saved])
    [seen] = _page(tmp_path, [chat], [{"at": 300, "look": "opened"}])
    assert _spoken(seen) == [("user", "a long one"), ("assistant", "still going")]
    assert seen["send"] == button


@pytest.mark.parametrize("second", ["chat Y", "chat A"])
def test_two_quick_clicks_in_the_chat_list_land_on_the_second(tmp_path: Path, second: str) -> None:
    """Opening a chat waits for its GET /sessions/{id}, and a long chat answers slower than a
    short one. The page opened whichever answer came back LAST, so clicking a long chat and then
    another one, or the one on screen, left it on the long one."""
    chats = [_chat(A, "chat A"), _chat(X, "chat X", delay=250), _chat(Y, "chat Y", delay=20)]
    steps = [
        {"at": 150, "click": "chat X"},
        {"at": 180, "click": second},
        {"at": 700, "look": "settled"},
    ]
    [seen] = _page(tmp_path, chats, steps)
    assert seen["chip"] == {"chat Y": Y, "chat A": A}[second][:8]


def test_a_chat_that_fails_to_open_after_a_later_click_says_nothing_in_the_chat_on_screen(
    tmp_path: Path,
) -> None:
    """A chat asked for before the one on screen, which then fails to open, drew "could not open
    the chat" into the chat the page had moved on to, as if that one had failed."""
    chats = [_chat(A, "chat A"), _chat(X, "chat X", delay=250, fails=True), _chat(Y, "chat Y")]
    steps = [
        {"at": 150, "click": "chat X"},
        {"at": 180, "click": "chat Y"},
        {"at": 700, "look": "settled"},
    ]
    [seen] = _page(tmp_path, chats, steps)  # _page refuses a page with an error card on it
    assert seen["chip"] == Y[:8]


def test_a_chat_clicked_before_another_is_made_active_does_not_open_over_it(
    tmp_path: Path,
) -> None:
    """Make this chat active opens a chat without the chat list, and counts as the last one
    asked for too: a chat clicked before it, and slower to answer, opened over the chat just
    made active."""
    chats = [_chat(A, "chat A"), _chat(X, "chat X", delay=250), _chat(Y, "chat Y")]
    steps = [
        {"at": 150, "click": "chat Y"},
        {"at": 300, "click": "chat X"},
        {"at": 330, "press": "make-active"},  # Y, the chat on screen
        {"at": 900, "look": "settled"},
    ]
    [seen] = _page(tmp_path, chats, steps)
    assert seen["chip"] == Y[:8]
    assert seen["banner_hidden"] is True  # Y is the active chat now


def test_a_chat_whose_history_fails_after_another_opened_says_nothing_in_that_one(
    tmp_path: Path,
) -> None:
    """Opening a chat draws its saved conversation first. When that failed after another chat
    had opened, the "could not load" card went into the chat then on screen, as if it had."""
    failing = _chat(X, "chat X", transcripts=[{"status": 500, "delay": 250}])
    chats = [_chat(A, "chat A"), failing, _chat(Y, "chat Y")]
    steps = [
        {"at": 150, "click": "chat X"},
        {"at": 200, "click": "chat Y"},
        {"at": 700, "look": "settled"},
    ]
    [seen] = _page(tmp_path, chats, steps)  # _page refuses a page with an error card on it
    assert seen["chip"] == Y[:8]


def test_a_chat_list_that_answers_late_does_not_undo_a_later_one(tmp_path: Path) -> None:
    """The chat list is asked for whenever what it shows can change, and an answer can come back
    after a later one's. The page drew each as it came, so a list asked for before a chat was
    made active, and answered after, marked the old chat active again and the chat now active
    as one only viewed."""
    chats = [_chat(A, "chat A"), _chat(Y, "chat Y")]
    steps = [
        {"at": 150, "click": "chat Y"},  # the list it asks for (A active) answers 300 ms later
        {"at": 250, "press": "make-active"},  # the list it asks for (Y active) answers at once
        {"at": 800, "look": "settled"},
    ]
    [seen] = _page(tmp_path, chats, steps, list_delays=[0, 300])
    assert seen["chip"] == Y[:8]
    assert seen["banner_hidden"] is True
    assert seen["placeholder"] == "Message Zak Code…"


def _said(*lines: str) -> dict[str, Any]:
    """A saved conversation: ``lines`` spoken in turn, the user's first."""
    roles = ("user", "assistant")
    return {"messages": [{"role": roles[i % 2], "text": t, "at": _T0} for i, t in enumerate(lines)]}


@pytest.mark.parametrize("opens", ["a click in the list", "New chat"])
def test_a_chat_opened_from_one_with_a_conversation_on_screen_shows_its_own(
    tmp_path: Path, opens: str
) -> None:
    """The first line drawn takes the start screen out of the page, and opening a chat put its
    model and id on the start screen by looking that up: it found nothing, threw, and stopped
    half way, the chat it left still on screen and nothing watching the one it opened. Every
    switch away from a chat with anything in it did that, the next one too."""
    other = (
        _chat(Y, "chat Y", fresh=True)
        if opens == "New chat"
        else _chat(Y, "chat Y", transcripts=[_said("second", "and again")])
    )
    chats = [_chat(A, "chat A", transcripts=[_said("hello", "hi there")]), other]
    leave = {"press": "new-chat"} if opens == "New chat" else {"click": "chat Y"}
    steps = [
        {"at": 150, **leave},
        {"at": 450, "look": "opened"},
        {"at": 500, "click": "chat A"},
        {"at": 800, "look": "back"},
    ]
    opened, back = _page(tmp_path, chats, steps)
    assert opened["chip"] == Y[:8]
    assert opened["watched"].get(Y) == 1
    if opens == "New chat":
        assert _spoken(opened) == []
        assert "fake-model · session " + Y[:8] in [row["text"] for row in opened["rows"]]
    else:
        assert _spoken(opened) == [("user", "second"), ("assistant", "and again")]
    assert back["chip"] == A[:8]
    assert _spoken(back) == [("user", "hello"), ("assistant", "hi there")]


@pytest.mark.parametrize("drawn_from", ["saved", "replayed"])
def test_a_chat_with_a_conversation_opens_with_no_start_screen_on_the_way(
    tmp_path: Path, drawn_from: str
) -> None:
    """Opening a chat put the start screen back at once, and the first row drawn took it out, so
    every switch to a chat with a conversation flashed it until the rows came: from the saved
    conversation, or, for a chat used since the server started, from its bus's replay."""
    if drawn_from == "saved":
        other = _chat(Y, "chat Y", transcripts=[{**_said("second", "and again"), "delay": 300}])
    else:
        turn = [
            _frame(1, event="user_message", text="second", at=_T0),
            _frame(2, event="text", text="and again", at=_T0 + 1),
            _frame(
                3, event="done", stop_reason="completed", iterations=1, usage=_USAGE, at=_T0 + 2
            ),
        ]
        on_the_bus = {**_said("second", "and again"), "bus_from": _T0, "bus_since": 0}
        watch = {"0": {"frames": turn, "delay": 300}}
        other = _chat(Y, "chat Y", watch=watch, transcripts=[on_the_bus])
    chats = [_chat(A, "chat A", transcripts=[_said("hello", "hi there")]), other]
    steps = [
        {"at": 150, "click": "chat Y"},
        {"at": 300, "look": "on its way"},
        {"at": 800, "look": "drawn"},
    ]
    on_its_way, drawn = _page(tmp_path, chats, steps)
    assert all("fake-model · session" not in row["text"] for row in on_its_way["rows"])
    assert _spoken(drawn) == [("user", "second"), ("assistant", "and again")]


def test_making_the_chat_on_screen_active_takes_down_what_the_status_line_said_of_it(
    tmp_path: Path,
) -> None:
    """A send from a chat that is only viewed is held, with a line saying to make the chat
    active first. Making it active opens nothing (it is on screen already), so nothing took
    that line down, and it went on asking for what had just been done."""
    chats = [_chat(A, "chat A"), _chat(Y, "chat Y")]
    steps = [
        {"at": 150, "click": "chat Y"},
        {"at": 300, "type": "hello"},
        {"at": 320, "press": "send"},
        {"at": 400, "look": "held"},
        {"at": 450, "press": "make-active"},
        {"at": 700, "look": "active"},
    ]
    held, active = _page(tmp_path, chats, steps)
    assert held["status"] == "you're viewing this chat — make it active to send here"
    assert held["posted"] == []  # nothing went to /say
    assert active["banner_hidden"] is True
    assert active["status"] == ""


@pytest.mark.parametrize(
    ("viewing", "opened", "lands_on"),
    [("chat A", Y, Y), ("chat A", None, A), ("chat X", Y, X)],
    ids=["active-chat-follows-the-fresh-one", "kept-chat-starts-over", "parked-page-stays"],
)
def test_a_page_left_open_while_the_server_restarts_lands_where_a_fresh_page_would(
    tmp_path: Path, viewing: str, opened: str | None, lands_on: str
) -> None:
    """A server opens a fresh chat as it starts, so a page opened then lands on it. A page left
    open across the restart came back on the chat it showed, still marked active, and its next
    message was refused there. It follows the active chat now, as a fresh page would, unless it
    was parked on another chat, which it keeps; nothing marks the restart in the conversation."""
    resumed = {
        "null": {"frames": [_frame(1, event="text", text="still here", at=_T0 + 5)]},
        "1": {"frames": []},
    }
    chats = [
        _chat(A, "chat A", watch=resumed, transcripts=[_said("hello", "hi there")]),
        _chat(X, "chat X", watch=resumed, transcripts=[_said("parked", "ok")]),
        _chat(Y, "chat Y"),
    ]
    steps: list[dict[str, Any]] = [{"at": 150, "click": "chat X"}] if viewing == "chat X" else []
    steps += [{"at": 400, "restart": opened}, {"at": 800, "look": "after"}]
    [after] = _page(tmp_path, chats, steps)
    assert after["chip"] == lands_on[:8]
    assert all("session restarted" not in row["text"] for row in after["rows"])
    spoken = {Y: [], A: [("user", "hello"), ("assistant", "hi there")]}
    assert _spoken(after) == spoken.get(lands_on, [("user", "parked"), ("assistant", "ok")])
    assert after["banner_hidden"] is (lands_on != X)  # a parked page still says it is not live


@pytest.mark.parametrize(
    ("then", "on_screen"),
    [({"click": "chat A"}, A), ({"press": "make-active"}, X)],
    ids=["back-to-the-active-chat", "the-chat-on-screen-made-active"],
)
def test_the_banner_follows_the_chat_on_screen_even_when_the_chat_list_cannot_load(
    tmp_path: Path, then: dict[str, str], on_screen: str
) -> None:
    """The banner that says the chat on screen is not the active one was set only once the chat
    list loaded, so while the list could not load it went on saying "Make this chat active" over
    the active chat: after going back to it, and after making the chat on screen active."""
    chats = [_chat(A, "chat A"), _chat(X, "chat X")]
    steps = [
        {"at": 150, "click": "chat X"},
        {"at": 300, "look": "on X"},
        {"at": 350, "set": {"listFails": True}},
        {"at": 400, **then},
        {"at": 700, "look": "after"},
    ]
    on_x, after = _page(tmp_path, chats, steps)
    assert on_x["banner_hidden"] is False  # X is not the active chat: the banner says so
    assert after["chip"] == on_screen[:8]
    assert after["banner_hidden"] is True
    assert after["placeholder"] == "Message Zak Code…"


# ── a rotation the page replayed, on a real server (ADR-0325) ─────────────────────────────────


def _put_current(base: str, session_id: str | None) -> str:
    deadline = time.monotonic() + 10
    while True:  # a turn's reservation is released just after its save: a 409 can come first
        resp = httpx.put(base + "/sessions/current", json={"session_id": session_id}, timeout=5)
        if resp.status_code != 409 or time.monotonic() > deadline:
            break
        time.sleep(0.05)
    assert resp.status_code == 200, resp.text
    opened: str = resp.json()["id"]
    return opened


def test_a_page_opening_a_chat_made_current_again_does_not_see_it_restart(tmp_path: Path) -> None:
    """A, then a new chat, then A again; then a page opens A. A's bus held the rotation its
    pages were told when it was left, and the page replayed it, found A still current, marked
    A "session restarted" and ended the turn on screen."""
    with _LiveServer(_app(tmp_path)) as base:
        _converse(base, ["one"])
        a = httpx.get(base + "/sessions/current", timeout=5).json()["id"]
        assert _put_current(base, None) != a
        assert _put_current(base, a) == a
        assert _NODE is not None
        script = tmp_path / "page.js"
        script.write_text(_DOM + _TAIL + _page_script(), encoding="utf-8")
        proc = subprocess.run(
            [_NODE, str(script), base, "2", "", "0"], capture_output=True, timeout=60, check=False
        )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    drawn = json.loads(proc.stdout.decode("utf-8"))
    assert drawn["errs"] == [], drawn["errs"]
    texts = [row["text"] for row in drawn["rows"]]
    assert "echo one" in texts  # the page drew the chat
    assert not any("session restarted" in text for text in texts)
