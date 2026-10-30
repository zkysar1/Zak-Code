"""A turn's footer times the turn, whatever the watch connection does (ADR-0219).

The page times a turn from its user_message's publish stamp. A dropped watch connection is
not a turn end: the page resumes from its cursor and the same turn's done frame arrives on
the new connection, so that footer must still carry the duration. A turn that DOES end
stops the clock, so a later turn that opens without a user_message (one joined after the bus
evicted it) prints no duration rather than one spanning two turns.

These run the page's OWN script, cut from the shipped page, under node, against a DOM just
deep enough for it and a fetch that serves a scripted watch stream per connection. Not a
browser: headless Chrome's virtual clock, which a browser test needs to skip the page's
reconnect backoff, hangs the full Chrome binary the CI runners carry (measured 2026-09-23:
the headless shell drew the footer in 0.4s, full Chromium never exited in 40s). Node is
what CI already runs the page's grammar under (test_webclient_parity.py).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

_INDEX = (
    Path(__file__).resolve().parents[1] / "src" / "zakcode" / "server" / "static" / "index.html"
)
_NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(_NODE is None, reason="needs node on PATH (CI runners carry it)")

_T0 = 1_790_000_000.0  # any fixed epoch: these tests read durations, never the HH:MM stamp
_USAGE = {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10, "cost_usd": 0.0}

#: Runs before the page's script. `__STREAMS` maps the `since` cursor each watch connection
#: asks with ("null" on the first) to the frames that connection carries, served once each.
#: A connection asking for anything else never answers, so once the script is spent the
#: page sits waiting and node's event loop drains, and `beforeExit` reports what was drawn.
_PRELUDE = """
const __created = [];
class __Node {
  constructor(tag, text) {
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.parent = null;
    this.ownText = text === undefined ? "" : String(text);
    this.classes = new Set();
    this.hidden = false;
    this.disabled = false;
    this.value = "";
    this.style = {};
    this.dataset = {};
    this.scrollTop = 0;
    this.scrollHeight = 0;
    this.clientHeight = 0;
    const own = this.classes;
    this.classList = {
      add: (...names) => names.forEach((n) => own.add(n)),
      remove: (...names) => names.forEach((n) => own.delete(n)),
      contains: (n) => own.has(n),
      toggle: (n, force) => {
        const on = force === undefined ? !own.has(n) : !!force;
        if (on) own.add(n); else own.delete(n);
        return on;
      },
    };
    __created.push(this);
  }
  get className() { return [...this.classes].join(" "); }
  set className(v) {
    this.classes.clear();
    for (const n of String(v).split(/\\s+/)) if (n) this.classes.add(n);
  }
  get textContent() { return this.ownText + this.children.map((c) => c.textContent).join(""); }
  set textContent(v) { this.ownText = String(v); this.children = []; }
  appendChild(child) {
    child.remove();
    child.parent = this;
    this.children.push(child);
    return child;
  }
  append(...kids) {
    for (const k of kids) this.appendChild(typeof k === "string" ? new __Node("#text", k) : k);
  }
  insertBefore(child) { return this.appendChild(child); }
  remove() {
    if (this.parent) this.parent.children = this.parent.children.filter((c) => c !== this);
    this.parent = null;
  }
  replaceChildren(...kids) { this.children = []; this.append(...kids); }
  addEventListener() {}
  removeEventListener() {}
  setAttribute(k, v) { this[k] = String(v); }
  getAttribute(k) { return k in this ? this[k] : null; }
  removeAttribute(k) { delete this[k]; }
  focus() {}
  blur() {}
  click() {}
  scrollTo() {}
  scrollIntoView() {}
  querySelector() { return null; }
  querySelectorAll() { return []; }
  closest() { return null; }
  getBoundingClientRect() { return { top: 0, bottom: 0, left: 0, right: 0, width: 0, height: 0 }; }
}
// The page's own elements, each made when first asked for: in the document, or inside the one
// the page's markup puts it in. Found by id only while in the document, as a browser finds
// them: once add() takes the start screen out of the page, it and what is inside it are not.
const __body = new __Node("body");
const __INSIDE = { "empty-meta": "empty" };
const __byId = new Map();
function __element(id) {
  if (!__byId.has(id)) {
    const node = new __Node("div");
    (id in __INSIDE ? __element(__INSIDE[id]) : __body).appendChild(node);
    __byId.set(id, node);
  }
  return __byId.get(id);
}
const document = {
  getElementById(id) {
    const node = __element(id);
    for (let n = node; n; n = n.parent) if (n === __body) return node;
    return null;
  },
  createElement: (tag) => new __Node(tag),
  createTextNode: (text) => new __Node("#text", text),
  createDocumentFragment: () => new __Node("#fragment"),
  querySelectorAll: () => [],
  addEventListener() {},
  activeElement: null,
};
const navigator = { clipboard: { writeText: async () => {} } };

const __STREAMS = JSON.parse(process.argv[2]);
// The watch bus that serves each connection, keyed like `__STREAMS` and sent as X-Bus-Epoch. A
// connection with no entry is served with no name, as an older server does (ADR-0289).
const __EPOCHS = JSON.parse(process.argv[3] || "{}");
const __SESSION = { id: "abcdef0123456789", model: "fake-model" };
function __sse(frames) {
  // CRLF framing, as sse-starlette sends it.
  return frames.map(([id, f]) => `id: ${id}\\r\\ndata: ${JSON.stringify(f)}\\r\\n\\r\\n`).join("");
}
globalThis.fetch = async (url) => {
  const u = new URL(url, "http://page.test");
  if (u.pathname === "/sessions/current") {
    return { ok: true, status: 200, json: async () => __SESSION };
  }
  if (u.pathname === "/sessions/" + __SESSION.id + "/transcript") {
    // A session with nothing saved: the page draws no history, only the scripted stream.
    const empty = { session_id: __SESSION.id, messages: [], message_count: 0, bus_from: null };
    return { ok: true, status: 200, json: async () => empty };
  }
  if (u.pathname === "/watch/" + __SESSION.id) {
    const key = String(u.searchParams.get("since"));
    if (!(key in __STREAMS)) return new Promise(() => {});
    const bytes = new TextEncoder().encode(__sse(__STREAMS[key]));
    delete __STREAMS[key];
    let sent = false;
    const read = async () => {
      if (sent) return { done: true };
      sent = true;
      return { done: false, value: bytes };
    };
    // A connection with no name scripted is served as an older server serves it, with no
    // X-Bus-Epoch header: the page must resume by cursor alone, as before servers named buses.
    const epoch = key in __EPOCHS ? __EPOCHS[key] : null;
    const headers = { get: (name) => (name.toLowerCase() === "x-bus-epoch" ? epoch : null) };
    return { ok: true, status: 200, headers, body: { getReader: () => ({ read }) } };
  }
  return { ok: false, status: 404, json: async () => ({}) };
};
// The reconnect backoff is a wait, not behaviour under test: take every timer at once.
const __setTimeout = setTimeout;
globalThis.setTimeout = (fn, _ms, ...args) => __setTimeout(fn, 0, ...args);

process.once("beforeExit", () => {
  const attached = __created.filter((n) => n.parent !== null);
  const footers = attached
    .filter((n) => n.classes.has("turnend"))
    .map((row) => row.children.find((c) => c.classes.has("receipt")).textContent);
  const errors = attached.filter((n) => n.classes.has("errcard")).map((n) => n.textContent);
  process.stdout.write(JSON.stringify({ footers, errors }));
});
"""


def _page_script() -> str:
    html = _INDEX.read_text(encoding="utf-8")
    scripts = re.findall(r"<script>\n(.*?)</script>", html, re.S)
    assert len(scripts) == 1, "the page carries exactly one inline script"
    return scripts[0]


def _footers(
    streams: dict[str, list[tuple[str, dict[str, Any]]]],
    tmp_path: Path,
    epochs: dict[str, str] | None = None,
) -> list[str]:
    """Run the page against the scripted watch streams; return every turn footer it drew.

    `epochs` names the watch bus that serves each connection, keyed like `streams`; a connection
    it does not list is served with no bus name, as an older server serves it."""
    assert _NODE is not None
    script = tmp_path / "page.js"
    script.write_text(_PRELUDE + _page_script(), encoding="utf-8")
    proc = subprocess.run(
        [_NODE, str(script), json.dumps(streams), json.dumps(epochs or {})],
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    drawn = json.loads(proc.stdout.decode("utf-8"))
    # A page error surfaces as an error card, and a DOM this fake lacks would too, as
    # "[bad frame]": either one means the footers below were not the page's whole story.
    assert drawn["errors"] == [], drawn["errors"]
    footers: list[str] = drawn["footers"]
    return footers


def _is_duration(part: str) -> bool:
    # The terminal's duration shapes: "0.0s", "12.3s", "1m 15s".
    return re.fullmatch(r"(\d+m )?\d+(\.\d+)?s", part) is not None


def _done(at: float) -> dict[str, Any]:
    return {
        "event": "done",
        "stop_reason": "completed",
        "iterations": 2,
        "usage": _USAGE,
        "at": at,
    }


def test_a_turn_that_survives_a_reconnect_keeps_its_duration(tmp_path: Path) -> None:
    streams = {
        "null": [
            ("1", {"event": "user_message", "text": "hello", "at": _T0}),
            ("2", {"event": "text", "text": "working on it", "at": _T0 + 1.0}),
        ],
        # The first connection drops after frame 2; the resume carries the rest of the SAME
        # turn, so its done frame is timed from the user_message on the first connection.
        "2": [("3", _done(_T0 + 75.3))],
    }
    [footer] = _footers(streams, tmp_path)
    assert footer.startswith("done · 2 iterations · 10 tokens · "), footer
    assert "1m 15s" in footer.split(" · "), footer


def test_a_turn_that_ends_stops_its_clock(tmp_path: Path) -> None:
    streams = {
        "null": [
            ("1", {"event": "user_message", "text": "hello", "at": _T0}),
            ("2", _done(_T0 + 75.3)),
            # A second turn with no user_message of its own (one joined after the bus evicted it).
            ("3", {"event": "text", "text": "a turn nobody timed", "at": _T0 + 200.0}),
            ("4", _done(_T0 + 230.0)),
        ],
    }
    first, second = _footers(streams, tmp_path)
    # Positive control: the timed turn prints its duration, so an untimed footer below is
    # the clock having stopped, not the page failing to draw durations at all.
    assert "1m 15s" in first.split(" · "), first
    assert not any(_is_duration(part) for part in second.split(" · ")), second
    assert len(second.split(" · ")) == len(first.split(" · ")) - 1, (first, second)


def test_a_turn_the_old_bus_took_with_it_does_not_lend_its_clock_or_meter_to_the_next(
    tmp_path: Path,
) -> None:
    """A server restart ends the turn that was running: nothing will finish it. The page that
    starts over on the new bus must not carry that turn's clock and token meter into the next
    turn, which may open with no user_message of its own: its footer would time and count from a
    turn that no longer exists (ADR-0289). On the SAME bus the turn is the same turn, and both
    carry on (above)."""
    usage = {"event": "usage", "usage": {"total_tokens": 7, "prompt_tokens": 5}, "at": _T0 + 1.0}
    dead_turn = [
        ("1", {"event": "user_message", "text": "hello", "at": _T0}),
        ("2", usage),
        ("3", {"event": "text", "text": "working on it", "at": _T0 + 2.0}),
    ]

    def next_turn(first_cursor: int) -> list[tuple[str, dict[str, Any]]]:
        text = {"event": "text", "text": "a turn nobody timed", "at": _T0 + 200.0}
        done = {"event": "done", "stop_reason": "completed", "iterations": 2, "at": _T0 + 230.0}
        return [(str(first_cursor), text), (str(first_cursor + 1), done)]

    # The connection ends after cursor 3 and the page asks to resume past it, on a bus that is
    # the same (cursors go on) or another (cursors start again from 1).
    [same_bus] = _footers(
        {"null": dead_turn, "3": next_turn(4)}, tmp_path, {"null": "bus-a", "3": "bus-a"}
    )
    [restarted] = _footers(
        {"null": dead_turn, "3": next_turn(1)}, tmp_path, {"null": "bus-a", "3": "bus-b"}
    )
    [opened_on_the_new_bus] = _footers({"null": next_turn(1)}, tmp_path)
    # Positive control: the same turn carries its clock and meter across a reconnect to its bus,
    # so the restarted page's footer being free of both is the page starting over.
    assert "3m 50s" in same_bus.split(" · "), same_bus
    assert "7 tokens" in same_bus, same_bus
    assert restarted == opened_on_the_new_bus


def test_a_cancelled_turn_says_when_its_cost_could_not_be_priced(tmp_path: Path) -> None:
    """A cancel sends no done frame, so the page draws that footer from the usage frames it
    summed. The sum keeps `cost_unpriced` as the terminal's `Usage.__add__` does (ADR-0302): one
    call that could not be priced makes the figure a floor, and none priced leaves no figure."""

    def cancelled(*usages: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        frames = [("1", {"event": "user_message", "text": "hello", "at": _T0})]
        for n, usage in enumerate(usages, start=2):
            frames.append((str(n), {"event": "usage", "usage": usage, "at": _T0 + n}))
        end = {"event": "status", "message": "interrupted", "at": _T0 + 10.0}
        return [*frames, (str(len(frames) + 1), end)]

    priced = {"total_tokens": 10, "prompt_tokens": 8, "cost_usd": 0.0213}
    unpriced = {"total_tokens": 10, "prompt_tokens": 8, "cost_usd": 0.0, "cost_unpriced": True}
    [both_priced] = _footers({"null": cancelled(priced, priced)}, tmp_path)
    [none_priced] = _footers({"null": cancelled(unpriced, unpriced)}, tmp_path)
    [one_priced] = _footers({"null": cancelled(priced, unpriced)}, tmp_path)
    # Positive control: a priced cancel prints its figure, so the words below are the flag.
    assert "$0.0426" in both_priced.split(" · "), both_priced
    assert "unpriced" not in both_priced, both_priced
    assert "unpriced" in none_priced.split(" · "), none_priced
    assert "at least $0.0213, some calls unpriced" in one_priced.split(" · "), one_priced
