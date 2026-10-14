"""A web page left open while the server restarts catches up instead of going deaf (ADR-0289).

The watch bus (``zakcode.server.event_bus``) is in memory and numbers its events from 1, so a
cursor means something only to the bus that issued it. A page that stays open across a server
restart (``zakcode update`` restarts the server) reconnects holding its old cursor, and the new
bus only reaches that number after that many more events: the page drew nothing new until then.
The server now names each bus (``X-Bus-Epoch``) and the page sends the name back, so a cursor from
a bus that is gone is dropped, and a page that finds itself on a new bus starts over as a freshly
opened one would.

These run the page's OWN script under node against a REAL loopback server, as
test_webclient_reload_history.py does, but keep the page open while the test acts on the server
and on the page's connection. What a restart does to a bus is what the registry's ``discard``
does: the old one is closed and forgotten, the next request makes a new one. The page process is
driven over stdin (drop the connection, go offline, come back online, snapshot what is drawn).
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from types import TracebackType
from typing import Any

import pytest

from tests.test_server_consumer import _LiveServer
from tests.test_webclient_reload_history import (
    _CLOCK,
    _ONE_AND_TWO,
    _app,
    _converse,
    _open_page,
    _transcript,
)
from tests.test_webclient_turn_clock import _NODE, _PRELUDE, _page_script
from zakcode.server import app as server_app
from zakcode.server.event_bus import EventBusRegistry

pytestmark = pytest.mark.skipif(_NODE is None, reason="needs node on PATH (CI runners carry it)")

_ALL_SIX = [*_ONE_AND_TWO, ("user", "three"), ("assistant", "echo three")]

#: Runs after the page's DOM and before its script. `fetch` becomes the real one, pointed at the
#: live server, and records every URL the page asks for. Commands arrive on stdin, one per line,
#: and each is answered with one JSON line: DROP aborts the page's watch connection (a network
#: blip), OFFLINE does that and refuses every later watch connection (a laptop asleep), ONLINE
#: lifts that, SNAP reports the rows drawn, the error cards and the URLs asked for.
_TAIL = """
const __BASE = process.argv[2];
const __realFetch = globalThis.fetch;
const __trace = [];
let __offline = false;
let __watch = null;
globalThis.fetch = (url, opts) => {
  const full = new URL(url, __BASE);
  __trace.push(full.pathname + full.search);
  if (!full.pathname.startsWith("/watch/")) return __realFetch(full, opts);
  if (__offline) return Promise.reject(new TypeError("network is down"));
  __watch = new AbortController();
  return __realFetch(full, { ...opts, signal: __watch.signal });
};
const __log = document.getElementById("log");
const __say = (obj) => process.stdout.write(JSON.stringify(obj) + "\\n");
let __pending = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => {
  __pending += chunk;
  let end;
  while ((end = __pending.indexOf("\\n")) >= 0) {
    const cmd = __pending.slice(0, end).trim();
    __pending = __pending.slice(end + 1);
    if (cmd === "DROP" || cmd === "OFFLINE") {
      __offline = cmd === "OFFLINE";
      if (__watch) __watch.abort();
      __say({ ok: true });
    } else if (cmd === "ONLINE") {
      __offline = false;
      __say({ ok: true });
    } else if (cmd === "SNAP") {
      const rows = __log.children.map((r) => ({
        cls: [...r.classes].filter((c) => c !== "row").join(" "),
        text: r.textContent,
      }));
      const errs = __created
        .filter((n) => n.parent !== null && n.classes.has("errcard"))
        .map((n) => n.textContent);
      __say({ rows, errs, trace: __trace });
    }
  }
});
process.stdin.on("end", () => process.exit(0));
"""


class _OpenPage:
    """The page's script running under node and held open, so the test can act while it is there."""

    def __init__(self, base: str, workdir: Path) -> None:
        assert _NODE is not None
        script = workdir / f"open-page-{time.monotonic_ns()}.js"
        dom = _PRELUDE.split("const __STREAMS = ")[0]
        script.write_text(dom + _TAIL + _page_script(), encoding="utf-8")
        self._errors = workdir / f"{script.stem}.err"
        self._proc = subprocess.Popen(
            [_NODE, str(script), base],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._errors.open("wb"),
            text=True,
            encoding="utf-8",
        )

    def __enter__(self) -> _OpenPage:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.close()  # the page process exits when its stdin ends
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()

    def ask(self, command: str) -> dict[str, Any]:
        """Send one command and return the JSON line the page answers it with."""
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(command + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        assert line, f"the page process ended: {self._errors.read_text(errors='replace')}"
        answer: dict[str, Any] = json.loads(line)
        return answer

    def settle(self, want: int, timeout: float = 15.0) -> dict[str, Any]:
        """Snapshot once `want` user and assistant rows are drawn and nothing has changed for a
        beat, so a late duplicate would still show. Fails with what was drawn if it never is."""
        deadline = time.monotonic() + timeout
        seen, changed = "", time.monotonic()
        while True:
            snap = self.ask("SNAP")
            key = json.dumps(snap["rows"])
            if key != seen:
                seen, changed = key, time.monotonic()
            if len(_spoken(snap)) >= want and time.monotonic() - changed > 0.8:
                return snap
            assert time.monotonic() < deadline, (
                f"wanted {want} rows, the page drew {_spoken(snap)}; asked for {snap['trace']}"
            )
            time.sleep(0.05)

    def wait_for_watches(self, count: int, timeout: float = 15.0) -> dict[str, Any]:
        """Snapshot once the page has asked for the watch stream `count` times."""
        deadline = time.monotonic() + timeout
        while True:
            snap = self.ask("SNAP")
            if len(_watches(snap)) >= count:
                return snap
            assert time.monotonic() < deadline, f"the page asked for {snap['trace']}"
            time.sleep(0.05)


def _spoken(snap: dict[str, Any]) -> list[tuple[str, str]]:
    """The user and assistant rows of a snapshot, as the other page tests read them."""
    rows: list[tuple[str, str]] = []
    for row in snap["rows"]:
        if row["cls"] == "user":
            rows.append(("user", _CLOCK.sub("", row["text"]).removeprefix("›")))
        elif row["cls"] == "assistant":
            rows.append(("assistant", row["text"]))
    return rows


def _watches(snap: dict[str, Any]) -> list[str]:
    return [url for url in snap["trace"] if url.startswith("/watch/")]


def _transcripts(snap: dict[str, Any]) -> list[str]:
    return [url for url in snap["trace"] if url.endswith("/transcript")]


@pytest.fixture
def registries(monkeypatch: pytest.MonkeyPatch) -> list[EventBusRegistry]:
    """Every watch-bus registry the app under test makes, so a test can discard a session's bus."""
    made: list[EventBusRegistry] = []

    def make() -> EventBusRegistry:
        registry = EventBusRegistry()
        made.append(registry)
        return registry

    monkeypatch.setattr(server_app, "EventBusRegistry", make)
    return made


def _restart_the_bus(registries: list[EventBusRegistry], base: str) -> None:
    """What a server restart does to the session's watch bus: it is gone, and the next request
    makes a new one that numbers its events from 1."""
    registries[0].discard(_transcript(base)["session_id"])


@pytest.mark.parametrize("said_before_the_page_reconnects", [False, True])
def test_a_page_left_open_across_a_bus_restart_shows_the_next_turn_as_a_fresh_page_does(
    tmp_path: Path, registries: list[EventBusRegistry], said_before_the_page_reconnects: bool
) -> None:
    with _LiveServer(_app(tmp_path)) as base:
        _converse(base, ["one", "two"])
        with _OpenPage(base, tmp_path) as page:
            assert _spoken(page.settle(4)) == _ONE_AND_TWO
            _restart_the_bus(registries, base)
            if not said_before_the_page_reconnects:
                page.wait_for_watches(2)  # the page is back, on the new (empty) bus
                time.sleep(0.5)
            _converse(base, ["three"])
            drawn = page.settle(6)
        assert drawn["errs"] == []
        assert _spoken(drawn) == _ALL_SIX
        # The page asked for its place by the name of the bus it had held a cursor from, and
        # drew the saved conversation again for the new bus.
        assert "&epoch=" in _watches(drawn)[1]
        assert len(_transcripts(drawn)) == 2
        # What it shows is what a page opened now shows.
        assert _open_page(base, tmp_path, want=6) == _ALL_SIX


def test_a_page_that_was_offline_while_the_server_restarted_catches_up_from_the_saved_session(
    tmp_path: Path, registries: list[EventBusRegistry]
) -> None:
    """The turn the page missed was published only to the bus that died with the restart, so no
    replay can bring it back; the saved session is the one place it survives."""
    with _LiveServer(_app(tmp_path)) as base:
        _converse(base, ["one", "two"])
        with _OpenPage(base, tmp_path) as page:
            assert _spoken(page.settle(4)) == _ONE_AND_TWO
            page.ask("OFFLINE")
            _converse(base, ["three"])  # the page cannot see this turn go by
            _restart_the_bus(registries, base)
            page.ask("ONLINE")
            drawn = page.settle(6)
        assert drawn["errs"] == []
        assert _spoken(drawn) == _ALL_SIX
        assert _open_page(base, tmp_path, want=6) == _ALL_SIX


def test_a_page_whose_connection_drops_resumes_from_its_cursor_on_the_same_bus(
    tmp_path: Path, registries: list[EventBusRegistry]
) -> None:
    """No restart: the same bus answers, so the cursor is good. The page asks only for what
    follows it, keeps what it drew, and draws nothing twice."""
    with _LiveServer(_app(tmp_path)) as base:
        _converse(base, ["one", "two"])
        with _OpenPage(base, tmp_path) as page:
            assert _spoken(page.settle(4)) == _ONE_AND_TWO
            page.ask("DROP")
            reconnected = page.wait_for_watches(2)
            _converse(base, ["three"])
            drawn = page.settle(6)
    assert len(registries) == 1  # the one app; its bus was never replaced
    assert drawn["errs"] == []
    assert _spoken(drawn) == _ALL_SIX
    assert "&since=" in _watches(reconnected)[1] and "&epoch=" in _watches(reconnected)[1]
    assert len(_transcripts(drawn)) == 1  # it never started over
