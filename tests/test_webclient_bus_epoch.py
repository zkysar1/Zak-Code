"""A web page left open while the server restarts catches up instead of going deaf (ADR-0289).

The watch bus (``zakcode.server.event_bus``) is in memory and numbers its events from 1, so a
cursor means something only to the bus that issued it. A page that stays open across a server
restart reconnects holding its old cursor, and the new bus only reaches that number after that
many more events: the page drew nothing new until then.
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
from collections.abc import Callable, Iterator
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
from zakcode.server.event_bus import EventBusRegistry, SessionEventBus

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


#: Seconds between two snapshots of the page.
_POLL = 0.05


class _OpenPage:
    """The page's script running under node and held open, so the test can act while it is there."""

    def __init__(self, base: str, workdir: Path, tail: str = _TAIL) -> None:
        assert _NODE is not None
        script = workdir / f"open-page-{time.monotonic_ns()}.js"
        dom = _PRELUDE.split("const __STREAMS = ")[0]
        script.write_text(dom + tail + _page_script(), encoding="utf-8")
        self._errors = workdir / f"{script.stem}.err"
        self._errors_out = self._errors.open("wb")
        self._proc = subprocess.Popen(
            [_NODE, str(script), base],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._errors_out,
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
        try:
            assert self._proc.stdin is not None
            self._proc.stdin.close()  # the page process exits when its stdin ends
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        finally:
            # Both ends this class opened, closed here and not left to the garbage collector,
            # which reports each as a ResourceWarning under `-X dev`.
            if self._proc.stdout is not None:
                self._proc.stdout.close()
            self._errors_out.close()

    def ask(self, command: str) -> dict[str, Any]:
        """Send one command and return the JSON line the page answers it with."""
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(command + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        assert line, f"the page process ended: {self._errors.read_text(errors='replace')}"
        answer: dict[str, Any] = json.loads(line)
        return answer

    def _snapshots(self, timeout: float) -> Iterator[dict[str, Any]]:
        """Snapshots of the page, one every `_POLL` seconds, as many as `timeout` seconds hold.

        The wait is counted in snapshots and not read off the clock. A snapshot is a round trip
        to the page's process, and a runner that holds that process up makes the answer late; a
        deadline on the clock charged that delay to the page. On one Windows run the first
        answer came back 23 s after a 15 s deadline began, showing a page that had drawn nothing
        and asked for one URL, and the test failed a page that had only just started. A late
        answer is now one snapshot, and a page that never gets there still runs out.
        """
        for _ in range(round(timeout / _POLL)):
            yield self.ask("SNAP")
            time.sleep(_POLL)

    def settle(self, want: int, timeout: float = 15.0) -> dict[str, Any]:
        """Snapshot once `want` user and assistant rows are drawn and nothing has changed for a
        beat, so a late duplicate would still show. Fails with what was drawn if it never is."""
        started = time.monotonic()
        seen, changed = "", started
        snap: dict[str, Any] = {"rows": [], "trace": []}
        for snap in self._snapshots(timeout):
            key = json.dumps(snap["rows"])
            if key != seen:
                seen, changed = key, time.monotonic()
            if len(_spoken(snap)) >= want and time.monotonic() - changed > 0.8:
                return snap
        raise AssertionError(
            f"wanted {want} rows, the page drew {_spoken(snap)}; asked for {snap['trace']}; "
            f"waited {time.monotonic() - started:.0f} s"
        )

    def wait_for_watches(self, count: int, timeout: float = 15.0) -> dict[str, Any]:
        """Snapshot once the page has asked for the watch stream `count` times."""
        return self._wait_for(lambda snap: len(_watches(snap)) >= count, timeout)

    def wait_for_transcripts(self, count: int, timeout: float = 15.0) -> dict[str, Any]:
        """Snapshot once the page has asked for the saved conversation `count` times. Its second
        ask is the page starting over: the start screen is back by then, the history not yet."""
        return self._wait_for(lambda snap: len(_transcripts(snap)) >= count, timeout)

    def _wait_for(self, done: Callable[[dict[str, Any]], bool], timeout: float) -> dict[str, Any]:
        started = time.monotonic()
        snap: dict[str, Any] = {"trace": []}
        for snap in self._snapshots(timeout):
            if done(snap):
                return snap
        raise AssertionError(
            f"the page asked for {snap['trace']}; waited {time.monotonic() - started:.0f} s"
        )


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


class _Registry(EventBusRegistry):
    """The app's watch-bus registry, which a test can also arm to lose a bus at a chosen moment.

    A restart between the page's two first requests (the saved conversation, then the watch)
    answers the first from the old bus and the second from a new one. Armed, this forgets the
    session's bus the instant the transcript route has read it: the route is handed the old bus
    and the next request makes a new one."""

    def __init__(self) -> None:
        super().__init__()
        self.lose_the_bus_once_it_is_read = False

    def get(self, session_id: str) -> SessionEventBus | None:
        bus = super().get(session_id)
        if bus is not None and self.lose_the_bus_once_it_is_read:
            self.lose_the_bus_once_it_is_read = False
            self.discard(session_id)
        return bus


@pytest.fixture
def registries(monkeypatch: pytest.MonkeyPatch) -> list[_Registry]:
    """Every watch-bus registry the app under test makes, so a test can discard a session's bus."""
    made: list[_Registry] = []

    def make() -> _Registry:
        registry = _Registry()
        made.append(registry)
        return registry

    monkeypatch.setattr(server_app, "EventBusRegistry", make)
    return made


def _restart_the_bus(registries: list[_Registry], base: str) -> None:
    """What a server restart does to the session's watch bus: it is gone, and the next request
    makes a new one that numbers its events from 1."""
    registries[0].discard(_transcript(base)["session_id"])


@pytest.mark.parametrize("said_before_the_page_reconnects", [False, True])
def test_a_page_left_open_across_a_bus_restart_shows_the_next_turn_as_a_fresh_page_does(
    tmp_path: Path, registries: list[_Registry], said_before_the_page_reconnects: bool
) -> None:
    with _LiveServer(_app(tmp_path)) as base:
        _converse(base, ["one", "two"])
        with _OpenPage(base, tmp_path) as page:
            assert _spoken(page.settle(4)) == _ONE_AND_TWO
            _restart_the_bus(registries, base)
            if not said_before_the_page_reconnects:
                page.wait_for_transcripts(2)  # the page is back on the new (empty) bus ...
                assert _spoken(page.settle(4)) == _ONE_AND_TWO  # ... and has started over
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


def test_a_page_that_started_over_forgets_the_old_cursor_when_its_connection_drops_again(
    tmp_path: Path, registries: list[_Registry]
) -> None:
    """The new bus is empty when the page starts over, so the page has no cursor of its own yet.
    If its connection drops again before the first frame, the reconnect must not pair the OLD
    bus's cursor with the NEW bus's name: the name matches, so the server would honour the cursor
    and the page would skip every event up to a number it never saw on this bus: deaf again."""
    with _LiveServer(_app(tmp_path)) as base:
        _converse(base, ["one", "two"])
        with _OpenPage(base, tmp_path) as page:
            assert _spoken(page.settle(4)) == _ONE_AND_TWO
            _restart_the_bus(registries, base)
            page.wait_for_transcripts(2)  # the page is back on the new (empty) bus ...
            assert _spoken(page.settle(4)) == _ONE_AND_TWO  # ... and has started over
            page.ask("DROP")  # its connection drops before the new bus has sent it a frame
            reconnected = page.wait_for_watches(3)
            _converse(base, ["three"])
            drawn = page.settle(6)
    assert drawn["errs"] == []
    assert _spoken(drawn) == _ALL_SIX
    third = _watches(reconnected)[2]
    assert "&epoch=" in third and "&since=" not in third, third


def test_a_bus_replaced_between_the_pages_history_and_its_watch_does_not_cost_it_the_history(
    tmp_path: Path, registries: list[_Registry]
) -> None:
    """The page asks for the saved conversation, then for the watch. The history leaves the turns
    the bus holds to the bus's replay, so a bus replaced between the two requests (a restart) is
    one that holds none of them: the page would show an empty conversation. The transcript names
    the bus it was cut against, so the page sees the watch is another and starts over."""
    with _LiveServer(_app(tmp_path)) as base:
        _converse(base, ["one", "two"])
        registries[0].lose_the_bus_once_it_is_read = True  # the next read is the page's
        with _OpenPage(base, tmp_path) as page:
            drawn = page.settle(4)
        assert drawn["errs"] == []
        assert _spoken(drawn) == _ONE_AND_TWO
        assert len(_transcripts(drawn)) == 2  # the second history is cut against the new bus
        assert _open_page(base, tmp_path, want=4) == _ONE_AND_TWO


def test_a_page_that_was_offline_while_the_server_restarted_catches_up_from_the_saved_session(
    tmp_path: Path, registries: list[_Registry]
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
    tmp_path: Path, registries: list[_Registry]
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


# The page harness's own waits. These drive the real `settle` and `wait_for_watches` over answers
# the test scripts and a clock it moves by hand, so a runner's stall is reproduced exactly and
# costs no time.


class _HandClock:
    """Time that moves only when the test moves it: sleeping moves it, and so does a late answer."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def hand_clock(monkeypatch: pytest.MonkeyPatch) -> _HandClock:
    """This module's `time`, swapped for a clock that moves only when told to."""
    clock = _HandClock()
    monkeypatch.setattr(f"{__name__}.time", clock)
    return clock


class _StubPage(_OpenPage):
    """The real waits over scripted answers, with no node process behind them."""

    def __init__(self, clock: _HandClock, answers: list[dict[str, Any]], held_up: float) -> None:
        self._clock = clock
        self._answers = answers  # the last one repeats
        self._held_up = held_up  # seconds the runner keeps the page before its first answer
        self._asked = 0

    def ask(self, command: str) -> dict[str, Any]:
        if self._asked == 0:
            self._clock.now += self._held_up
        answer = self._answers[min(self._asked, len(self._answers) - 1)]
        self._asked += 1
        return answer


def _page_showing(rows: list[tuple[str, str]], asked: tuple[str, ...] = ()) -> dict[str, Any]:
    """A snapshot with these rows drawn, spelled the way the page's own rows are."""
    return {
        "rows": [
            {"cls": role, "text": ("›" + text) if role == "user" else text} for role, text in rows
        ],
        "trace": list(asked),
    }


def test_a_first_answer_the_runner_held_up_is_not_the_page_failing_to_settle(
    hand_clock: _HandClock,
) -> None:
    """The run that failed on Windows: a 15 s wait, a first snapshot back 23 s later that shows a
    page that has only just started, and the rows drawn by the next."""
    just_started = _page_showing([], ("/sessions/current",))
    page = _StubPage(hand_clock, [just_started, _page_showing(_ONE_AND_TWO)], held_up=23.0)
    assert _spoken(page.settle(4)) == _ONE_AND_TWO


def test_a_first_answer_the_runner_held_up_is_not_the_page_never_asking_for_the_watch(
    hand_clock: _HandClock,
) -> None:
    just_started = _page_showing([], ("/sessions/current",))
    watching = _page_showing([], ("/sessions/current", "/watch/s1"))
    page = _StubPage(hand_clock, [just_started, watching], held_up=23.0)
    assert _watches(page.wait_for_watches(1)) == ["/watch/s1"]


def test_a_page_that_never_draws_the_rows_still_runs_out_of_time(hand_clock: _HandClock) -> None:
    page = _StubPage(hand_clock, [_page_showing(_ONE_AND_TWO[:2])], held_up=0.0)
    with pytest.raises(AssertionError, match=r"wanted 4 rows, .*waited 15 s"):
        page.settle(4)


def test_a_page_that_never_asks_for_the_watch_still_runs_out_of_time(
    hand_clock: _HandClock,
) -> None:
    page = _StubPage(hand_clock, [_page_showing([], ("/sessions/current",))], held_up=0.0)
    with pytest.raises(AssertionError, match=r"the page asked for .*waited 15 s"):
        page.wait_for_watches(1)
