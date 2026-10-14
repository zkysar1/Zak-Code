"""THROWAWAY diagnostic, never merged.

A Windows CI run of test_a_page_reopened_when_the_bus_kept_only_the_newest_turn_draws_the_older_ones
_from_disk drew the newest turn twice. This repeats that test and records, for every run whose rows
differ from the expected ones, what the page fetched and when, the transcript the page was given, and
the watch replay the server holds. It also records, for every run, how the saved messages' stamps sit
against the start of the bus, so a tie or an inversion shows even when no run goes wrong.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.test_server_consumer import _LiveServer
from tests.test_webclient_reload_history import _CLOCK, _ONE_AND_TWO, _TAIL, _app, _converse
from tests.test_webclient_turn_clock import _NODE, _PRELUDE, _page_script
from zakcode.server import app as server_app
from zakcode.server.event_bus import EventBusRegistry

pytestmark = pytest.mark.skipif(_NODE is None, reason="needs node on PATH")

RUNS = 240
_EXPECTED = [*_ONE_AND_TWO, ("user", "three"), ("assistant", "echo three")]
_FETCH = "globalThis.fetch = (url, opts) => __realFetch(new URL(url, __BASE), opts);"
_WRITE = "process.stdout.write(JSON.stringify({ rows, errs }));"

_TRACED = _TAIL.replace(
    _FETCH,
    "const __trace = []; const __t00 = Date.now();\n"
    "globalThis.fetch = (url, opts) => {\n"
    "  __trace.push([Date.now() - __t00, 'fetch', String(url)]);\n"
    "  const p = __realFetch(new URL(url, __BASE), opts);\n"
    "  if (String(url).includes('/transcript')) {\n"
    "    p.then((r) => r.clone().json()).then((j) => __trace.push([Date.now() - __t00,\n"
    "      'transcript', j.bus_from, j.messages.map((m) => [m.role, m.text, m.at])]));\n"
    "  }\n"
    "  return p;\n"
    "};",
).replace(_WRITE, "process.stdout.write(JSON.stringify({ rows, errs, trace: __trace }));")


def _open_traced(base: str, workdir: Path, want: int) -> dict[str, Any]:
    assert _NODE is not None
    script = workdir / f"page-{time.monotonic_ns()}.js"
    dom = _PRELUDE.split("const __STREAMS = ")[0]
    script.write_text(dom + _TRACED + _page_script(), encoding="utf-8")
    proc = subprocess.run(
        [_NODE, str(script), base, str(want), "", "0"],
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    out: dict[str, Any] = json.loads(proc.stdout.decode("utf-8"))
    return out


def _rows(drawn: dict[str, Any]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for row in drawn["rows"]:
        if row["cls"] == "user":
            rows.append(("user", _CLOCK.sub("", row["text"]).removeprefix("›")))
        elif row["cls"] == "assistant":
            rows.append(("assistant", row["text"]))
    return rows


def _snapshot(base: str) -> dict[str, Any]:
    """The transcript and the watch replay as the server holds them now."""
    resp = httpx.get(base + "/sessions/current/transcript", timeout=5)
    transcript: dict[str, Any] = resp.json()
    frames: list[list[Any]] = []
    with httpx.stream(
        "GET",
        f"{base}/watch/{transcript['session_id']}?full=1",
        timeout=httpx.Timeout(10.0, read=1.5),
    ) as stream:
        cursor = None
        try:
            for line in stream.iter_lines():
                if line.startswith("id:"):
                    cursor = line[3:].strip()
                elif line.startswith("data:"):
                    doc = json.loads(line[5:])
                    frames.append([cursor, doc.get("at"), doc.get("event")])
        except httpx.ReadTimeout:
            pass
    return {
        "bus_from": transcript["bus_from"],
        "messages": [[m["role"], m["text"], m["at"]] for m in transcript["messages"]],
        "frames": frames,
    }


def test_the_ring_of_three_page_repeated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_app, "EventBusRegistry", lambda: EventBusRegistry(maxlen=3))
    print("DEBUG python", sys.version.split()[0], "clock", time.get_clock_info("time"), flush=True)
    bad: list[dict[str, Any]] = []
    margins: list[list[float | None]] = []
    for i in range(RUNS):
        workdir = tmp_path / f"run{i}"
        workdir.mkdir()
        with _LiveServer(_app(workdir)) as base:
            _converse(base, ["one", "two", "three"])
            drawn = _open_traced(base, workdir, want=6)
            rows = _rows(drawn)
            seen = next((t for t in drawn["trace"] if t[1] == "transcript"), None)
            if seen is not None and seen[2] is not None:
                start = seen[2]
                # messages: one, echo one, two, echo two, three, echo three
                margins.append([round((m[2] - start) * 1e6) for m in seen[3]])
            if rows != _EXPECTED:
                bad.append(
                    {
                        "run": i,
                        "rows": rows,
                        "errs": drawn["errs"],
                        "trace": drawn["trace"],
                        "after": _snapshot(base),
                    }
                )
    ties = sum(1 for m in margins if m[4] == 0 or m[5] == 0)
    before = sum(1 for m in margins if m[4] < 0 or m[5] < 0)
    print(
        f"DEBUG-RELOAD runs={RUNS} bad={len(bad)} turn3-ties={ties} turn3-before-bus={before}",
        flush=True,
    )
    print("DEBUG-RELOAD margins (microseconds from bus start, per message):", flush=True)
    for m in margins[:20]:
        print("   ", m, flush=True)
    assert not bad, json.dumps(bad)[:24000]
