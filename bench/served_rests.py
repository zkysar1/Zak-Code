"""Idle windows of one served run, read off its logs the way the sample-5 control block read them.

A WINDOW opens at a ``turn ended:`` line in serve.log and closes at the next turn's boot (the
routing line the web app writes as a turn starts) or at the run-stop request, whichever comes
first. Nothing after the run-stop request is counted: that is teardown, and the control block
excluded it too. The run's length is launch (the ``[run]`` line in progress.log) to the run-stop
request; when no stop was ever requested, to the last timestamped line, and the output says so.

The rest reader registered with served sample 6 (bench/results/served-luna-preregistration.log).
Its known answer is the sample-5 control: windows of 600.30, 600.29 and 302.06 s, 1502.65 of
2054.25 s.

Reads timestamps and stop reasons only. Prints no prompt text.

usage: python bench/served_rests.py <world-dir>
"""

from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3}) ")
ENDED = re.compile(r"zakcode\.agent\.loop turn ended: stop_reason=(\w+) iterations=(\d+)")
BOOT = "zakcode zakpick: routing per task category"
STOP = "zakcode.session.framework_stop framework stop requested"
LAUNCH = re.compile(r"^\[run\] (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})Z zakcode source=(\S+)")


def _at(line: str) -> float | None:
    m = STAMP.match(line)
    if not m:
        return None
    base = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    return base.timestamp() + int(m.group(2)) / 1000.0


def _clock(t: float) -> str:
    return datetime.fromtimestamp(t, tz=UTC).strftime("%H:%M:%S.%f")[:-3]


def read(world: Path) -> dict:
    launch = source = None
    for line in (world / "logs" / "progress.log").read_text(errors="replace").splitlines():
        m = LAUNCH.match(line)
        if m:
            launch = (
                datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC).timestamp()
            )
            source = m.group(2)
            break
    events: list[tuple[float, str, dict]] = []
    last = None
    for line in (world / "logs" / "serve.log").read_text(errors="replace").splitlines():
        t = _at(line)
        if t is None:
            continue
        last = t
        m = ENDED.search(line)
        if m:
            events.append((t, "end", {"stop_reason": m.group(1), "iterations": int(m.group(2))}))
        elif BOOT in line:
            events.append((t, "boot", {}))
        elif STOP in line:
            events.append((t, "stop", {}))
    stop_at = next((t for t, kind, _ in events if kind == "stop"), None)
    run_end = stop_at if stop_at is not None else last
    turns, windows = [], []
    open_window = None
    for t, kind, data in events:
        if kind == "end":
            turns.append(
                {
                    "n": len(turns) + 1,
                    "ended": _clock(t),
                    **data,
                    "before_run_stop": stop_at is None or t <= stop_at,
                }
            )
            if stop_at is None or t <= stop_at:
                open_window = (t, len(turns), data["stop_reason"])
        elif kind in ("boot", "stop") and open_window is not None:
            start, n, reason = open_window
            windows.append(
                {
                    "after_turn": n,
                    "stop_reason": reason,
                    "from": _clock(start),
                    "to": _clock(t),
                    "closed_by": "next_turn_boot" if kind == "boot" else "run_stop",
                    "seconds": round(t - start, 2),
                }
            )
            open_window = None
    if open_window is not None and run_end is not None:
        start, n, reason = open_window
        windows.append(
            {
                "after_turn": n,
                "stop_reason": reason,
                "from": _clock(start),
                "to": _clock(run_end),
                "closed_by": "log_end",
                "seconds": round(run_end - start, 2),
            }
        )
    run_seconds = round(run_end - launch, 2) if launch is not None and run_end is not None else None
    idle = round(sum(w["seconds"] for w in windows), 2)
    idle_stall = round(sum(w["seconds"] for w in windows if w["stop_reason"] == "veto_stall"), 2)
    return {
        "world": world.name,
        "zakcode_source": source,
        "launch": _clock(launch) if launch is not None else None,
        "run_stop_requested": _clock(stop_at) if stop_at is not None else None,
        "run_seconds": run_seconds,
        "turns": turns,
        "veto_stall_turns_before_run_stop": sum(
            1 for x in turns if x["stop_reason"] == "veto_stall" and x["before_run_stop"]
        ),
        "windows": windows,
        "idle_seconds": idle,
        "idle_share": round(idle / run_seconds, 4) if run_seconds else None,
        "idle_after_veto_stall_seconds": idle_stall,
        "idle_after_veto_stall_share": round(idle_stall / run_seconds, 4) if run_seconds else None,
    }


if __name__ == "__main__":
    print(json.dumps(read(Path(sys.argv[1])), indent=2))
