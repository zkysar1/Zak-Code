"""What each refused stop on a served run ANSWERED, and what getting back to work cost.

A reader for runs already made (bench/results/served-luna-preregistration.log). It reads three
things the run wrote itself and prints one JSON reading. Labels, counts and clock times only: no
prompt, completion or command text is read or printed.

usage: python bench/served_stops.py <world-dir> [--skill NAME]
       python bench/served_stops.py --selftest          known-answer checks on synthetic logs

REQUESTS (per-turn traces). Every main-loop request is a `usage` event, and since ADR-0204 and
ADR-0205 it says which plan reminder rode that request: the finished plan's answer-now line SENT
(`plan_complete` in `rails`), that line kept SILENT (`plan_complete` in `rails_silenced`), the
OPEN checklist (`plan` in `rails`), a RESTED tail (`rails_rested`), or NONE. What answered it is
read from the events up to the next request: TOOLS; a stop IN WORDS that a turn-end hook refused
(a `turn_end_skill` delivery note with no tool event before it); a HARNESS stop the hook refused
(the same note after tool events: the stuck ladder ended the turn, the model did not); the turn's
END; or OTHER. A trace whose usage events carry no `rails` key is older than ADR-0204 and is
refused, not guessed.

RE-ENTRIES (serve.log joined to the traces). The trace names the TOOL of each call, never the
skill; serve.log names each skill as it answers, in order. The k-th Skill call the trace marks ok
is the k-th skill line of serve.log, and the join is REFUSED unless the two counts are equal. A
MODEL-MADE RE-ENTRY is an ok Skill call for --skill (the loop's own skill) that was answered with
a body, not the already-loaded pointer. For each: whether a refused stop came before the next
Skill call, how many requests went out before it, and the reminder state of the last of them.

BACK TO WORK (serve.log). An interval opens at a refused stop and closes at the next load of a
skill OTHER than --skill: the loop has moved past re-entry. Refused stops inside an open interval
do not reopen it. Only intervals that open before the run-stop request count, and one still open
then is closed by it. The run is launch (progress.log) to the run-stop request, as in
served_rests.py.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from zakcode.agent.loop import _SKILL_TOOLS

STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3}) ")
SKILL_NAME = re.compile(r"skill '([^']+)'")
LAUNCH = re.compile(r"^\[run\] (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})Z zakcode source=")
DOOR = "TURN_END hook vetoed"
LOADED = "invoked via use_skill"
POINTERS = ("use_skill deduped", "use_skill answered")
RUN_STOP = "zakcode.session.framework_stop framework stop requested"
STATES = ("sent", "silent", "open", "rested", "none")
ANSWERS = ("tools", "words_refused", "harness_stop_refused", "turn_end", "other")


def _at(line: str) -> float | None:
    m = STAMP.match(line)
    if not m:
        return None
    base = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    return base.timestamp() + int(m.group(2)) / 1000.0


def _clock(t: float) -> str:
    return datetime.fromtimestamp(t, tz=UTC).strftime("%H:%M:%S")


def _state(data: dict) -> str:
    """Which plan reminder rode one request, from its own `usage` event."""
    rails = data.get("rails") or []
    if "plan_complete" in rails:
        return "sent"
    if "plan_complete" in (data.get("rails_silenced") or []):
        return "silent"
    if "plan" in rails:
        return "open"
    return "rested" if data.get("rails_rested") else "none"


def _is_door(event: dict) -> bool:
    return event.get("kind") == "intervention" and (event.get("data") or {}).get("kind") == (
        "turn_end_skill"
    )


def _is_skill_call(event: dict) -> bool:
    name = str(event.get("detail") or "").split(" ", 1)[0].split(":", 1)[0].split("(", 1)[0]
    return event.get("kind") == "tool" and name in _SKILL_TOOLS


def read_events(world: Path) -> list[dict]:
    """Every trace event of the run, turns in order (turn_2 before turn_10)."""
    files = sorted(
        (world / "logs" / "traces").rglob("turn_*.jsonl"),
        key=lambda p: (str(p.parent), int(re.sub(r"\D", "", p.stem) or 0)),
    )
    events: list[dict] = []
    for path in files:
        for line in path.read_text(errors="replace").splitlines():
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def read_requests(events: list[dict]) -> dict:
    """The reminder state of every request against what answered it, and the same per door."""
    usage = [i for i, e in enumerate(events) if e.get("kind") == "usage"]
    if usage and not any("rails" in (events[i].get("data") or {}) for i in usage):
        return {"refused": "no usage event carries `rails`: a build older than ADR-0204"}
    table: Counter[tuple[str, str]] = Counter()
    doors: Counter[tuple[str, str]] = Counter()
    for n, i in enumerate(usage):
        state = _state(events[i].get("data") or {})
        end = usage[n + 1] if n + 1 < len(usage) else len(events)
        tools = False
        answer = None
        for event in events[i + 1 : end]:
            if event.get("kind") == "tool":
                tools = True
            elif _is_door(event) and answer is None:
                answer = "harness_stop_refused" if tools else "words_refused"
                doors[(state, answer)] += 1
            elif event.get("kind") == "stop" and answer is None and not tools:
                answer = "turn_end"
        table[(state, answer or ("tools" if tools else "other"))] += 1
    return {
        "requests": len(usage),
        "by_state": {
            s: {a: table[(s, a)] for a in ANSWERS if table[(s, a)]}
            for s in STATES
            if any(table[(s, a)] for a in ANSWERS)
        },
        "refused_stops": sum(doors.values()),
        "refused_stops_by_state": {
            s: {a: doors[(s, a)] for a in ANSWERS if doors[(s, a)]}
            for s in STATES
            if any(doors[(s, a)] for a in ANSWERS)
        },
    }


def read_serve_log(world: Path) -> list[tuple[float, str, str]]:
    """(time, kind, skill) for every refused stop, skill answer and the run-stop request."""
    out: list[tuple[float, str, str]] = []
    for line in (world / "logs" / "serve.log").read_text(errors="replace").splitlines():
        t = _at(line)
        if t is None:
            continue
        name = SKILL_NAME.search(line)
        if DOOR in line:
            out.append((t, "door", ""))
        elif RUN_STOP in line:
            out.append((t, "run_stop", ""))
        elif LOADED in line and name:
            out.append((t, "load", name.group(1)))
        elif any(p in line for p in POINTERS) and name:
            out.append((t, "pointer", name.group(1)))
    return out


def read_reentries(events: list[dict], log: list[tuple[float, str, str]], skill: str) -> dict:
    answers = [(kind, name) for _, kind, name in log if kind in ("load", "pointer")]
    calls = [i for i, e in enumerate(events) if _is_skill_call(e)]
    ok = [i for i in calls if (events[i].get("data") or {}).get("ok")]
    if len(ok) != len(answers):
        return {
            "refused": "the join needs equal counts",
            "trace_ok_skill_calls": len(ok),
            "serve_log_skill_lines": len(answers),
        }
    rows: Counter[tuple[str, str]] = Counter()
    requests_before_the_stop: list[int] = []
    for index, (kind, name) in zip(ok, answers, strict=True):
        if kind != "load" or name != skill:
            continue
        last = "none"
        followed = "went_on"
        requests = 0
        for event in events[index + 1 :]:
            if event.get("kind") == "usage":
                last = _state(event.get("data") or {})
                requests += 1
            elif _is_door(event):
                followed = "refused_stop"
                requests_before_the_stop.append(requests)
                break
            elif _is_skill_call(event) or event.get("kind") == "stop":
                break
        rows[(followed, last)] += 1
    return {
        "trace_ok_skill_calls": len(ok),
        "serve_log_skill_lines": len(answers),
        "model_made_reentries": sum(rows.values()),
        "then": {
            f: {s: rows[(f, s)] for s in STATES if rows[(f, s)]}
            for f in ("refused_stop", "went_on")
            if any(rows[(f, s)] for s in STATES)
        },
        "requests_before_the_stop": sorted(requests_before_the_stop),
    }


def read_back_to_work(world: Path, log: list[tuple[float, str, str]], skill: str) -> dict:
    launch = None
    for line in (world / "logs" / "progress.log").read_text(errors="replace").splitlines():
        m = LAUNCH.match(line)
        if m:
            stamp = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC)
            launch = stamp.timestamp()
            break
    stop_at = next((t for t, kind, _ in log if kind == "run_stop"), None)
    if launch is None or stop_at is None:
        return {"refused": "no launch line or no run-stop request in the logs"}
    intervals: list[dict] = []
    opened: float | None = None
    inside = 0
    for t, kind, name in log:
        if kind == "door" and opened is None and t < stop_at:
            opened, inside = t, 1
        elif kind == "door" and opened is not None:
            inside += 1
        elif opened is not None and (kind == "run_stop" or (kind == "load" and name != skill)):
            end = min(t, stop_at)
            intervals.append(
                {"from": _clock(opened), "seconds": round(end - opened, 1), "refused_stops": inside}
            )
            opened = None
    total = round(sum(x["seconds"] for x in intervals), 1)
    run = round(stop_at - launch, 1)
    return {
        "run_seconds": run,
        "back_to_work_seconds": total,
        "back_to_work_share": round(total / run, 4) if run else None,
        "intervals": intervals,
    }


def read_world(world: Path, skill: str) -> dict:
    events = read_events(world)
    log = read_serve_log(world)
    return {
        "world": world.name,
        "skill": skill,
        **read_requests(events),
        "reentries": read_reentries(events, log, skill),
        "back_to_work": read_back_to_work(world, log, skill),
    }


# ── known answers ───────────────────────────────────────────────────────────────────────────────


def _ev(event: str, detail: str = "", **data: object) -> dict:
    return {"kind": event, "detail": detail, "data": data}


def _use(**data: object) -> dict:
    """One request's `usage` event as ADR-0205 writes it: all three keys always present."""
    return _ev("usage", "", **{"rails": [], "rails_silenced": [], "rails_rested": False, **data})


def selftest() -> int:
    door = _ev("intervention", "", kind="turn_end_skill")
    skill_ok = _ev("tool", "Skill", ok=True)
    skill_bad = _ev("tool", "Skill", ok=False)
    bash = _ev("tool", "Bash", ok=True)
    events = [
        _use(rails=["plan"]), bash,                                   # open      -> tools
        _use(rails=["plan"]), skill_ok,                               # re-entry 1 (body)
        _use(rails=["plan"]), bash,                                   #   open
        _use(rails=["plan_complete"]), door,                          #   sent -> words, refused
        _use(rails_silenced=["plan_complete"]), skill_ok,             # the pointer after the door
        _use(rails_silenced=["plan_complete"]), bash,                 # silent    -> tools
        _use(rails_rested=True), bash, door,                          # rested    -> harness stop
        _use(), skill_bad,                                            # a failed Skill call: no line
        _use(rails=["plan"]), skill_ok,                               # re-entry 2 (body)
        _use(rails=["plan"]), skill_ok,                               #   went on: another skill
        _use(rails=["plan_complete"]), bash,                          # sent      -> tools
        _use(rails=["plan"]), skill_ok,                               # open      -> tools (a skill)
        _use(rails=["plan_complete"]), _ev("stop", "completed"),      # sent      -> turn end
    ]  # fmt: skip
    day = "2026-01-01"

    def at(clock: str, text: str) -> str:
        return f"{day} {clock},000 INFO {text}"

    def loaded(skill: str) -> str:
        return f"zakcode skill '{skill}' {LOADED} (1 this turn, 1 this session)"

    def refused(reason: str) -> str:
        return f"zakcode.agent.loop {DOOR} stop_reason='{reason}' (veto 1 this turn)"

    log_lines = [
        at("10:00:10", loaded("cycle")),
        at("10:00:20", refused("completed")),
        at("10:00:21", "zakcode skill 'cycle' use_skill deduped (already loaded this turn)"),
        at("10:00:30", refused("stuck")),
        at("10:00:50", "zakcode skill 'cycle' asked for again after work — a re-entry"),
        at("10:00:50", loaded("cycle")),
        at("10:01:00", loaded("select")),
        at("10:01:30", refused("completed")),
        at("10:01:40", RUN_STOP),
        at("10:01:50", refused("completed")),
        at("10:02:00", loaded("stop")),
    ]
    failures: list[str] = []
    checks = 0

    def check(name: str, got: object, want: object) -> None:
        nonlocal checks
        checks += 1
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")

    with tempfile.TemporaryDirectory() as tmp:
        world = Path(tmp) / "w"
        (world / "logs" / "traces" / "s").mkdir(parents=True)
        (world / "logs" / "traces" / "s" / "turn_1.jsonl").write_text(
            "\n".join(json.dumps(e) for e in events) + "\n"
        )
        (world / "logs" / "serve.log").write_text("\n".join(log_lines) + "\n")
        (world / "logs" / "progress.log").write_text(f"[run] {day}T10:00:00Z zakcode source=/x\n")
        got = read_world(world, "cycle")
        check("requests", got["requests"], 13)
        check("sent", got["by_state"].get("sent"), {"tools": 1, "words_refused": 1, "turn_end": 1})
        check("silent", got["by_state"].get("silent"), {"tools": 2})
        check("open", got["by_state"].get("open"), {"tools": 6})
        check("rested", got["by_state"].get("rested"), {"harness_stop_refused": 1})
        check("none", got["by_state"].get("none"), {"tools": 1})
        check("doors", got["refused_stops"], 2)
        check(
            "doors by state",
            got["refused_stops_by_state"],
            {"sent": {"words_refused": 1}, "rested": {"harness_stop_refused": 1}},
        )
        joined = got["reentries"]
        check("join", (joined["trace_ok_skill_calls"], joined["serve_log_skill_lines"]), (5, 5))
        check("re-entries", got["reentries"]["model_made_reentries"], 2)
        check(
            "re-entries then",
            got["reentries"]["then"],
            {"refused_stop": {"sent": 1}, "went_on": {"open": 1}},
        )
        check("requests before the stop", got["reentries"]["requests_before_the_stop"], [2])
        work = got["back_to_work"]
        check("run", work["run_seconds"], 100.0)
        check(
            "intervals",
            work["intervals"],
            [
                {"from": "10:00:20", "seconds": 40.0, "refused_stops": 2},
                {"from": "10:01:30", "seconds": 10.0, "refused_stops": 1},
            ],
        )
        check("share", work["back_to_work_share"], 0.5)
        # The join must refuse, not guess, when a skill line is missing.
        (world / "logs" / "serve.log").write_text("\n".join(log_lines[1:]) + "\n")
        try:
            join_refused = "refused" in read_world(world, "cycle")["reentries"]
        except ValueError:  # a reader that joined unequal sides anyway
            join_refused = False
        check("join refused", join_refused, True)
        # A trace from before ADR-0204 is refused as well.
        old = [{"kind": "usage", "detail": "", "data": {}}, bash]
        check("old trace refused", "refused" in read_requests(old), True)
    for failure in failures:
        print("FAIL", failure)
    print(f"selftest: {checks - len(failures)} of {checks} known answers hold")
    return 1 if failures else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    argv = sys.argv[1:]
    name = "aspirations"  # the loop skill of the framework the served samples run
    if "--skill" in argv:
        at = argv.index("--skill")
        name = argv[at + 1]
        del argv[at : at + 2]
    print(json.dumps(read_world(Path(argv[0]), name), indent=2))
