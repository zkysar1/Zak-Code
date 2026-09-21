"""What followed each refused stop on a served run, read from its per-turn traces.

The door reader registered with served sample 6 (bench/results/served-luna-preregistration.log).

usage: python bench/served_doors.py <world-dir>     prints one JSON reading
       python bench/served_doors.py --selftest      known-answer checks on synthetic traces

A DOOR EVENT is an honoured refusal of a stop (a `turn_end_skill` delivery note). Its segment runs
to the next door event or the end of the turn. It RESUMED if the segment holds at least one WORK
call, by the product's own definition (a tool call that did not error and is not the plan, the
skill tool or the wake-up: the three sets are IMPORTED from the product, never re-typed here); it
STOPPED AGAIN otherwise. It was taken on a FINISHED PLAN if the first main request after it says
so: `plan_complete` in `rails` (the line was sent) or in `rails_silenced` (the line was kept
silent). A trace with no `rails` key is a build older than ADR-0204 and cannot be classified:
said, not guessed. It LOADED A SKILL if the segment holds a Skill call that was answered with a
body: an ok Skill call that no `skill_pointer` note claims. That, and not a work call, is what the
veto-stall fence counts (ADR-0187): three refusals running with no body loaded between them, and
the fourth ends the turn. So a door can RESUME and still leave the fence counting. Each turn
reports its trailing run of doors with no load; a turn the fence ended must show exactly three,
which is the reader's check on its own reading of the fence. Labels and counts only: no prompt,
completion or command text is read or printed.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from zakcode.agent.loop import _PLAN_TOOLS, _SKILL_TOOLS, _WAKEUP_TOOLS

NOT_WORK = _PLAN_TOOLS | _SKILL_TOOLS | _WAKEUP_TOOLS


def _tool_name(event: dict) -> str:
    return str(event.get("detail") or "").split(" ", 1)[0].split(":", 1)[0].split("(", 1)[0]


def read_turn(events: list[dict]) -> dict:
    doors: list[dict] = []
    current: dict | None = None
    usage = [e for e in events if e.get("kind") == "usage"]
    for event in events:
        kind, data = event.get("kind"), event.get("data") or {}
        if kind == "intervention" and data.get("kind") == "turn_end_skill":
            current = {
                "work": 0,
                "tools": Counter(),
                "first_request": None,
                "skill_ok": 0,
                "pointers": 0,
            }
            doors.append(current)
        elif current is None:
            continue
        elif kind == "usage" and current["first_request"] is None:
            current["first_request"] = {
                "has_rails": "rails" in data,
                "rails": list(data.get("rails") or []),
                "silenced": list(data.get("rails_silenced") or []),
                "has_silenced_key": "rails_silenced" in data,
            }
        elif kind == "intervention" and data.get("kind") == "skill_pointer":
            current["pointers"] += 1
        elif kind == "tool":
            name = _tool_name(event)
            current["tools"][name] += 1
            if data.get("ok") and name in _SKILL_TOOLS:
                current["skill_ok"] += 1
            if data.get("ok") and name not in NOT_WORK:
                current["work"] += 1
    stop = [e for e in events if e.get("kind") == "stop"]
    notes = Counter(
        (e.get("data") or {}).get("kind") for e in events if e.get("kind") == "intervention"
    )
    out = []
    for door in doors:
        first = door["first_request"]
        if first is None:
            plan = "no-request-followed"  # the fence ended the turn, or the run stopped
        elif not first["has_rails"]:
            plan = "unclassifiable-older-build"
        elif "plan_complete" in first["rails"] or "plan_complete" in first["silenced"]:
            plan = "finished"
        else:
            plan = "not-finished"
        out.append(
            {
                "plan": plan,
                "line": None
                if first is None
                else (
                    "silent"
                    if "plan_complete" in first["silenced"]
                    else "sent"
                    if "plan_complete" in first["rails"]
                    else "none"
                ),
                "resumed": door["work"] > 0,
                "loaded": door["skill_ok"] - door["pointers"] > 0,
                "tools": dict(door["tools"]),
            }
        )
    trailing = 0
    for door in reversed(out):
        if door["loaded"]:
            break
        trailing += 1
    return {
        "stop": (stop[-1].get("detail") if stop else None),
        "doors": out,
        "trailing_doors_without_load": trailing,
        "usage_events": len(usage),
        "usage_with_rails": sum(1 for e in usage if "rails" in (e.get("data") or {})),
        "usage_with_silenced_key": sum(
            1 for e in usage if "rails_silenced" in (e.get("data") or {})
        ),
        "notes": {
            k: v
            for k, v in notes.items()
            if k
            in (
                "veto_stall",
                "veto_plan_silenced",
                "turn_end_skill",
                "skill_pointer",
                "plan_gate",
                "stuck",
                "compaction",
            )
        },
    }


def read_world(world: Path) -> dict:
    files = sorted(
        (world / "logs" / "traces").rglob("turn_*.jsonl"),
        key=lambda p: (p.parent.name, int(p.stem.split("_")[1])),
    )
    turns = []
    for path in files:
        events = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        turns.append({"turn": f"{path.parent.name[:6]}/{path.stem}", **read_turn(events)})
    doors = [d for t in turns for d in t["doors"]]
    finished = [d for d in doors if d["plan"] == "finished"]
    other = [d for d in doors if d["plan"] == "not-finished"]

    def table(rows: list[dict]) -> dict:
        return {
            "n": len(rows),
            "stopped_again": sum(1 for d in rows if not d["resumed"]),
            "resumed_no_load": sum(1 for d in rows if d["resumed"] and not d["loaded"]),
            "loaded": sum(1 for d in rows if d["loaded"]),
        }

    return {
        "trace_files": len(files),
        "bytes": sum(p.stat().st_size for p in files),
        "turns": len(turns),
        "stops": dict(Counter(str(t["stop"]) for t in turns)),
        "veto_stall_turns": sum(1 for t in turns if t["stop"] == "veto_stall"),
        "usage_events": sum(t["usage_events"] for t in turns),
        "usage_with_rails": sum(t["usage_with_rails"] for t in turns),
        "usage_with_silenced_key": sum(t["usage_with_silenced_key"] for t in turns),
        "door_events": len(doors),
        "door_plan": dict(Counter(d["plan"] for d in doors)),
        "all_doors": table(doors),
        "finished_plan_doors": {
            **table(finished),
            "line": dict(Counter(str(d["line"]) for d in finished)),
        },
        "other_doors": table(other),
        "fence_check": [
            {
                "turn": t["turn"],
                "stop": t["stop"],
                "trailing_doors_without_load": t["trailing_doors_without_load"],
                "consistent": (t["trailing_doors_without_load"] == 3)
                if t["stop"] == "veto_stall"
                else (t["trailing_doors_without_load"] <= 3),
            }
            for t in turns
        ],
        "notes": dict(sum((Counter(t["notes"]) for t in turns), Counter())),
        "per_turn": [
            {
                "turn": t["turn"],
                "stop": t["stop"],
                "doors": [
                    (
                        "F"
                        if d["plan"] == "finished"
                        else "o"
                        if d["plan"] == "not-finished"
                        else "?"
                    )
                    + ("+" if d["resumed"] else "-")
                    + ("*" if d["loaded"] else "")
                    for d in t["doors"]
                ],
            }
            for t in turns
        ],
    }


def _ev(event: str, detail: str = "", **data) -> dict:
    """One trace event; ``note=`` stands for an intervention's own ``kind`` (the key collides)."""
    if "note" in data:
        data["kind"] = data.pop("note")
    return {"kind": event, "detail": detail, "data": data}


def selftest() -> int:
    failed = 0

    def check(label: str, got, want) -> None:
        nonlocal failed
        ok = got == want
        failed += not ok
        print(
            ("ok   " if ok else "FAIL ") + label + ("" if ok else f": got {got!r}, wanted {want!r}")
        )

    door = _ev("intervention", "", note="turn_end_skill")
    sent = _ev(
        "usage",
        "",
        rails=["prompt_context", "plan_complete"],
        rails_rested=False,
        rails_silenced=[],
    )
    silent = _ev(
        "usage", "", rails=["prompt_context"], rails_rested=False, rails_silenced=["plan_complete"]
    )
    open_plan = _ev("usage", "", rails=["plan"], rails_rested=False, rails_silenced=[])
    old = _ev("usage", "", prompt_tokens=1)
    bash_ok, bash_err = _ev("tool", "Bash", ok=True), _ev("tool", "Bash", ok=False)
    skill, plan, wake = (
        _ev("tool", "Skill", ok=True),
        _ev("tool", "update_plan", ok=True),
        _ev("tool", "ScheduleWakeup", ok=True),
    )
    stall = _ev("stop", "veto_stall")

    t = read_turn(
        [
            door,
            sent,
            skill,
            old,
            door,
            sent,
            door,
            sent,
            _ev("intervention", "", note="veto_stall"),
            stall,
        ]
    )
    check(
        "three doors, line sent, none resumed, turn ended veto_stall",
        ([d["plan"] + ":" + d["line"] + ":" + str(d["resumed"]) for d in t["doors"]], t["stop"]),
        (["finished:sent:False"] * 3, "veto_stall"),
    )
    t = read_turn([door, silent, skill, silent, bash_ok, _ev("stop", "completed")])
    check(
        "line silent, a work call followed: resumed",
        [(d["plan"], d["line"], d["resumed"]) for d in t["doors"]],
        [("finished", "silent", True)],
    )
    t = read_turn([door, silent, skill, plan, wake, bash_err, _ev("stop", "completed")])
    check(
        "the plan, the skill tool, the wake-up and a FAILED command are not work",
        [d["resumed"] for d in t["doors"]],
        [False],
    )
    t = read_turn([door, open_plan, bash_ok])
    check(
        "a refusal on an open plan is its own class",
        [(d["plan"], d["line"]) for d in t["doors"]],
        [("not-finished", "none")],
    )
    t = read_turn([door, old, bash_ok])
    check(
        "no rails key: an older build, said and not guessed",
        [d["plan"] for d in t["doors"]],
        ["unclassifiable-older-build"],
    )
    t = read_turn([bash_ok, door])
    check("a door nothing followed", [d["plan"] for d in t["doors"]], ["no-request-followed"])
    check("work before the first door belongs to no door", t["doors"][0]["resumed"], False)
    check(
        "the product's sets were imported, not re-typed",
        (
            "Skill" in NOT_WORK,
            "update_plan" in NOT_WORK,
            "ScheduleWakeup" in NOT_WORK,
            "Bash" in NOT_WORK,
        ),
        (True, True, True, False),
    )
    pointer = _ev("intervention", "", note="skill_pointer")
    t = read_turn(
        [
            door,
            silent,
            skill,
            pointer,
            bash_ok,
            door,
            silent,
            skill,
            bash_ok,
            door,
            silent,
            door,
            silent,
            door,
            silent,
            _ev("intervention", "", note="veto_stall"),
            stall,
        ]
    )
    check(
        "a pointer is not a load, a body is; the trailing run is counted from the last load",
        (
            [("+" if d["resumed"] else "-") + ("*" if d["loaded"] else "") for d in t["doors"]],
            t["trailing_doors_without_load"],
        ),
        (["+", "+*", "-", "-", "-"], 3),
    )
    t = read_turn([door, silent, _ev("tool", "Skill", ok=False), bash_ok])
    check("a FAILED Skill call loads nothing", [d["loaded"] for d in t["doors"]], [False])
    print(f"{'FAILED' if failed else 'passed'}: {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        raise SystemExit(selftest())
    print(json.dumps(read_world(Path(sys.argv[1])), indent=1))
