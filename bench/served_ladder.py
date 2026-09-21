"""What the stuck ladder fired on in a served loop: a replay of the product's own tracker.

usage: python bench/served_ladder.py <world-dir> [<world-dir> ...]   one JSON reading
       python bench/served_ladder.py --selftest                      known answers, synthetic runs

Labels, counts and distances only: no command, output or prompt text is printed.

WHY. The ladder's repeated-outcome signal (ADR-0038) counts identical (tool, epoch, output)
observations over a whole TURN, and only a successful workspace-write tool call opens a new
epoch. A served perpetual loop is one very long turn that changes its world through shell
scripts, so the count can only grow: the third return of any output lands a nudge, the fourth a
read-only iteration, the fifth a step back, the sixth a stop. Whether the rungs a served run drew
were the loop's own regular repeats or a model circling could not be read while the transcript
was a live window. Since ADR-0206 it is append-only, so every call and result is on disk.

SINCE ADR-0209 the product counts per LAP, which is what this file's first reading licensed
(sample 7: the lap rule removed 34 of 47 rungs, and the 13 it left were all on the plan tool's
receipt). The whole-turn count above is now the rule `turn`, a replay of what the product did
BEFORE; what it does now is the rule `shipped`. A world is read under both, and says which of
the two reproduces the stuck notes its own product wrote: that names the build that ran.

WHAT IT DOES. It feeds `zakcode.agent.stuck.StuckTracker`, the product's tracker, the calls and
results the transcript holds, one tool-call iteration at a time, the way the loop does: a fresh
tracker per turn; the epoch is the turn's count of successful workspace-write calls, read after
the batch ran; and the tracker is reset where the loop resets it -- when a completion that called
no tool was answered by the harness and the turn went on, and when a stop rung was refused. For
every rung reached on a repeated outcome it says: the tool, the repeat count, how many iterations
apart the identical observations were, whether a LAP boundary, a skill body or any Skill call lay
between them, and whether the repeated output was an error.

A LAP. The product learns which skill the loop IS when a turn-end hook refuses a stop and names
it: the harness then delivers that skill itself and keeps its name as the session's loop skill
(ADR-0187). A lap boundary here is that skill delivered again: by the harness at a refused stop,
or as the BODY answering the model's own Skill call for it (a pointer is not a delivery). Until
the first refused stop no skill is known to be the loop, and nothing is a boundary -- which is
also all a rule in the product could know.

THE CONTROL. A replay is believed only if it reproduces what the product itself noted. (1) Each
`stuck` note on the run's decision trace names its signals, tool and repeat count (ADR-0038
amended): per world the sequence of (tool, repeats) over the repeated-outcome notes must match
the replay's under the shipped rule, at least 80% in order, counted against the LONGER of the
two. (2) The refused stops found in the transcript must be the deliveries the trace noted, give
or take one (a run cut off between the two writes). A world failing either is NOT READ, and named.
That control is sample 7's and stays as registered: under `turn`, and a world with no note has
nothing to reproduce. `control_by_count` puts the same question to BOTH counts, and there a
world with no note is reproduced by a replay that draws none: under the lap count a healthy run
is expected to draw none, and `turn` beside it says what the whole-turn count would have drawn.

COUNTING RULES replayed on the same calls:
  turn      identical observations counted over the whole turn. The product's until ADR-0209.
  shipped   the product's since ADR-0209, fed what its loop feeds the tracker: the lap count
            (the count starts again at a lap boundary, unless the lap that just ended showed
            nothing new to the turn), the work count, and the plan tool's receipt flag (a
            "Plan updated" receipt repeats only while no work call succeeded in between). The
            transcript keeps no result data, so the flag is put back from the receipt's first
            words; the selftest asks the real tool for them.
  lap       the count starts again at EVERY lap boundary, with no bound and no receipt rule.
            The candidate sample 7's reading was taken on.
  body      ...whenever ANY Skill call is answered with a body. Described only: a model that hops
            between skills while it circles would never climb under it.
  reentry   ...at ANY Skill call, pointer or body. Described only, for the same reason.
  window:N  only the last N tool-call iterations are remembered.
A replay under a candidate rule shows where that rule would have fired on the SAME calls. It
cannot show what the model would have done without the rungs it really drew.

THE READING (fixed in results/served-luna-preregistration.log before any sample 7 transcript was
read with this file) is `reading()` below, over the worlds that pass the control.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import tempfile
from collections import Counter, deque
from fractions import Fraction
from pathlib import Path

from zakcode.agent.loop import (
    _PLAN_TOOLS,
    _SKILL_TOOLS,
    _UNOBSERVING_TOOLS,
    _VETO_SKILL_NOTE,
    _WAKEUP_TOOLS,
    _composed_skill_name,
    harness_skill_turn_text,
)
from zakcode.agent.stuck import SIG_REPEATED_OUTCOME, StuckAction, StuckTracker, outcome_signature
from zakcode.hooks.transcript import render_claude_code_transcript
from zakcode.messages import Message, TextBlock, ToolResultBlock, ToolUseBlock
from zakcode.providers.base import ToolCall
from zakcode.tasks import TaskNetwork
from zakcode.tools.base import RECEIPT_OF_CHANGE, PermissionTier, ToolContext
from zakcode.tools.builtins.default_registry import default_registry
from zakcode.tools.builtins.update_plan import UpdatePlanTool
from zakcode.wakeup import LOOP_WAKE_NOTE

#: The product's own sets: the skill tool under both spellings, and the tools whose results are
#: the harness's own delivery and are never counted as an outcome.
SKILL_TOOLS = _SKILL_TOOLS
UNCOUNTED = _UNOBSERVING_TOOLS
#: ...and the tools a successful call to is NOT work (ADR-0196): keeping the plan, loading a
#: skill, re-arming the wake-up. The loop's work count is every other successful call.
NOT_WORK = _PLAN_TOOLS | _SKILL_TOOLS | _WAKEUP_TOOLS
#: How the plan tool's receipt of a CHANGE begins (ADR-0209). The product flags that result in
#: its data, which a transcript does not keep, so the reader puts the flag back from these
#: words. The selftest asks the real tool for both of its receipts and checks them against it.
RECEIPT_HEAD = "Plan updated"
POINTER_HEAD = "[already loaded"  # how every pointer the skill tool answers with begins
#: How the `<command-message>` line of a refused stop's delivery reads (the product's own note).
VETO_HEAD = "[harness] " + " ".join(_VETO_SKILL_NOTE.split("{")[0].split())[:48]
#: The registered rule.
RULE = {
    "control_match": Fraction(80, 100),  # a world under this share of matching notes is not read
    "refused_stops_may_differ_by": 1,  # ...nor one whose refused stops differ by more than this
    "min_rungs": 10,  # pooled repeated-outcome rungs the reading needs
    "regular": Fraction(70, 100),  # the lap rule removes at least this share: LOOP REGULARITY
    "circling": Fraction(30, 100),  # ...at most this share: CIRCLING
}


def _write_tier_tools() -> frozenset[str]:
    """Every name and alias of a tool whose success opens a new epoch: the workspace-write tier,
    asked of the product's registry (two builtins of that tier register only when configured)."""
    registry = default_registry()
    names = {"save_rule", "save_skill"}
    for name in registry.names():
        spec = getattr(registry.get(name), "spec", None)
        if getattr(spec, "required_permission", None) is PermissionTier.WORKSPACE_WRITE:
            names |= {name, *registry.aliases_of(name)}
    return frozenset(names)


EDIT_TOOLS = _write_tier_tools()
#: The plan tool under every name the product's registry gives it.
PLAN_RECEIPT_TOOLS = frozenset({"update_plan", *default_registry().aliases_of("update_plan")})


def _text(content: object) -> str:
    """A tool result's text, whichever of the two shapes the transcript used."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(b.get("text", "")) for b in content if isinstance(b, dict))
    return "" if content is None else str(content)


def _said(row: dict) -> str:
    """What a user row SAYS: a typed message or a harness delivery. Tool results say nothing
    here."""
    content = (row.get("message") or {}).get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(b.get("text", ""))
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def _result(tool: str, use_id: str, block: dict) -> ToolResultBlock:
    """A tool result as the tracker is handed one. The transcript keeps its text and whether it
    was an error; the one piece of result DATA the tracker reads, the plan tool's receipt flag
    (ADR-0209), is put back from how the receipt begins."""
    output, failed = _text(block.get("content")), bool(block.get("is_error"))
    receipt = tool in PLAN_RECEIPT_TOOLS and not failed and output.lstrip().startswith(RECEIPT_HEAD)
    return ToolResultBlock(
        tool_use_id=use_id,
        output=output,
        is_error=failed,
        data={RECEIPT_OF_CHANGE: True} if receipt else None,
    )


def read_transcript(transcript: Path) -> dict:
    """Tool-call iterations in order -- one assistant row's calls and the results answering them
    -- each with what came since the one before it: completions that called no tool, and refused
    stops (the skill each delivered). Plus how many refused stops the file holds in all."""
    rows = []
    with transcript.open(errors="replace") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    results: dict[str, dict] = {}
    for row in rows:
        content = (row.get("message") or {}).get("content") if row.get("type") == "user" else None
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    results[str(block.get("tool_use_id"))] = block
    iterations: list[dict] = []
    wordy, refused, refused_in_all = 0, [], 0
    for row in rows:
        if row.get("type") == "user":
            said = _said(row)
            skill = _composed_skill_name(said)
            if skill is not None and VETO_HEAD in said.split("\n", 1)[0]:
                refused.append(skill)
                refused_in_all += 1
            continue
        if row.get("type") != "assistant":
            continue
        content = (row.get("message") or {}).get("content")
        blocks = [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []
        uses = [b for b in blocks if b.get("type") == "tool_use"]
        if not uses:
            wordy += 1  # the harness answered it and the turn went on, or the turn ended
            continue
        iterations.append(
            {
                "text": "".join(str(b.get("text", "")) for b in blocks if b.get("type") == "text"),
                "calls": [
                    ToolCall(
                        id=str(u.get("id")), name=str(u.get("name")), arguments=u.get("input") or {}
                    )
                    for u in uses
                ],
                "results": [
                    _result(str(u.get("name")), str(u.get("id")), results[str(u.get("id"))])
                    for u in uses
                    if str(u.get("id")) in results
                ],
                "wordy_completions_before": wordy,
                "refused_stops": refused,
            }
        )
        wordy, refused = 0, []
    return {"iterations": iterations, "refused_stops": refused_in_all}


def read_trace(world: Path, session: str) -> dict:
    """The product's own record of one session: per turn, how many tool calls it noted; the (tool,
    repeats) of every `stuck` note that names a repeated outcome, in order; and how many refused
    stops it noted as DELIVERED (one it could not deliver went out as a plain rail)."""
    files = sorted(
        (world / "logs" / "traces" / session).glob("turn_*.jsonl"),
        key=lambda p: int(re.sub(r"\D", "", p.stem) or 0),
    )
    sizes: list[int] = []
    notes: list[tuple[str, int]] = []
    delivered = on_a_receipt = 0
    for path in files:
        tools = 0
        for line in path.open(errors="replace"):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            data = event.get("data") if isinstance(event.get("data"), dict) else {}
            if event.get("kind") == "tool":
                tools += 1
            if data.get("kind") == "stuck" and SIG_REPEATED_OUTCOME in str(data.get("signals")):
                notes.append((str(data.get("tool")), int(data.get("repeats") or 0)))
                on_a_receipt += bool(data.get("receipt"))  # plan churn, said so (ADR-0209)
            if data.get("kind") == "turn_end_skill" and not data.get("refused"):
                delivered += 1
        sizes.append(tools)
    return {
        "sizes": sizes,
        "notes": notes,
        "notes_on_a_receipt": on_a_receipt,
        "refused_stops_delivered": delivered,
    }


def _skill_asked(call: ToolCall) -> str:
    """The skill a Skill call names, under either spelling of the tool's argument."""
    asked = call.arguments.get("skill") or call.arguments.get("name") or ""
    return str(asked).strip().lstrip("/")


def _body_delivered(iteration: dict, only: str | None = None) -> bool:
    """A Skill call answered with a BODY: a procedure begins. With `only`, the body of THAT
    skill; and no skill at all while `only` is still unknown (the empty name)."""
    by_id = {r.tool_use_id: r for r in iteration["results"]}
    for call in iteration["calls"]:
        result = by_id.get(call.id)
        if call.name not in SKILL_TOOLS or result is None or result.is_error:
            continue
        if result.output.lstrip().startswith(POINTER_HEAD):
            continue
        if only is None or (only and _skill_asked(call) == only):
            return True
    return False


def replay(turns: list[list[dict]], rule: str = "turn") -> dict:
    """Feed the product's tracker each turn's iterations under a counting rule."""
    window = int(rule.split(":", 1)[1]) if rule.startswith("window:") else 0
    rungs: list[dict] = []
    index = laps = own_loads = refusals = bodies = skill_calls = edits = work = 0
    loop_skill = ""  # the session's: learned at a refused stop, kept from turn to turn
    for iterations in turns:
        tracker = StuckTracker(uncounted_outcome_tools=UNCOUNTED)  # one per turn, as the loop
        epoch = 0
        # outcome signature -> [(iteration index, laps, own loads of the loop skill, refused
        # stops, bodies, Skill calls) at each sighting]
        seen: dict[str, list[tuple[int, ...]]] = {}
        recent: deque[list[str]] = deque()  # what each remembered iteration added (window rule)
        narrowed = False  # the previous iteration drew the read-only rung
        stopped = False  # ...or the stop rung, and the turn went on: the stop was refused
        for it in iterations:
            index += 1
            if it["refused_stops"]:
                loop_skill = it["refused_stops"][-1]  # the hook's last word on it (ADR-0187)
            own_load = _body_delivered(it, only=loop_skill)
            lap = bool(it["refused_stops"]) or own_load
            body = _body_delivered(it)
            calls_skill = any(c.name in SKILL_TOOLS for c in it["calls"])
            if it["wordy_completions_before"] or stopped:
                tracker.reset()  # where the loop resets it: the harness answered, the turn went on
            if (
                (rule == "lap" and lap)
                or (rule == "body" and body)
                or (rule == "reentry" and calls_skill)
            ):
                tracker._outcome_counts.clear()  # the candidate rule: count afresh from here
            laps += lap
            own_loads += own_load
            refusals += len(it["refused_stops"])
            bodies += body
            skill_calls += sum(c.name in SKILL_TOOLS for c in it["calls"])
            by_id = {r.tool_use_id: r for r in it["results"]}
            work += sum(  # the loop's work count (ADR-0196), which it never resets
                1
                for call in it["calls"]
                if call.name not in NOT_WORK
                and (done := by_id.get(call.id)) is not None
                and not done.is_error
            )
            if narrowed and rungs:
                rungs[-1]["errors_in_the_next_iteration"] = sum(r.is_error for r in it["results"])
            for call in it["calls"]:  # the loop hands the tracker the epoch AFTER the batch ran
                result = by_id.get(call.id)
                if call.name in EDIT_TOOLS and result is not None and not result.is_error:
                    epoch += 1
                    edits += 1
            added = []
            for call in it["calls"]:
                result = by_id.get(call.id)
                if result is None:
                    continue
                sig = _keyed(rule, call.name, result, epoch, work)
                if sig is not None:
                    seen.setdefault(sig, []).append(
                        (index, laps, own_loads, refusals, bodies, skill_calls)
                    )
                    added.append(sig)
            # `shipped`: the lap and work counts as the loop hands them over, read after the
            # batch ran. Only a CHANGE in either means anything to the tracker, so counting a
            # boundary once where the loop counted two doors is the same reading. Every other
            # rule hands over neither, which is the whole-turn count.
            counts = {"lap": laps, "work": work} if rule == "shipped" else {}
            tracker.observe(
                it["calls"], it["results"], assistant_text=it["text"], epoch=epoch, **counts
            )
            action = tracker.next_action()
            narrowed = action is StuckAction.NARROW
            stopped = action is StuckAction.STOP
            if action is not StuckAction.CONTINUE:
                evidence = tracker.evidence()
                row: dict = {
                    "iteration": index,
                    "rung": action.value,
                    "signals": evidence.get("signals"),
                    "tool": evidence.get("tool"),
                    "repeats": evidence.get("repeats"),
                    "receipt": bool(evidence.get("receipt")),
                }
                if SIG_REPEATED_OUTCOME in str(row["signals"]) and added:
                    # the observation the TRACKER counted highest under this rule (its own tie
                    # break: the first such call), not the one sighted most in the turn; a
                    # harness delivery it never counts stands at zero and cannot be chosen
                    worst = max(added, key=lambda s: tracker._outcome_counts[s])
                    where = seen[worst][-int(row["repeats"] or 1) :]
                    row["gaps_in_iterations"] = [
                        b[0] - a[0] for a, b in zip(where, where[1:], strict=False)
                    ]
                    row["laps_between"] = where[-1][1] - where[0][1]
                    row["own_loads_between"] = where[-1][2] - where[0][2]
                    row["refused_stops_between"] = where[-1][3] - where[0][3]
                    row["bodies_between"] = where[-1][4] - where[0][4]
                    row["skill_calls_between"] = where[-1][5] - where[0][5]
                    row["was_an_error"] = any(
                        by_id[c.id].is_error
                        for c in it["calls"]
                        if c.id in by_id and _keyed(rule, c.name, by_id[c.id], epoch, work) == worst
                    )
                rungs.append(row)
            if window:
                recent.append(added)
                while len(recent) > window:
                    for sig in recent.popleft():
                        tracker._outcome_counts[sig] -= 1
                        if tracker._outcome_counts[sig] <= 0:
                            del tracker._outcome_counts[sig]
    repeated = [r for r in rungs if SIG_REPEATED_OUTCOME in str(r["signals"])]
    return {
        "rule": rule,
        "iterations": index,
        "file_edits_that_opened_an_epoch": edits,
        "lap_boundaries": laps,
        "skill_bodies_delivered": bodies,
        "skill_calls": skill_calls,
        "rungs": dict(Counter(r["rung"] for r in rungs)),
        "repeated_outcome_rungs": len(repeated),
        "of_them_a_lap_boundary_lay_between": sum(
            (r.get("laps_between") or 0) > 0 for r in repeated
        ),
        # ...and the ones a lap rule could not tell from a model that stops in words after
        # every probe: nothing but refused stops lay between the repeats.
        "of_them_only_refused_stops_lay_between": sum(
            (r.get("refused_stops_between") or 0) > 0 and not r.get("own_loads_between")
            for r in repeated
        ),
        "of_them_a_body_lay_between": sum((r.get("bodies_between") or 0) > 0 for r in repeated),
        "of_them_a_skill_call_lay_between": sum(
            (r.get("skill_calls_between") or 0) > 0 for r in repeated
        ),
        "of_them_the_output_was_an_error": sum(bool(r.get("was_an_error")) for r in repeated),
        # ...and the ones on a tool's receipt for a change (ADR-0209). The TRACKER says which,
        # from the tool's own flag, under every rule here. On a TRACE only a product since
        # ADR-0209 writes the key, so an older world's own notes count none.
        "of_them_on_a_receipt": sum(bool(r.get("receipt")) for r in repeated),
        "by_tool": dict(Counter(str(r["tool"]) for r in repeated)),
        "errors_in_the_iteration_after_a_read_only_rung": [
            r["errors_in_the_next_iteration"] for r in rungs if "errors_in_the_next_iteration" in r
        ],
        "detail": rungs,
    }


def _keyed(rule: str, tool: str, result: ToolResultBlock, epoch: int, work: int) -> str | None:
    """The signature of one result AS THE TRACKER KEYED IT under `rule`, so that what is said of
    a rung (its gaps, what lay between, whether it was an error) is said of the sightings the
    tracker counted as one. Since ADR-0209 the product keys a RECEIPT on the work count too.
    The signature itself is the product's; only the choice of `work` is made here, by the
    tracker's own test, and the mutation proof withholds it."""
    receipt = (
        rule == "shipped"
        and not result.is_error
        and bool((result.data or {}).get(RECEIPT_OF_CHANGE))
    )
    return outcome_signature(tool, result.output or "", epoch, work=work if receipt else None)


def _matches(noted: list[tuple[str, int]], replayed: list[tuple[str, int]]) -> int:
    """How many of the product's notes the replay reproduces IN ORDER (longest common
    subsequence), so one missing or extra rung costs one match and not every later one."""
    table = [[0] * (len(replayed) + 1) for _ in range(len(noted) + 1)]
    for i, a in enumerate(noted, 1):
        for j, b in enumerate(replayed, 1):
            table[i][j] = (
                table[i - 1][j - 1] + 1 if a == b else max(table[i - 1][j], table[i][j - 1])
            )
    return table[-1][-1]


#: `turn` stays first and `shipped` goes last: sample 7's reading indexes the others by place.
RULES = ("turn", "lap", "body", "reentry", "window:40", "shipped")


def read_world(world: Path) -> dict:
    files = sorted((world / "mind-workspace" / ".zakcode" / "transcripts").glob("*.jsonl"))
    if len(files) != 1:  # a served run is ONE session; the order of several is not on disk
        return {"world": world.name, "refused": f"{len(files)} transcript files, not one"}
    read = read_transcript(files[0])
    iterations = read["iterations"]
    trace = read_trace(world, files[0].stem)
    sizes, noted = trace["sizes"], trace["notes"]
    calls = sum(len(it["calls"]) for it in iterations)
    out: dict = {
        "world": world.name,
        "tool_call_iterations": len(iterations),
        "tool_calls_in_transcript": calls,
        "tool_calls_on_the_trace": sum(sizes),
        "tool_calls": dict(Counter(c.name for it in iterations for c in it["calls"]).most_common()),
    }
    # Split into turns by the trace's own per-turn tool-call counts. If the two records disagree
    # on the total the split cannot be trusted, and the run is replayed as one turn and says so.
    turns: list[list[dict]] = []
    if sizes and sum(sizes) == calls:
        rest = list(iterations)
        for size in sizes:
            turn, held = [], 0
            while rest and held < size:
                held += len(rest[0]["calls"])
                turn.append(rest.pop(0))
            turns.append(turn)
        out["turns"] = len(turns)
    else:
        turns = [iterations]
        out["turns"] = "one (the trace and the transcript disagree on the call total)"
    out["replays"] = [replay(turns, rule) for rule in RULES]
    whole_turn = out["replays"][0]  # sample 7's control is against the whole-turn count
    mine = [
        (str(r["tool"]), int(r["repeats"] or 0))
        for r in whole_turn["detail"]
        if SIG_REPEATED_OUTCOME in str(r["signals"])
    ]
    matched = _matches(noted, mine)
    in_transcript = read["refused_stops"]
    # The share is taken of the LONGER list, so a note the replay did not reproduce and a
    # rung the product never noted both cost a match. The 1 is for a run that drew no rung
    # at all: it matched nothing, so it does not pass -- there was nothing to reproduce.
    notes_match = Fraction(matched, max(len(noted), len(mine), 1)) >= RULE["control_match"]
    stops_match = (
        abs(trace["refused_stops_delivered"] - in_transcript) <= RULE["refused_stops_may_differ_by"]
    )
    by_count = {}
    for count in ("turn", "shipped"):
        drawn = [
            (str(r["tool"]), int(r["repeats"] or 0))
            for r in out["replays"][RULES.index(count)]["detail"]
            if SIG_REPEATED_OUTCOME in str(r["signals"])
        ]
        agreed, longer = _matches(noted, drawn), max(len(noted), len(drawn))
        by_count[count] = {
            "repeated_outcome_rungs_replayed": len(drawn),
            "matched_in_order": agreed,
            # No note and no rung replayed IS agreement here: a healthy run under the lap
            # count draws none, and a replay that draws none has reproduced that.
            "reproduces_the_notes": longer == 0
            or Fraction(agreed, longer) >= RULE["control_match"],
        }
    told = [count for count, c in by_count.items() if c["reproduces_the_notes"]]
    out["control_by_count"] = {
        "repeated_outcome_notes_on_the_trace": len(noted),
        "of_them_on_a_receipt": trace["notes_on_a_receipt"],
        **by_count,
        # Which count the product that wrote this world used, read off its own notes. `either`:
        # the two counts draw the same rungs on these calls, so the notes cannot tell.
        "written_under": told[0] if len(told) == 1 else ("either" if told else "neither"),
        "refused_stops_match": stops_match,
    }
    out["control"] = {
        "repeated_outcome_notes_on_the_trace": len(noted),
        "repeated_outcome_rungs_replayed": len(mine),
        "matched_in_order": matched,
        "refused_stops_on_the_trace": trace["refused_stops_delivered"],
        "refused_stops_in_the_transcript": in_transcript,
        "passes": notes_match and stops_match,
    }
    return out


def reading(worlds: list[dict]) -> dict:
    """Sample 7's registered reading over every world given: the share of the whole-turn
    count's rungs that the plain lap rule removes. A world is left out, and named, if it has no
    transcript or its replay under `turn` does not reproduce the product's own record, which is
    what a world written since ADR-0209 will do: this reading is for worlds written before."""
    out: dict = {"rule": {k: str(v) for k, v in RULE.items()}}
    read, left_out = [], {}
    for w in worlds:
        if w.get("refused"):
            left_out[w["world"]] = w["refused"]
        elif not w["control"]["passes"]:
            c = w["control"]
            left_out[w["world"]] = (
                f"replay matched {c.get('matched_in_order')} of "
                f"{c.get('repeated_outcome_notes_on_the_trace')} notes "
                f"({c.get('repeated_outcome_rungs_replayed')} replayed); refused stops "
                f"{c.get('refused_stops_in_the_transcript')} in the transcript, "
                f"{c.get('refused_stops_on_the_trace')} on the trace"
            )
        else:
            read.append(w)
    out["worlds_read"], out["worlds_left_out"] = [w["world"] for w in read], left_out
    if len(read) * 2 < len(worlds) or not read:
        return out | {"batch": "REPLAY NOT TRUSTED"}
    by_rule = {
        rule: sum(w["replays"][i]["repeated_outcome_rungs"] for w in read)
        for i, rule in enumerate(RULES)
    }
    whole_turn = by_rule["turn"]
    out["repeated_outcome_rungs_by_rule"] = by_rule
    for key in (
        "of_them_a_lap_boundary_lay_between",
        "of_them_only_refused_stops_lay_between",
        "of_them_a_body_lay_between",
        "of_them_a_skill_call_lay_between",
        "of_them_the_output_was_an_error",
    ):
        out[key] = sum(w["replays"][0][key] for w in read)
    if whole_turn < RULE["min_rungs"]:
        return out | {"batch": "TOO FEW"}
    removed = Fraction(whole_turn - by_rule["lap"], whole_turn)
    out["share_the_lap_rule_removes"] = str(removed)
    if removed >= RULE["regular"]:
        return out | {"batch": "LOOP REGULARITY"}
    if removed <= RULE["circling"]:
        return out | {"batch": "CIRCLING"}
    return out | {"batch": "MIXED"}


# ── known answers ────────────────────────────────────────────────


class _Run:
    """A synthetic served run. Its transcript is rendered by the product's own writer from the
    product's own messages, and its trace is what the product's tracker says of the SAME calls,
    reset where the loop resets it. A stop rung the script runs on from was a refused stop: the
    harness delivers the loop skill, as it does in a served run.

    WHICH product is `write`'s to say: the one before ADR-0209 handed its tracker the epoch and
    nothing else (`turn`), the one since also hands it the lap and work counts (`shipped`),
    counted HERE from the script, the way the loop counts them, and not the way `replay` reads
    them back off a transcript. The two are kept apart so that each can catch the other."""

    LOOP = "loop"
    BODY = "# the loop\n" + "step: do the next thing in the procedure\n" * 8

    def __init__(self) -> None:
        self.turns: list[list[tuple[str, object]]] = [[]]
        self._plan = TaskNetwork()  # the session's plan, which the REAL plan tool keeps

    def batch(self, *calls: tuple) -> None:
        """One completion calling several tools: (tool, output, is_error, arguments) each, and
        the result's DATA as a fifth where the tool that answered sets any."""
        self.turns[-1].append(("calls", list(calls)))

    def call(
        self,
        name: str,
        out: str,
        err: bool = False,
        args: dict | None = None,
        data: dict | None = None,
    ) -> None:
        self.batch((name, out, err, args or {}, data))

    def points_at_the_loop(self, text: str) -> None:
        """The model's own Skill call for the loop skill, answered with the pointer, flagged the
        way the skill tool flags one (ADR-0203)."""
        self.call("Skill", text, args={"skill": self.LOOP}, data={"pointer": True})

    def plans(self, note: str) -> None:
        """The model journals into its plan, and the REAL plan tool answers: the receipt's words
        and its flag are the product's own. Three steps, the second in progress, `note` on it:
        every call changes the plan, and none moves the done count or the current step."""
        tasks = [
            {"title": "Select the goal", "status": "done", "outcome": "selected"},
            {"title": "Carry out the goal", "status": "in_progress", "note": note},
            {"title": "Close the iteration"},
        ]
        ctx = ToolContext(workspace_root=Path("."), task_network=self._plan)
        result = asyncio.run(UpdatePlanTool().execute({"tasks": tasks}, ctx))
        self.batch(("update_plan", result.output, result.is_error, {"tasks": tasks}, result.data))

    def loads_the_loop(self) -> None:
        """The model's own Skill call for the loop skill, answered with its body."""
        self.call("Skill", self.BODY, args={"skill": self.LOOP})

    def says_only(self) -> None:
        """A completion that calls no tool; the harness answers with a rail, the turn goes on."""
        self.turns[-1].append(("says", None))

    def refused_stop(self, skill: str = LOOP) -> None:
        """The model stops in words, a turn-end hook refuses and names `skill`, the harness
        delivers it."""
        self.turns[-1].append(("refused", skill))

    def undeliverable_stop(self) -> None:
        """...and the hook names something that is no skill here: its words go out as a rail."""
        self.turns[-1].append(("rail", None))

    def new_turn(self, *, woken: bool = False) -> None:
        """The next turn; `woken`: the wake-up door opened it by delivering the loop skill,
        which is the harness's delivery too and is NOT a refused stop."""
        self.turns.append([("woken", None)] if woken else [])

    def write(
        self,
        root: Path,
        name: str,
        *,
        trace: str = "true",
        forget_refusals: int = 0,
        product: str = "turn",
    ) -> Path:
        """`product` is which product wrote the world (see the class). `trace` is what the
        product's trace holds: "true" every stuck note the calls drew,
        "none" no note, "half" every other one, "but-first" all but the first, "lies" only
        notes the calls cannot have drawn, "padded" the true notes and those false ones.
        `forget_refusals` leaves that many refused stops off the trace."""
        world = root / name
        folder = world / "mind-workspace" / ".zakcode" / "transcripts"
        folder.mkdir(parents=True)
        traces = world / "logs" / "traces" / "s1"
        traces.mkdir(parents=True)
        messages: list[Message] = []
        n = drawn = 0
        lap = work = 0  # the loop's two counts (ADR-0209, ADR-0196): neither is ever reset
        loop_skill = ""  # the session's: whatever the last refused stop named (ADR-0187)

        def framed(skill: str, note: str) -> Message:
            frame = (
                f"<command-message>{skill} is running</command-message>\n"
                f"<command-name>/{skill}</command-name>\n\n{self.BODY}"
            )
            return Message.user(harness_skill_turn_text(frame, note))

        def deliver(skill: str, events: list[dict]) -> None:
            nonlocal forget_refusals, lap, loop_skill
            messages.append(framed(skill, _VETO_SKILL_NOTE.format(reason="the loop goes on")))
            lap, loop_skill = lap + 1, skill  # door one: the harness delivers the loop skill
            if forget_refusals > 0:
                forget_refusals -= 1
            else:
                events.append(
                    {"kind": "intervention", "data": {"kind": "turn_end_skill", "skill": skill}}
                )

        for t, turn in enumerate(self.turns, 1):
            tracker = StuckTracker(uncounted_outcome_tools=UNCOUNTED)
            events: list[dict] = []
            epoch = 0
            for at, (op, what) in enumerate(turn):
                if op == "woken":
                    messages.append(framed(self.LOOP, LOOP_WAKE_NOTE))
                    continue
                if op != "calls":
                    messages.append(Message.assistant_text("that is all for now"))
                    if op == "refused":
                        deliver(str(what), events)
                    else:
                        messages.append(Message.user("[harness] there is more to do"))
                    if op == "rail":
                        skill_note = {"kind": "turn_end_skill", "skill": "nowhere", "refused": True}
                        events.append({"kind": "intervention", "data": skill_note})
                    tracker.reset()
                    continue
                uses, blocks = [], []
                for tool, out, err, args, *rest in what:  # type: ignore[union-attr]
                    n += 1
                    given = args or {"command": f"probe {n}"}
                    data = rest[0] if rest else None
                    uses.append(ToolUseBlock(id=f"c{n}", name=tool, input=given))
                    blocks.append(
                        ToolResultBlock(tool_use_id=f"c{n}", output=out, is_error=err, data=data)
                    )
                    events.append({"kind": "tool", "detail": tool, "ok": not err})
                    epoch += tool in EDIT_TOOLS and not err
                    work += tool not in NOT_WORK and not err
                    # Door two: the BODY of the session's loop skill answers the model's own
                    # call. The tool's flag says what is a pointer, as it does for the loop.
                    lap += (
                        tool in SKILL_TOOLS
                        and not err
                        and not (data or {}).get("pointer")
                        and bool(loop_skill)
                        and given.get("skill") == loop_skill
                    )
                messages.append(
                    Message(role="assistant", blocks=[TextBlock(text="working"), *uses])
                )
                messages.append(Message.tool_results(blocks))
                counts = {"lap": lap, "work": work} if product == "shipped" else {}
                tracker.observe(
                    [ToolCall(id=u.id, name=u.name, arguments=u.input) for u in uses],
                    blocks,
                    assistant_text="working",
                    epoch=epoch,
                    **counts,
                )
                action = tracker.next_action()
                if action is not StuckAction.CONTINUE:
                    drawn += 1
                    keep = (
                        trace in ("true", "padded")
                        or (trace == "half" and drawn % 2 == 1)
                        or (trace == "but-first" and drawn > 1)
                    )
                    if keep:
                        note = {"kind": "stuck", **tracker.evidence()}
                        events.append({"kind": "intervention", "data": note})
                if action is StuckAction.STOP and at + 1 < len(turn):
                    deliver(self.LOOP, events)  # the script runs on: the stop was refused
                    tracker.reset()
            if trace in ("lies", "padded"):  # notes the calls cannot have produced
                events += [
                    {
                        "kind": "intervention",
                        "data": {
                            "kind": "stuck",
                            "signals": SIG_REPEATED_OUTCOME,
                            "tool": "Read",
                            "repeats": 9,
                        },
                    }
                ] * 3
            (traces / f"turn_{t}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
        (folder / "s1.jsonl").write_text(render_claude_code_transcript(messages, session_id="s1"))
        return world


def selftest() -> int:
    results: list[bool] = []

    def check(name: str, got: object, want: object) -> None:
        ok = got == want
        results.append(ok)
        print(
            f"  {'ok  ' if ok else 'FAIL'} {name}{'' if ok else f' -- got {got!r}, want {want!r}'}"
        )

    banner = "iteration opened; nothing claimed; the queue holds what it held before"
    pointer = "[already loaded] the skill is running this turn: go on from where you are"

    def work(tag: str) -> str:
        return f"distinct work {tag}, with an output long enough to count as an outcome"

    def healthy(laps: int) -> _Run:
        """A served loop as the product knows one: the first stop in words is refused and the
        harness delivers the loop skill; from then on each lap the model loads it itself, prints
        the same banner ONCE and does new work."""
        run = _Run()
        run.refused_stop()
        for lap in range(laps):
            if lap:
                run.loads_the_loop()
            run.call("Bash", banner)
            run.call("Bash", work(f"of lap {lap}"))
        return run

    def circling(repeats: int) -> _Run:
        """The ADR-0038 incident's shape: one lap, one result again and again between probes."""
        run = _Run()
        run.refused_stop()
        for k in range(repeats):
            run.call("Bash", banner)
            run.call("Bash", work(f"of probe {k}"))
        return run

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        got = read_world(healthy(6).write(root, "healthy"))
        by = {r["rule"]: r for r in got["replays"]}
        check(
            "a healthy six-lap loop climbs the WHOLE ladder under the shipped rule",
            by["turn"]["rungs"],
            {"nudge": 1, "narrow": 1, "step_back": 1, "stop": 1},
        )
        check(
            "...each of its rungs has a lap boundary between the repeats",
            by["turn"]["of_them_a_lap_boundary_lay_between"],
            4,
        )
        check(
            "...the repeats are one lap (3 iterations) apart",
            by["turn"]["detail"][0].get("gaps_in_iterations"),
            [3, 3],
        )
        check(
            "...with the model's own loads of the loop skill between them, and no refused stop",
            (
                by["turn"]["detail"][0].get("own_loads_between"),
                by["turn"]["detail"][0].get("refused_stops_between"),
                by["turn"]["of_them_only_refused_stops_lay_between"],
            ),
            (2, 0, 0),
        )
        check("...under the lap rule it draws no rung", by["lap"]["rungs"], {})
        check("...nor under the body rule, nor the reentry rule", by["body"]["rungs"], {})
        check("...(the reentry rule)", by["reentry"]["rungs"], {})
        check(
            "...no epoch opened; 7 lap boundaries: 5 own loads, 2 refused stops (one its STOP)",
            (
                by["turn"]["file_edits_that_opened_an_epoch"],
                by["turn"]["lap_boundaries"],
                by["turn"]["skill_bodies_delivered"],
            ),
            (0, 7, 5),
        )
        check(
            "the control passes when the trace is the product's own",
            got["control"],
            {
                "repeated_outcome_notes_on_the_trace": 4,
                "repeated_outcome_rungs_replayed": 4,
                "matched_in_order": 4,
                "refused_stops_on_the_trace": 2,
                "refused_stops_in_the_transcript": 2,
                "passes": True,
            },
        )

        got = read_world(circling(5).write(root, "circling"))
        by = {r["rule"]: r for r in got["replays"]}
        check(
            "a circling model climbs under the shipped rule",
            by["turn"]["rungs"],
            {"nudge": 1, "narrow": 1, "step_back": 1},
        )
        check("...and just the same under the lap rule", by["lap"]["rungs"], by["turn"]["rungs"])
        check("...and under the body rule", by["body"]["rungs"], by["turn"]["rungs"])
        check(
            "...and under the reentry rule: no Skill call lies between its repeats",
            by["reentry"]["rungs"],
            by["turn"]["rungs"],
        )
        check(
            "...no lap boundary lies between its repeats, and the output was no error",
            (
                by["turn"]["of_them_a_lap_boundary_lay_between"],
                by["turn"]["of_them_the_output_was_an_error"],
            ),
            (0, 0),
        )
        check(
            "...a window of 40 iterations still sees it",
            by["window:40"]["rungs"],
            by["turn"]["rungs"],
        )

        # The loop skill is whatever the hook named. Until a stop has been refused nothing is.
        run = _Run()
        for lap in range(4):
            run.loads_the_loop()
            run.call("Bash", banner)
            run.call("Bash", work(f"of lap {lap}"))
        got = read_world(run.write(root, "never-refused"))
        by = {r["rule"]: r for r in got["replays"]}
        check(
            "a loop no hook has named has no lap: the lap rule climbs as the shipped rule does",
            (by["lap"]["rungs"], by["lap"]["lap_boundaries"], by["body"]["rungs"]),
            ({"nudge": 1, "narrow": 1}, 0, {}),
        )

        # A model that hops between OTHER skills while one result keeps coming back.
        run = _Run()
        run.refused_stop()
        for k in range(4):
            run.call("Skill", f"# another procedure {k}\n" + "step\n" * 6, args={"skill": f"s{k}"})
            run.call("Bash", banner)
        got = read_world(run.write(root, "hopping"))
        by = {r["rule"]: r for r in got["replays"]}
        check(
            "a body of ANOTHER skill is no lap: the lap rule still climbs, the body rule does not",
            (by["lap"]["rungs"], by["body"]["rungs"]),
            ({"nudge": 1, "narrow": 1}, {}),
        )

        run = _Run()
        run.refused_stop()
        for _ in range(4):
            run.call("Skill", pointer, args={"skill": _Run.LOOP})
            run.call("Bash", banner)
        got = read_world(run.write(root, "pointers"))
        by = {r["rule"]: r for r in got["replays"]}
        check(
            "a POINTER is not a lap: the lap rule still climbs, and so does the body rule",
            (by["lap"]["rungs"], by["body"]["rungs"]),
            ({"nudge": 1, "narrow": 1}, {"nudge": 1, "narrow": 1}),
        )
        check(
            "...the reentry rule does not (why it is described and not licensed)",
            by["reentry"]["rungs"],
            {},
        )
        check(
            "...Skill calls lie between the repeats; bodies and laps do not",
            (
                by["turn"]["of_them_a_skill_call_lay_between"],
                by["turn"]["of_them_a_body_lay_between"],
                by["turn"]["of_them_a_lap_boundary_lay_between"],
            ),
            (2, 0, 0),
        )

        # Every lap ends in words and is refused: the harness's delivery alone marks the laps.
        run = _Run()
        for lap in range(6):
            run.refused_stop()
            run.call("Bash", banner)
            run.call("Bash", work(f"of lap {lap}"))
        got = read_world(run.write(root, "all-refused"))
        by = {r["rule"]: r for r in got["replays"]}
        check(
            "laps marked by refused stops alone: the shipped rule climbs, the lap rule does not",
            (sum(by["turn"]["rungs"].values()), by["lap"]["rungs"], by["body"]["rungs"]),
            (4, {}, by["turn"]["rungs"]),
        )
        check(
            "...and every one of those rungs had ONLY refused stops between its repeats",
            (
                by["turn"]["detail"][0].get("own_loads_between"),
                by["turn"]["detail"][0].get("refused_stops_between"),
                by["turn"]["of_them_only_refused_stops_lay_between"],
            ),
            (0, 2, 4),
        )
        check(
            "...and every refused stop is on both records (six in words, one at the stop rung)",
            (
                got["control"]["refused_stops_on_the_trace"],
                got["control"]["refused_stops_in_the_transcript"],
                got["control"]["passes"],
            ),
            (7, 7, True),
        )

        run = _Run()
        run.refused_stop()
        for _ in range(5):
            run.call(
                "Edit",
                "edited one file; this output is long enough to count",
                args={"file_path": "a.py"},
            )
            run.call("Bash", banner)
        got = read_world(run.write(root, "edits"))
        check(
            "a file edit between repeats opens an epoch: Bash never climbs",
            got["replays"][0]["by_tool"].get("Bash"),
            None,
        )
        check(
            "...and a run that drew no rung has nothing to reproduce: its control does not pass",
            (got["control"]["matched_in_order"], got["control"]["passes"]),
            (0, False),
        )

        # The epoch the tracker is handed is read AFTER the batch ran: an edit and the banner in
        # ONE completion put that banner in the new epoch.
        run = _Run()
        run.refused_stop()
        run.call("Bash", banner)
        run.call("Bash", banner)
        run.batch(
            ("Edit", "edited one file; this output is long enough to count", False, {"p": "a"}),
            ("Bash", banner, False, {"command": "look"}),
        )
        got = read_world(run.write(root, "edit-in-the-batch"))
        check(
            "an edit in the same batch as the third banner: that banner is a first sighting",
            got["replays"][0]["rungs"],
            {},
        )

        run = healthy(3)
        run.new_turn(woken=True)
        for k in range(2):
            run.call("Bash", banner)
            run.call("Bash", work(f"of the second turn {k}"))
        got = read_world(run.write(root, "two-turns"))
        check(
            "a new TURN starts a new tracker: 3 + 2 returns of the banner draw ONE nudge",
            (got["turns"], got["replays"][0]["rungs"]),
            (2, {"nudge": 1}),
        )
        check(
            "...and the wake-up door's delivery that opened the second turn is no refused stop",
            (
                got["control"]["refused_stops_on_the_trace"],
                got["control"]["refused_stops_in_the_transcript"],
            ),
            (1, 1),
        )

        run = _Run()
        run.refused_stop()
        for k in range(4):  # the fourth banner draws the read-only rung...
            run.call("Bash", banner)
            if k < 3:
                run.call("Bash", work(f"of probe {k}"))
        run.call(  # ...and the very next iteration is refused
            "Bash", "permission denied: the iteration is limited to read-only tools", err=True
        )
        got = read_world(run.write(root, "narrowed"))
        check(
            "an error in the iteration after a read-only rung is counted",
            got["replays"][0]["errors_in_the_iteration_after_a_read_only_rung"],
            [1],
        )
        got = read_world(circling(4).write(root, "narrowed-quietly"))
        check(
            "...and an iteration after it that went through counts none",
            got["replays"][0]["errors_in_the_iteration_after_a_read_only_rung"],
            [0],
        )

        run = _Run()
        run.refused_stop()
        for k in range(3):
            run.call("Bash", "make: no rule to make target all; stop (exit status 2)", err=True)
            run.call("Bash", work(f"of probe {k}"))
        got = read_world(run.write(root, "same-error"))
        check(
            "a repeated output that was an ERROR is told apart",
            (
                got["replays"][0]["repeated_outcome_rungs"],
                got["replays"][0]["of_them_the_output_was_an_error"],
            ),
            (1, 1),
        )

        # The same short refusal over and over: the ladder climbs on its OTHER signals, and a
        # short output is never an outcome. Such rungs are drawn, and never counted here.
        run = _Run()
        run.refused_stop()
        for _ in range(5):
            run.call("Bash", "no", err=True, args={"command": "make"})
        got = read_world(run.write(root, "other-signals"))
        check(
            "rungs drawn by other signals are drawn, and none counts as a repeated outcome",
            (
                got["replays"][0]["rungs"],
                got["replays"][0]["repeated_outcome_rungs"],
                got["control"]["repeated_outcome_notes_on_the_trace"],
                got["control"]["repeated_outcome_rungs_replayed"],
            ),
            ({"nudge": 1, "narrow": 1}, 0, 0, 0),
        )
        check(
            "...by count too: no such note and no such rung replayed, so the notes tell nothing",
            (
                got["control_by_count"]["turn"]["repeated_outcome_rungs_replayed"],
                got["control_by_count"]["shipped"]["repeated_outcome_rungs_replayed"],
                got["control_by_count"]["written_under"],
            ),
            (0, 0, "either"),
        )
        # ...and a completion in words between them is where the loop resets the tracker: the
        # streak those signals had built starts over, in the replay as in the product.
        run = _Run()
        run.refused_stop()
        for k in range(7):  # unbroken, seven of them would climb three rungs
            run.call("Bash", "no", err=True, args={"command": "make"})
            if k == 2:
                run.says_only()
        got = read_world(run.write(root, "reset-in-words"))
        check(
            "a completion in words resets the streak: the four refusals after it draw ONE nudge",
            got["replays"][0]["rungs"],
            {"nudge": 1},
        )

        # ...and so is a stop rung the run went on from: the stop was refused. Sixteen of them:
        # the ladder climbs to its stop, starts over, and reaches one more nudge.
        run = _Run()
        run.refused_stop()
        for _ in range(16):
            run.call("Bash", "no", err=True, args={"command": "make"})
        got = read_world(run.write(root, "stop-refused"))
        check(
            "a stop rung the run went on from resets the tracker, as the loop does",
            got["replays"][0]["rungs"],
            {"nudge": 3, "narrow": 2, "step_back": 1, "stop": 1},
        )

        # Two counted calls in one batch, under a rule that starts again: the rung is about the
        # call the TRACKER counted to three since the lap began (a result that was fine), not
        # about the failure sighted five times in the turn.
        run = _Run()
        run.refused_stop()
        failure = "make: no rule to make target all; stop (exit status 2)"
        for k in range(2):
            run.call("Bash", failure, err=True, args={"command": f"early {k}"})
            run.call("Bash", work(f"between the early failures {k}"))
        run.loads_the_loop()
        for k in range(3):
            run.batch(
                ("Bash", banner, False, {"command": f"look {k}"}),
                ("Bash", failure, True, {"command": f"late {k}"}),
            )
            run.call("Bash", work(f"between the batches {k}"))
        got = read_world(run.write(root, "two-in-a-batch"))
        first = (got["replays"][RULES.index("lap")]["detail"] or [{}])[0]
        check(
            "under the lap rule the rung is about the call THAT rule counted to three",
            (first.get("repeats"), first.get("was_an_error"), first.get("laps_between")),
            (3, False, 0),
        )

        # Two laps in one turn: the banner twice in the first, three times in the second.
        run = _Run()
        run.refused_stop()
        run.call("Bash", banner)
        run.call("Bash", work("of the first lap"))
        run.call("Bash", banner)
        run.loads_the_loop()
        for k in range(3):
            run.call("Bash", banner)
            run.call("Bash", work(f"of the second lap {k}"))
        got = read_world(run.write(root, "two-laps"))
        first = (got["replays"][RULES.index("lap")]["detail"] or [{}])[0]
        check(
            "under the lap rule a rung is measured over the repeats THAT rule counted",
            (first.get("repeats"), first.get("gaps_in_iterations"), first.get("laps_between")),
            (3, [2, 2], 0),
        )

        liar = read_world(healthy(6).write(root, "liar", trace="lies"))
        check(
            "a trace the calls cannot have written fails the control",
            liar["control"]["passes"],
            False,
        )
        half = read_world(healthy(6).write(root, "half", trace="half"))
        check(
            "a replay with rungs the product never noted fails it (2 noted, 4 replayed)",
            (half["control"]["matched_in_order"], half["control"]["passes"]),
            (2, False),
        )
        padded = read_world(healthy(6).write(root, "padded", trace="padded"))
        check(
            "a trace with notes the replay does not reproduce fails it (4 of 7 matched)",
            (padded["control"]["matched_in_order"], padded["control"]["passes"]),
            (4, False),
        )
        nearly = read_world(healthy(7).write(root, "nearly", trace="but-first"))
        check(
            "a replay matching exactly 80% is believed (4 of 5)",
            (
                nearly["control"]["matched_in_order"],
                nearly["control"]["repeated_outcome_rungs_replayed"],
                nearly["control"]["passes"],
            ),
            (4, 5, True),
        )
        check(
            "...by count too: exactly 80% under `turn` reproduces the notes, so `turn` wrote it",
            (
                nearly["control_by_count"]["turn"]["reproduces_the_notes"],
                nearly["control_by_count"]["written_under"],
            ),
            (True, "turn"),
        )
        check(
            "...each stop rung the run went on from was a refused stop, on both records",
            (
                nearly["control"]["refused_stops_on_the_trace"],
                nearly["control"]["refused_stops_in_the_transcript"],
            ),
            (3, 3),
        )
        last = (nearly["replays"][0]["detail"] or [{}])[-1]
        check(
            "...a rung with own loads AND a refused stop between its repeats is not one of the "
            "refused-stops-only ones",
            (
                last.get("own_loads_between"),
                last.get("refused_stops_between"),
                nearly["replays"][0]["of_them_only_refused_stops_lay_between"],
            ),
            (6, 1, 0),
        )
        silent = read_world(healthy(6).write(root, "silent", trace="none"))
        check(
            "a trace with no stuck note fails it too (nothing to reproduce)",
            silent["control"]["passes"],
            False,
        )

        def refusals(name: str, forgotten: int) -> dict:
            run = healthy(6)
            for _ in range(3):
                run.refused_stop()
                run.call("Bash", work(f"after a refusal in {name}"))
            return read_world(run.write(root, name, forget_refusals=forgotten))["control"]

        one_off = refusals("one-refusal-off", 1)
        check(
            "one refused stop missing from the trace is a run cut off: still believed",
            (
                one_off["refused_stops_on_the_trace"],
                one_off["refused_stops_in_the_transcript"],
                one_off["passes"],
            ),
            (4, 5, True),
        )
        check("...two missing is not", refusals("two-refusals-off", 2)["passes"], False)
        run = healthy(6)
        run.undeliverable_stop()
        run.call("Bash", work("after the rail"))
        got = read_world(run.write(root, "undeliverable"))
        check(
            "a refused stop the harness could not deliver is no delivery, on either record",
            (
                got["control"]["refused_stops_on_the_trace"],
                got["control"]["refused_stops_in_the_transcript"],
            ),
            (2, 2),
        )
        (root / "empty").mkdir()
        check(
            "a world with no transcript is refused by name",
            read_world(root / "empty").get("refused"),
            "0 transcript files, not one",
        )
        twice = healthy(6).write(root, "two-sessions")
        held = twice / "mind-workspace" / ".zakcode" / "transcripts"
        (held / "s2.jsonl").write_text((held / "s1.jsonl").read_text())
        check(
            "...and so is one holding two sessions: their order is not on disk",
            read_world(twice).get("refused"),
            "2 transcript files, not one",
        )

        regular = [read_world(healthy(6).write(root, f"h{i}")) for i in range(3)]
        check(
            "reading: three healthy loops",
            (reading(regular)["batch"], reading(regular).get("share_the_lap_rule_removes")),
            ("LOOP REGULARITY", "1"),
        )
        stuck = [read_world(circling(6).write(root, f"c{i}")) for i in range(3)]
        check(
            "reading: three circling runs",
            (reading(stuck)["batch"], reading(stuck).get("share_the_lap_rule_removes")),
            ("CIRCLING", "0"),
        )
        check("reading: two of each", reading(regular[:2] + stuck[:2])["batch"], "MIXED")
        check("reading: too few rungs", reading([stuck[0]])["batch"], "TOO FEW")
        check(
            "reading: most replays unbelieved",
            reading([liar, silent, regular[0]])["batch"],
            "REPLAY NOT TRUSTED",
        )
        check(
            "reading: an unbelieved world is left out and named",
            sorted(reading([liar, regular[0], regular[1]])["worlds_left_out"]),
            ["liar"],
        )
        check(
            "reading: half the worlds believed is enough to read",
            reading([liar, regular[0]])["batch"],
            "TOO FEW",
        )

        # A batch of calls: the uncounted Skill call must never stand in for the repeated
        # observation. The pointer comes back every iteration; every second one also holds the
        # banner and a call that fails in a new way each time.
        run = _Run()
        run.refused_stop()
        for k in range(8):
            calls = [("Skill", pointer, False, {"skill": _Run.LOOP})]
            if k % 2:
                calls.append(("Bash", banner, False, {"command": f"probe {k}"}))
                calls.append(
                    (
                        "Bash",
                        f"attempt {k} failed in a way of its own, at length",
                        True,
                        {"command": f"try {k}"},
                    )
                )
            run.batch(*calls)
        got = read_world(run.write(root, "batches"))
        first = (got["replays"][0]["detail"] or [{}])[0]
        check(
            "in a batch the repeated observation is the counted call, two iterations apart",
            first.get("gaps_in_iterations"),
            [2, 2],
        )
        check(
            "...and it was no error, though the same batch held one",
            first.get("was_an_error"),
            False,
        )
        check(
            "a healthy loop under a window shorter than two laps draws no rung",
            replay(
                [read_transcript(next((root / "healthy").rglob("s1.jsonl")))["iterations"]],
                "window:5",
            )["rungs"],
            {},
        )

        # Thresholds met exactly are met, on counts alone (no transcript needed).
        def counts(name: str, whole_turn: int, lap: int) -> dict:
            blank = {
                "of_them_a_lap_boundary_lay_between": 0,
                "of_them_only_refused_stops_lay_between": 0,
                "of_them_a_body_lay_between": 0,
                "of_them_a_skill_call_lay_between": 0,
                "of_them_the_output_was_an_error": 0,
            }
            by_rule = {
                "turn": whole_turn,
                "lap": lap,
                "body": 0,
                "reentry": 0,
                "window:40": whole_turn,
                "shipped": 0,
            }
            return {
                "world": name,
                "control": {"passes": True},
                "replays": [{"repeated_outcome_rungs": by_rule[r], **blank} for r in RULES],
            }

        check("10 rungs are enough", reading([counts("a", 10, 3)])["batch"], "LOOP REGULARITY")
        check("9 are not", reading([counts("a", 9, 0)])["batch"], "TOO FEW")
        check(
            "the lap rule removing exactly 70% is LOOP REGULARITY",
            reading([counts("a", 20, 6)])["batch"],
            "LOOP REGULARITY",
        )
        check("...69% is MIXED", reading([counts("a", 100, 31)])["batch"], "MIXED")
        check("...exactly 30% is CIRCLING", reading([counts("a", 20, 14)])["batch"], "CIRCLING")
        check("...31% is MIXED", reading([counts("a", 100, 69)])["batch"], "MIXED")
        check(
            "rungs pool across worlds",
            reading([counts("a", 5, 0), counts("b", 5, 5)]).get("share_the_lap_rule_removes"),
            "1/2",
        )

        # ── ADR-0209: the product counts per lap, and this file replays both counts ──
        def both(world: dict) -> tuple[dict, dict, str]:
            by = {r["rule"]: r for r in world["replays"]}
            return by["turn"], by["shipped"], world["control_by_count"]["written_under"]

        whole = {"nudge": 1, "narrow": 1, "step_back": 1, "stop": 1}
        old, new, under = both(read_world(healthy(6).write(root, "healthy-before-0209")))
        check(
            "a healthy loop written BEFORE ADR-0209: its notes are the whole-turn count's",
            (old["rungs"], new["rungs"], under),
            (whole, {}, "turn"),
        )
        got = read_world(healthy(6).write(root, "healthy-since-0209", product="shipped"))
        old, new, under = both(got)
        check(
            "...and written SINCE: no note, none replayed under `shipped`, the ladder under `turn`",
            (old["rungs"], new["rungs"], under),
            (whole, {}, "shipped"),
        )
        check(
            "...sample 7's control leaves that world out: it has no note to reproduce",
            (got["control"]["repeated_outcome_notes_on_the_trace"], got["control"]["passes"]),
            (0, False),
        )
        got = read_world(circling(5).write(root, "circling-since-0209", product="shipped"))
        old, new, under = both(got)
        check(
            "circling inside one lap climbs under both counts, so its notes cannot say which",
            (new["rungs"], old["rungs"] == new["rungs"], under),
            ({"nudge": 1, "narrow": 1, "step_back": 1}, True, "either"),
        )

        # THE BOUND. A model that probes, stops and is sent round again: a lap boundary lies
        # between every two sightings, and no lap after the first shows anything new.
        run = _Run()
        for _ in range(6):
            run.call("Bash", banner)
            run.refused_stop()
        got = read_world(run.write(root, "stops-after-every-probe", product="shipped"))
        old, new, under = both(got)
        plain = {r["rule"]: r for r in got["replays"]}["lap"]
        check(
            "a model that stops after every probe still climbs under `shipped`, one lap late",
            (new["rungs"], old["rungs"], under),
            ({"nudge": 1, "narrow": 1, "step_back": 1}, whole, "shipped"),
        )
        check(
            "...and never under the plain lap rule: the bound is what catches it",
            plain["rungs"],
            {},
        )

        run = _Run()
        run.refused_stop()
        for _ in range(4):
            run.points_at_the_loop(pointer)
            run.call("Bash", banner)
        old, new, under = both(read_world(run.write(root, "pointers-since", product="shipped")))
        check(
            "a POINTER is no lap to the product either: `shipped` climbs as `turn` does",
            (new["rungs"], old["rungs"] == new["rungs"], under),
            ({"nudge": 1, "narrow": 1}, True, "either"),
        )

        # The plan tool's receipt, from the REAL tool. First the words this file knows it by.
        desk = _Run()
        desk.plans("the first note")
        desk.plans("the first note")  # the same plan again: the tool's OTHER receipt
        (changed,), (resent,) = (op[1] for op in desk.turns[0])
        check(
            "the real plan tool flags `Plan updated` and begins it with RECEIPT_HEAD",
            (changed[1].startswith(RECEIPT_HEAD), (changed[4] or {}).get(RECEIPT_OF_CHANGE)),
            (True, True),
        )
        check(
            "...and neither flags nor so begins the receipt for a plan sent back unchanged",
            (resent[1].startswith(RECEIPT_HEAD), RECEIPT_OF_CHANGE in (resent[4] or {})),
            (False, False),
        )
        check(
            "the flag is put back on exactly that result: not an error, not another tool's",
            [
                _result("update_plan", "c", {"content": changed[1]}).data,
                _result("update_plan", "c", {"content": resent[1]}).data,
                _result("update_plan", "c", {"content": changed[1], "is_error": True}).data,
                _result("Bash", "c", {"content": changed[1]}).data,
            ],
            [{RECEIPT_OF_CHANGE: True}, None, None, None],
        )

        run = _Run()
        run.refused_stop()
        for k in range(6):
            run.call("Bash", work(f"between two plan notes {k}"))
            run.plans(f"so far: {k} checks")
        old, new, under = both(read_world(run.write(root, "journalling", product="shipped")))
        check(
            "journalling into the plan between steps: no rung under `shipped`, four under `turn`",
            (new["rungs"], old["by_tool"], under),
            ({}, {"update_plan": 4}, "shipped"),
        )
        run = _Run()
        run.refused_stop()
        for k in range(6):
            run.plans(f"rewritten {k} times")
        old, new, under = both(read_world(run.write(root, "plan-churn", product="shipped")))
        check(
            "...rewriting the plan with NOTHING in between still climbs under `shipped`",
            (new["by_tool"], new["rungs"] == old["rungs"], under),
            ({"update_plan": 4}, True, "either"),
        )
        # Work that FAILED is no work (ADR-0196 counts successful calls): with nothing but
        # failures between them, the receipts are plan churn still.
        run = _Run()
        run.refused_stop()
        for k in range(6):
            run.call("Read", f"no such file: notes-{k}.md (tried {k + 1} places)", err=True)
            run.plans(f"looked in {k + 1} places")
        old, new, under = both(read_world(run.write(root, "failed-work", product="shipped")))
        check(
            "...work that FAILED between two plan notes is no work: the receipts still climb",
            (new["by_tool"].get("update_plan"), old["by_tool"].get("update_plan"), under),
            (4, 4, "either"),
        )
        # One batch holding a look that FAILED (new words each time) and the churned receipt.
        run = _Run()
        run.refused_stop()
        for k in range(4):
            desk = _Run()  # asked of the real tool, then put in one batch with the look
            desk._plan = run._plan
            desk.plans(f"tried {k + 1} places")
            ((receipt,),) = (op[1] for op in desk.turns[0])
            run.batch(("Read", f"no such file: notes-{k}.md (place {k + 1})", True, {}), receipt)
        got = read_world(run.write(root, "mixed-batch", product="shipped"))
        _, new, under = both(got)
        first = (new["detail"] or [{}])[0]  # no rung drawn is a FAILED check, never a crash
        check(
            "in a batch of a failed look and a churned receipt, the rung is said of the RECEIPT",
            tuple(first.get(k) for k in ("tool", "receipt", "was_an_error", "gaps_in_iterations")),
            ("update_plan", True, False, [1, 1]),
        )
        check(
            "...and receipts are counted apart: by the replay under either count, and on the trace",
            (
                new["of_them_on_a_receipt"],
                got["control_by_count"]["of_them_on_a_receipt"],
                got["replays"][0]["of_them_on_a_receipt"],
                under,
            ),
            (2, 2, 2, "either"),
        )
        check(
            "a trace the calls cannot have written is reproduced by NEITHER count",
            liar["control_by_count"]["written_under"],
            "neither",
        )

    print(f"{sum(results)} ok, {len(results) - sum(results)} failed")
    return 0 if all(results) else 1


def main() -> int:
    if sys.argv[1:] == ["--selftest"]:
        return selftest()
    if not sys.argv[1:]:
        print(__doc__)
        return 2
    worlds = [read_world(Path(arg)) for arg in sys.argv[1:]]
    print(json.dumps({"worlds": worlds, "reading": reading(worlds)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
