#!/usr/bin/env python
"""Decision-point bench: does a LOW reasoning effort change what a small gpt-5.6-tier model does
at the four points where the served loop saw it go wrong? Direct provider calls, stdlib only, no
client library in between (a library's 200 is not the provider's acceptance; see the
pre-registration log).

Arms (the route and the effort are separated on purpose):
  A  chat       /v1/chat/completions  reasoning_effort=none   what Zak-Code sends today
  B  resp-low   /v1/responses         reasoning.effort=low    tools WITH reasoning: only here
  C  resp-none  /v1/responses         reasoning.effort=none   route control (B's route, A's effort)

Cells (one fixed history each; scored on the model's NEXT response only, mechanically):
  order    a dependency ("deploy only if the build succeeds"): PASS = the first response runs
           the build and does not run the deploy in the same batch.
  refusal  the same call refused twice with advice that names the call just refused: PASS = the
           next action is neither the identical re-issue nor the other queue (a different item).
  pointer  a skill asked for again gets the product's "[already loaded]" pointer: PASS = the
           first step's command next (not the skill tool again, not text only, not plan only).
  body     a skill body whose first step is a command: PASS = that command is run next.
  typed    the same body, arriving IN the operator's typed /command message (the product's own
           frame), with no call made yet: PASS = the first step's command is the first action.
           This is the first decision of a served run (sample 3, 2026-09-18).

Every outcome carries a label that names WHICH way it went, so a FAIL is never just a FAIL and
a reader who disagrees with a PASS line can re-score from the rows.

Usage:
  effort_decision_points.py --selftest                       scorer control, no network
  effort_decision_points.py --model gpt-5.6-luna --n 20 --out results.jsonl
The key is read from OPENAI_API_KEY and never printed. Output: one JSON row per call, without
any prompt text.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any

CHAT_URL = "https://api.openai.com/v1/chat/completions"
RESP_URL = "https://api.openai.com/v1/responses"

ARMS: dict[str, tuple[str, str]] = {
    "A-chat-none": ("chat", "none"),
    "B-resp-low": ("resp", "low"),
    "C-resp-none": ("resp", "none"),
}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "bash",
        "description": "Run a shell command in the workspace and return its output.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file from the workspace.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "skill",
        "description": "Load the instructions of a named skill. Loading a skill does not run it.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "update_plan",
        "description": "Record or update your plan as a list of steps.",
        "parameters": {
            "type": "object",
            "properties": {"steps": {"type": "array", "items": {"type": "string"}}},
            "required": ["steps"],
        },
    },
]

SYSTEM = (
    "You are a coding agent working in a repository through tools. Act through the tools; "
    "when you are done, say so in one or two sentences."
)

#: Zak-Code's own pointer text (src/zakcode/__init__.py), verbatim, with the skill name filled in.
POINTER = (
    "[already loaded] Nothing new was loaded: the full instructions for skill 'release-check' are "
    "already in your context THIS turn, unchanged — in the /command message you were given, or an "
    "earlier Skill result — and you have run no tool on them since they arrived. Loading a skill "
    "does not run it. Carry those instructions out now, starting from their first step: your next "
    "action is "
    "that step's tool call, not another Skill call and not a summary."
)

#: A synthesized skill body (NOT any private framework text): purpose, a table, then the steps.
BODY = """# release-check

Purpose: decide whether the current branch may be released. This skill is a procedure, not a report.

| Signal | Where it comes from | What it means |
|---|---|---|
| preflight | scripts/preflight.sh | environment and lockfile are consistent |
| unit | scripts/unit.sh | the fast tests pass |
| notes | docs/RELEASE.md | the release notes name this version |

Rules: run the steps in order; stop at the first failure and say which step failed; do not summarize
the procedure instead of running it; do not skip a step because its result seems obvious.

## Steps

Step 1. Run `bash scripts/preflight.sh` and read its output. This is your FIRST action.
Step 2. Only if step 1 passed, run `bash scripts/unit.sh`.
Step 3. Only if step 2 passed, read docs/RELEASE.md and check it names the version from step 1.
Step 4. Report: released or not, and the step that decided it.
"""

#: The product's command-expansion frame for a typed ``/<skill>`` (compose_skill_turn): the
#: body follows it in the SAME user message, so the instructions arrive with no call made.
TYPED_FRAME = (
    "<command-message>release-check is running</command-message>\n"
    "<command-name>/release-check</command-name>\n\n"
)

REFUSED = json.dumps(
    {
        "error": "ambiguous_item",
        "detail": (
            "item-7 exists in BOTH the main queue and the side queue. The claim is ambiguous: "
            "name the queue explicitly and re-issue. --queue main claims the main copy, "
            "--queue side claims the "
            "side copy."
        ),
    }
)
CLAIM_CMD = "bash scripts/claim-item.sh item-7 --queue main"

# An abstract history is a list of steps:
#   ("user", text) | ("call", id, name, args) | ("result", id, output)
History = list[tuple[Any, ...]]

CELLS: dict[str, History] = {
    "order": [
        (
            "user",
            "Ship the current branch: run `bash scripts/build.sh`, and only if the build succeeds "
            "run "
            "`bash scripts/deploy.sh`. Never deploy a build you have not seen succeed.",
        )
    ],
    "refusal": [
        (
            "user",
            "You selected item-7 from the MAIN queue (title: 'Rotate the staging certificate'). "
            "Claim it with scripts/claim-item.sh, then start on it. The side queue holds unrelated "
            "items that may "
            "reuse ids.",
        ),
        ("call", "c1", "bash", {"command": CLAIM_CMD}),
        ("result", "c1", REFUSED),
        ("call", "c2", "bash", {"command": CLAIM_CMD}),
        ("result", "c2", REFUSED),
    ],
    "pointer": [
        ("user", "/release-check"),
        ("call", "s1", "skill", {"name": "release-check"}),
        ("result", "s1", BODY),
        ("call", "s2", "skill", {"name": "release-check"}),
        ("result", "s2", POINTER),
    ],
    "body": [
        ("user", "/release-check"),
        ("call", "s1", "skill", {"name": "release-check"}),
        ("result", "s1", BODY),
    ],
    "typed": [("user", TYPED_FRAME + BODY)],
}


def chat_payload(model: str, effort: str, history: History) -> dict[str, Any]:
    msgs: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM}]
    for step in history:
        if step[0] == "user":
            msgs.append({"role": "user", "content": step[1]})
        elif step[0] == "call":
            _, cid, name, args = step
            msgs.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": cid,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }
                    ],
                }
            )
        else:
            _, cid, output = step
            msgs.append({"role": "tool", "tool_call_id": cid, "content": output})
    return {
        "model": model,
        "messages": msgs,
        "tools": [{"type": "function", "function": t} for t in TOOLS],
        "reasoning_effort": effort,
    }


def resp_payload(model: str, effort: str, history: History) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for step in history:
        if step[0] == "user":
            items.append({"role": "user", "content": step[1]})
        elif step[0] == "call":
            _, cid, name, args = step
            items.append(
                {
                    "type": "function_call",
                    "call_id": cid,
                    "name": name,
                    "arguments": json.dumps(args),
                }
            )
        else:
            _, cid, output = step
            items.append({"type": "function_call_output", "call_id": cid, "output": output})
    return {
        "model": model,
        "instructions": SYSTEM,
        "input": items,
        "tools": [{"type": "function", **t} for t in TOOLS],
        "reasoning": {"effort": effort},
        "store": False,
    }


def normalise(route: str, body: dict[str, Any]) -> dict[str, Any]:
    """One shape for both routes: tool calls (name + parsed args), text length, usage counts."""
    calls: list[dict[str, Any]] = []
    text = ""
    if route == "chat":
        msg = body["choices"][0]["message"]
        text = msg.get("content") or ""
        for tc in msg.get("tool_calls") or []:
            calls.append(
                {"name": tc["function"]["name"], "args": _args(tc["function"].get("arguments"))}
            )
        usage = body.get("usage") or {}
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        tokens_in, tokens_out = usage.get("prompt_tokens"), usage.get("completion_tokens")
    else:
        for item in body.get("output") or []:
            if item.get("type") == "function_call":
                calls.append({"name": item.get("name"), "args": _args(item.get("arguments"))})
            elif item.get("type") == "message":
                for part in item.get("content") or []:
                    text += part.get("text") or ""
        usage = body.get("usage") or {}
        reasoning = (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
        tokens_in, tokens_out = usage.get("input_tokens"), usage.get("output_tokens")
    return {
        "calls": calls,
        "text_chars": len(text.strip()),
        "reasoning_tokens": reasoning,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }


def _args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {"_unparsed": str(raw)[:200]}
    return parsed if isinstance(parsed, dict) else {"_unparsed": str(parsed)[:200]}


_BUILD_THEN_DEPLOY = re.compile(r"build\.sh\b[^;&|\n]*&&[^;|\n]*deploy\.sh")


def _commands(calls: list[dict[str, Any]]) -> list[str]:
    return [str(c["args"].get("command") or "") for c in calls if c["name"] == "bash"]


def score(cell: str, resp: dict[str, Any]) -> tuple[bool, str]:
    """(PASS?, outcome label). The label names WHICH way it went, so a FAIL is never just a FAIL."""
    calls = resp["calls"]
    cmds = _commands(calls)
    names = [c["name"] for c in calls]
    if not calls:
        return False, "text-only"
    if cell == "order":
        if any("deploy" in c for c in cmds):
            # `build && deploy` lets the SHELL gate the deploy, but the task says never deploy
            # a build you have not SEEN succeed: still a FAIL, labelled apart from the
            # unconditional forms so the two are never confused.
            gated = all(_BUILD_THEN_DEPLOY.search(c) for c in cmds if "deploy" in c)
            return False, "deploy-chained-after-build" if gated else "deploy-in-first-response"
        if any("build" in c for c in cmds):
            return True, "build-first"
        return False, "other-first-action"
    if cell == "refusal":
        squashed = [" ".join(c.split()) for c in cmds]
        if any(c == CLAIM_CMD or c.endswith("claim-item.sh item-7 --queue main") for c in squashed):
            return False, "identical-reissue"
        if any("claim-item.sh" in c and "--queue side" in c for c in squashed):
            return False, "switched-to-other-queue"
        if any("claim-item.sh" in c and "item-7" in c and "--queue" not in c for c in squashed):
            return False, "unnamed-reissue"
        return True, "different-action"
    if cell in ("pointer", "body", "typed"):
        # The first step ran: progress, whatever rode along. A skill call in the same
        # response is labelled, not failed — the loop moves either way.
        if any("preflight" in c for c in cmds):
            return True, "first-step-run+skill-again" if "skill" in names else "first-step-run"
        if "skill" in names:
            return False, "skill-again"
        if set(names) <= {"update_plan"}:
            return False, "plan-only"
        return False, "other-work-call" if cell == "pointer" else "other-first-action"
    raise ValueError(cell)


def post(url: str, payload: dict[str, Any], key: str) -> tuple[int, dict[str, Any] | str]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 - fixed https URL
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, str(json.loads(raw)["error"]["message"])[:300]
        except (ValueError, KeyError, TypeError):
            return exc.code, raw[:300]
    except (urllib.error.URLError, TimeoutError) as exc:
        return 0, f"{type(exc).__name__}: {str(exc)[:200]}"


# ── the pre-registered reading of a finished batch ────────────────────────────────────────
#: Thresholds are in PASSES PER 20 (a rate times 20), so an arm that lost a call or two to
#: HTTP errors is still read on the same scale. Fixed before the first scored call; see
#: bench/results/effort-decision-points-preregistration.log.
GAIN = 7  # B - ref >= GAIN on a cell supports BUILD
REPLICATION_GAIN = 5  # the bar a single supporting cell must clear again in a second sample
HARM = 4  # B <= ref - HARM on any cell is HARM
FLAT = 2  # |B - ref| <= FLAT on every cell is NO BUILD
ROUTE = 3  # |C - A| >= ROUTE: the route itself moved the cell, so ref becomes C
CEILING = 17  # ref >= CEILING: no room for a GAIN, the cell cannot support BUILD
MIN_SCORED = 18  # fewer scored calls than this in an arm x cell: NOT MEASURED


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]] (sum of tables no likelier than this)."""
    from math import comb

    n1, n2, k = a + b, c + d, a + c
    total = comb(n1 + n2, k)

    def prob(x: int) -> float:
        return comb(n1, x) * comb(n2, k - x) / total

    observed = prob(a)
    lo, hi = max(0, k - n2), min(k, n1)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= observed * (1 + 1e-9)))


def read_batch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-cell tables and the verdict the pre-registration names. Pure: rows in, reading out."""
    cells: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        arm = cells.setdefault(row["cell"], {}).setdefault(
            row["arm"],
            {
                "n": 0,
                "pass": 0,
                "errors": 0,
                "labels": {},
                "calls": 0,
                "reasoning": 0,
                "seconds": 0.0,
                "tokens_in": 0,
                "tokens_out": 0,
            },
        )
        if row.get("pass") is None:
            arm["errors"] += 1
            continue
        arm["n"] += 1
        arm["pass"] += 1 if row["pass"] else 0
        arm["labels"][row["label"]] = arm["labels"].get(row["label"], 0) + 1
        arm["calls"] += len(row.get("calls") or [])
        arm["reasoning"] += row.get("reasoning_tokens") or 0
        arm["seconds"] += row.get("seconds") or 0.0
        arm["tokens_in"] += row.get("tokens_in") or 0
        arm["tokens_out"] += row.get("tokens_out") or 0
    per_cell: dict[str, dict[str, Any]] = {}
    for cell, arms in cells.items():
        a, b, c = (arms.get(k) for k in ("A-chat-none", "B-resp-low", "C-resp-none"))
        if not a or not b or not c or min(a["n"], b["n"], c["n"]) < MIN_SCORED:
            per_cell[cell] = {"reading": "NOT MEASURED"}
            continue
        per20 = {k: 20 * v["pass"] / v["n"] for k, v in (("A", a), ("B", b), ("C", c))}
        route_moved = abs(per20["C"] - per20["A"]) >= ROUTE
        ref_name = "C" if route_moved else "A"
        ref = c if route_moved else a
        diff = per20["B"] - per20[ref_name]
        if diff <= -HARM:
            reading = "HARM"
        elif diff >= GAIN and per20[ref_name] < CEILING:
            reading = "GAIN"
        elif abs(diff) <= FLAT:
            reading = "FLAT"
        else:
            reading = "BETWEEN"
        per_cell[cell] = {
            "reading": reading,
            "ref": ref_name,
            "per20": {k: round(v, 1) for k, v in per20.items()},
            "diff": round(diff, 1),
            "ceiling": per20[ref_name] >= CEILING,
            "route_moved": route_moved,
            "p_B_vs_ref": round(
                fisher_two_sided(
                    b["pass"], b["n"] - b["pass"], ref["pass"], ref["n"] - ref["pass"]
                ),
                4,
            ),
        }
    readings = [v["reading"] for v in per_cell.values()]
    gains = readings.count("GAIN")
    if "NOT MEASURED" in readings:
        verdict = "NOT MEASURED"
    elif "HARM" in readings:
        verdict = "HARM"
    elif gains >= 2:
        verdict = "BUILD-CANDIDATE"
    elif gains == 1:
        verdict = "ONE-CELL GAIN: REPLICATE BEFORE READING"
    elif all(r == "FLAT" for r in readings):
        verdict = "NO BUILD"
    else:
        verdict = "MIXED"
    return {"verdict": verdict, "cells": per_cell, "tables": cells}


def report(path: str) -> int:
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    reading = read_batch(rows)
    print(f"rows: {len(rows)}  served models: {sorted({str(r.get('served_model')) for r in rows})}")
    for cell, arms in reading["tables"].items():
        print(f"\n[{cell}] {json.dumps(reading['cells'][cell])}")
        for arm, t in sorted(arms.items()):
            n = t["n"] or 1
            compatible = t["pass"] + t["labels"].get("plan-only", 0)
            print(
                f"  {arm:<12} pass {t['pass']:>2}/{t['n']:<2} progress-compatible {compatible:>2} "
                f"errors {t['errors']} calls/response {t['calls'] / n:.2f} "
                f"reasoning/response {t['reasoning'] / n:.0f} s/response {t['seconds'] / n:.1f} "
                f"tokens in/out {t['tokens_in']}/{t['tokens_out']}"
            )
            print(f"               {json.dumps(t['labels'], sort_keys=True)}")
    print(f"\nVERDICT: {reading['verdict']}")
    return 0


def selftest() -> int:
    """Every PASS form and every FAIL form of every cell, read back exactly. No network."""

    def r(*calls: tuple[str, dict[str, Any]], text: int = 0) -> dict[str, Any]:
        return {"calls": [{"name": n, "args": a} for n, a in calls], "text_chars": text}

    bash = lambda c: ("bash", {"command": c})  # noqa: E731
    cases: list[tuple[str, dict[str, Any], bool, str]] = [
        ("order", r(bash("bash scripts/build.sh")), True, "build-first"),
        (
            "order",
            r(bash("bash scripts/build.sh && bash scripts/deploy.sh")),
            False,
            "deploy-chained-after-build",
        ),
        (
            "order",
            r(bash("bash scripts/build.sh; bash scripts/deploy.sh")),
            False,
            "deploy-in-first-response",
        ),
        (
            "order",
            r(bash("bash scripts/deploy.sh && bash scripts/build.sh")),
            False,
            "deploy-in-first-response",
        ),
        (
            "order",
            r(bash("bash scripts/build.sh"), bash("bash scripts/deploy.sh")),
            False,
            "deploy-in-first-response",
        ),
        ("order", r(("read_file", {"path": "scripts/build.sh"})), False, "other-first-action"),
        ("order", r(text=40), False, "text-only"),
        ("refusal", r(bash(CLAIM_CMD)), False, "identical-reissue"),
        (
            "refusal",
            r(bash("bash  scripts/claim-item.sh item-7   --queue main")),
            False,
            "identical-reissue",
        ),
        (
            "refusal",
            r(bash("bash scripts/claim-item.sh item-7 --queue side")),
            False,
            "switched-to-other-queue",
        ),
        ("refusal", r(bash("bash scripts/claim-item.sh item-7")), False, "unnamed-reissue"),
        ("refusal", r(("read_file", {"path": "scripts/claim-item.sh"})), True, "different-action"),
        ("refusal", r(bash("bash scripts/list-queue.sh main")), True, "different-action"),
        ("refusal", r(text=200), False, "text-only"),
        ("pointer", r(bash("bash scripts/preflight.sh")), True, "first-step-run"),
        ("pointer", r(("skill", {"name": "release-check"})), False, "skill-again"),
        (
            "pointer",
            r(("skill", {"name": "release-check"}), bash("bash scripts/preflight.sh")),
            True,
            "first-step-run+skill-again",
        ),
        ("pointer", r(("update_plan", {"steps": ["run preflight"]})), False, "plan-only"),
        ("pointer", r(bash("ls scripts")), False, "other-work-call"),
        ("pointer", r(text=300), False, "text-only"),
        ("body", r(bash("bash scripts/preflight.sh")), True, "first-step-run"),
        (
            "body",
            r(("update_plan", {"steps": ["a"]}), bash("bash scripts/preflight.sh")),
            True,
            "first-step-run",
        ),
        ("body", r(("update_plan", {"steps": ["a"]})), False, "plan-only"),
        ("body", r(("skill", {"name": "release-check"})), False, "skill-again"),
        ("body", r(bash("bash scripts/unit.sh")), False, "other-first-action"),
        ("body", r(text=500), False, "text-only"),
        ("typed", r(bash("bash scripts/preflight.sh")), True, "first-step-run"),
        (
            "typed",
            r(("skill", {"name": "release-check"}), bash("bash scripts/preflight.sh")),
            True,
            "first-step-run+skill-again",
        ),
        ("typed", r(("skill", {"name": "release-check"})), False, "skill-again"),
        ("typed", r(("update_plan", {"steps": ["run preflight"]})), False, "plan-only"),
        ("typed", r(("read_file", {"path": "scripts/preflight.sh"})), False, "other-first-action"),
        ("typed", r(text=300), False, "text-only"),
    ]
    bad = 0
    for cell, resp, want_pass, want_label in cases:
        got = score(cell, resp)
        if got != (want_pass, want_label):
            bad += 1
            print(f"SELFTEST MISMATCH {cell}: want {(want_pass, want_label)} got {got}")
    # The two payload builders must carry the same history: same number of steps, same order.
    for cell, history in CELLS.items():
        chat = chat_payload("m", "none", history)["messages"][1:]
        resp = resp_payload("m", "none", history)["input"]
        if len(chat) != len(resp) or len(chat) != len(history):
            bad += 1
            print(
                f"SELFTEST MISMATCH payload lengths for {cell}: "
                f"{len(chat)} {len(resp)} {len(history)}"
            )

    # Every verdict the pre-registration names, from synthetic batches: (A, B, C) passes of 20.
    def batch(spec: dict[str, tuple[int, int, int]], scored: int = 20) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for cell, passes in spec.items():
            for arm, k in zip(ARMS, passes, strict=True):
                rows += [
                    {"cell": cell, "arm": arm, "pass": i < k, "label": "x", "calls": []}
                    for i in range(scored)
                ]
                rows += [{"cell": cell, "arm": arm, "pass": None} for _ in range(20 - scored)]
        return rows

    flat = dict.fromkeys(("order", "refusal", "pointer", "body"), (10, 11, 10))
    verdicts: list[tuple[str, dict[str, tuple[int, int, int]], int, str]] = [
        ("all flat", flat, 20, "NO BUILD"),
        ("two gains", {**flat, "typed": (8, 16, 9), "order": (9, 17, 9)}, 20, "BUILD-CANDIDATE"),
        ("one gain", {**flat, "typed": (8, 16, 9)}, 20, "ONE-CELL GAIN: REPLICATE BEFORE READING"),
        (
            "harm beats gains",
            {**flat, "typed": (8, 16, 9), "order": (9, 17, 9), "body": (14, 10, 14)},
            20,
            "HARM",
        ),
        ("a gain at the ceiling is not a gain", {**flat, "typed": (17, 20, 17)}, 20, "MIXED"),
        ("route moved: read against C", {**flat, "typed": (8, 16, 15)}, 20, "NO BUILD"),
        ("between", {**flat, "typed": (8, 13, 8)}, 20, "MIXED"),
        ("too few scored", flat, 17, "NOT MEASURED"),
    ]
    for name, spec, scored, want in verdicts:
        got_verdict = read_batch(batch(spec, scored))["verdict"]
        if got_verdict != want:
            bad += 1
            print(f"SELFTEST MISMATCH verdict {name!r}: want {want!r} got {got_verdict!r}")
    # The exact test against two hand-checked tables.
    for table, want_p in (((8, 12, 16, 4), 0.0225), ((10, 10, 10, 10), 1.0)):
        got_p = round(fisher_two_sided(*table), 4)
        if abs(got_p - want_p) > 0.0006:
            bad += 1
            print(f"SELFTEST MISMATCH fisher {table}: want {want_p} got {got_p}")
    print(
        f"selftest: {len(cases)} scorer cases, {len(CELLS)} payload pairs, "
        f"{len(verdicts)} verdict branches, 2 exact-test tables, {bad} mismatches"
    )
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="effort-decision-points.jsonl")
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--report", default="", help="read a finished batch and print the verdict")
    ns = ap.parse_args()
    if ns.selftest:
        return selftest()
    if ns.report:
        return report(ns.report)
    key = os.environ.get("OPENAI_API_KEY") or ""
    if not key:
        print("OPENAI_API_KEY is not set", file=sys.stderr)
        return 2
    cells, arms = ns.cells.split(","), ns.arms.split(",")
    total = done = 0
    with open(ns.out, "a", encoding="utf-8") as fh:
        # Interleave arms inside every repetition, so provider-side drift hits all arms alike.
        for rep in range(ns.n):
            for cell in cells:
                for arm in arms:
                    route, effort = ARMS[arm]
                    payload = (chat_payload if route == "chat" else resp_payload)(
                        ns.model, effort, CELLS[cell]
                    )
                    t0 = time.monotonic()
                    status, body = post(CHAT_URL if route == "chat" else RESP_URL, payload, key)
                    row: dict[str, Any] = {
                        "rep": rep,
                        "cell": cell,
                        "arm": arm,
                        "status": status,
                        "seconds": round(time.monotonic() - t0, 2),
                    }
                    total += 1
                    if status == 200 and isinstance(body, dict):
                        resp = normalise(route, body)
                        ok, label = score(cell, resp)
                        row.update(resp)
                        row.update({"pass": ok, "label": label, "served_model": body.get("model")})
                        done += 1
                    else:
                        row.update({"pass": None, "label": "http-error", "error": body})
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
            print(f"rep {rep + 1}/{ns.n}: {done}/{total} calls scored", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
