#!/usr/bin/env python3
"""Replay every dumped run's ``update_plan`` calls through the tool and count unchanged receipts.

The positive control for the unchanged-plan rail (ADR-0168): each run's recorded ``update_plan``
argument sequence is fed, in order, to a fresh :class:`~zakcode.tasks.TaskNetwork` through the
REAL :class:`~zakcode.tools.builtins.update_plan.UpdatePlanTool`, and the receipts are classified
(U updated, N unchanged, C cleared, E error). Other tools are not replayed — no evidence lines
reach the network — so an ADR-0116 challenge cannot fire here, which only under-counts changes.

Measured 2026-09-13 over 163 pod runs (313 plan calls): the rail fires in exactly the three
arm-K doom-loop runs (four receipts each) and in no other run.

Usage: ``python bench/plan_resends.py <dumps-root> [cell ...]`` — reads the LARGEST wire dump per
run (the conversation grows monotonically; the last file is not the longest). Prints per-run
receipt strings and totals only; never the prompts.
"""

from __future__ import annotations

import asyncio
import collections
import glob
import json
import os
import sys
from pathlib import Path

from zakcode.tasks import TaskNetwork
from zakcode.tools.base import ToolContext
from zakcode.tools.builtins.update_plan import UpdatePlanTool


def plan_calls(run_dir: str) -> list[str]:
    """The run's ``update_plan`` argument strings, in order, from its largest wire dump."""
    wires = glob.glob(os.path.join(run_dir, "wire-*.json"))
    if not wires:
        return []
    with open(max(wires, key=os.path.getsize), encoding="utf-8") as fh:
        dump = json.load(fh)
    out: list[str] = []
    for message in dump.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            if function.get("name") == "update_plan":
                out.append(str(function.get("arguments") or ""))
    return out


async def replay(calls: list[str]) -> str:
    """Feed the calls to one network; one receipt letter per call."""
    network = TaskNetwork()
    ctx = ToolContext(workspace_root=Path("/tmp"), task_network=network)
    tool = UpdatePlanTool()
    letters: list[str] = []
    for raw in calls:
        try:
            parsed = json.loads(raw)
        except ValueError:
            letters.append("E")
            continue
        args = parsed if isinstance(parsed, dict) else {"tasks": parsed}
        result = await tool.execute(args, ctx)
        output = result.output
        if result.is_error:
            letters.append("E")
        elif output.startswith("Plan unchanged"):
            letters.append("N")
        elif output.startswith("Plan cleared"):
            letters.append("C")
        elif output.startswith("Plan updated"):
            letters.append("U")
        else:
            letters.append("?")
    return "".join(letters)


def main(root: str, cells: list[str]) -> None:
    totals: collections.Counter[str] = collections.Counter()
    for cell in cells or sorted(os.listdir(root)):
        for run_dir in sorted(glob.glob(os.path.join(root, cell, "run-*"))):
            calls = plan_calls(run_dir)
            if not calls:
                continue
            receipts = asyncio.run(replay(calls))
            unchanged = receipts.count("N")
            totals["runs"] += 1
            totals["plan_calls"] += len(calls)
            totals["unchanged"] += unchanged
            totals["runs_fired"] += unchanged > 0
            print(f"{cell}/{os.path.basename(run_dir)}: {receipts} unchanged={unchanged}")
    print("---", dict(totals))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/root/zb-dumps", sys.argv[2:])
