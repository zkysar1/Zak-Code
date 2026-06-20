#!/usr/bin/env python
"""Live branching-skills demo: one skill picks between two next-skills based on what it reads.

Where run_skill_chain.py proves a FIXED relay (A -> B -> C always), this proves CONDITIONAL routing:
`triage-start` reads `INPUT.txt` and calls `use_skill('handle-urgent')` OR `use_skill('handle-normal')`
depending on the content. We run it twice — an urgent input and a routine one — and check each time
that the RIGHT branch fired and the WRONG one did NOT. That's the model making a decision mid-chain,
not following a script.

Usage:
    ./.venv/Scripts/python.exe bench/run_skill_branch.py
Config: ZSKILL_MODEL (default openai/gpt-4o-mini — reliable native tool-calling).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_task  # noqa: E402 — sibling bench script (interpreter-on-path + usage snapshot)

MODEL = os.environ.get("ZSKILL_MODEL", "openai/gpt-4o-mini")
SKILLS_DIR = Path(__file__).resolve().parent / "skill_chain" / "branch"
PROMPT = "Triage the request in INPUT.txt by using the triage-start skill, then follow its routing."

# scenario id -> (INPUT.txt contents, the skill that SHOULD fire, the RESULT.md marker)
SCENARIOS = {
    "urgent": ("urgent: the production server is down and customers are affected", "handle-urgent"),
    "normal": ("weekly status: all systems green, routine maintenance scheduled", "handle-normal"),
}
HANDLERS = {"handle-urgent", "handle-normal"}


def _build_agent(workspace: Path):
    from zakcode import Agent
    from zakcode.config import load_settings

    run_task._ensure_interpreter_on_path()
    settings = load_settings().model_copy(
        update={
            "workspace_root": str(workspace),
            "default_model": MODEL,
            "permission_mode": "autonomous",
            "max_cost_usd": 0.50,
            "max_iterations": 30,
            "api_base": None,
        }
    )
    return Agent(
        settings=settings,
        enable_skills=True,
        extra_skill_dirs=[str(SKILLS_DIR)],
        enable_compaction=True,
        enable_rules=False,
        enable_subagents=False,
        enable_mcp=False,
        enable_plugins=False,
    )


def _run_scenario(scenario: str, input_text: str, expected: str) -> dict:
    from zakcode.hooks import HookEvent, LifecyclePayload

    ws = Path(tempfile.mkdtemp(prefix=f"zskill-branch-{scenario}-"))
    (ws / "INPUT.txt").write_text(input_text, encoding="utf-8")
    agent = _build_agent(ws)

    fired: list[str] = []

    def _on_skill(payload: LifecyclePayload) -> None:
        name = str(payload.data.get("skill", ""))
        fired.append(name)
        print(f"  [{scenario}] ▸ skill fired: {name}", file=sys.stderr)

    agent.hook_manager.register_lifecycle(HookEvent.ON_SKILL_SELECTED, _on_skill)

    stop_reason = None
    crashed = None
    t0 = time.perf_counter()
    try:
        stop_reason = agent.run_turn(PROMPT).stop_reason
    except Exception as e:  # noqa: BLE001 — a crash is a (failed) result, not a runner error
        crashed = f"{type(e).__name__}: {e}"
    elapsed = time.perf_counter() - t0

    result_md = (ws / "RESULT.md").read_text(encoding="utf-8") if (ws / "RESULT.md").is_file() else ""
    shutil.rmtree(ws, ignore_errors=True)

    wrong = (HANDLERS - {expected}).pop()
    expected_marker = "URGENT" if expected == "handle-urgent" else "NORMAL"
    wrong_marker = "NORMAL" if expected == "handle-urgent" else "URGENT"
    routed_right = (
        "triage-start" in fired
        and expected in fired
        and wrong not in fired
        and expected_marker in result_md
        and wrong_marker not in result_md
    )
    return {
        "scenario": scenario,
        "input": input_text,
        "expected_branch": expected,
        "skills_fired": fired,
        "result_md": result_md.strip(),
        "stop_reason": stop_reason,
        "crashed": crashed,
        "elapsed_s": round(elapsed, 1),
        "routed_correctly": routed_right,
    }


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="  %(message)s", stream=sys.stderr)
    results = []
    for scenario, (input_text, expected) in SCENARIOS.items():
        print(f"=== scenario {scenario!r}: expect -> {expected} ===", file=sys.stderr)
        results.append(_run_scenario(scenario, input_text, expected))

    all_ok = all(r["routed_correctly"] for r in results)
    report = {"model": MODEL, "scenarios": results, "result": {"both_routed_correctly": all_ok}}
    print(json.dumps(report, indent=2))
    for r in results:
        print(
            f"\n[branch:{r['scenario']}] fired={r['skills_fired']} "
            f"-> {'PASS' if r['routed_correctly'] else 'FAIL'} ({r['stop_reason']})",
            file=sys.stderr,
        )
    print(f"\n[skill-branch] both routed correctly -> {'PASS' if all_ok else 'FAIL'}", file=sys.stderr)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
