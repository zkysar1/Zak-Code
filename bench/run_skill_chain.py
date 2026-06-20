#!/usr/bin/env python
"""Live skills-chaining demo: watch a model invoke three skills that daisy-chain into each other.

The skills system surfaces an L0 catalog in the prompt and lets the MODEL invoke a skill by name
via the ``use_skill`` tool (M7). This harness proves a *chain*: a kickoff prompt makes the model
call ``use_skill('relay-start')``; that skill's body tells it to write the relay log and then call
``use_skill('relay-middle')``; which hands off to ``use_skill('relay-finish')``. Three skills, one
turn, each handing off to the next — driven entirely by the skill bodies, not the harness.

We watch it two ways:
  * the ``ON_SKILL_SELECTED`` signal fires once per ``use_skill`` call — printed live, in order;
  * each skill appends a marked line to ``RELAY.md``, so the workspace proves every skill's
    instructions actually RAN (not just loaded).

Usage:
    ./.venv/Scripts/python.exe bench/run_skill_chain.py
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
SKILLS_DIR = Path(__file__).resolve().parent / "skill_chain" / "skills"
EXPECTED_CHAIN = ["relay-start", "relay-middle", "relay-finish"]
MARKERS = ["step-1 relay-start ran", "step-2 relay-middle ran", "step-3 relay-finish ran"]
PROMPT = (
    "Run the relay demo. Begin by using the relay-start skill, then follow wherever it leads. "
    "Do not stop until the relay reports it is complete."
)


def _build_agent(workspace: Path):
    from zakcode import Agent
    from zakcode.config import load_settings

    run_task._ensure_interpreter_on_path()
    settings = load_settings().model_copy(
        update={
            "workspace_root": str(workspace),
            "default_model": MODEL,
            "permission_mode": "autonomous",  # headless: file writes never prompt
            "max_cost_usd": 0.50,
            "max_iterations": 30,  # chain is ~3 skills x (load + write) + kickoff
            "api_base": None,
        }
    )
    return Agent(
        settings=settings,
        enable_skills=True,  # the feature under test
        extra_skill_dirs=[str(SKILLS_DIR)],  # the three relay skills live here
        enable_compaction=True,
        enable_rules=False,
        enable_subagents=False,
        enable_mcp=False,
        enable_plugins=False,
    )


def _run() -> dict:
    from zakcode.hooks import HookEvent, LifecyclePayload

    ws = Path(tempfile.mkdtemp(prefix="zskill-chain-"))
    agent = _build_agent(ws)

    fired: list[dict[str, str]] = []

    def _on_skill(payload: LifecyclePayload) -> None:
        name = str(payload.data.get("skill", ""))
        source = str(payload.data.get("source", ""))
        fired.append({"skill": name, "source": source})
        print(f"  ▸ skill fired #{len(fired)}: {name}  (source={source})", file=sys.stderr)

    agent.hook_manager.register_lifecycle(HookEvent.ON_SKILL_SELECTED, _on_skill)

    print(f"=== skills in catalog: {agent.skill_registry.names()} ===", file=sys.stderr)
    print(f"=== kickoff: {PROMPT!r} ===", file=sys.stderr)

    stop_reason = None
    crashed = None
    t0 = time.perf_counter()
    try:
        result = agent.run_turn(PROMPT)
        stop_reason = result.stop_reason
    except Exception as e:  # noqa: BLE001 — a crash is a (failed) result, not a runner error
        crashed = f"{type(e).__name__}: {e}"
    elapsed = time.perf_counter() - t0

    relay = ws / "RELAY.md"
    relay_text = relay.read_text(encoding="utf-8") if relay.is_file() else ""
    try:
        snap = run_task._usage_snapshot(agent)
        cost = round(snap["session_cost_usd"], 6)
    except Exception:  # noqa: BLE001 — cost is nice-to-have, never fail the demo on it
        cost = None
    shutil.rmtree(ws, ignore_errors=True)

    chain_order = list(dict.fromkeys(f["skill"] for f in fired))  # unique, first-seen order
    markers_present = [m for m in MARKERS if m in relay_text]
    return {
        "model": MODEL,
        "stop_reason": stop_reason,
        "crashed": crashed,
        "elapsed_s": round(elapsed, 1),
        "cost_usd": cost,
        "skills_fired": fired,
        "chain_order": chain_order,
        "relay_markers_present": markers_present,
        "relay_md": relay_text,
        "result": {
            "chain_complete": chain_order == EXPECTED_CHAIN,  # all three, invoked in order
            "all_skills_executed": len(markers_present) == len(MARKERS),  # bodies actually ran
            "sources_all_tool": bool(fired) and all(f["source"] == "tool" for f in fired),
        },
    }


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="  %(message)s", stream=sys.stderr)
    report = _run()
    print(json.dumps(report, indent=2))
    r = report["result"]
    passed = r["chain_complete"] and r["all_skills_executed"]
    print(
        f"\n[skill-chain] chain={report['chain_order']} complete={r['chain_complete']} "
        f"executed={r['all_skills_executed']} ({len(report['relay_markers_present'])}/3 markers) "
        f"-> {'PASS' if passed else 'FAIL'} ({report['stop_reason']})",
        file=sys.stderr,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
