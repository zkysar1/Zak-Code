#!/usr/bin/env python
"""Bench experiment: does the quality gate earn its tokens? — the activation evidence (increment 6e).

The answer to *"when does flipping the engine on win?"*. For ONE task it runs the agent twice on a
SMALL model — gate **OFF** (today's baseline) vs gate **ON** (seam A, the quality engine) — grades
both with the held-out ``verify.py``, and reports the delta: pass, $/task, wall-clock. Run it across
the suite and the pattern shows where quality-on raises pass-rate per dollar — i.e. *when an operator
should flip it on*. That's the data the small-model preset (the activation story) gets baked from.

Config (env, optional):
    ZBENCH_SMALL_MODEL        default groq/qwen/qwen3-32b   the model under test (the bet: small)
    ZBENCH_QUALITY_THRESHOLD  default 0.8                   the gate's ship threshold

Usage (from the repo root, with the repo venv):
    ./.venv/Scripts/python.exe bench/run_quality.py bench/tasks/04-todo-cli
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_task  # noqa: E402 — sibling bench script; reuse its interpreter-PATH + usage helpers

SMALL_MODEL = os.environ.get("ZBENCH_SMALL_MODEL", "groq/qwen/qwen3-32b")
THRESHOLD = float(os.environ.get("ZBENCH_QUALITY_THRESHOLD", "0.8"))


def _build_agent(workspace: Path, spec: dict, *, quality_gate: bool):
    """An autonomous headless agent on the small model, with the quality gate ON or OFF."""
    from zakcode import Agent
    from zakcode.config import load_settings

    run_task._ensure_interpreter_on_path()
    base = load_settings()
    settings = base.model_copy(
        update={
            "workspace_root": str(workspace),
            "default_model": SMALL_MODEL,
            "permission_mode": "autonomous",
            "max_cost_usd": spec.get("max_cost_usd", 1.0),
            "max_iterations": spec.get("max_iterations", 50),
            "api_base": None,
            "quality_gate": quality_gate,  # the only difference between the two runs
            "quality_gate_threshold": THRESHOLD,
        }
    )
    return Agent(
        settings=settings,
        enable_compaction=True,
        enable_memory=False,
        enable_rules=False,
        enable_skills=False,
        enable_subagents=False,
        enable_mcp=False,
        enable_plugins=False,
    )


def _grade(task_dir: Path, workspace: Path, spec: dict) -> bool:
    vf = task_dir / "verify.py"
    if not vf.is_file():
        return False
    try:
        proc = subprocess.run(
            [sys.executable, str(vf)],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=spec.get("verify_timeout_s", 120),
        )
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def _run(task_dir: Path, spec: dict, *, quality_gate: bool) -> dict:
    """One agent attempt in a fresh seeded workspace with the gate on/off; run + grade. Never raises."""
    tag = "on" if quality_gate else "off"
    ws = Path(tempfile.mkdtemp(prefix=f"zq-{spec['id']}-{tag}-"))
    seed = task_dir / "workspace"
    if seed.is_dir():
        shutil.copytree(seed, ws, dirs_exist_ok=True)
    agent = _build_agent(ws, spec, quality_gate=quality_gate)
    stop_reason = None
    crashed = None
    t0 = time.perf_counter()
    try:
        result = agent.run_turn(spec["prompt"])
        stop_reason = result.stop_reason
    except Exception as e:  # noqa: BLE001 — a crash is a (failed) result, not a runner error
        crashed = f"{type(e).__name__}: {e}"
    elapsed = time.perf_counter() - t0
    snap = run_task._usage_snapshot(agent)
    passed = _grade(task_dir, ws, spec) if crashed is None else False
    shutil.rmtree(ws, ignore_errors=True)
    return {
        "quality_gate": quality_gate,
        "passed": passed,
        "stop_reason": stop_reason,
        "crashed": crashed,
        "cost_usd": round(snap["session_cost_usd"], 6),
        "elapsed_s": round(elapsed, 1),
    }


def run_experiment(task_dir: Path) -> dict:
    spec = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    off = _run(task_dir, spec, quality_gate=False)
    on = _run(task_dir, spec, quality_gate=True)
    return {
        "task": spec["id"],
        "model": SMALL_MODEL,
        "threshold": THRESHOLD,
        "gate_off": off,
        "gate_on": on,
        "result": {
            "off_passed": off["passed"],
            "on_passed": on["passed"],
            "gate_helped": on["passed"] and not off["passed"],  # flipping it on rescued the task
            "gate_hurt": off["passed"] and not on["passed"],  # a regression (should be rare)
            "extra_cost_usd": round(on["cost_usd"] - off["cost_usd"], 6),  # the tokens it cost
            "extra_time_s": round(on["elapsed_s"] - off["elapsed_s"], 1),
        },
    }


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: run_quality.py <task_dir>", file=sys.stderr)
        return 2
    report = run_experiment(Path(argv[0]).resolve())
    print(json.dumps(report, indent=2))
    r = report["result"]
    print(
        f"\n[{report['task']}] gate OFF passed={r['off_passed']} -> ON passed={r['on_passed']} "
        f"(helped={r['gate_helped']}, hurt={r['gate_hurt']}, +${r['extra_cost_usd']:.4f}, "
        f"+{r['extra_time_s']}s) on {SMALL_MODEL}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
