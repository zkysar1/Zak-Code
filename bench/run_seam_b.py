#!/usr/bin/env python
"""Live seam B demo: watch best-of-N rescue a stalled turn (the last unconfirmed piece).

Runs ONE task on a small model with ``verify_command`` set, ``best_of_attempts=1`` (baseline) vs N
(seam B on). With a verifier set, a turn that can't pass it STALLS (``verification_failed``); seam B
then fans out N isolated attempts and adopts the first that verifies — applying only its diff. The
``[seam B]`` INFO lines surface the retry firing live.

The task's held-out ``verify.py`` is used as the verifier (it runs in the workspace; the agent never
reads it). Usage:
    ./.venv/Scripts/python.exe bench/run_seam_b.py bench/tasks/04-todo-cli
Config: ZBENCH_SMALL_MODEL (default groq/qwen/qwen3-32b), ZBENCH_ATTEMPTS (default 3).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_task  # noqa: E402 — sibling bench script

SMALL_MODEL = os.environ.get("ZBENCH_SMALL_MODEL", "groq/qwen/qwen3-32b")
ATTEMPTS = int(os.environ.get("ZBENCH_ATTEMPTS", "3"))


def _build_agent(workspace: Path, spec: dict, *, best_of_attempts: int, verify_command: str):
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
            "best_of_attempts": best_of_attempts,  # 1 = baseline (off); >1 = seam B on
            "verify_command": verify_command,  # the loop verify gate + seam B both use this
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


def _run(task_dir: Path, spec: dict, *, best_of_attempts: int) -> dict:
    ws = Path(tempfile.mkdtemp(prefix=f"zsb-{spec['id']}-n{best_of_attempts}-"))
    seed = task_dir / "workspace"
    if seed.is_dir():
        shutil.copytree(seed, ws, dirs_exist_ok=True)
    verify_command = f'"{sys.executable}" "{(task_dir / "verify.py").resolve()}"'
    agent = _build_agent(ws, spec, best_of_attempts=best_of_attempts, verify_command=verify_command)
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
        "best_of_attempts": best_of_attempts,
        "passed": passed,
        "stop_reason": stop_reason,
        "crashed": crashed,
        "cost_usd": round(snap["session_cost_usd"], 6),
        "elapsed_s": round(elapsed, 1),
    }


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: run_seam_b.py <task_dir>", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.WARNING, format="  %(message)s", stream=sys.stderr)
    logging.getLogger("zakcode.agent.best_of_attempts").setLevel(logging.INFO)
    task_dir = Path(argv[0]).resolve()
    spec = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))

    print("=== baseline (best_of_attempts=1, seam B off) ===", file=sys.stderr)
    off = _run(task_dir, spec, best_of_attempts=1)
    print(f"=== seam B on (best_of_attempts={ATTEMPTS}) ===", file=sys.stderr)
    on = _run(task_dir, spec, best_of_attempts=ATTEMPTS)

    report = {
        "task": spec["id"],
        "model": SMALL_MODEL,
        "attempts": ATTEMPTS,
        "baseline": off,
        "seam_b": on,
        "result": {
            "baseline_passed": off["passed"],
            "seam_b_passed": on["passed"],
            "seam_b_rescued": on["passed"] and not off["passed"],  # the win we're after
        },
    }
    print(json.dumps(report, indent=2))
    r = report["result"]
    print(
        f"\n[{spec['id']}] baseline passed={r['baseline_passed']} ({off['stop_reason']}) -> "
        f"seam B passed={r['seam_b_passed']} ({on['stop_reason']}); rescued={r['seam_b_rescued']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
