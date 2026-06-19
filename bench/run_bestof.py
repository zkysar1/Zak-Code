#!/usr/bin/env python
"""Bench experiment: does N-small fan-out beat one big call? (the quality engine's central bet)

For ONE task this runs **N attempts with a SMALL model** (diverse via temperature) and **one attempt
with a BIG model**, then uses the quality engine's pairwise tournament
(:func:`zakcode.quality.judge.best_of`) to **judge-select the best small attempt** — reading the
candidates' source, NOT running them (judges, not oracles). The held-out oracle (`verify.py`) then
grades every attempt, so we can separate three things:

* **generation ceiling** — did *any* of the N small attempts pass? (best the fan-out could do)
* **best-of-N result** — did the JUDGE-selected small attempt pass? (the real product outcome)
* **judge quality** — when a passing attempt existed, did the judge pick one? (selection skill)

…and compare all of that, on pass-rate / $ / wall-clock, against the single big-model run. That is
the falsifiable test: if the judge-selected small solution passes where (or as often as) the big one
does, at less cost, the bet holds; if not, we just saved ourselves from gearing the loop toward a
loss.

Config (env, all optional):
    ZBENCH_SMALL_MODEL   default groq/qwen/qwen3-32b      the cheap fan-out model
    ZBENCH_BIG_MODEL     default openai/gpt-4o-mini       the single-shot capable model
    ZBENCH_JUDGE_MODEL   default = ZBENCH_SMALL_MODEL      the selector (default: all-small pipeline)
    ZBENCH_N             default 3                         attempts in the fan-out
    ZBENCH_SMALL_TEMP    default 0.7                       sampling temperature for diversity

Usage (from the repo root, with the repo venv):
    ./.venv/Scripts/python.exe bench/run_bestof.py bench/tasks/04-todo-cli
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_task  # noqa: E402 — sibling bench script; reuse its interpreter-PATH + usage helpers

SMALL_MODEL = os.environ.get("ZBENCH_SMALL_MODEL", "groq/qwen/qwen3-32b")
BIG_MODEL = os.environ.get("ZBENCH_BIG_MODEL", "openai/gpt-4o-mini")
JUDGE_MODEL = os.environ.get("ZBENCH_JUDGE_MODEL", SMALL_MODEL)
N = int(os.environ.get("ZBENCH_N", "3"))
SMALL_TEMP = float(os.environ.get("ZBENCH_SMALL_TEMP", "0.7"))

#: Source extensions serialized as a "solution" for the judge to read (it can't run the code).
_SRC_EXTS = {".py", ".js", ".ts", ".sh", ".rb", ".go", ".rs", ".java", ".txt", ".md", ".json", ".toml"}


def _build_agent_for(workspace: Path, spec: dict, model: str, temperature: float):
    """An autonomous headless agent pinned to ONE model (not zakpick) at ``temperature``.

    Pinning ``default_model`` to a concrete litellm string makes the whole attempt run on that one
    model — a clean single-model candidate, no per-category routing or escalation.
    """
    from zakcode import Agent
    from zakcode.config import load_settings

    run_task._ensure_interpreter_on_path()
    base = load_settings()  # loads the repo .env -> provider keys into os.environ for litellm
    settings = base.model_copy(
        update={
            "workspace_root": str(workspace),
            "default_model": model,
            "temperature": temperature,
            "permission_mode": "autonomous",
            "max_cost_usd": spec.get("max_cost_usd", 1.0),
            "max_iterations": spec.get("max_iterations", 50),
            "api_base": None,
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


def _serialize_solution(workspace: Path, *, max_per_file: int = 3000, max_total: int = 12000) -> str:
    """Concatenate the workspace's source files (capped) — the artifact the judge reads."""
    parts: list[str] = []
    total = 0
    for f in sorted(workspace.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in _SRC_EXTS or ".git" in f.parts:
            continue
        try:
            content = f.read_text(encoding="utf-8", errors="replace")[:max_per_file]
        except OSError:
            continue
        chunk = f"--- {f.relative_to(workspace)} ---\n{content}\n\n"
        if total + len(chunk) > max_total:
            break
        parts.append(chunk)
        total += len(chunk)
    return "".join(parts) or "(no source files produced)"


def _grade(task_dir: Path, workspace: Path, spec: dict) -> bool:
    """Run the held-out oracle (verify.py) against ``workspace``; True iff it exits 0."""
    import subprocess

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


def _run_attempt(task_dir: Path, spec: dict, model: str, temperature: float, tag: str) -> dict:
    """One agent attempt in a fresh seeded workspace: run, grade, serialize. Never raises."""
    ws = Path(tempfile.mkdtemp(prefix=f"zbof-{spec['id']}-{tag}-"))
    seed = task_dir / "workspace"
    if seed.is_dir():
        shutil.copytree(seed, ws, dirs_exist_ok=True)
    agent = _build_agent_for(ws, spec, model, temperature)
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
    return {
        "tag": tag,
        "model": model,
        "passed": passed,
        "stop_reason": stop_reason,
        "crashed": crashed,
        "cost_usd": round(snap["session_cost_usd"], 6),
        "elapsed_s": round(elapsed, 1),
        "solution": _serialize_solution(ws),
        "_ws": ws,
    }


def run_experiment(task_dir: Path) -> dict:
    spec = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))

    # N small attempts (diverse), one big attempt. Run sequentially for rate-limit safety; record
    # per-attempt time so both the sequential (sum) and the parallelizable (max) wall-clock are known.
    small = [_run_attempt(task_dir, spec, SMALL_MODEL, SMALL_TEMP, f"small{i}") for i in range(N)]
    big = _run_attempt(task_dir, spec, BIG_MODEL, 0.0, "big")

    # Judge-SELECT the best small attempt by reading its source (judges, not oracles).
    judge_provider = _build_agent_for(
        Path(tempfile.mkdtemp(prefix="zbof-judge-")), spec, JUDGE_MODEL, 0.0
    ).provider
    from zakcode.quality import best_of

    selected_index, judge_usage = asyncio.run(
        best_of(judge_provider, criteria=spec["prompt"], candidates=[a["solution"] for a in small])
    )

    any_small_passed = any(a["passed"] for a in small)
    selected_passed = small[selected_index]["passed"]
    small_cost = sum(a["cost_usd"] for a in small)
    judge_cost = round(judge_usage.cost_usd, 6)
    bestof_cost = round(small_cost + judge_cost, 6)

    for a in (*small, big):  # tidy temp workspaces; the report has the data
        shutil.rmtree(a.pop("_ws"), ignore_errors=True)
        a.pop("solution")  # drop the (large) serialized source from the report

    return {
        "task": spec["id"],
        "config": {
            "small_model": SMALL_MODEL,
            "big_model": BIG_MODEL,
            "judge_model": JUDGE_MODEL,
            "n": N,
            "small_temp": SMALL_TEMP,
        },
        "small_attempts": small,
        "big": big,
        "result": {
            "any_small_passed": any_small_passed,  # the fan-out's generation ceiling
            "selected_index": selected_index,
            "selected_passed": selected_passed,  # the real best-of-N product outcome
            "judge_optimal": (selected_passed if any_small_passed else None),  # picked a winner?
            "big_passed": big["passed"],
            "bestof_cost_usd": bestof_cost,
            "bestof_judge_cost_usd": judge_cost,
            "bestof_time_sequential_s": round(sum(a["elapsed_s"] for a in small), 1),
            "bestof_time_parallel_est_s": round(max(a["elapsed_s"] for a in small), 1),
            "big_cost_usd": big["cost_usd"],
            "big_time_s": big["elapsed_s"],
            # The headline: did best-of-N match/beat big, and was it cheaper?
            "bestof_won_where_big_lost": selected_passed and not big["passed"],
            "bestof_lost_where_big_won": big["passed"] and not selected_passed,
            "bestof_cheaper_than_big": bestof_cost < big["cost_usd"],
        },
    }


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: run_bestof.py <task_dir>", file=sys.stderr)
        return 2
    report = run_experiment(Path(argv[0]).resolve())
    print(json.dumps(report, indent=2))
    r = report["result"]
    print(
        f"\n[{report['task']}] best-of-{N}({SMALL_MODEL}) selected_passed={r['selected_passed']} "
        f"(ceiling any_passed={r['any_small_passed']}) ${r['bestof_cost_usd']:.4f}  vs  "
        f"big({BIG_MODEL}) passed={r['big_passed']} ${r['big_cost_usd']:.4f}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
