#!/usr/bin/env python
"""Lane-D benchmark runner: run ONE zak-code task headless against Groq, capture cost.

Usage (from the Zak-Code repo root, with the repo venv):
    ./.venv/Scripts/python.exe bench/run_task.py bench/tasks/01-wordfreq
    ./.venv/Scripts/python.exe bench/run_task.py --preflight bench/tasks/01-wordfreq

A task dir contains:
    task.json   {id, title, prompt, max_iterations?, max_cost_usd?, verify_timeout_s?}
    workspace/  (optional) seed files copied into a fresh temp workspace
    verify.py   exits 0 on success; run with cwd = the temp workspace (held-out oracle)

The runner sets up an isolated temp workspace, drives the agent to completion with
permission_mode=autonomous (no prompts; dangerous ops hard-DENY), runs verify.py, and
emits a single JSON result to stdout. --preflight constructs the agent and checks the
cost/permission API WITHOUT any LLM call (cheap API smoke).
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


def _ensure_interpreter_on_path() -> None:
    """Put THIS runner's interpreter dir first on PATH for the agent's SUBPROCESS tools.

    The agent verifies its work by shelling out (`python -m pytest ...`) via bash/powershell,
    which resolve `python` from PATH. The isolated temp workspace has no project venv, so a bare
    `python` hits the *system* interpreter — which typically lacks pytest. The model's correct
    test-suite verification then fails with "No module named pytest" (and `pip install` is
    hard-denied in autonomous mode), so the recipe gate stalls a turn whose code is actually
    fine — while the held-out oracle, run with THIS interpreter (the repo venv, which HAS
    pytest), passes. That mismatch is exactly the bench's `recipe_stalled`-but-oracle-passes
    artifact. A real user runs zakcode inside an activated project venv, so the representative
    fix is to expose the runner's own (pytest-capable) environment to the subprocess. Idempotent.
    """
    scripts_dir = str(Path(sys.executable).parent)
    path = os.environ.get("PATH", "")
    if scripts_dir not in path.split(os.pathsep):
        os.environ["PATH"] = scripts_dir + os.pathsep + path


def _build_agent(workspace: Path, spec: dict):
    """Mirror the CLI's canonical construction, adapted for headless reproducible runs."""
    from zakcode import Agent
    from zakcode.config import load_settings

    # Make `python`/`pytest` resolve to this runner's (pytest-capable) interpreter for the
    # agent's subprocess verification, mirroring an activated project venv (see the helper).
    _ensure_interpreter_on_path()
    # load_settings() (cwd = repo root) loads the repo .env -> GROQ_API_KEY into os.environ,
    # which litellm reads directly for groq/ models. api_base stays None (commented out in .env).
    base = load_settings()
    update = {
        "workspace_root": str(workspace),
        "default_model": "zakpick",        # Groq per-category routing
        "permission_mode": "autonomous",   # headless: never prompts, dangerous=hard-DENY
        "max_cost_usd": spec.get("max_cost_usd", 1.0),
        "max_iterations": spec.get("max_iterations", 50),
        "api_base": None,                  # belt-and-suspenders: never a local base
    }
    # Model-comparison override (Lane D): ZBENCH_DEEP_MODEL swaps the deep_code category's
    # model so the same tasks can be benchmarked across candidates (e.g. llama-3.3-70b-versatile
    # vs the default openai/gpt-oss-120b). tool_calling_mode flows from ZAKCODE_TOOL_CALLING_MODE
    # via base settings (text|native|auto) — no override needed here.
    deep_model = os.environ.get("ZBENCH_DEEP_MODEL")
    if deep_model:
        from zakcode.providers.routing import ZakpickModel

        # ZBENCH_DEEP_SOURCE picks the provider/runtime: "groq" (default), "openai",
        # "gemini", "deepseek", "fireworks_ai", "together_ai", "local"/"ollama", etc.
        # The litellm string is "<source>/<model>" so any litellm-supported supplier
        # can be benchmarked for the deep_code category by config alone.
        deep_source = os.environ.get("ZBENCH_DEEP_SOURCE", "groq")
        existing = dict(getattr(base, "zakpick_models", None) or {})
        existing["deep_code"] = ZakpickModel(model=deep_model, source=deep_source)
        existing["delegate"] = ZakpickModel(model=deep_model, source=deep_source)
        update["zakpick_models"] = existing
    settings = base.model_copy(update=update)
    agent = Agent(
        settings=settings,
        enable_compaction=True,   # needed so long tasks survive the context window
        enable_memory=False,      # OFF: cross-session memory would make cost non-reproducible
        enable_rules=False,       # OFF: no rule injection in the bare temp workspace
        enable_skills=False,      # OFF: no skill dirs
        enable_subagents=False,   # OFF: keep the baseline single-agent (enable later for a "full" run)
        enable_mcp=False,
        enable_plugins=False,
    )
    return agent


def _usage_snapshot(agent) -> dict:
    sess = agent.session.cumulative_usage()
    per_model = {
        m: {
            "cost_usd": round(u.cost_usd, 6),
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens,
        }
        for m, u in agent.session.usage_by_model().items()
    }
    return {
        "session_cost_usd": round(sess.cost_usd, 6),
        "session_prompt_tokens": sess.prompt_tokens,
        "session_completion_tokens": sess.completion_tokens,
        "session_total_tokens": sess.total_tokens,
        "cache_read_tokens": sess.cache_read_tokens,
        "per_model": per_model,
    }


def preflight(task_dir: Path) -> int:
    """Construct the agent and exercise the cost/permission API with NO LLM call."""
    spec = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    ws = Path(tempfile.mkdtemp(prefix=f"zbench-pf-{spec['id']}-"))
    try:
        agent = _build_agent(ws, spec)
        snap = _usage_snapshot(agent)  # must work on a fresh session -> all zero
        perm = agent.settings.permission_mode
        print(
            json.dumps(
                {
                    "preflight": "ok",
                    "permission_mode": str(perm),
                    "default_model": agent.settings.default_model,
                    "max_cost_usd": agent.settings.max_cost_usd,
                    "max_iterations": agent.settings.max_iterations,
                    "fresh_session_cost_usd": snap["session_cost_usd"],
                    "has_run_turn": hasattr(agent, "run_turn"),
                },
                indent=2,
            )
        )
        return 0
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def run(task_dir: Path) -> int:
    spec = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    ws = Path(tempfile.mkdtemp(prefix=f"zbench-{spec['id']}-"))
    seed = task_dir / "workspace"
    if seed.is_dir():
        shutil.copytree(seed, ws, dirs_exist_ok=True)

    agent = _build_agent(ws, spec)

    err = None
    stop_reason = iterations = routed_category = routed_escalated = degraded = None
    turn_error = None
    turn_cost = turn_tokens = 0
    t0 = time.perf_counter()
    try:
        result = agent.run_turn(spec["prompt"])
        stop_reason = result.stop_reason
        iterations = result.iterations
        routed_category = result.routed_category
        routed_escalated = result.routed_escalated
        degraded = result.degraded
        turn_error = result.error  # TurnResult.error: the provider/loop error detail behind stop_reason
        turn_cost = result.usage.cost_usd
        turn_tokens = result.usage.total_tokens
    except Exception as e:  # noqa: BLE001 - a crash is a result (a bug to file), not a runner failure
        err = f"{type(e).__name__}: {e}"
    elapsed = time.perf_counter() - t0

    snap = _usage_snapshot(agent)

    verify_rc = None
    verify_out = ""
    vf = task_dir / "verify.py"
    if vf.is_file() and err is None:
        try:
            proc = subprocess.run(
                [sys.executable, str(vf)],
                cwd=ws,
                capture_output=True,
                text=True,
                timeout=spec.get("verify_timeout_s", 120),
            )
            verify_rc = proc.returncode
            verify_out = (proc.stdout + proc.stderr).strip()[-800:]
        except subprocess.TimeoutExpired:
            verify_rc = 124
            verify_out = "verify timed out"

    success = verify_rc == 0
    report = {
        "id": spec["id"],
        "title": spec.get("title", ""),
        "success": success,
        "stop_reason": stop_reason,
        "iterations": iterations,
        "routed_category": routed_category,
        "routed_escalated": routed_escalated,
        "degraded": degraded,
        "turn_error": turn_error,
        "elapsed_s": round(elapsed, 1),
        "tok_per_s": round((snap["session_completion_tokens"] / elapsed), 1) if elapsed > 0 else None,
        "turn_cost_usd": round(turn_cost, 6),
        "turn_tokens": turn_tokens,
        **snap,
        "verify_rc": verify_rc,
        "verify_out": verify_out,
        "error": err,
        "workspace": str(ws) if not success else "(cleaned)",
    }
    print(json.dumps(report, indent=2))

    # Keep the workspace for inspection on failure/crash; clean it on success.
    if success:
        shutil.rmtree(ws, ignore_errors=True)
    return 0


def main(argv: list[str]) -> int:
    args = [a for a in argv if a != "--preflight"]
    if not args:
        print("usage: run_task.py [--preflight] <task_dir>", file=sys.stderr)
        return 2
    task_dir = Path(args[0]).resolve()
    if "--preflight" in argv:
        return preflight(task_dir)
    return run(task_dir)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
