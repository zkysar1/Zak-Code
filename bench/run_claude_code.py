#!/usr/bin/env python
"""Reference-agent arm: run one bench task through the Claude Code CLI, verify it identically.

WHY THIS EXISTS. The campaign's directive is "as good as claude code at being an agent", and
until now nothing here measured Claude Code at all -- zakcode's scores were reported against no
reference. A score of 20/20 means something very different if the reference agent also scores
20/20 (the suite cannot discriminate) than if it scores 14/20 (zakcode is ahead).

WHAT THIS CANNOT DO, stated up front because the result is worthless if this is forgotten. It is
NOT a loop-vs-loop comparison. Claude Code runs Claude models; zakcode's passes ran
``zds-qwen3.5/3.6-35b`` on a local pod. Holding the model fixed is impossible on this box: there
is no ANTHROPIC_API_KEY for zakcode to call a Claude model, and Claude Code cannot be pointed at
the pod. ``providers/claude_code.py`` does not close the gap -- it is a text-in/text-out bridge
needing a callable supplied by a live Claude Code session, not a headless provider. So every
number this produces is CONFOUNDED BY MODEL, and the only claims it supports are about the
SUITE's discriminating power, never about which agent loop is better.

Usage:  ./.venv/bin/python bench/run_claude_code.py bench/tasks/02-median-bug
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parent


def run(task_dir: Path, timeout_s: int = 900) -> int:
    spec = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    ws = Path(tempfile.mkdtemp(prefix=f"zbench-cc-{spec['id']}-"))
    seed = task_dir / "workspace"
    if seed.is_dir():
        shutil.copytree(seed, ws, dirs_exist_ok=True)

    # --allowedTools rather than --dangerously-skip-permissions: the task only needs to read,
    # write and run code inside a throwaway workspace, and a scoped grant is the smaller blast
    # radius. In -p mode a tool outside the list is denied, not prompted, so the run stays headless.
    cmd = [
        "claude", "-p", spec["prompt"],
        "--allowedTools", "Read,Write,Edit,Bash,Glob,Grep",
        "--output-format", "json",
    ]
    t0 = time.perf_counter()
    err = None
    stdout = ""
    try:
        proc = subprocess.run(cmd, cwd=ws, capture_output=True, text=True, timeout=timeout_s)
        stdout = proc.stdout
        rc = proc.returncode
        stderr_tail = proc.stderr.strip()[-400:]
    except subprocess.TimeoutExpired:
        rc, stderr_tail = 124, f"exceeded {timeout_s}s"
    except Exception as e:  # noqa: BLE001 - a crash is a result, not a runner failure
        rc, stderr_tail = -1, f"{type(e).__name__}: {e}"
        err = stderr_tail
    elapsed = time.perf_counter() - t0

    # The CLI's own JSON report carries cost/turns when --output-format json succeeds.
    meta: dict = {}
    try:
        meta = json.loads(stdout) if stdout.strip().startswith("{") else {}
    except json.JSONDecodeError:
        meta = {}

    verify_rc, verify_out = None, ""
    vf = task_dir / "verify.py"
    if vf.is_file():
        try:
            vp = subprocess.run(
                [sys.executable, str(vf)], cwd=ws, capture_output=True, text=True,
                timeout=spec.get("verify_timeout_s", 120),
            )
            verify_rc = vp.returncode
            verify_out = (vp.stdout + vp.stderr).strip()[-800:]
        except subprocess.TimeoutExpired:
            verify_rc, verify_out = 124, "verify timed out"

    report = {
        "id": spec["id"],
        "agent": "claude-code",
        "success": verify_rc == 0,
        "cli_rc": rc,
        "elapsed_s": round(elapsed, 1),
        "num_turns": meta.get("num_turns"),
        "total_cost_usd": meta.get("total_cost_usd"),
        "model": (meta.get("modelUsage") and list(meta["modelUsage"].keys())) or meta.get("model"),
        "verify_rc": verify_rc,
        "verify_out": verify_out,
        "stderr_tail": stderr_tail,
        "error": err,
        "workspace": str(ws) if verify_rc != 0 else "(cleaned)",
    }
    print(json.dumps(report, indent=2))
    if verify_rc == 0:
        shutil.rmtree(ws, ignore_errors=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: run_claude_code.py <task_dir>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(run(Path(sys.argv[1]).resolve()))
