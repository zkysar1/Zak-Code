#!/usr/bin/env python
"""Lane-D benchmark aggregator: run every task in bench/tasks/ in parallel, summarize cost.

Usage (from the Zak-Code repo root, with the repo venv):
    ./.venv/Scripts/python.exe bench/run_suite.py            # all tasks
    ./.venv/Scripts/python.exe bench/run_suite.py 01 03 05   # only matching task-id prefixes
    ./.venv/Scripts/python.exe bench/run_suite.py --jobs 4   # cap concurrency (default 3)

Each task runs in its OWN process (so each gets an isolated Agent/Session/workspace, per
the scout's parallelism finding). Results are aggregated into a table + a results JSON at
bench/results/suite-<n>tasks.json. Groq rate limits are why concurrency defaults low.
"""
from __future__ import annotations

import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
TASKS_DIR = BENCH / "tasks"
RESULTS_DIR = BENCH / "results"


def discover(filters: list[str]) -> list[Path]:
    dirs = sorted(p for p in TASKS_DIR.iterdir() if p.is_dir() and (p / "task.json").is_file())
    if filters:
        dirs = [d for d in dirs if any(d.name.startswith(f) or f in d.name for f in filters)]
    return dirs


def run_one(task_dir: Path) -> dict:
    """Run one task in a child process; parse its JSON report from stdout."""
    proc = subprocess.run(
        [sys.executable, str(BENCH / "run_task.py"), str(task_dir)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    out = proc.stdout.strip()
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        # Recover the last balanced {...} block if stdout carried extra noise.
        start = out.rfind("\n{")
        if start != -1:
            try:
                return json.loads(out[start:].strip())
            except json.JSONDecodeError:
                pass
        return {
            "id": task_dir.name,
            "success": False,
            "stop_reason": "runner_error",
            "error": f"could not parse report; rc={proc.returncode}; stderr={proc.stderr.strip()[-300:]}",
        }


def fmt_row(r: dict) -> str:
    ok = "PASS" if r.get("success") else "FAIL"
    cat = (r.get("routed_category") or "-")[:10]
    return (
        f"{r.get('id','?'):16} {ok:5} {str(r.get('stop_reason'))[:15]:15} "
        f"it={str(r.get('iterations','-')):>3} {cat:10} "
        f"${r.get('session_cost_usd',0):.5f}  "
        f"tok={r.get('session_total_tokens',0):>7} "
        f"(in {r.get('session_prompt_tokens',0)}/out {r.get('session_completion_tokens',0)}) "
        f"{r.get('elapsed_s','-')}s"
    )


def main(argv: list[str]) -> int:
    jobs = 3
    filters: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--jobs":
            jobs = int(argv[i + 1])
            i += 2
        else:
            filters.append(argv[i])
            i += 1

    task_dirs = discover(filters)
    if not task_dirs:
        print("no tasks found", file=sys.stderr)
        return 2
    print(f"running {len(task_dirs)} task(s), {jobs} at a time: {[d.name for d in task_dirs]}\n")

    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(run_one, d): d for d in task_dirs}
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result()
            results.append(r)
            print("  done:", fmt_row(r))

    results.sort(key=lambda r: r.get("id", ""))

    # Aggregate.
    n = len(results)
    passed = sum(1 for r in results if r.get("success"))
    total_cost = sum(r.get("session_cost_usd", 0) or 0 for r in results)
    total_tok = sum(r.get("session_total_tokens", 0) or 0 for r in results)
    total_in = sum(r.get("session_prompt_tokens", 0) or 0 for r in results)
    total_out = sum(r.get("session_completion_tokens", 0) or 0 for r in results)
    total_cache = sum(r.get("cache_read_tokens", 0) or 0 for r in results)
    total_iters = sum(r.get("iterations", 0) or 0 for r in results)

    # Per-model rollup across all tasks.
    per_model: dict[str, dict] = {}
    for r in results:
        for m, u in (r.get("per_model") or {}).items():
            agg = per_model.setdefault(m, {"cost_usd": 0.0, "total_tokens": 0})
            agg["cost_usd"] += u.get("cost_usd", 0)
            agg["total_tokens"] += u.get("total_tokens", 0)

    print("\n" + "=" * 100)
    print(f"SUITE: {passed}/{n} passed")
    for r in results:
        print("  " + fmt_row(r))
    print("-" * 100)
    print(f"  total cost      ${total_cost:.5f}   (avg ${total_cost/n:.5f}/task)")
    print(f"  total tokens    {total_tok:,}   (in {total_in:,} / out {total_out:,} / cached {total_cache:,})")
    print(f"  input fraction  {(total_in/total_tok*100) if total_tok else 0:.1f}%   cache-hit fraction {(total_cache/total_in*100) if total_in else 0:.1f}% of input")
    print(f"  total iters     {total_iters}   (avg {total_iters/n:.1f}/task)   cost/iter ${(total_cost/total_iters) if total_iters else 0:.5f}")
    print("  per-model:")
    for m, u in sorted(per_model.items()):
        print(f"    {m:30} ${u['cost_usd']:.5f}   {u['total_tokens']:,} tok")
    print("=" * 100)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"suite-{n}tasks.json"
    out_path.write_text(
        json.dumps(
            {
                "summary": {
                    "n": n, "passed": passed, "total_cost_usd": total_cost,
                    "total_tokens": total_tok, "input_tokens": total_in, "output_tokens": total_out,
                    "cache_read_tokens": total_cache, "total_iterations": total_iters,
                    "per_model": per_model,
                },
                "tasks": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
