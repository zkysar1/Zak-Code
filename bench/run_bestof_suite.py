#!/usr/bin/env python
"""Run the best-of-N experiment across the whole suite + aggregate — the generalization test.

The activation measurement showed the quality GATE is a targeted tool; the open question is whether
best-of-N (generation fan-out) is the broad small-model win it looked like on `04`. This calls
``run_bestof.run_experiment`` on each task and aggregates: how often does the fan-out CONTAIN a passer
(the generation ceiling), does the oracle-grounded best-of-N MATCH/BEAT one big call, and at what
cost — across tasks, not just one.

Usage (from the repo root, with the repo venv):
    ./.venv/Scripts/python.exe bench/run_bestof_suite.py            # all tasks
    ./.venv/Scripts/python.exe bench/run_bestof_suite.py 01 03 04   # by id-prefix
Tunable via the same ZBENCH_* env as run_bestof (SMALL/BIG/JUDGE model, N, temperature, votes).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_bestof  # noqa: E402 — sibling bench script

_TASKS_DIR = Path(__file__).resolve().parent / "tasks"


def _select(argv: list[str]) -> list[Path]:
    tasks = sorted(
        p for p in _TASKS_DIR.iterdir() if p.is_dir() and (p / "task.json").is_file()
    )
    if argv:
        tasks = [t for t in tasks if any(t.name.startswith(a) for a in argv)]
    return tasks


def main(argv: list[str]) -> int:
    tasks = _select(argv)
    if not tasks:
        print("no matching tasks", file=sys.stderr)
        return 2

    results: list[dict] = []
    for task in tasks:
        print(f"=== {task.name} ===", file=sys.stderr)
        try:
            results.append(run_bestof.run_experiment(task))
        except Exception as e:  # noqa: BLE001 — one task crashing must not sink the sweep
            print(f"  {task.name} crashed: {type(e).__name__}: {e}", file=sys.stderr)

    n = len(results)
    rs = [r["result"] for r in results]
    summary = {
        "tasks": n,
        # The fan-out's GENERATION ceiling: did any of the N small attempts pass?
        "generation_ceiling_passed": sum(1 for r in rs if r["any_small_passed"]),
        # The product outcome: oracle-grounded best-of-N (the hybrid selector).
        "bestof_hybrid_passed": sum(1 for r in rs if r["hybrid_passed"]),
        # The judge-alone selector (the selection gap the oracle fixes).
        "bestof_judge_only_passed": sum(1 for r in rs if r["judge_only_passed"]),
        # Where oracle-grounding rescued a generated win the judge alone discarded.
        "oracle_rescued": sum(1 for r in rs if r["oracle_rescued_a_win"]),
        # The single-big-call baseline.
        "big_passed": sum(1 for r in rs if r["big_passed"]),
        "bestof_total_cost_usd": round(sum(r["bestof_cost_usd"] for r in rs), 4),
        "big_total_cost_usd": round(sum(r["big_cost_usd"] for r in rs), 4),
    }
    print(json.dumps({"per_task": results, "summary": summary}, indent=2))
    s = summary
    print(
        f"\nSUITE ({n} tasks): generation-ceiling {s['generation_ceiling_passed']}/{n} · "
        f"best-of-N(hybrid) {s['bestof_hybrid_passed']}/{n} (judge-only "
        f"{s['bestof_judge_only_passed']}/{n}, oracle rescued {s['oracle_rescued']}) · "
        f"1-big {s['big_passed']}/{n} · ${s['bestof_total_cost_usd']} vs ${s['big_total_cost_usd']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
