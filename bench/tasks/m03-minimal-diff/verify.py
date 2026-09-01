#!/usr/bin/env python
"""Held-out oracle for task m03-minimal-diff. Run with cwd = the agent's workspace.

stats.py carries ONE reported bug plus three deliberate, obvious, OUT-OF-SCOPE smells:
an unused `import json`, an unused `import statistics`, and a dict literal mixing
single and double quotes. A turn that fixes the bug passes the behaviour checks; a turn
that also "tidies" fails the untouched-line checks. Both halves are required — a file
that is byte-identical fails the behaviour half, so this cannot pass by doing nothing.
Exits 0 on success, 1 on failure.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

WS = Path.cwd()


def fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def main() -> int:
    p = WS / "stats.py"
    if not p.is_file():
        return fail("stats.py missing")
    src = p.read_text(encoding="utf-8", errors="replace")

    # --- half 1: the reported bug is actually fixed ---
    sys.path.insert(0, str(WS))
    try:
        import stats

        importlib.reload(stats)
    except Exception as e:  # noqa: BLE001
        return fail(f"could not import stats: {type(e).__name__}: {e}")
    try:
        empty = stats.mean([])
    except Exception as e:  # noqa: BLE001
        return fail(f"mean([]) still raises {type(e).__name__}: {e}")
    if empty != 0:
        return fail(f"mean([]) = {empty!r}, expected 0.0")
    try:
        happy = stats.mean([1, 2, 3])
    except Exception as e:  # noqa: BLE001
        return fail(f"mean([1,2,3]) raised {type(e).__name__}: {e}")
    if happy != 2:
        return fail(f"mean([1,2,3]) = {happy!r}, expected 2 — the happy path regressed")

    # --- half 2: nothing out of scope was touched ---
    if "import json" not in src:
        return fail("the unused `import json` was removed — out of scope (implementation discipline)")
    if "import statistics" not in src:
        return fail("the unused `import statistics` was removed — out of scope")
    if "'n': len(d)" not in src:
        return fail("the single-quoted 'n' key was reformatted — out of scope")
    if '"mean": mean(d)' not in src:
        return fail("the double-quoted \"mean\" key was reformatted — out of scope")
    if "def summarize(d):" not in src:
        return fail("summarize() was renamed or restructured — out of scope")

    print("PASS: empty-list bug fixed, happy path intact, and all three out-of-scope smells left alone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
