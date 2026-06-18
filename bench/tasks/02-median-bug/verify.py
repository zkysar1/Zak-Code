#!/usr/bin/env python
"""Held-out oracle for task 02-median-bug. Run with cwd = the agent's workspace.

Imports median directly and checks values (independent of the seeded test file, so a
fix that cheated by editing tests still fails here). Exits 0 on success, 1 on failure.
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
    if not (WS / "stats.py").is_file():
        return fail("stats.py missing")
    sys.path.insert(0, str(WS))
    try:
        import stats  # noqa: F401

        importlib.reload(stats)
        median = stats.median
    except Exception as e:  # noqa: BLE001
        return fail(f"could not import stats.median: {type(e).__name__}: {e}")

    cases = [
        ([3, 1, 2], 2),
        ([1, 2, 3, 4], 2.5),
        ([10, 20], 15.0),
        ([42], 42),
        ([5, 3, 1, 2, 4], 3),
        ([4, 1, 3, 2], 2.5),
    ]
    for nums, expected in cases:
        got = median(list(nums))
        if got != expected:
            return fail(f"median({nums}) = {got}, expected {expected}")

    print(f"PASS: median correct on {len(cases)} held-out cases (incl. even-length averaging)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
