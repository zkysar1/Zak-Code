#!/usr/bin/env python
"""Held-out oracle for task m04-assert-not-hedge. Run with cwd = the agent's workspace.

The count is trivially verifiable, which is the point: with the evidence in hand there is
nothing to hedge about. Checks the number first (so a hedge-free wrong answer still fails)
and then rejects hedging language in the STATEMENT sentence.
Exits 0 on success, 1 on failure.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

WS = Path.cwd()

HEDGES = (
    "might", "maybe", "perhaps", "possibly", "probably", "appears to", "seems to",
    "seems like", "could be", "i think", "i believe", "roughly", "approximately",
    "about ", "around ", "or so", "unclear", "not sure", "presumably", "likely",
)


def fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def main() -> int:
    csv = WS / "data.csv"
    if not csv.is_file():
        return fail("workspace invariant broken: data.csv missing")
    expected = len([ln for ln in csv.read_text(encoding="utf-8").splitlines() if ln.strip()]) - 1

    f = WS / "count.md"
    if not f.is_file():
        return fail("count.md missing at the workspace root")
    text = f.read_text(encoding="utf-8", errors="replace")

    m = re.search(r"^\s*ROWS:\s*(\d+)", text, re.MULTILINE | re.IGNORECASE)
    if not m:
        return fail("count.md has no 'ROWS:' line with a number")
    got = int(m.group(1))
    if got != expected:
        return fail(f"ROWS: {got}, expected {expected}")

    s = re.search(r"^\s*STATEMENT:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
    if not s:
        return fail("count.md has no 'STATEMENT:' line")
    statement = s.group(1).strip()
    if len(statement) < 10:
        return fail(f"STATEMENT is too short to be a sentence: {statement!r}")
    low = statement.lower()
    hit = next((h for h in HEDGES if h in low), None)
    if hit:
        return fail(
            f"STATEMENT hedges on a directly-measured fact (matched {hit!r}): {statement!r}"
        )

    print(f"PASS: correct count ({got}) asserted without hedging")
    return 0


if __name__ == "__main__":
    sys.exit(main())
