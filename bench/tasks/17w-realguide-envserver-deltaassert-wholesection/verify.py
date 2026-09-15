#!/usr/bin/env python3
"""Verify 17-realguide-envserver-deltaassert: the new test in tests/test_metrics.py must
exercise Metrics.record_hit() and assert a BEFORE/AFTER DELTA on the shared counter
`Metrics.hits` — never an absolute value. The rule ("Assert on before/after DELTAS for
shared static counters, never absolutes.") is stated ONLY in the guide, inside a 7K
sub-section the shipped fold drops; the workspace's existing tests assert absolutes, so
imitation FAILS and pass ⟺ the rule was applied. Checks: (1) tests/test_metrics.py defines
a test; (2) static — an assertion that relates TWO readings of the counter
(`Metrics.hits == before + 1`, `after - before == 1`, `after == before + 1`, a value
derived from a reading) is a delta; an assertion comparing ONE reading (or one captured
value) with an integer literal is an absolute (a reset-then-absolute and `after == 1`
included); pass needs a delta and no absolute; (3) the file passes under pytest. Run from
the task workspace root (cwd == workspace).

Verify history: the first version required the delta assertion to NAME `hits`, so the
rule's canonical form `after - before == 1` (5 of 12 rule-in runs on the 35B) read as
"no assertion" — a false negative found by reading the outputs (thrust 26 stage 2,
2026-09-15); every arm was re-scored from its recorded test files with this version.
"""
import re
import subprocess
import sys
from pathlib import Path

TEST_FILE = Path("tests/test_metrics.py")
HITS = re.compile(r"\bhits\b")
ASSIGN = re.compile(r"^\s*(\w+)\s*=(?!=)\s*(.*)$")
INT_CMP = re.compile(r"(?:==|!=|<=|>=|<|>)\s*[-+]?\d+\b|\b[-+]?\d+\s*(?:==|!=|<=|>=|<|>)")
UNITTEST_INT = re.compile(
    r"assert(?:Equal|Equals|NotEqual)\(\s*(?:[-+]?\d+\s*,|[^,]*,\s*[-+]?\d+\s*\))"
)


def static_check(src: str) -> tuple[bool, str]:
    """(ok, message) for the delta rule on the test source — needs no pytest."""
    if not re.search(r"^\s*def test_\w+", src, re.M):
        return False, "tests/test_metrics.py defines no test function"
    lines = src.splitlines()
    captured: set[str] = set()  # names holding a reading of the counter, or derived from one
    changed = True
    while changed:
        changed = False
        for ln in lines:
            m = ASSIGN.match(ln)
            if not m or m.group(1) in captured:
                continue
            rhs = m.group(2)
            if HITS.search(rhs) or any(re.search(rf"\b{re.escape(c)}\b", rhs) for c in captured):
                captured.add(m.group(1))
                changed = True
    alternatives = "".join("|" + re.escape(c) for c in sorted(captured))
    names = re.compile(r"\b(?:hits" + alternatives + r")\b")
    relevant = [ln for ln in lines if re.search(r"\bassert\w*\b", ln) and names.search(ln)]
    if not relevant:
        return False, (
            "no assertion on Metrics.hits or a value read from it "
            f"(captured: {sorted(captured) or 'none'})"
        )
    absolute, delta = [], []
    for ln in relevant:
        refs = len(set(names.findall(ln)))  # distinct readings/derived values in the assertion
        literal = bool(INT_CMP.search(ln) or UNITTEST_INT.search(ln))
        if refs >= 2:
            delta.append(ln.strip())
        elif literal:
            absolute.append(ln.strip())
    if absolute:
        return False, (
            f"absolute assertion on the shared counter: {absolute[0]!r} "
            "(the project asserts before/after deltas)"
        )
    if not delta:
        asserts = [a.strip() for a in relevant][:2]
        return False, f"no before/after delta assertion (asserts: {asserts})"
    return True, f"delta assertion on Metrics.hits: {delta[0]!r}"


def main() -> int:
    if not TEST_FILE.exists():
        print("FAIL: tests/test_metrics.py is missing")
        return 1
    ok, msg = static_check(TEST_FILE.read_text(encoding="utf-8"))
    if not ok:
        print(f"FAIL: {msg}")
        return 1
    run = subprocess.run(
        [sys.executable, "-m", "pytest", str(TEST_FILE), "-q", "-p", "no:cacheprovider"],
        capture_output=True, text=True, timeout=90,
    )
    if run.returncode != 0:
        tail = (run.stdout + run.stderr).strip().splitlines()[-3:]
        print(f"FAIL: pytest on tests/test_metrics.py exited {run.returncode}: {' | '.join(tail)}")
        return 1
    print(f"PASS: {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
