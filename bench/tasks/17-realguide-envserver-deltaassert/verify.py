#!/usr/bin/env python3
"""Verify 17-realguide-envserver-deltaassert: the new test in tests/test_metrics.py must exercise Metrics.record_hit()
and assert a BEFORE/AFTER DELTA on the shared counter `Metrics.hits` — never an absolute value. The rule ("Assert on
before/after DELTAS for shared static counters, never absolutes.") is stated ONLY in the guide, inside a 7K
sub-section the fold drops; the workspace's existing tests assert absolutes, so imitation FAILS and pass ⟺ the rule
was applied. Checks: (1) tests/test_metrics.py defines a test; (2) that file passes under pytest; (3) at least one
assertion mentions `hits`; (4) no assertion compares `hits` with an integer literal (absolute; a reset-then-absolute
counts as absolute); (5) some assertion on `hits` references a name captured from `hits` beforehand (the delta).
Run from the task workspace root (cwd == workspace).
"""
import re
import subprocess
import sys
from pathlib import Path

TEST_FILE = Path("tests/test_metrics.py")
ABSOLUTE = [
    re.compile(r"\bhits\b\s*(?:==|!=|>=|<=|>|<)\s*\d+\b"),
    re.compile(r"^\s*assert\s+\d+\s*(?:==|!=)\s*.*\bhits\b"),
    re.compile(r"assert(?:Equal|Equals|NotEqual)\(\s*(?:[\w.]*\bhits\b\s*,\s*[-+]?\d+\b|[-+]?\d+\s*,\s*[\w.]*\bhits\b)"),
]
CAPTURE = re.compile(r"^\s*(\w+)\s*=(?!=)\s*.*\bhits\b")


def main() -> int:
    if not TEST_FILE.exists():
        print("FAIL: tests/test_metrics.py is missing")
        return 1
    src = TEST_FILE.read_text(encoding="utf-8")
    if not re.search(r"^\s*def test_\w+", src, re.M):
        print("FAIL: tests/test_metrics.py defines no test function")
        return 1
    run = subprocess.run(
        [sys.executable, "-m", "pytest", str(TEST_FILE), "-q", "-p", "no:cacheprovider"],
        capture_output=True, text=True, timeout=90,
    )
    if run.returncode != 0:
        tail = (run.stdout + run.stderr).strip().splitlines()[-3:]
        print(f"FAIL: pytest on tests/test_metrics.py exited {run.returncode}: {' | '.join(tail)}")
        return 1
    lines = src.splitlines()
    asserts = [ln for ln in lines if re.search(r"\bassert\w*\b", ln) and re.search(r"\bhits\b", ln)]
    if not asserts:
        print("FAIL: no assertion on Metrics.hits in tests/test_metrics.py")
        return 1
    absolute = [ln.strip() for ln in asserts if any(p.search(ln) for p in ABSOLUTE)]
    if absolute:
        print(f"FAIL: absolute assertion on the shared counter: {absolute[0]!r} (the project asserts before/after deltas)")
        return 1
    captured = {m.group(1) for ln in lines for m in [CAPTURE.match(ln)] if m}
    delta = [ln.strip() for ln in asserts if any(re.search(rf"\b{re.escape(c)}\b", ln) for c in captured)]
    if not delta:
        print(f"FAIL: no before/after delta assertion on hits (captured names: {sorted(captured) or 'none'}; asserts: {[a.strip() for a in asserts][:2]})")
        return 1
    print(f"PASS: delta assertion on Metrics.hits: {delta[0]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
