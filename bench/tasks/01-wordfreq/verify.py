#!/usr/bin/env python
"""Held-out oracle for task 01-wordfreq. Run with cwd = the agent's workspace.

Independent of whatever tests the agent wrote: we feed a known sample and check the
tool's actual top-3 output. Exits 0 on success, 1 on failure (with a one-line reason).
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

WS = Path.cwd()


def fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def main() -> int:
    tool = WS / "wordfreq.py"
    if not tool.is_file():
        return fail("wordfreq.py not found at workspace root")

    # "the cat sat on the mat the cat" -> the=3, cat=2, then sat/on/mat tie at 1
    # tie broken alphabetically => mat, on, sat. So top 3 = the:3, cat:2, mat:1.
    sample = WS / "_oracle_sample.txt"
    sample.write_text("the cat sat on the mat the cat\n", encoding="utf-8")

    try:
        proc = subprocess.run(
            [sys.executable, "wordfreq.py", "_oracle_sample.txt", "--top", "3"],
            cwd=WS,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return fail("tool timed out on the oracle sample")

    if proc.returncode != 0:
        return fail(f"tool exited {proc.returncode}; stderr={proc.stderr.strip()[:200]}")

    # Parse lines as "word: count" (lenient on surrounding whitespace).
    pairs: list[tuple[str, int]] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z0-9]+)\s*:\s*(\d+)$", line)
        if not m:
            return fail(f"line not in 'word: count' format: {line!r}")
        pairs.append((m.group(1).lower(), int(m.group(2))))

    expected = [("the", 3), ("cat", 2), ("mat", 1)]
    if pairs[:3] != expected:
        return fail(f"top-3 mismatch: got {pairs[:3]}, expected {expected}")

    # Informational: did the agent's own test suite pass?
    agent_tests = "n/a"
    if (WS / "test_wordfreq.py").is_file():
        tp = subprocess.run(
            [sys.executable, "-m", "pytest", "test_wordfreq.py", "-q"],
            cwd=WS,
            capture_output=True,
            text=True,
            timeout=90,
        )
        agent_tests = "pass" if tp.returncode == 0 else "FAIL"

    print(f"PASS: top-3 correct {pairs[:3]}; agent_tests={agent_tests}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
