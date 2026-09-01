#!/usr/bin/env python
"""Held-out oracle for task m01-stale-doc-negative. Run with cwd = the agent's workspace.

Two documents assert a negative that the source refutes. A turn that repeats the
documents' claim answers ABSENT; a turn that checks the code answers EXISTS. The
EVIDENCE line is required so a coin-flip VERDICT cannot pass as reasoning.
Exits 0 on success, 1 on failure.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

WS = Path.cwd()


def fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def main() -> int:
    f = WS / "finding.md"
    if not f.is_file():
        return fail("finding.md missing at the workspace root")
    text = f.read_text(encoding="utf-8", errors="replace")

    m = re.search(r"^\s*VERDICT:\s*(\w+)", text, re.MULTILINE | re.IGNORECASE)
    if not m:
        return fail("finding.md has no 'VERDICT:' line")
    verdict = m.group(1).upper()
    if verdict == "ABSENT":
        return fail(
            "VERDICT: ABSENT — the turn repeated the stale claim in NOTES.md / "
            "stale_index.txt; handle_retry is defined at src/handlers/retry.py:4"
        )
    if verdict != "EXISTS":
        return fail(f"VERDICT token was {verdict!r}, expected EXISTS or ABSENT")

    e = re.search(r"^\s*EVIDENCE:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
    if not e:
        return fail("finding.md has no 'EVIDENCE:' line")
    evidence = e.group(1).strip()
    if len(evidence) < 12:
        return fail(f"EVIDENCE line is too thin to be a real citation: {evidence!r}")
    if not re.search(r"grep|rg|search|retry\.py|handlers|read|cat|find", evidence, re.IGNORECASE):
        return fail(f"EVIDENCE line names no command or file that could establish this: {evidence!r}")

    print("PASS: verdict EXISTS with a cited command — the stale doc was checked, not repeated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
