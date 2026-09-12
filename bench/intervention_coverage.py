#!/usr/bin/env python
"""Which of the engine's intervention paths has the bench ever actually exercised?

ADR-0146 reported FOUR robustness paths that never fired in any bench run. That number was bounded
by what happened to be instrumented, not by the engine: a full census says the engine can emit ~52
distinct intervention kinds and the bench has ever recorded SEVEN. A mechanism whose trigger the
test never reaches is untested, and it reports identically to one that works.

Two modes, and the second is the point:

  --census   Enumerate every `kind="..."` the engine can emit, diff against every kind recorded in
             bench/results/**/*.json, and list what has never fired. A map of the blind spots.

  --ratchet  Fail (exit 1) when a kind that WAS exercised in the recorded corpus no longer appears
             in the newest results file. This is the detector the campaign kept wishing it had: an
             instrument that silently stops measuring reads exactly like a clean world, and the
             only thing that distinguishes them is a second number read beside the first. Here the
             second number is yesterday's coverage.

Usage:
    ./.venv/bin/python bench/intervention_coverage.py --census
    ./.venv/bin/python bench/intervention_coverage.py --ratchet bench/results/suite-10tasks.json
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOOP = ROOT / "src" / "zakcode" / "agent"
RESULTS = ROOT / "bench" / "results"

# Kinds the bench's workload structurally cannot reach, with the reason. Listing them is what keeps
# the census honest: "45 paths untested" is alarming and partly unfair, and an unexplained
# exclusion list is how a coverage number gets quietly gamed.
UNREACHABLE = {
    "skill_page": "no skills (enable_skills=False)",
    "skill_paging": "no skills",
    "skill_page_body_missing": "no skills",
    "skill_sections_dropped": "no skills",
    "skill_sections_reopened": "no skills",
    "skill_sections_restored": "no skills",
    "skill_skeleton": "no skills",
    "skill_coverage": "no skills",
    "user_only_skill": "no skills",
    "await_user": "non-interactive",
    "awaiting_user": "non-interactive",
    "awaiting_refused": "non-interactive",
    "wakeup": "no scheduler in the bench",
    "say": "no voice surface",
    "slash_text_routed": "no slash surface",
    "restart": "single-turn runner",
}


def emitted_kinds() -> dict[str, list[str]]:
    out: dict[str, list[str]] = collections.defaultdict(list)
    for path in sorted(LOOP.rglob("*.py")):
        for i, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
            for m in re.finditer(r'kind="([a-z_]+)"', line):
                out[m.group(1)].append(f"{path.relative_to(ROOT)}:{i}")
    return dict(out)


def recorded_kinds(files: list[Path]) -> collections.Counter:
    seen: collections.Counter = collections.Counter()
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows = d.get("tasks") if isinstance(d, dict) and "tasks" in d else [d]
        for row in rows or []:
            if isinstance(row, dict):
                for k, v in (row.get("trace_interventions") or {}).items():
                    seen[k] += v
    return seen


def main(argv: list[str]) -> int:
    emitted = emitted_kinds()
    if "--ratchet" in argv:
        idx = argv.index("--ratchet")
        newest = Path(argv[idx + 1]) if len(argv) > idx + 1 else None
        if newest is None or not newest.is_file():
            print("--ratchet needs a results JSON path", file=sys.stderr)
            return 2
        corpus = sorted(p for p in RESULTS.rglob("*.json") if p != newest)
        was = set(recorded_kinds(corpus))
        now = set(recorded_kinds([newest]))
        gone = sorted(was - now)
        print(f"previously exercised: {len(was)}   in {newest.name}: {len(now)}")
        if gone:
            print("WENT DARK -- these fired in the recorded corpus and do not appear here:")
            for k in gone:
                print(f"  {k}")
            print("\nA path that stopped firing is either a fixed bug or a BROKEN INSTRUMENT, and")
            print("the two are indistinguishable from this output alone. Establish which before")
            print("citing any pass that contains none of them.")
            return 1
        print("no previously-exercised kind went dark.")
        return 0

    seen = recorded_kinds(sorted(RESULTS.rglob("*.json")))
    never = sorted(set(emitted) - set(seen))
    blocked = [k for k in never if k in UNREACHABLE]
    gap = [k for k in never if k not in UNREACHABLE]
    print(f"kinds the engine can emit:     {len(emitted)}")
    print(f"kinds the bench has recorded:  {len(seen)}  {dict(seen)}")
    print(f"never fired:                   {len(never)}")
    print(f"  structurally unreachable:    {len(blocked)}")
    print(f"  REACHABLE BUT UNTESTED:      {len(gap)}")
    for k in gap:
        print(f"    {k:26} emitted at {', '.join(emitted[k][:2])}")
    print()
    for k in blocked:
        print(f"  (unreachable) {k:24} {UNREACHABLE[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
