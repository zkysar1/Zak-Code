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


#: Knobs that change WHICH intervention paths a run can reach. A baseline drawn from rows with
#: different knobs is not a baseline: a forced-compaction arm or a low-max_tokens arm reaches paths
#: an ordinary pass cannot, so comparing across them reports "went dark" on every ordinary pass --
#: an alarm that always fires is exactly as useless as one that never does. Measured: the first
#: version of this ratchet flagged six kinds against a clean 7-task pass.
_KNOB_KEYS = ("compact_fraction", "tool_deny", "max_completion_tokens", "pin_identity")


def _knob_signature(row: dict) -> tuple | None:
    """The row's knob configuration, or ``None`` when the row does not record one.

    ABSENCE OF THE FIELD IS NOT EVIDENCE OF ITS VALUE. Rows predating the `knobs` key would map to
    an all-``None`` tuple and become indistinguishable from a modern default-knob row -- measured:
    that put 183 rows in a baseline and flagged three kinds that had only ever fired under knobs the
    rows never recorded. An unknown configuration is excluded from every baseline instead.
    """
    k = row.get("knobs")
    if not isinstance(k, dict):
        return None
    return tuple(k.get(name) for name in _KNOB_KEYS)


def _rows(files: list[Path]) -> list[dict]:
    out: list[dict] = []
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows = d.get("tasks") if isinstance(d, dict) and "tasks" in d else [d]
        out.extend(r for r in (rows or []) if isinstance(r, dict))
    return out


def recorded_kinds(files: list[Path], signature: tuple | None = None) -> collections.Counter:
    """Intervention kinds recorded across ``files``.

    ``signature`` restricts the tally to rows whose knob configuration matches -- required for the
    ratchet, where a mismatched baseline manufactures false alarms.
    """
    seen: collections.Counter = collections.Counter()
    for row in _rows(files):
        if signature is not None and _knob_signature(row) != signature:
            continue
        for k, v in (row.get("trace_interventions") or {}).items():
            seen[k] += v
    return seen


def main(argv: list[str]) -> int:
    emitted = emitted_kinds()
    if "--compare" in argv:
        # Compare bench coverage against a PRODUCTION census: {kind: count}. The bench's own
        # history cannot tell you which untested paths matter, and the exclusion list above is a
        # statement about this harness's configuration, never about importance -- measured, the
        # paths it calls structurally unreachable are exactly the ones production leans on.
        idx = argv.index("--compare")
        prod = set(json.loads(Path(argv[idx + 1]).read_text(encoding="utf-8")))
        bench = set(recorded_kinds(sorted(RESULTS.rglob("*.json"))))
        emitted = set(emitted_kinds())
        print(f"engine can emit   : {len(emitted)}")
        print(f"bench recorded    : {len(bench)}")
        print(f"production        : {len(prod)}")
        print(f"BOTH              : {len(bench & prod)}  {sorted(bench & prod)}")
        print(f"PRODUCTION ONLY   : {len(prod - bench)}  {sorted(prod - bench)}")
        print("  ^ live in production and never exercised by any test here. This is the risk")
        print("    surface: an untested path that real users reach. Rank work by THIS list.")
        print(f"BENCH ONLY        : {len(bench - prod)}  {sorted(bench - prod)}")
        print(f"NEITHER           : {len(emitted - bench - prod)}")
        union = len((bench | prod) & emitted)
        print(f"union coverage    : {union}/{len(emitted)} = {union / len(emitted) * 100:.0f}%")
        return 0
    if "--ratchet" in argv:
        idx = argv.index("--ratchet")
        newest = Path(argv[idx + 1]) if len(argv) > idx + 1 else None
        if newest is None or not newest.is_file():
            print("--ratchet needs a results JSON path", file=sys.stderr)
            return 2
        corpus = sorted(p for p in RESULTS.rglob("*.json") if p != newest)
        new_rows = _rows([newest])
        sigs = {sig for r in new_rows if (sig := _knob_signature(r)) is not None}
        if not sigs:
            # Two different diagnoses, and conflating them sends the reader to the wrong fix: an
            # unparseable file is usually a run still in flight (measured while writing this), a
            # parseable one without `knobs` is a pass from before the field existed.
            why = ("could not be parsed -- is a run still writing it?" if not new_rows
                   else "records no `knobs` for any row")
            print(f"{newest.name} {why}, so its configuration is UNKNOWN.")
            print("A ratchet cannot compare an unknown configuration to anything: BLIND, not clean.")
            return 2
        if len(sigs) != 1:
            print(f"MIXED KNOBS in {newest.name}: {len(sigs)} configurations in one pass.")
            print("No single baseline is comparable, so this ratchet is BLIND here, not clean.")
            print("Report the pass per-arm, or re-run each arm as its own pass.")
            return 2
        signature = next(iter(sigs))
        baseline_rows = [r for r in _rows(corpus) if _knob_signature(r) == signature]
        was = set(recorded_kinds(corpus, signature))
        now = set(recorded_kinds([newest], signature))
        print(f"knobs {dict(zip(_KNOB_KEYS, signature, strict=True))}")
        print(f"baseline rows with these knobs: {len(baseline_rows)}")
        if not baseline_rows:
            print("NO COMPARABLE BASELINE EXISTS -- this configuration has never been recorded")
            print("before, so nothing can have gone dark. That is BLINDNESS, not a clean report:")
            print("this pass BECOMES the baseline, and the next one is the first real reading.")
            return 0
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
