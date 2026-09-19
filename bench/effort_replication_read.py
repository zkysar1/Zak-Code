"""Reads a REPLICATION batch of bench/effort_decision_points.py, one cell at a time.

The first batch is read by that script's own --report. A replication asks three narrower
questions of each cell it repeats, and each has its own line here so that none of them can
borrow a verdict from another:

  against today's product   B against A   does effort low on the Responses route beat what
                                          the product sends today?
  within the route          B against C   does effort move the cell on the Responses route?
  route effect              A against C   does the route itself move the cell?

Every (A, B, C) in 0..20 maps to exactly one label on each line; --selftest proves it by
enumeration. Counts are passes per 20: a rate times 20, so an arm that lost a call to an
HTTP error is read on the same scale. Nothing here makes a network call. Stdlib only.

usage: effort_replication_read.py <batch.jsonl> [<first-batch.jsonl>]
       effort_replication_read.py --selftest
"""

from __future__ import annotations

import json
import sys
from typing import Any

ARMS = ("A-chat-none", "B-resp-low", "C-resp-none")
REPLICATES = 5  # passes per 20, the first batch's R5 line
REVERSED = 4  # passes per 20, the first batch's R3 (HARM) line
ROUTE = 3  # passes per 20, the first batch's R1 line
NO_ROOM = 16  # a reference at or above this cannot be beaten by REPLICATES
MIN_SCORED = 18


def product_line(a: float, b: float) -> str:
    if b - a >= REPLICATES:
        return "REPLICATES"
    if a - b >= REVERSED:
        return "REVERSED"
    return "DOES NOT REPLICATE (NO ROOM)" if a >= NO_ROOM else "DOES NOT REPLICATE"


def route_line(b: float, c: float) -> str:
    if b - c >= REPLICATES:
        return "REPLICATES"
    if c - b >= REVERSED:
        return "REVERSED"
    return "DOES NOT REPLICATE (NO ROOM)" if c >= NO_ROOM else "DOES NOT REPLICATE"


def route_effect_line(a: float, c: float) -> str:
    if a - c >= ROUTE:
        return "REPLICATES"
    if c - a >= ROUTE:
        return "REVERSED"
    return "ABSENT"


def per20(rows: list[dict[str, Any]], cell: str, arm: str) -> tuple[float | None, int, int]:
    scored = [r for r in rows if r.get("cell") == cell and r.get("arm") == arm]
    ok = [r for r in scored if r.get("status") == 200 and r.get("pass") is not None]
    if len(ok) < MIN_SCORED:
        return None, len(ok), len(scored)
    return 20.0 * sum(1 for r in ok if r["pass"]) / len(ok), len(ok), len(scored)


def read_cell(rows: list[dict[str, Any]], cell: str) -> dict[str, Any]:
    counts = {arm: per20(rows, cell, arm) for arm in ARMS}
    if any(v[0] is None for v in counts.values()):
        return {
            "cell": cell,
            "reading": "NOT MEASURED",
            "scored": {k: v[1] for k, v in counts.items()},
        }
    a, b, c = (counts[arm][0] for arm in ARMS)
    assert a is not None and b is not None and c is not None
    return {
        "cell": cell,
        "per20": {"A": a, "B": b, "C": c},
        "against_todays_product": product_line(a, b),
        "within_the_route": route_line(b, c),
        "route_effect": route_effect_line(a, c),
    }


def load(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def report(path: str, first: str | None) -> int:
    rows = load(path)
    before = load(first) if first else []
    served = sorted({str(r.get("served_model")) for r in rows if r.get("status") == 200})
    errors = sum(1 for r in rows if r.get("status") != 200)
    print(f"rows: {len(rows)}  served models: {served}  http errors: {errors}")
    for cell in sorted({str(r.get("cell")) for r in rows}):
        print(json.dumps(read_cell(rows, cell), sort_keys=True))
        if before:
            print(
                "  first batch: " + json.dumps(read_cell(before, cell).get("per20"), sort_keys=True)
            )
    return 0


def selftest() -> int:
    bad = 0
    labels: dict[str, set[str]] = {"product": set(), "route": set(), "effect": set()}
    for a in range(21):
        for b in range(21):
            for c in range(21):
                got = (product_line(a, b), route_line(b, c), route_effect_line(a, c))
                labels["product"].add(got[0])
                labels["route"].add(got[1])
                labels["effect"].add(got[2])
                # one label per line, and the two directions of a line never both hold
                if (b - a >= REPLICATES) and (a - b >= REVERSED):
                    bad += 1
                if got[0] == "REPLICATES" and b - a < REPLICATES:
                    bad += 1
                if got[0].startswith("DOES NOT") and (b - a >= REPLICATES or a - b >= REVERSED):
                    bad += 1
                if got[2] == "ABSENT" and abs(a - c) >= ROUTE:
                    bad += 1
    cases = [
        ((6, 20, 0), ("REPLICATES", "REPLICATES", "REPLICATES")),  # the first batch's refusal
        ((17, 19, 6), ("DOES NOT REPLICATE (NO ROOM)", "REPLICATES", "REPLICATES")),  # its order
        ((10, 14, 10), ("DOES NOT REPLICATE", "DOES NOT REPLICATE", "ABSENT")),
        ((12, 8, 12), ("REVERSED", "REVERSED", "ABSENT")),
        ((5, 10, 9), ("REPLICATES", "DOES NOT REPLICATE", "REVERSED")),
        ((20, 20, 20), ("DOES NOT REPLICATE (NO ROOM)", "DOES NOT REPLICATE (NO ROOM)", "ABSENT")),
        # Each threshold, one pass either side of it, in literal numbers: a changed constant
        # cannot satisfy these, where the enumeration above would follow it.
        ((6, 10, 6), ("DOES NOT REPLICATE", "DOES NOT REPLICATE", "ABSENT")),  # +4
        ((6, 11, 6), ("REPLICATES", "REPLICATES", "ABSENT")),  # +5
        ((10, 7, 10), ("DOES NOT REPLICATE", "DOES NOT REPLICATE", "ABSENT")),  # -3
        ((10, 6, 10), ("REVERSED", "REVERSED", "ABSENT")),  # -4
        ((10, 10, 8), ("DOES NOT REPLICATE", "DOES NOT REPLICATE", "ABSENT")),  # route 2
        ((10, 10, 7), ("DOES NOT REPLICATE", "DOES NOT REPLICATE", "REPLICATES")),  # route 3
        ((8, 10, 10), ("DOES NOT REPLICATE", "DOES NOT REPLICATE", "ABSENT")),  # route -2
        ((7, 10, 10), ("DOES NOT REPLICATE", "DOES NOT REPLICATE", "REVERSED")),  # route -3
        ((15, 17, 15), ("DOES NOT REPLICATE", "DOES NOT REPLICATE", "ABSENT")),  # room
        ((16, 18, 16), ("DOES NOT REPLICATE (NO ROOM)", "DOES NOT REPLICATE (NO ROOM)", "ABSENT")),
    ]
    for (a, b, c), want in cases:
        got = (product_line(a, b), route_line(b, c), route_effect_line(a, c))
        if got != want:
            bad += 1
            print(f"SELFTEST MISMATCH {(a, b, c)}: want {want} got {got}")
    for n, want_reading in ((17, "NOT MEASURED"), (18, None)):
        few = [{"cell": "x", "arm": arm, "status": 200, "pass": True} for arm in ARMS] * n
        if read_cell(few, "x").get("reading") != want_reading:
            bad += 1
            print(f"SELFTEST MISMATCH: {n} scored calls per arm must read {want_reading}")
    lost = [
        {"cell": "y", "arm": arm, "status": 200 if i else 500, "pass": True if i else None}
        for arm in ARMS
        for i in range(20)
    ]
    if read_cell(lost, "y").get("per20") != {"A": 20.0, "B": 20.0, "C": 20.0}:
        bad += 1
        print("SELFTEST MISMATCH: an arm that lost one call is read on the same 20 scale")
    want_labels = {
        "product": {"REPLICATES", "REVERSED", "DOES NOT REPLICATE", "DOES NOT REPLICATE (NO ROOM)"},
        "route": {"REPLICATES", "REVERSED", "DOES NOT REPLICATE", "DOES NOT REPLICATE (NO ROOM)"},
        "effect": {"REPLICATES", "REVERSED", "ABSENT"},
    }
    if labels != want_labels:
        bad += 1
        print(f"SELFTEST MISMATCH labels reached: {labels}")
    summary = f"9261 count triples, {len(cases)} named cases, 3 coverage cases"
    print(f"selftest: {summary}, {bad} mismatches")
    return 1 if bad else 0


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        return selftest()
    if len(sys.argv) not in (2, 3):
        print(__doc__, file=sys.stderr)
        return 2
    return report(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else None)


if __name__ == "__main__":
    raise SystemExit(main())
