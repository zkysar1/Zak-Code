"""Reads a REPLICATION batch of bench/effort_product_route.py.

The first batch is read by that script's own --report. A replication asks one narrower question
of each cell it repeats: on the product's OWN route, does a configured depth of low beat what the
product sends today, a second time?

  the product's route   P-low against P-none   REPLICATES / REVERSED / DOES NOT REPLICATE

The thresholds are the first bench's replication reader's, imported and not copied, so the two
readers cannot drift (bench/effort_replication_read.py: a gain of 5 replicates, a loss of 4 is a
reversal, a reference at 16 or above has no room, fewer than 18 scored calls is not measured).
The wire gate comes first and is the call script's own: if any scored product call's last request
did not go to /v1/responses carrying the arm's own effort, nothing is read. The three raw arms
are printed beside each cell as same-batch controls and decide nothing.

Over the repeated cells: REPLICATED only if every cell REPLICATES; REVERSED if any cell is;
NOT MEASURED if any cell is; otherwise NOT REPLICATED. Nothing here makes a network call.

usage: effort_product_route_replication_read.py <batch.jsonl> [<first-batch.jsonl>]
       effort_product_route_replication_read.py --selftest
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import effort_product_route as epr  # noqa: E402 — sibling bench script: arms, per20, the wire gate
import effort_replication_read as err  # noqa: E402 — sibling reader: the replication thresholds

NOT_READ = "NOT THE BRIDGE: NOT READ"


def read_cell(rows: list[dict[str, Any]], cell: str) -> dict[str, Any]:
    rates = {arm: epr.per20(rows, cell, arm) for arm in epr.ARMS}
    out: dict[str, Any] = {
        "cell": cell,
        "per20": {a: (None if v[0] is None else round(v[0], 1)) for a, v in rates.items()},
        "scored": {a: v[1] for a, v in rates.items()},
    }
    p_none, p_low = rates["P-none"][0], rates["P-low"][0]
    if p_none is None or p_low is None:
        out["the_products_route"] = "NOT MEASURED"
    else:
        out["the_products_route"] = err.product_line(p_none, p_low)
    return out


def decide(labels: list[str]) -> str:
    if not labels or any(label == "NOT MEASURED" for label in labels):
        return "NOT MEASURED"
    if any(label == "REVERSED" for label in labels):
        return "REVERSED"
    if all(label == "REPLICATES" for label in labels):
        return "REPLICATED"
    return "NOT REPLICATED"


def read_batch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wire = epr.wire_line(rows)
    cells = [read_cell(rows, c) for c in sorted({str(r.get("cell")) for r in rows})]
    decision = decide([c["the_products_route"] for c in cells])
    return {
        "decision": decision if wire["all_on_bridge"] else NOT_READ,
        "wire": wire,
        "cells": cells,
    }


def load(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def report(path: str, first: str | None) -> int:
    rows = load(path)
    served = sorted({str(r.get("served_model")) for r in rows if r.get("status") == 200})
    errors = sum(1 for r in rows if r.get("status") != 200)
    print(f"rows: {len(rows)}  served models: {served}  http errors: {errors}")
    print(json.dumps(read_batch(rows), indent=1, sort_keys=True))
    if first:
        before = load(first)
        for cell in sorted({str(r.get("cell")) for r in rows}):
            print("first batch: " + json.dumps(read_cell(before, cell), sort_keys=True))
    return 0


def _batch(counts: dict[str, dict[str, int]], **extra: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell, arms in counts.items():
        for arm, passes in arms.items():
            wire = epr._bridge(epr.PRODUCT_ARMS[arm]) if arm in epr.PRODUCT_ARMS else {}
            rows += epr._rows(cell, arm, passes, status=200, **{**wire, **extra})
    return rows


def selftest() -> int:
    bad: list[str] = []

    def check(name: str, got: Any, want: Any) -> None:
        if got != want:
            bad.append(f"{name}: got {got!r}, want {want!r}")

    # Every (P-none, P-low) pair maps to exactly one of the three labels, and the boundaries sit
    # where the imported constants say: one pass either side of each.
    known = {"REPLICATES", "REVERSED", "DOES NOT REPLICATE", "DOES NOT REPLICATE (NO ROOM)"}
    for none in range(21):
        for low in range(21):
            if err.product_line(none, low) not in known:
                bad.append(f"unlabelled pair ({none}, {low})")
    for none, low, want in (
        (5, 10, "REPLICATES"),
        (5, 9, "DOES NOT REPLICATE"),
        (9, 5, "REVERSED"),
        (8, 5, "DOES NOT REPLICATE"),
        (16, 20, "DOES NOT REPLICATE (NO ROOM)"),
        (15, 20, "REPLICATES"),
        (0, 20, "REPLICATES"),
        (20, 16, "REVERSED"),
    ):
        check(f"line({none},{low})", err.product_line(none, low), want)

    for labels, want in (
        (["REPLICATES", "REPLICATES"], "REPLICATED"),
        (["REPLICATES"], "REPLICATED"),
        (["REPLICATES", "DOES NOT REPLICATE"], "NOT REPLICATED"),
        (["REPLICATES", "DOES NOT REPLICATE (NO ROOM)"], "NOT REPLICATED"),
        (["REPLICATES", "REVERSED"], "REVERSED"),
        (["REVERSED", "NOT MEASURED"], "NOT MEASURED"),
        (["REPLICATES", "NOT MEASURED"], "NOT MEASURED"),
        ([], "NOT MEASURED"),
    ):
        check(f"decide({labels})", decide(labels), want)

    controls = {"A-chat-none": 19, "B-resp-low": 18, "C-resp-none": 6}
    first = _batch(
        {
            "order": {**controls, "P-none": 5, "P-low": 19},
            "refusal": {**controls, "P-none": 0, "P-low": 20},
        }
    )
    got = read_batch(first)
    check("first batch decision", got["decision"], "REPLICATED")
    check("first batch order per20", got["cells"][0]["per20"]["P-low"], 19.0)

    # The arms must not be read the wrong way round: a product that beats its own low arm is a
    # reversal, never a replication.
    swapped = _batch({"order": {**controls, "P-none": 19, "P-low": 5}})
    check("swapped arms", read_batch(swapped)["decision"], "REVERSED")

    mixed = _batch(
        {
            "order": {**controls, "P-none": 17, "P-low": 19},
            "refusal": {**controls, "P-none": 0, "P-low": 20},
        }
    )
    check("one cell with no room", read_batch(mixed)["decision"], "NOT REPLICATED")

    # The wire gate outranks every count: one scored P-low call that carried effort none.
    off = [dict(r) for r in first]
    for r in off:
        if r["arm"] == "P-low" and r["cell"] == "order":
            r["wire"] = [{"path": epr.BRIDGE_PATH, "effort": "none"}]
            break
    check("one call off its own effort", read_batch(off)["decision"], NOT_READ)
    chat = [dict(r) for r in first]
    for r in chat:
        if r["arm"] == "P-none" and r["cell"] == "refusal":
            r["wire"] = [{"path": "/v1/chat/completions", "effort": "none"}]
            break
    check("one call off the bridge", read_batch(chat)["decision"], NOT_READ)

    # Under 18 scored calls in either product arm the cell, and so the decision, is not measured;
    # a thin CONTROL arm decides nothing.
    thin = [r for r in first if not (r["arm"] == "P-low" and r["cell"] == "order")]
    thin += epr._rows("order", "P-low", 17, n=17, status=200, **epr._bridge("low"))
    check("17 scored P-low calls", read_batch(thin)["decision"], "NOT MEASURED")
    thin_control = [r for r in first if not (r["arm"] == "A-chat-none" and r["cell"] == "order")]
    check("a missing control arm", read_batch(thin_control)["decision"], "REPLICATED")

    for line in bad:
        print("FAIL " + line)
    print(f"selftest: {8 + 8 + 8} literal cases, 441 pairs, {len(bad)} mismatches")
    return 1 if bad else 0


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        return selftest()
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        return 2
    return report(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else None)


if __name__ == "__main__":
    sys.exit(main())
