#!/usr/bin/env python
"""Held-out oracle for 08-mutation-leak. Run with cwd = the agent's workspace.

WHY THIS TASK EXISTS. Two prior attempts at ADR-0151's unblock both passed. The measured lesson
(ADR-0151 addendum) is that EXPLORATION costs turns while fully-specified intricacy costs almost
nothing -- 06-plugin-conventions took 2.7x baseline turns, 07-ttl-cache took 1.1x. So this task
maximises DISTANCE between symptom and cause rather than local intricacy: the symptom surfaces in
audit.py, the defect is in normalize.py, and the mechanism that connects them (a deliberately shared
cache) is in load.py -- three modules apart, none of which is wrong on its own reading.

The oracle is OUTCOME-based, not location-based: any fix achieving the invariant is accepted, and
nothing mandates which file changes. But check 5 pins the ROOT rather than the symptom -- it calls
normalize() on a caller-owned list and requires that list to be untouched -- so silencing the audit
or reloading around the problem cannot pass.

Exits 0 on success, 1 on the first failure.
"""
from __future__ import annotations

import copy
import importlib
import json
import sys
from pathlib import Path

WS = Path.cwd()
EXPECTED = {"count": 4, "total_usd": 139.22}


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    sys.path.insert(0, str(WS))
    for m in [k for k in list(sys.modules) if k == "pipeline" or k.startswith("pipeline.")]:
        del sys.modules[m]
    pipeline = importlib.import_module("pipeline")

    on_disk = json.loads((WS / "records.json").read_text(encoding="utf-8"))

    # 1. the pipeline runs clean
    summary, problems = pipeline.run()
    if problems:
        fail(f"run() still reports audit problems: {problems[:3]}")
    if summary != EXPECTED:
        fail(f"summary is {summary}, expected {EXPECTED}")

    # 2. the audit still has teeth -- it must FAIL on a deliberately corrupted raw record.
    #    This is what stops "make the audit quiet" from passing: a weakened or bypassed audit
    #    reports clean here too, and clean is exactly what a correct fix also reports.
    raw = pipeline.load_records()
    if not isinstance(raw, list) or not raw:
        fail("load_records() did not return a non-empty list")
    victim = raw[0]
    saved = victim.get("amount")
    victim["amount"] = 999.99  # a float where the source has a string
    still_detects = pipeline.audit(summary)
    victim["amount"] = saved
    if not still_detects:
        fail("audit() no longer detects a corrupted raw record, so it was weakened, bypassed or "
             "made to read around the problem. Rule 3 requires it to keep checking what it checks.")

    # 3. the source data on disk is untouched
    if json.loads((WS / "records.json").read_text(encoding="utf-8")) != on_disk:
        fail("records.json on disk was modified; the pipeline must not rewrite its source")

    # 4. the records handed out by the loader are still the source values
    for r in pipeline.load_records():
        if not isinstance(r["amount"], str):
            fail(f"after run(), raw record {r['id']} has a {type(r['amount']).__name__} amount; "
                 f"the source records were altered (rule 2)")

    # 5. ROOT CAUSE: normalize() must not mutate a caller-owned list. Location-agnostic -- it
    #    tests the behaviour, so a fix anywhere that achieves it is accepted, and a fix that only
    #    hides the symptom downstream is not.
    mine = [{"id": 99, "amount": "7.25", "currency": " eur "}]
    before = copy.deepcopy(mine)
    out = pipeline.normalize(mine)
    if mine != before:
        fail(f"normalize() mutated the list it was given: {before} -> {mine}. The root cause is "
             f"in-place mutation of the caller's records, not anything downstream of it.")
    if not out or out[0]["amount"] != 7.25 or out[0]["currency"] != "EUR":
        fail(f"normalize() no longer produces the coerced values; got {out}")

    print("PASS: audit clean and still has teeth, summary correct, source records untouched, "
          "and normalize() no longer mutates its input")


if __name__ == "__main__":
    main()
