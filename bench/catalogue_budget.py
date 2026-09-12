#!/usr/bin/env python3
"""CAT cells (catalogue budget, pre-registered 19:51 2026-09-12): apply the rules to the arm JSONs.

Usage:
    ./.venv/bin/python bench/catalogue_budget.py bench/results

Reads ``determinism-zakcode-pinworkspace-temp0-<task>.CAT-<ARM>-<task>.json`` for the five tasks
and two arms, prints per-task rows (pass, elapsed, turns, byte-identity), the pooled numbers, and
then the pre-registered rules R1-R5. R6 (use_skill calls) is dump-derived and is counted beside
the dumps on the bench box, never here (the dumps carry full prompts and are not committed).
"""

import json
import os
import statistics
import sys

TASKS = ["02-median-bug", "03-lru", "06-plugin-conventions", "07-ttl-cache", "08-mutation-leak"]
ARMS = ["NONE", "FULL"]


def load(d: str, arm: str, task: str) -> dict | None:
    p = os.path.join(d, f"determinism-zakcode-pinworkspace-temp0-{task}.CAT-{arm}-{task}.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def main(d: str) -> None:
    rows: dict[tuple[str, str], dict] = {}
    for task in TASKS:
        for arm in ARMS:
            j = load(d, arm, task)
            if j is None:
                continue
            runs = j["runs"]
            digs = [json.dumps(r.get("digests"), sort_keys=True) for r in runs]
            rows[(task, arm)] = {
                "n": len(runs),
                "pass": sum(1 for r in runs if r.get("verify_rc") == 0),
                "no_report": sum(1 for r in runs if r.get("no_report")),
                "elapsed": [r.get("elapsed_s") for r in runs],
                "turns": [r.get("num_turns") for r in runs],
                "identical": len(set(digs)) == 1 and all(r.get("digests") for r in runs),
                "states": len(set(digs)),
            }
    head = f"{'task':22} {'arm':5} {'pass':>5} {'noRep':>5} {'elapsed_s':>24} {'turns':>12}"
    print(head + " identical")
    for task in TASKS:
        for arm in ARMS:
            r = rows.get((task, arm))
            if not r:
                print(f"{task:22} {arm:5}  (not yet)")
                continue
            left = f"{task:22} {arm:5} {r['pass']:>2}/{r['n']:<2} {r['no_report']:>5}"
            mid = f"{str(r['elapsed']):>24} {str(r['turns']):>12}"
            print(f"{left} {mid} {r['identical']} ({r['states']} states)")
    pooled: dict[str, dict] = {}
    for arm in ARMS:
        rs = [rows[k] for k in rows if k[1] == arm]
        if not rs:
            continue
        el = [e for r in rs for e in r["elapsed"] if e is not None]
        tu = [t for r in rs for t in r["turns"] if t is not None]
        pooled[arm] = {
            "runs": sum(r["n"] for r in rs),
            "pass": sum(r["pass"] for r in rs),
            "no_report": sum(r["no_report"] for r in rs),
            "med_elapsed": statistics.median(el) if el else None,
            "mean_turns": statistics.mean(tu) if tu else None,
            "identical_tasks": [k[0] for k in rows if k[1] == arm and rows[k]["identical"]],
        }
    print("\n--- pooled")
    for arm, p in pooled.items():
        print(
            f"{arm:5} pass {p['pass']}/{p['runs']}  no_report {p['no_report']}  "
            f"median elapsed {p['med_elapsed']}s  mean turns {p['mean_turns']:.2f}  "
            f"identical 3/3 on: {p['identical_tasks']}"
        )
    complete = set(pooled) == set(ARMS) and all(pooled[a]["runs"] == 15 for a in ARMS)
    if not complete:
        print("\n(verdict withheld: cells incomplete)")
        return
    print("\n--- VERDICT (pre-registered rules, catalogue-budget-preregistration.log 19:51)")
    dp = pooled["FULL"]["pass"] - pooled["NONE"]["pass"]
    if dp <= -3:
        print(
            f"R1: FULL - NONE = {dp:+d} passes -> the catalogue costs OUTCOMES; build the shortlist"
        )
    elif dp >= 3:
        print(
            f"R3: FULL - NONE = {dp:+d} passes -> surprising; record; no claim without replication."
        )
    else:
        print(f"R2: FULL - NONE = {dp:+d} passes -> no outcome effect at N=15; latency/cost only.")
    me_n, me_f = pooled["NONE"]["med_elapsed"], pooled["FULL"]["med_elapsed"]
    ratio = me_f / me_n if me_n else float("nan")
    verdict4 = "material" if ratio > 1.25 else "not material"
    print(f"R4: median elapsed FULL/NONE = {me_f}/{me_n} = {ratio:.2f} -> {verdict4} latency cost")
    kept = pooled["FULL"]["identical_tasks"]
    lost = [t for t in pooled["NONE"]["identical_tasks"] if t not in kept]
    print(f"R5: identity lost in FULL only on {lost or 'none'}")
    print("R6: use_skill / save_skill calls -> dump-derived, counted beside the dumps.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bench/results")
