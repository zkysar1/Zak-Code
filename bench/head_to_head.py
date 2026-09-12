#!/usr/bin/env python3
"""H2H (head-to-head, pre-registered 22:22 2026-09-12): per-task verdicts across three arms.

Usage:
    ./.venv/bin/python bench/head_to_head.py bench/results

Reads ``determinism-claude-code-asships-<task>.H2H-CC-<task>.json`` and
``determinism-zakcode-pinworkspace-temp0-<task>.H2H-<35B|27B>-<task>.json`` and prints, per task
and per zakcode arm, the pre-registered verdict: PARITY (CC >= 1/2 and zakcode >= 2/3), GAP
(CC >= 1/2 and zakcode <= 1/3), CEILING (CC 0/2), EDGE (CC 0/2 and zakcode >= 2/3). The task is
the replication unit; no pooled parity score is printed on purpose (rb-10837).
"""

import json
import os
import statistics
import sys

TASKS = [
    "m01-stale-doc-negative",
    "m02-ambiguous-zero",
    "m03-minimal-diff",
    "m04-assert-not-hedge",
    "m05-read-before-edit",
    "06-plugin-conventions",
]
ZAK = ["35B", "27B"]


def _load(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _reason(r: dict) -> str:
    """Verifier tail, plus the stop reason when the JSON carries one (determinism_arm.py
    stored the stop reason UNDER verify_out for zakcode runs before 2026-09-12)."""
    tail = (r.get("verify_out") or "")[:70]
    stop = r.get("stop_reason")
    return f"{tail} [stop={stop}]" if stop else tail


def _cell(runs: list[dict]) -> dict:
    passes = sum(1 for r in runs if r.get("verify_rc") == 0)
    reasons = sorted({_reason(r) for r in runs if r.get("verify_rc") != 0})
    return {
        "n": len(runs),
        "pass": passes,
        "no_report": sum(1 for r in runs if r.get("no_report")),
        "med_s": statistics.median(r.get("elapsed_s") or 0 for r in runs) if runs else None,
        "turns": [r.get("num_turns") for r in runs],
        "cost": round(sum(r.get("total_cost_usd") or 0 for r in runs), 3),
        "reasons": reasons,
    }


def verdict(cc: dict | None, zak: dict | None) -> str:
    if cc is None or zak is None:
        return "(pending)"
    if zak["no_report"]:
        return "REFUSED (no report)"
    cc_ok = cc["pass"] >= 1
    if cc_ok and zak["pass"] >= 2:
        return "PARITY"
    if cc_ok and zak["pass"] <= 1:
        return "GAP"
    if not cc_ok and zak["pass"] >= 2:
        return "EDGE"
    return "CEILING"


def main(d: str) -> None:
    print(
        f"{'task':24} {'CC pass':>8} {'CC $':>6} {'CC s':>6} | {'35B':>5} {'s':>6} verdict"
        f"        | {'27B':>5} {'s':>6} verdict"
    )
    for task in TASKS:
        cc_j = _load(os.path.join(d, f"determinism-claude-code-asships-{task}.H2H-CC-{task}.json"))
        cc = _cell(cc_j["runs"]) if cc_j else None
        cols = []
        for arm in ZAK:
            p = os.path.join(
                d, f"determinism-zakcode-pinworkspace-temp0-{task}.H2H-{arm}-{task}.json"
            )
            zj = _load(p)
            z = _cell(zj["runs"]) if zj else None
            v = verdict(cc, z)
            passes = f"{z['pass']}/{z['n']}" if z else "-"
            secs = f"{z['med_s']:.0f}" if z else "-"
            cols.append(f"{passes:>5} {secs:>6} {v:14}")
        if cc:
            cc_pass = f"{cc['pass']}/{cc['n']}"
            cc_txt = f"{cc_pass:>8} {cc['cost']:>6.2f} {cc['med_s']:>6.1f}"
        else:
            cc_txt = f"{'-':>8} {'-':>6} {'-':>6}"
        print(f"{task:24} {cc_txt} | {' | '.join(cols)}")
        for arm in ZAK:
            p = os.path.join(
                d, f"determinism-zakcode-pinworkspace-temp0-{task}.H2H-{arm}-{task}.json"
            )
            zj = _load(p)
            if zj:
                z = _cell(zj["runs"])
                for reason in z["reasons"]:
                    print(f"{'':24}   {arm} fail: {reason}")
        if cc:
            for reason in cc["reasons"]:
                print(f"{'':24}   CC fail: {reason}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bench/results")
