#!/usr/bin/env python
"""Is the 85.2% 'conditional accuracy' biased upward? -- the confound named in
retrieval-lever-preregistration.log, tested with NO model calls.

THE WORRY, stated there before any of this was computed: conditional accuracy (arm accuracy
divided by arm recall) assumes the queries BM25 MISSES are no harder for the model than the ones
it HITS. BM25 plausibly misses exactly the queries whose wording is furthest from the skill
description -- which are plausibly also the ones the model gets wrong from the full catalogue. If
so, the misses were doomed anyway, dividing by recall credits the shortlist for queries it never
had to answer, and the 85.2% is inflated.

THE TEST. Join per-skill BM25 retrieval hits (computable locally) against per-skill FULL accuracy
(choosability-ni-matrix.json). If FULL scores the SAME on skills BM25 finds and skills BM25
misses, the misses are not systematically harder and the conditional number is not inflated by
this mechanism. If FULL scores WORSE on the missed skills, it is.

POPULATION CHECK FIRST: the two runs must cover the same skills and FULL must reproduce, or the
join is meaningless. Measured: 140/140 overlap, FULL 60.2% in both runs, 0.0 points apart.
"""
from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ctopk", BENCH / "choosability_topk.py")
ctopk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ctopk)

K = 20  # the pre-registered primary


def main() -> int:
    ni = json.load(open(BENCH / "results/choosability-ni-matrix.json"))["per_skill"]
    cat = ctopk.entries()
    qmap = {n: qs for n, _ in cat if len(qs := ctopk.queries_for(n)) == 3}

    overlap = set(ni) & set(qmap)
    ni_full = sum(v["FULL"] for v in ni.values()) / len(ni)
    print(f"population: {len(overlap)}/{len(qmap)} skills overlap; FULL {ni_full:.1%} (NI run) "
          f"vs {253/420:.1%} (TOPK run)")
    if len(overlap) != len(qmap) or abs(ni_full - 253 / 420) > 0.02:
        print("INSTRUMENT FAILURE: the two runs do not share a population. Refusing to join.")
        return 4

    bm = ctopk.BM25(cat)
    # Per-skill retrieval hit count at K: how many of this skill's 3 queries put it in the top K.
    hit_n, full_acc = {}, {}
    for name, qs in qmap.items():
        ranked_hits = 0
        for q in qs:
            ranked = sorted(bm.docs, key=lambda n: -bm.score(ctopk._toks(q), bm.docs[n]))[:K]
            ranked_hits += name in ranked
        hit_n[name] = ranked_hits
        full_acc[name] = ni[name]["FULL"]

    print(f"\n--- FULL's accuracy, split by how well BM25 retrieves that skill (K={K})")
    print(f"{'BM25 hits (of 3)':>18}  {'skills':>7}  {'FULL accuracy':>14}")
    buckets = {}
    for name, h in hit_n.items():
        buckets.setdefault(h, []).append(full_acc[name])
    for h in sorted(buckets):
        v = buckets[h]
        print(f"{h:>18}  {len(v):>7}  {statistics.mean(v):>13.1%}")

    found = [full_acc[n] for n, h in hit_n.items() if h == 3]
    missed = [full_acc[n] for n, h in hit_n.items() if h == 0]
    print(f"\n  always-retrieved skills (3/3): n={len(found):>3}  FULL {statistics.mean(found):.1%}")
    print(f"  never-retrieved skills  (0/3): n={len(missed):>3}  FULL {statistics.mean(missed):.1%}")
    gap = (statistics.mean(found) - statistics.mean(missed)) * 100
    print(f"  gap: {gap:+.1f} points")

    print("\n--- VERDICT")
    if gap > 10:
        print(f"  CONFOUND CONFIRMED. FULL scores {gap:.0f} points WORSE on the skills BM25 loses, so")
        print("  those queries were disproportionately going to be missed anyway. Dividing accuracy")
        print("  by recall credits the shortlist for questions it never had to answer: the 85.2%")
        print("  conditional figure is BIASED UPWARD and must not be used to project an arm score.")
    elif gap < -10:
        print(f"  INVERSE CONFOUND. FULL scores {-gap:.0f} points BETTER on the skills BM25 loses --")
        print("  the conditional figure is biased DOWNWARD and the shortlist is being undersold.")
    else:
        print(f"  NO MATERIAL CONFOUND from this mechanism ({gap:+.1f} points). The skills BM25 misses")
        print("  are not systematically harder for the model, so conditional accuracy is not")
        print("  inflated by the effect the pre-registration worried about. It remains an ARM-LEVEL")
        print("  aggregate, not a per-query result, and that limit stands.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
