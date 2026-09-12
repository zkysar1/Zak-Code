#!/usr/bin/env python
"""Does SIBLING DENSITY explain both halves of the retrieval confound? -- NO MODEL CALLS.

ADR-0155's fifth addendum recorded sibling density as the next hypothesis and said explicitly that
"a catalogue's family structure is measurable without any model call". This is that measurement.

THE CHAIN BEING TESTED. The confound (retrieval_confound.py) found that BM25 loses precisely the
queries the model gets wrong from the full catalogue -- a 53-point gradient. That is a CORRELATION
with no mechanism. Sibling density is a candidate mechanism that would produce exactly it: a skill
sitting in a large family of same-prefix siblings has near-duplicate descriptions, so BM25 cannot
separate it from its siblings (retrieval miss) AND the model cannot separate it either (selection
miss). One cause, both symptoms.

IF IT HOLDS, the campaign's object changes again: not catalogue size (falsified), not description
length (retracted), not retrieval quality alone (confounded) -- but family structure, which is a
property of how skills are NAMED and SCOPED, and is fixable by the catalogue's author rather than
by any amount of prompt or retriever engineering.

IF IT FAILS, sibling density joins size and length as a falsified driver, and the 53-point
gradient still has no mechanism -- which is worth knowing before more work is spent assuming one.
"""
from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

BENCH = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ctopk", BENCH / "choosability_topk.py")
ctopk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ctopk)

K = 20


def family(name: str) -> str:
    """First hyphen-separated token. 'access-aws-services' and 'access-email' are one family."""
    return name.split("-", 1)[0]


def main() -> int:
    ni = json.load(open(BENCH / "results/choosability-ni-matrix.json"))["per_skill"]
    cat = ctopk.entries()
    qmap = {n: qs for n, _ in cat if len(qs := ctopk.queries_for(n)) == 3}
    ni_full = sum(v["FULL"] for v in ni.values()) / len(ni)
    if len(set(ni) & set(qmap)) != len(qmap) or abs(ni_full - 253 / 420) > 0.02:
        print("INSTRUMENT FAILURE: populations differ. Refusing to join.")
        return 4

    fam_size = Counter(family(n) for n, _ in cat)
    bm = ctopk.BM25(cat)

    rows = []
    for name, qs in qmap.items():
        hits = sum(
            name in sorted(bm.docs, key=lambda n: -bm.score(ctopk._toks(q), bm.docs[n]))[:K]
            for q in qs
        )
        rows.append((name, fam_size[family(name)], hits / 3, ni[name]["FULL"]))

    print(f"{len(rows)} skills; {len(fam_size)} families; "
          f"largest family {max(fam_size.values())} members\n")
    print(f"{'family size':>12}  {'skills':>7}  {'BM25 recall@20':>15}  {'FULL accuracy':>14}")
    buckets: dict[str, list] = {}
    for _, fs, rec, acc in rows:
        key = "1 (no siblings)" if fs == 1 else "2" if fs == 2 else "3-4" if fs <= 4 else "5+"
        buckets.setdefault(key, []).append((rec, acc))
    order = ["1 (no siblings)", "2", "3-4", "5+"]
    for key in order:
        if key not in buckets:
            continue
        v = buckets[key]
        print(f"{key:>12}  {len(v):>7}  {statistics.mean(r for r, _ in v):>14.1%}  "
              f"{statistics.mean(a for _, a in v):>13.1%}")

    solo = buckets.get("1 (no siblings)", [])
    big = buckets.get("5+", [])
    print("\n--- VERDICT")
    if not solo or not big:
        print("  Not enough spread in family size to test the mechanism.")
        return 0
    d_rec = (statistics.mean(r for r, _ in solo) - statistics.mean(r for r, _ in big)) * 100
    d_acc = (statistics.mean(a for _, a in solo) - statistics.mean(a for _, a in big)) * 100
    print(f"  solo skills vs 5+-member families:  recall {d_rec:+.1f} points,  "
          f"FULL accuracy {d_acc:+.1f} points")
    # Two checks the extreme contrast cannot make, both of which fired on the first real run:
    #   (1) MONOTONICITY -- if middle buckets beat the solo bucket, "more siblings is worse" is
    #       false whatever the extremes say.
    #   (2) WITHIN-BUCKET HOMOGENEITY -- if the 5+ bucket is driven by a few families rather than
    #       by size, then SIZE is not the variable, and the largest family is the test case.
    mono = True
    prev = None
    for key in order:
        if key not in buckets:
            continue
        m = statistics.mean(a for _, a in buckets[key])
        if prev is not None and m > prev + 0.02:
            mono = False
        prev = m
    by_fam: dict[str, list] = {}
    for name, fs, _, acc in rows:
        if fs >= 5:
            by_fam.setdefault(family(name), []).append(acc)
    fam_means = {f: statistics.mean(v) for f, v in by_fam.items()}
    largest = max(by_fam, key=lambda f: len(by_fam[f])) if by_fam else None
    overall = statistics.mean(a for _, _, _, a in rows)
    spread = (max(fam_means.values()) - min(fam_means.values())) * 100 if fam_means else 0.0
    print(f"  monotonic across buckets: {mono}")
    if largest:
        print(f"  largest family ({largest}, {len(by_fam[largest])} scored): "
              f"{fam_means[largest]:.1%} vs overall {overall:.1%}")
        print(f"  spread ACROSS the 5+ families: {spread:.0f} points")

    if not mono or spread > 20:
        print("  MECHANISM NOT SUPPORTED BY SIZE, despite the extreme contrast above.")
        if not mono:
            print("    A middle bucket beats the solo bucket, so 'more siblings is worse' is false.")
        if spread > 20:
            print(f"    The 5+ bucket spreads {spread:.0f} points across its own families, and the")
            print("    LARGEST family is not the worst -- so what hurts is a property of particular")
            print("    families, not their size. Family SIZE joins catalogue size and description")
            print("    length as a falsified driver; the gradient still has no mechanism.")
    elif d_rec > 10 and d_acc > 10:
        print("  MECHANISM SUPPORTED on both halves: large families are harder to RETRIEVE and")
        print("  harder to CHOOSE, which is exactly the shape that produces the 53-point confound")
        print("  from a single cause. Family structure -- how skills are named and scoped -- becomes")
        print("  the object, and it is fixable by the catalogue's author.")
    elif d_acc > 10:
        print("  PARTIAL: family size predicts SELECTION difficulty but not retrieval difficulty.")
        print("  Sibling confusion is real for the model; it is not what BM25 is tripping on, so")
        print("  the two halves of the confound need different explanations.")
    elif d_rec > 10:
        print("  PARTIAL: family size predicts RETRIEVAL difficulty but not selection difficulty --")
        print("  the opposite of the sibling-confusion story, which was about the model.")
    else:
        print("  MECHANISM NOT SUPPORTED. Family size predicts neither half materially. Sibling")
        print("  density joins catalogue SIZE and description LENGTH as a falsified driver, and the")
        print("  53-point gradient still has no mechanism. Say so rather than assuming one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
