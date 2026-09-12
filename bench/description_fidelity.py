#!/usr/bin/env python
"""H-FIDELITY: does a skill's DESCRIPTION represent its own BODY? -- NO MODEL CALLS.

Pre-registered in bench/results/description-fidelity-preregistration.log at 11:16, before any
statistic here was computed.

WHY THIS CANDIDATE. A 53-point monotonic gradient links retrieval difficulty to selection
difficulty (retrieval_confound.py), and three drivers are falsified: catalogue SIZE, description
LENGTH, family SIZE. The gradient has a correlation and no mechanism. Queries in this benchmark are
extracted from each skill's BODY while the catalogue shows only its one-line DESCRIPTION -- so where
a description misrepresents its own skill, BM25 cannot match the query to it AND the model cannot
recognise it. One cause, both symptoms.

F1 is a SANITY CHECK, not evidence: BM25 *is* lexical overlap, so overlap must predict recall or the
measure is broken. F2 is the real test -- the model reads the full catalogue and can reason
semantically, so it need not depend on lexical overlap at all.
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

K = 20


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation -- the relationship need not be linear, and overlap is bounded."""
    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for t in range(i, j + 1):
                r[order[t]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def main() -> int:
    ni = json.load(open(BENCH / "results/choosability-ni-matrix.json"))["per_skill"]
    cat = ctopk.entries()
    desc = dict(cat)
    qmap = {n: qs for n, _ in cat if len(qs := ctopk.queries_for(n)) == 3}
    ni_full = sum(v["FULL"] for v in ni.values()) / len(ni)
    if len(set(ni) & set(qmap)) != len(qmap) or abs(ni_full - 253 / 420) > 0.02:
        print("INSTRUMENT FAILURE: populations differ. Refusing to join.")
        return 4

    bm = ctopk.BM25(cat)
    names, cover, recall, acc = [], [], [], []
    for name, qs in qmap.items():
        # Same tokenizer BM25 uses, so the retrieval half is measured on its own terms.
        dtok = set(ctopk._toks(name.replace("-", " ") + " " + desc[name]))
        covs = []
        for q in qs:
            qtok = set(ctopk._toks(q))
            covs.append(len(qtok & dtok) / len(qtok) if qtok else 0.0)
        hits = sum(
            name in sorted(bm.docs, key=lambda n: -bm.score(ctopk._toks(q), bm.docs[n]))[:K]
            for q in qs
        )
        names.append(name)
        cover.append(statistics.mean(covs))
        recall.append(hits / 3)
        acc.append(ni[name]["FULL"])

    f1 = spearman(cover, recall)
    f2 = spearman(cover, acc)
    print(f"{len(names)} skills; description-query token coverage vs outcomes\n")
    print(f"  F1  coverage vs BM25 recall@{K}   spearman rho = {f1:+.3f}   (SANITY CHECK)")
    print(f"  F2  coverage vs FULL accuracy     spearman rho = {f2:+.3f}   (THE REAL TEST)")

    print(f"\n{'coverage decile':>16}  {'skills':>7}  {'BM25 recall':>12}  {'FULL accuracy':>14}")
    paired = sorted(zip(cover, recall, acc))
    n = len(paired)
    for lab, lo, hi in (("lowest third", 0, n // 3), ("middle third", n // 3, 2 * n // 3),
                        ("highest third", 2 * n // 3, n)):
        chunk = paired[lo:hi]
        print(f"{lab:>16}  {len(chunk):>7}  {statistics.mean(r for _, r, _ in chunk):>11.1%}  "
              f"{statistics.mean(a for _, _, a in chunk):>13.1%}")

    print("\n--- VERDICT (against the pre-registered rules)")
    if f1 < 0.15:
        print(f"  F1 FAILS (rho {f1:+.3f}). BM25 is lexical overlap, so overlap MUST predict its")
        print("  recall. The measure is broken; no verdict on F2 or anything else.")
        return 4
    print(f"  F1 holds (rho {f1:+.3f}) -- the coverage measure behaves as it must.")
    if f2 >= 0.15:
        print(f"  F2 HOLDS (rho {f2:+.3f}): H-FIDELITY SUPPORTED. Description CONTENT predicts")
        print("  full-catalogue accuracy even though the model sees every skill and could reason")
        print("  semantically. The object becomes whether a description says what the skill does in")
        print("  the words a user would use -- a catalogue-authoring action, and the first")
        print("  actionable driver this arc has found. Next step is a rewrite arm on the worst decile.")
    else:
        print(f"  F2 FAILS (rho {f2:+.3f}): H-FIDELITY REJECTED for the SELECTION half. Lexical")
        print("  fidelity explains retrieval and NOT the model's failures, so the 53-point gradient")
        print("  still has no single mechanism. Recording that rather than reaching for a fourth")
        print("  candidate in the same breath.")
    if f2 >= f1:
        print(f"\n  F3 FAILS: the model's dependence on lexical overlap (rho {f2:+.3f}) is not weaker")
        print(f"  than bag-of-words BM25's ({f1:+.3f}). On this task the 35b model is doing no better")
        print("  than lexical matching -- a finding about the model worth more than the fidelity one.")
    else:
        print(f"\n  F3 holds: the model ({f2:+.3f}) depends on lexical overlap LESS than BM25 ({f1:+.3f}),")
        print("  so it is doing something beyond bag-of-words -- just not enough to be independent of it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
