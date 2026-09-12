#!/usr/bin/env python
"""R4 from retrieval-lever-preregistration.log: is the BM25 retriever trivially improvable?

NO MODEL CALLS. recall@K is a property of the retriever and the query set alone, so this runs in
seconds against zero pod load -- which is exactly why the pre-registration names it the first test
to run and the cheapest way to falsify H-RETRIEVAL early.

PRE-REGISTERED RULE (R4): a cheap tokenizer change lifts recall@20 by >= 3 points over the
measured 72.9% baseline. Below 3 points, the cheap route is shut and any further retrieval work
needs a real embedding retriever.

The BASELINE arm here must reproduce the pre-registration's measured ceilings
(@10 68.3 / @20 72.9 / @40 81.0) or this script is not measuring the same thing the live run did,
and no variant number from it means anything. That check is the first thing printed.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ctopk", BENCH / "choosability_topk.py")
ctopk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ctopk)

KS = (10, 20, 40)
# The pre-registration's measured ceilings -- the reproduction target for the baseline arm.
REGISTERED = {10: 0.683, 20: 0.729, 40: 0.810}


def tok_baseline(s: str) -> list[str]:
    """The shipped tokenizer: alphanumeric runs, drop anything <= 2 chars."""
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2]


def tok_keep2(s: str) -> list[str]:
    """Keep 2-character tokens. 'ui', 'db', 'ci', 'qa', 'id' are content words in skill names."""
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 1]


_SUF = ("ing", "ers", "er", "ies", "es", "s")


def _stem(w: str) -> str:
    for suf in _SUF:
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def tok_stem(s: str) -> list[str]:
    """Suffix-strip so 'deploying' matches 'deploy' and 'queries' matches 'query'."""
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 1]


def tok_stem_bigram(s: str) -> list[str]:
    """Stemming plus adjacent bigrams -- gives the retriever a little word-order signal."""
    ws = tok_stem(s)
    return ws + [f"{a}_{b}" for a, b in zip(ws, ws[1:])]


VARIANTS = {
    "baseline (shipped)": tok_baseline,
    "keep 2-char tokens": tok_keep2,
    "stemmed": tok_stem,
    "stemmed + bigrams": tok_stem_bigram,
}


def recall_at(tok, cat, qmap) -> dict[int, float]:
    ctopk._toks = tok  # BM25 reads the module-level tokenizer for both docs and queries
    bm = ctopk.BM25(cat)
    hits = dict.fromkeys(KS, 0)
    total = 0
    for name, qs in qmap.items():
        for q in qs:
            total += 1
            ranked = sorted(bm.docs, key=lambda n: -bm.score(tok(q), bm.docs[n]))
            for k in KS:
                hits[k] += name in ranked[:k]
    return {k: hits[k] / total for k in KS}


def main() -> int:
    cat = ctopk.entries()
    qmap = {n: qs for n, _ in cat if len(qs := ctopk.queries_for(n)) == 3}
    total = sum(len(v) for v in qmap.values())
    print(f"{len(cat)} skills, {len(qmap)} with 3 queries, {total} queries -- no model calls\n")

    rows = {}
    for label, tok in VARIANTS.items():
        rows[label] = recall_at(tok, cat, qmap)
        r = rows[label]
        print(f"  {label:22} recall@10 {r[10]:6.1%}  @20 {r[20]:6.1%}  @40 {r[40]:6.1%}")

    base = rows["baseline (shipped)"]
    print("\n--- REPRODUCTION CHECK (baseline must match the pre-registered ceilings)")
    ok = True
    for k, want in REGISTERED.items():
        got = base[k]
        good = abs(got - want) <= 0.01
        ok &= good
        print(f"  recall@{k:<3} registered {want:6.1%}  measured {got:6.1%}  "
              f"{'match' if good else 'MISMATCH'}")
    if not ok:
        print("\nINSTRUMENT FAILURE: the baseline does not reproduce the pre-registered ceilings, so "
              "this script is not measuring what the live run measured. Every variant number above "
              "is void. Refusing to render an R4 verdict.")
        return 4

    print("\n--- R4 VERDICT (pre-registered: a cheap tokenizer change lifts recall@20 by >= 3 points)")
    best_label = max((lbl for lbl in rows if lbl != "baseline (shipped)"), key=lambda x: rows[x][20])
    gain = (rows[best_label][20] - base[20]) * 100
    print(f"  best cheap variant: {best_label}  recall@20 {rows[best_label][20]:.1%}  "
          f"gain {gain:+.1f} points")
    if gain >= 3.0:
        print("  R4 HOLDS -- the retriever is trivially improvable. The cheap route to H-RETRIEVAL "
              "is open; re-run the TOPK arm with this tokenizer before building anything larger.")
    else:
        print("  R4 FAILS -- the cheap route is SHUT. Tokenizer tweaks do not move recall "
              "materially, so any further retrieval work needs a real embedding retriever, which "
              "is a build rather than a tweak. H-RETRIEVAL survives; its cheap test does not.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
