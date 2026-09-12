#!/usr/bin/env python3
"""Coach replication of the description-FIDELITY finding. Runs ON zc-03, stdlib only.

Pre-registered in bench/results/fidelity-coach-preregistration.log (11:24) before any coach
per-skill data existed. The prior coach FULL arm (58.3%) persisted only aggregates.

Coach's skill text never leaves the box: this prints correlations, tertile means and counts --
never a description, a query, or a skill name. The per-skill table is written to /root ON-BOX and
stays there. Extraction/rendering/asking are IMPORTED from the coach_choosability.py that actually
ran, so the arm is the same arm; coverage, BM25 and spearman are ported from
description_fidelity.py / choosability_topk.py with the same tokenizer.
"""
import json, math, re, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/root")
import coach_choosability as cc  # noqa: E402  -- the script that produced the 58.3% arm

K = 20
PRIOR_FULL = 0.583


def toks(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2]


class BM25:
    def __init__(self, cat):
        self.docs = {n: toks(n.replace("-", " ") + " " + d) for n, d, _ in cat}
        self.df = Counter(w for t in self.docs.values() for w in set(t))
        self.N = len(self.docs)
        self.avgdl = sum(len(v) for v in self.docs.values()) / self.N

    def score(self, q, d, k1=1.5, b=0.75):
        tf = Counter(d)
        s = 0.0
        for w in q:
            if w not in tf:
                continue
            idf = math.log(1 + (self.N - self.df[w] + 0.5) / (self.df[w] + 0.5))
            s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / self.avgdl))
        return s

    def top(self, query, k):
        q = toks(query)
        return sorted(self.docs, key=lambda n: -self.score(q, self.docs[n]))[:k]


def spearman(xs, ys):
    def ranks(v):
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
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def mean(v):
    return sum(v) / len(v) if v else 0.0


def main():
    env = cc._dotenv()
    base, key = env["ZAKCODE_API_BASE"], env["ZAKCODE_API_KEY"]
    model = "zds-qwen3.6-35b"
    try:
        cfg_model = json.loads(env["ZAKCODE_ZAKPICK_MODELS"])["deep_code"]["model"]
    except Exception:
        cfg_model = None
    if cfg_model and cfg_model != model:
        raise SystemExit("MODEL MISMATCH: config says %r, prior runs used %r." % (cfg_model, model))

    cat = cc.entries()
    desc = {n: d for n, d, _ in cat}
    qmap = {n: qs for n, _, p in cat if len(qs := cc.queries_for(p, n)) == 3}
    total = sum(len(v) for v in qmap.values())
    print("coach skills=%d  with 3 queries=%d  queries=%d  model=%s" % (len(cat), len(qmap), total, model))

    bm = BM25(cat)
    rendered = cc.render(cat, "FULL")
    per = {}
    hits = 0
    for i, (name, qs) in enumerate(qmap.items(), 1):
        dtok = set(toks(name.replace("-", " ") + " " + desc[name]))
        cov = mean([len(set(toks(q)) & dtok) / len(set(toks(q))) for q in qs if toks(q)])
        rec = sum(name in bm.top(q, K) for q in qs) / 3
        got = [cc.ask(base, key, model, rendered, q) for q in qs]
        s = sum(1 for g in got if g == name)
        hits += s
        per[name] = {"coverage": cov, "recall20": rec, "FULL": s / 3}
        if i % 16 == 0:
            print("  .. %d/%d skills" % (i, len(qmap)), flush=True)
    Path("/root/coach-fidelity-per-skill.json").write_text(json.dumps(per, indent=1))
    print("per-skill table written ON-BOX to /root/coach-fidelity-per-skill.json (never printed)")

    full = hits / total
    cov = [per[n]["coverage"] for n in per]
    rec = [per[n]["recall20"] for n in per]
    acc = [per[n]["FULL"] for n in per]
    r1, r2 = spearman(cov, rec), spearman(cov, acc)
    print("\nC0  FULL %d/%d = %.1f%%   (prior run 58.3%%, VOID below %.0f%%)" % (hits, total, full * 100, cc.VOID_BELOW * 100))
    print("C1  coverage vs BM25 recall@%d   rho = %+.3f   (sanity)" % (K, r1))
    print("C2  coverage vs FULL accuracy     rho = %+.3f   (the test)" % r2)
    paired = sorted(zip(cov, rec, acc))
    n = len(paired)
    print("\n%16s  %7s  %12s  %14s" % ("coverage tertile", "skills", "BM25 recall", "FULL accuracy"))
    thirds = {}
    for lab, lo, hi in (("lowest third", 0, n // 3), ("middle third", n // 3, 2 * n // 3), ("highest third", 2 * n // 3, n)):
        ch = paired[lo:hi]
        thirds[lab] = mean([a for _, _, a in ch])
        print("%16s  %7d  %11.1f%%  %13.1f%%" % (lab, len(ch), mean([r for _, r, _ in ch]) * 100, thirds[lab] * 100))
    spread = (thirds["highest third"] - thirds["lowest third"]) * 100
    print("C3  tertile spread = %+.1f points   (pre-registered floor 25)" % spread)
    print("C4  model less lexical than BM25: %s" % (r2 < r1))

    print("\n--- VERDICT (pre-registered rules)")
    if full < cc.VOID_BELOW or abs(full - PRIOR_FULL) > 0.05:
        print("C0 FAILS: FULL %.1f%% is not within 5 points of 58.3%% (or below VOID). No verdict." % (full * 100))
        return 4
    if r1 < 0.15:
        print("C1 FAILS (rho %+.3f): coverage port broken. No verdict on C2-C4." % r1)
        return 4
    if r2 >= 0.15 and spread >= 25:
        print("REPLICATES: rho %+.3f, spread %+.1f. Fidelity holds on both catalogues measured -- 'both', never 'generalizes'." % (r2, spread))
    elif r2 >= 0.15:
        print("DIRECTION REPLICATES, MAGNITUDE DOES NOT: rho %+.3f but spread %+.1f < 25. Report both." % (r2, spread))
    else:
        print("DOES NOT REPLICATE: rho %+.3f. Fidelity is SCOPED to ayoai-mind's catalogue; headline, not footnote." % r2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
