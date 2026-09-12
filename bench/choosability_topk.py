#!/usr/bin/env python
"""Does a RETRIEVED shortlist beat the full catalogue -- cheaper AND better?

Shortening descriptions was retracted (ADR-0155 fourth addendum: -8.8pp). But the 62% ceiling comes
from SIBLING CONFUSION, and a shortlist attacks that directly: choosing among 20 candidates removes
most of the right answer's near-identical siblings. So the prediction is better AND cheaper, where
first-sentences were cheaper and worse.

Pre-registration: bench/results/choosability-topk-preregistration.log
Usage:  source bench/results/_podenv.sh && ./.venv/bin/python bench/choosability_topk.py
"""
from __future__ import annotations

import json
import math
import os
import random
import time
import re
import sys
from collections import Counter
from pathlib import Path

import httpx

SKILLS = Path("/opt/ayoai-mind/.claude/skills")
VOID_BELOW = 0.45
KS = (10, 20, 40)
PRIMARY_K = 20


def entries() -> list[tuple[str, str]]:
    out = []
    for sk in sorted(SKILLS.glob("*/SKILL.md")):
        h = sk.read_text(encoding="utf-8", errors="replace")[:6000]
        m = re.search(r"^description:\s*(.+)$", h, re.M)
        if m:
            out.append((sk.parent.name, m.group(1).strip()))
    return out


def _toks(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2]


class BM25:
    """Retrieves over name+description ONLY -- exactly what the catalogue already carries. It never
    sees the answer, and the shortlist is rendered in the same format as the full catalogue, so the
    single variable between arms is WHICH skills appear."""

    def __init__(self, cat: list[tuple[str, str]]):
        self.docs = {n: _toks(n.replace("-", " ") + " " + d) for n, d in cat}
        self.df = Counter(w for t in self.docs.values() for w in set(t))
        self.N = len(self.docs)
        self.avgdl = sum(len(v) for v in self.docs.values()) / self.N

    def score(self, q: list[str], d: list[str], k1: float = 1.5, b: float = 0.75) -> float:
        tf = Counter(d)
        s = 0.0
        for w in q:
            if w not in tf:
                continue
            idf = math.log(1 + (self.N - self.df[w] + 0.5) / (self.df[w] + 0.5))
            s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / self.avgdl))
        return s

    def top(self, query: str, k: int) -> list[str]:
        q = _toks(query)
        return sorted(self.docs, key=lambda n: -self.score(q, self.docs[n]))[:k]


def render(pairs: list[tuple[str, str]]) -> str:
    return "\n".join(f"- {n}: {d}" for n, d in pairs)


def queries_for(name: str, k: int = 3) -> list[str]:
    txt = (SKILLS / name / "SKILL.md").read_text(encoding="utf-8", errors="replace")
    body = re.sub(r"^---.*?\n---\n", "", txt, flags=re.S)
    body = re.sub(r"^#.*$", "", body, flags=re.M)
    body = re.sub(r"`[^`]*`", "", body)
    paras = [p.strip() for p in body.split("\n\n") if len(p.strip()) > 200]
    out = []
    for chunk in paras[1 : 1 + k]:
        c = re.sub(r"\s+", " ", chunk)[:600]
        out.append(c.replace(name, "this") if name in c else c)
    return out


def ask(client: httpx.Client, catalogue: str, query: str) -> str:
    base = os.environ["OPENAI_BASE_URL"].rstrip("/")
    model = json.loads(os.environ["ZAKCODE_ZAKPICK_MODELS"])["deep_code"]["model"]
    r = client.post(
        f"{base}/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content":
                 "You choose exactly one skill from a catalogue. Reply with the skill name only, "
                 "nothing else.\n\nCatalogue:\n" + catalogue},
                {"role": "user", "content":
                 "Which single skill from the catalogue does this text belong to?\n\n" + query},
            ],
            "temperature": 0.0,
            "max_tokens": 64,
            "chat_template_kwargs": {"enable_thinking": False},  # top-level, never extra_body
        },
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        timeout=300,
    )
    r.raise_for_status()
    ch = r.json()["choices"][0]
    content = (ch["message"].get("content") or "").strip().strip("`/ ")
    if not content:
        raise SystemExit(
            f"INSTRUMENT FAILURE: empty content (finish={ch.get('finish_reason')!r}). "
            f"Fix the request shape before reading any score."
        )
    return content


def boot(per_skill, a, b, iters=40000):
    names = list(per_skill)
    diffs = [per_skill[n][a] - per_skill[n][b] for n in names]
    pt = sum(diffs) / len(diffs)
    rng = random.Random(20260912)
    reps = sorted(sum(s) / len(s) for s in
                  ([diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))]
                   for _ in range(iters)))
    return pt, reps[int(0.025 * len(reps))], reps[int(0.975 * len(reps))]


def main(argv: list[str]) -> int:
    cat = entries()
    bm = BM25(cat)
    desc = dict(cat)
    qmap = {n: qs for n, _ in cat if len(qs := queries_for(n)) == 3}
    total = sum(len(v) for v in qmap.values())
    arms = ["FULL"] + [f"TOPK{k}" for k in KS]
    print(f"{len(cat)} skills   {len(qmap)} with 3 queries   {total} queries   "
          f"{total * len(arms)} requests")

    full_cat = render(cat)
    _done = [0]
    per_skill = {n: {} for n in qmap}
    recall = {a: 0 for a in arms}
    chars = {a: [] for a in arms}
    with httpx.Client() as client:
        for arm in arms:
            hits = 0
            for name, qs in qmap.items():
                got = []
                for q in qs:
                    _done[0] += 1
                    if _done[0] % 25 == 0:
                        print(f"  .. {arm} {_done[0]} queries  {time.strftime('%H:%M:%S')}",
                              flush=True)
                    if arm == "FULL":
                        c = full_cat
                        recall[arm] += 1
                    else:
                        k = int(arm[4:])
                        short = bm.top(q, k)
                        recall[arm] += name in short
                        c = render([(n, desc[n]) for n in short])
                    chars[arm].append(len(c))
                    got.append(ask(client, c, q))
                s = sum(1 for g in got if g == name)
                per_skill[name][arm] = s / 3
                hits += s
            mc = sum(chars[arm]) / len(chars[arm])
            print(f"{arm:7} {hits}/{total} = {hits/total:5.1%}   recall@K {recall[arm]/total:5.1%}   "
                  f"mean catalogue {mc:8,.0f} ch ({mc/len(full_cat)*100:4.0f}% of FULL)")

    Path("bench/results/choosability-topk-matrix.json").write_text(
        json.dumps({"per_skill": per_skill,
                    "recall": {a: recall[a] / total for a in arms},
                    "mean_chars": {a: sum(chars[a]) / len(chars[a]) for a in arms}}, indent=1),
        encoding="utf-8")

    full_acc = sum(v["FULL"] for v in per_skill.values()) / len(per_skill)
    print("\n--- verdict against the pre-registered rules ---")
    if full_acc < VOID_BELOW:
        print(f"VOID: FULL control {full_acc:.1%} < {VOID_BELOW:.0%}.")
        return 3
    print(f"control OK: FULL {full_acc:.1%} (VOID below {VOID_BELOW:.0%}).")
    for k in KS:
        arm = f"TOPK{k}"
        pt, lo, hi = boot(per_skill, arm, "FULL")
        tag = "PRIMARY" if k == PRIMARY_K else "secondary"
        print(f"\n{arm} minus FULL [{tag}]: {pt:+.1%}  95% CI ({lo:+.1%}, {hi:+.1%})")
        if k == PRIMARY_K:
            if lo > 0:
                print("  TOPK BEATS the full catalogue -- better AND cheaper.")
            elif hi < 0:
                print("  TOPK is WORSE. The recall headroom was a ceiling, not an achievement.")
            else:
                print("  No accuracy gain demonstrated; the claim reduces to COST only, which must")
                print("  then clear the same 5-point non-inferiority margin as the retracted run.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
