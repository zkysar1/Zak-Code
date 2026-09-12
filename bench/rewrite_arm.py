#!/usr/bin/env python
"""The REWRITE arm: is description fidelity CAUSAL? Pre-registered in
bench/results/rewrite-arm-preregistration.log before any rewrite existed.

Three phases, each persisting its output before the next runs (a killed run keeps what finished):
  prepare  no model calls: pick the lowest-coverage tertile, build HOLDOUT bodies, count exclusions
  rewrite  one call per target -- the model writes a new description from the holdout body
  score    840 calls -- ORIG and REWRITE catalogues over all 420 queries; W0-W4 verdicts
  verdict  no model calls -- re-apply W0-W4 to the saved rewrite-scores.json

THE HOLDOUT IS THE DESIGN. Queries are paragraphs 2-4 of each skill's body (queries_for). The
rewriter is shown the skill name, the original description, and the body with those three
paragraphs REMOVED -- it never sees a query -- so a coverage gain comes from describing the skill
better, not from copying the test. Skills with fewer than 2 long paragraphs left are excluded and
counted, never silently kept.

Usage:  source bench/results/_podenv.sh &&
        ./.venv/bin/python bench/rewrite_arm.py prepare|rewrite|score
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import re
import statistics
import sys
from pathlib import Path

import httpx

BENCH = Path(__file__).resolve().parent
RES = BENCH / "results"
spec = importlib.util.spec_from_file_location("ctopk", BENCH / "choosability_topk.py")
ctopk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ctopk)

TARGETS_F = RES / "rewrite-targets.json"
REWRITES_F = RES / "rewrite-descriptions.json"
SCORES_F = RES / "rewrite-scores.json"
VOID_BELOW = 0.45
MIN_HOLDOUT_PARAS = 2
MARGIN = 0.05  # W3 non-inferiority margin, same as the retracted shortening run
PREDICTED_LIFT = 0.10  # W2 point-estimate floor, set inside the predicted direction
# W0 reference: the FULL control measured on the CURRENT catalogue instrument. It was 0.602 on
# the defective loader (ADR-0158, first addendum) and is 0.643 on the fixed one (third addendum);
# a control is a control within an instrument version, so this moves whenever the loader does.
W0_REFERENCE = 0.643


def coverage(name: str, desc: str, qs: list[str]) -> float:
    dtok = set(ctopk._toks(name.replace("-", " ") + " " + desc))
    vals = [
        len(set(ctopk._toks(q)) & dtok) / len(set(ctopk._toks(q))) for q in qs if ctopk._toks(q)
    ]
    return statistics.mean(vals) if vals else 0.0


def holdout_body(name: str) -> tuple[str, int]:
    """The body with the three query-source paragraphs removed. Mirrors queries_for's parsing
    exactly so the removed paragraphs are the SAME ones the queries came from."""
    path = ctopk.SKILLS / name / "SKILL.md"
    body = re.sub(
        r"^---.*?\n---\n", "", path.read_text(encoding="utf-8", errors="replace"), flags=re.S
    )
    body = re.sub(r"^#.*$", "", body, flags=re.M)
    body = re.sub(r"`[^`]*`", "", body)
    paras = [p.strip() for p in body.split("\n\n") if p.strip()]
    long_idx = [i for i, p in enumerate(paras) if len(p) > 200]
    query_idx = set(long_idx[1:4])  # queries_for: paras[1:1+k] of the >200-char paragraphs
    kept = [p for i, p in enumerate(paras) if i not in query_idx]
    remaining_long = len(long_idx) - len(query_idx)
    text = re.sub(r"\s+", " ", "\n\n".join(kept))[:3000]
    return text, remaining_long


def prepare() -> int:
    cat = ctopk.entries()
    desc = dict(cat)
    qmap = {n: qs for n, _ in cat if len(qs := ctopk.queries_for(n)) == 3}
    cov = {n: coverage(n, desc[n], qs) for n, qs in qmap.items()}
    ranked = sorted(cov, key=lambda n: cov[n])
    tertile = ranked[: len(ranked) // 3]
    targets, excluded = [], []
    for n in tertile:
        text, remaining = holdout_body(n)
        (targets if remaining >= MIN_HOLDOUT_PARAS else excluded).append(
            {
                "name": n,
                "coverage_orig": cov[n],
                "holdout_chars": len(text),
                "remaining_long_paras": remaining,
            }
        )
    TARGETS_F.write_text(
        json.dumps(
            {
                "targets": targets,
                "excluded": excluded,
                "tertile_size": len(tertile),
                "n_skills": len(qmap),
            },
            indent=1,
        )
    )
    print(
        f"{len(qmap)} skills; lowest-coverage tertile = {len(tertile)}; "
        f"targets with >= {MIN_HOLDOUT_PARAS} long paragraphs after holdout = {len(targets)}; "
        f"EXCLUDED = {len(excluded)}"
    )
    print(
        f"  target coverage: mean {statistics.mean(t['coverage_orig'] for t in targets):.3f}, "
        f"max {max(t['coverage_orig'] for t in targets):.3f}   (rest of catalogue: "
        f"{statistics.mean(cov[n] for n in ranked[len(ranked) // 3 :]):.3f})"
    )
    print(f"  holdout chars: median {statistics.median(t['holdout_chars'] for t in targets):.0f}")
    print(f"wrote {TARGETS_F}")
    return 0


def ask_rewrite(client: httpx.Client, name: str, desc: str, holdout: str) -> str:
    r = client.post(
        os.environ["OPENAI_BASE_URL"].rstrip("/") + "/chat/completions",
        json={
            "model": "zds-qwen3.6-35b",
            "messages": [
                {
                    "role": "system",
                    "content": "You write one-line descriptions for a catalogue of agent "
                    "skills. A good description says concretely what the skill DOES and WHEN "
                    "an agent should reach for it, in the words a user asking for it would "
                    "use. Reply with the description only -- no name, "
                    "no quotes, no preamble.",
                },
                {
                    "role": "user",
                    "content": f"Skill name: {name}\nCurrent description: {desc}\n\n"
                    f"Skill body (excerpt):\n{holdout}"
                    "\n\nWrite a 1-3 sentence description of what this skill does and "
                    "when to use it.",
                },
            ],
            "temperature": 0.0,
            "max_tokens": 220,
            "chat_template_kwargs": {"enable_thinking": False},  # top-level, never extra_body
        },
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        timeout=300,
    )
    r.raise_for_status()
    ch = r.json()["choices"][0]
    content = (ch["message"].get("content") or "").strip().strip('"` ')
    if not content:
        raise SystemExit(
            f"INSTRUMENT FAILURE: empty rewrite for a target (finish={ch.get('finish_reason')!r})."
        )
    return re.sub(r"\s+", " ", content)


def rewrite() -> int:
    t = json.loads(TARGETS_F.read_text())
    cat = ctopk.entries()
    desc = dict(cat)
    qmap = {n: qs for n, _ in cat if len(qs := ctopk.queries_for(n)) == 3}
    out = {}
    with httpx.Client() as client:
        for i, tg in enumerate(t["targets"], 1):
            n = tg["name"]
            text, _ = holdout_body(n)
            new = ask_rewrite(client, n, desc[n], text)
            out[n] = {
                "orig": desc[n],
                "new": new,
                "coverage_orig": coverage(n, desc[n], qmap[n]),
                "coverage_new": coverage(n, new, qmap[n]),
            }
            REWRITES_F.write_text(json.dumps(out, indent=1))  # persist every step
            if i % 10 == 0:
                print(f"  .. {i}/{len(t['targets'])} rewritten", flush=True)
    improved = sum(1 for v in out.values() if v["coverage_new"] > v["coverage_orig"])
    print(
        f"\nW1 mechanism check: coverage rose on {improved}/{len(out)} targets "
        f"(pre-registered floor 35)  mean "
        f"{statistics.mean(v['coverage_orig'] for v in out.values()):.3f} "
        f"-> {statistics.mean(v['coverage_new'] for v in out.values()):.3f}"
    )
    print(
        "W1",
        "HOLDS"
        if improved >= 35
        else "FAILS -- no verdict on causality; do not widen the rewriter's view",
    )
    return 0


def boot(per_skill: dict, a: str, b: str, names: list[str], iters: int = 40000):
    diffs = [per_skill[n][a] - per_skill[n][b] for n in names]
    pt = sum(diffs) / len(diffs)
    rng = random.Random(20260912)
    reps = sorted(
        sum(s) / len(s)
        for s in (
            [diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))] for _ in range(iters)
        )
    )
    return pt, reps[int(0.025 * len(reps))], reps[int(0.975 * len(reps))]


def score() -> int:
    rw = json.loads(REWRITES_F.read_text())
    cat = ctopk.entries()
    qmap = {n: qs for n, _ in cat if len(qs := ctopk.queries_for(n)) == 3}
    targets = [n for n in rw if n in qmap]
    untouched = [n for n in qmap if n not in rw]
    cats = {
        "ORIG": ctopk.render(cat),
        "REWRITE": ctopk.render([(n, rw[n]["new"] if n in rw else d) for n, d in cat]),
    }
    per_skill = {n: {} for n in qmap}
    picks = {arm: {} for arm in cats}
    done = 0
    with httpx.Client() as client:
        for arm, rendered in cats.items():
            hits = 0
            for n, qs in qmap.items():
                got = [ctopk.ask(client, rendered, q) for q in qs]
                picks[arm][n] = got
                s = sum(1 for g in got if g == n)
                per_skill[n][arm] = s / 3
                hits += s
                done += 3
                if done % 60 == 0:
                    print(f"  .. {arm} {done} queries", flush=True)
            print(f"{arm:8} {hits}/{len(qmap) * 3} = {hits / (len(qmap) * 3):.1%}")
            SCORES_F.write_text(json.dumps({"per_skill": per_skill, "picks": picks}, indent=1))

    return _verdict(per_skill, picks, targets, untouched)


def _verdict(
    per_skill: dict[str, dict[str, float]],
    picks: dict[str, dict[str, list[str]]],
    targets: list[str],
    untouched: list[str],
) -> int:
    """W0-W4 over scored arms. Shared by ``score`` (live) and ``verdict`` (saved scores)."""
    cats = ("ORIG", "REWRITE")
    full = statistics.mean(v["ORIG"] for v in per_skill.values())
    print("\n--- VERDICT (pre-registered rules)")
    if full < VOID_BELOW or abs(full - W0_REFERENCE) > 0.03:
        print(f"W0 FAILS: ORIG {full:.1%} not within 3 points of {W0_REFERENCE:.1%}. No verdict.")
        return 4
    print(f"W0 holds: ORIG {full:.1%}")
    pt, lo, hi = boot(per_skill, "REWRITE", "ORIG", targets)
    print(
        f"W2 targets (n={len(targets)}): REWRITE - ORIG = {pt * 100:+.1f} points, "
        f"95% CI ({lo * 100:+.1f}, {hi * 100:+.1f})  "
        f"-> {'HOLDS' if (pt >= PREDICTED_LIFT and lo > 0) else 'FAILS'}"
    )
    pt3, lo3, hi3 = boot(per_skill, "REWRITE", "ORIG", untouched)
    w3_text = "HOLDS" if lo3 > -MARGIN else "FAILS (untouched skills fell past the 5-point margin)"
    print(
        f"W3 untouched (n={len(untouched)}): {pt3 * 100:+.1f} points, "
        f"CI ({lo3 * 100:+.1f}, {hi3 * 100:+.1f})  "
        f"-> {w3_text}"
    )
    tset = set(targets)
    decoy = {arm: sum(1 for n in untouched for g in picks[arm][n] if g in tset) for arm in cats}
    print(
        f"W4 decoy picks (untouched query -> target skill): "
        f"ORIG {decoy['ORIG']}  REWRITE {decoy['REWRITE']}"
    )
    w2 = pt >= PREDICTED_LIFT and lo > 0
    w3 = lo3 > -MARGIN
    if w2 and w3:
        print(
            "\nFIDELITY IS CAUSAL. File the fleet-wide rewrite goal; "
            "coach replication is the next check."
        )
    elif not w2:
        print(
            "\nCOVERAGE ROSE AND ACCURACY DID NOT (if W1 held): fidelity is a MARKER, "
            "not a cause. Retire the lever."
        )
    else:
        print(
            "\nLIFT PAID FOR BY THE UNTOUCHED SKILLS: report the net over all 140; "
            "do not ship a rewrite that moves errors."
        )
    return 0


def verdict() -> int:
    """Re-apply W0-W4 to the SAVED scores -- no model calls.

    The 2026-09-12 19:02 score run printed W0 against the stale 60.2% constant and returned
    before W2-W4 (ADR-0158 fourth addendum). The scores it saved are complete, so the
    pre-registered rules are applied to them here, as written, instead of by hand in a log.
    """
    rw = json.loads(REWRITES_F.read_text())
    saved = json.loads(SCORES_F.read_text())
    per_skill, picks = saved["per_skill"], saved["picks"]
    targets = [n for n in rw if n in per_skill]
    untouched = [n for n in per_skill if n not in rw]
    print(
        f"verdict from {SCORES_F.name}: {len(per_skill)} skills, "
        f"{len(targets)} targets, {len(untouched)} untouched"
    )
    return _verdict(per_skill, picks, targets, untouched)


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    sys.exit({"prepare": prepare, "rewrite": rewrite, "score": score, "verdict": verdict}[phase]())
