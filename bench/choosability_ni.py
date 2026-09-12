#!/usr/bin/env python
"""Non-inferiority test: does a name+first-sentence catalogue lose measurable choosability?

The n=142 run (ADR-0155 third addendum) could not settle FULL vs FIRST: 11-4 discordant pairs,
p=0.1185, a superiority test that failed to reject. Failing to reject is not evidence of
equivalence, and the catalogue is the whole population, so no larger sample and no amount of
repetition exists -- measurement noise there was exactly zero.

The available axis is queries PER SKILL. Each skill yields several body paragraphs, so a skill can
carry a graded 0-K score per arm instead of a single 0/1 flip. That converts the 127 skills that
tied uninformatively into evidence.

Queries from one skill are NOT independent, so the unit of analysis is the SKILL: score each skill
0-K per arm, compare skill-level means, and bootstrap by RESAMPLING SKILLS. Resampling queries
would discard the clustering and inflate significance.

Pre-registration: bench/results/choosability-noninferiority-preregistration.log
Usage:  source bench/results/_podenv.sh && ./.venv/bin/python bench/choosability_ni.py [k]
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # noqa: E402 -- sibling module
from _frontmatter import assert_sane, read_description  # noqa: E402

import httpx

SKILLS = Path("/opt/ayoai-mind/.claude/skills")
MARGIN = 0.05  # pre-registered non-inferiority margin, fixed before data
VOID_BELOW = 0.25


def entries() -> list[tuple[str, str]]:
    out = []
    for sk in sorted(SKILLS.glob("*/SKILL.md")):
        head = sk.read_text(encoding="utf-8", errors="replace")[:6000]
        desc = read_description(head)
        if desc:
            out.append((sk.parent.name, desc))
    assert_sane(out)
    return out


def render(cat: list[tuple[str, str]], arm: str) -> str:
    lines = []
    for name, desc in cat:
        if arm == "NAMES":
            lines.append(f"- {name}")
        elif arm == "FIRST":
            # Hoisted out of the f-string: this venv is 3.11 and a backslash inside an f-string
            # expression is a SyntaxError there (PEP 701 relaxed it only in 3.12).
            first = re.split(r"(?<=[.!?])\s", desc, maxsplit=1)[0]
            lines.append(f"- {name}: {first}")
        else:
            lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def queries_for(name: str, k: int) -> list[str]:
    """Up to k queries from a skill's BODY -- same extraction rule as the n=142 run, only the
    COUNT changes, so the two runs stay comparable. Paragraph 0 is skipped: a skill's opening
    paragraph often paraphrases its own description, which would hand FULL the answer."""
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
            # TOP-LEVEL, never nested under extra_body -- that is an SDK kwarg those clients
            # flatten; a raw POST sends it verbatim and the chat template never sees it, leaving
            # thinking ON and the cap consumed by reasoning_content (guard-6562).
            "chat_template_kwargs": {"enable_thinking": False},
        },
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        timeout=300,
    )
    r.raise_for_status()
    ch = r.json()["choices"][0]
    content = (ch["message"].get("content") or "").strip().strip("`/ ")
    if not content:
        raise SystemExit(
            f"INSTRUMENT FAILURE: empty content (finish={ch.get('finish_reason')!r}, "
            f"reasoning={len(ch['message'].get('reasoning_content') or '')} ch). Fix the request "
            f"shape before reading any score."
        )
    return content


def boot_ci(per_skill: dict[str, dict[str, float]], a: str, b: str, iters: int = 20000):
    """Paired bootstrap over SKILLS -- resampling skills, not queries, is what preserves the
    clustering. Returns (point estimate, one-sided 95% upper bound) for a-minus-b."""
    names = list(per_skill)
    diffs = [per_skill[n][a] - per_skill[n][b] for n in names]
    point = sum(diffs) / len(diffs)
    rng = random.Random(20260912)
    reps = []
    for _ in range(iters):
        s = [diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))]
        reps.append(sum(s) / len(s))
    reps.sort()
    return point, reps[int(0.95 * len(reps))]


def main(argv: list[str]) -> int:
    k = int(argv[0]) if argv else 3
    cat = entries()
    qmap = {n: qs for n, _ in cat if len(qs := queries_for(n, k)) == k}
    total = sum(len(v) for v in qmap.values())
    print(f"catalogue={len(cat)} skills   skills with {k} usable queries={len(qmap)}   "
          f"queries={total}   requests={total * 3}")

    per_skill: dict[str, dict[str, float]] = {n: {} for n in qmap}
    raw: dict[str, dict[str, list[str]]] = {n: {} for n in qmap}
    with httpx.Client() as client:
        for arm in ("FULL", "FIRST", "NAMES"):
            rendered = render(cat, arm)
            hits = 0
            for name, qs in qmap.items():
                got = [ask(client, rendered, q) for q in qs]
                raw[name][arm] = got
                s = sum(1 for g in got if g == name)
                per_skill[name][arm] = s / k
                hits += s
            print(f"{arm:6} {hits}/{total} = {hits / total:.1%}   catalogue {len(rendered):,} chars")

    Path("bench/results/choosability-ni-matrix.json").write_text(
        json.dumps({"k": k, "per_skill": per_skill, "raw": raw}, indent=1), encoding="utf-8")

    full_acc = sum(v["FULL"] for v in per_skill.values()) / len(per_skill)
    print("\n--- verdict against the pre-registered rules ---")
    if full_acc < VOID_BELOW:
        print(f"VOID: FULL control at {full_acc:.1%} (< {VOID_BELOW:.0%}).")
        return 3
    print(f"control OK: FULL {full_acc:.1%} (VOID below {VOID_BELOW:.0%}).")
    for arm in ("FIRST", "NAMES"):
        point, upper = boot_ci(per_skill, "FULL", arm)
        print(f"\nFULL minus {arm}: {point:+.1%}   one-sided 95% upper bound {upper:+.1%}   "
              f"(paired bootstrap over {len(per_skill)} skills)")
        if arm == "FIRST":
            if upper < MARGIN:
                print(f"  NON-INFERIOR at the pre-registered {MARGIN:.0%} margin. The 66% saving "
                      f"is safe to build; the third addendum's 'cannot exclude a 10pp loss' "
                      f"caveat is DISCHARGED and must be updated in place.")
            else:
                print(f"  NOT DEMONSTRATED: the bound reaches {upper:.1%}, at or beyond the "
                      f"{MARGIN:.0%} margin. The recommendation stays unproven -- do not ship the "
                      f"shortened catalogue on the strength of this ADR.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
