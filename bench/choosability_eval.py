#!/usr/bin/env python
"""Can a model still pick the right skill when the catalogue is shortened?

ADR-0155's second addendum recommends rendering the skill catalogue as name + first sentence (66%
cheaper) and asserts that names alone are "probably too cryptic". The ADR flags that sentence as a
judgement rather than a measurement. This measures it, because a cheaper catalogue the model cannot
choose from is not a saving -- it is a capability regression paid for in tokens.

Three arms over the SAME 145-skill catalogue, differing only in how each entry is rendered:
    FULL   name + full description   (today)
    FIRST  name + first sentence     (the recommendation)
    NAMES  name only                 (the untested guess)

The query for each skill is drawn from its BODY, never its description. If the query came from the
description, FULL would win by construction and the experiment would measure string matching rather
than choosability. Front matter is stripped and the opening lines skipped, since a skill's first
paragraph often paraphrases its own description.

FULL is the instrument's positive control: if FULL scores poorly the query construction is broken
and no arm comparison means anything, so a low FULL VOIDS the run rather than flattering the cheap
arms.

Usage:  source bench/results/_podenv.sh && ./.venv/bin/python bench/choosability_eval.py [n]
"""
from __future__ import annotations

import json
import os
import re
import sys
from math import comb, sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # noqa: E402 -- sibling module
from _frontmatter import assert_sane, read_description  # noqa: E402

import httpx

SKILLS = Path("/opt/ayoai-mind/.claude/skills")


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
            first = re.split(r"(?<=[.!?])\s", desc, maxsplit=1)[0]
            lines.append(f"- {name}: {first}")
        else:
            lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def query_for(name: str) -> str | None:
    """A query built from the skill's BODY, with front matter and opening paragraph removed."""
    txt = (SKILLS / name / "SKILL.md").read_text(encoding="utf-8", errors="replace")
    body = re.sub(r"^---.*?\n---\n", "", txt, flags=re.S)
    body = re.sub(r"^#.*$", "", body, flags=re.M)          # drop headings (often the skill's name)
    body = re.sub(r"`[^`]*`", "", body)                     # drop code spans (paths, script names)
    paras = [p.strip() for p in body.split("\n\n") if len(p.strip()) > 200]
    if len(paras) < 2:
        return None
    chunk = paras[1] if len(paras) > 1 else paras[0]        # skip the opening paraphrase
    chunk = re.sub(r"\s+", " ", chunk)[:600]
    return chunk if name not in chunk else chunk.replace(name, "this")


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
            # TOP-LEVEL, not nested under `extra_body`. `extra_body` is a litellm/openai-SDK
            # kwarg that those clients FLATTEN into the JSON body; this is a raw httpx POST, so
            # sending it verbatim hands the server a body key named `extra_body` that it ignores
            # and the chat template never sees `enable_thinking`. The first run of this eval did
            # exactly that: thinking stayed ON, the 32-token cap was consumed entirely by
            # reasoning_content, every arm returned "" and scored 0/30. Measured side by side --
            # nested: content='' reasoning=125ch finish=length; top-level: content='reflect'
            # reasoning=0ch finish=stop.
            "chat_template_kwargs": {"enable_thinking": False},
        },
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        timeout=300,
    )
    r.raise_for_status()
    choice = r.json()["choices"][0]
    msg = choice["message"]
    content = (msg.get("content") or "").strip().strip("`/ ")
    # An empty answer is an INSTRUMENT failure, never a wrong choice. Scoring it as a miss is how
    # the first run turned its own broken request shape into a finding about the catalogue: a blank
    # reply is indistinguishable from "chose the wrong skill" at the scoreboard, and both arms and
    # control degrade together, so nothing looks anomalous. Fail loud instead.
    if not content:
        raise SystemExit(
            f"INSTRUMENT FAILURE: empty content (finish_reason={choice.get('finish_reason')!r}, "
            f"reasoning_content={len(msg.get('reasoning_content') or '')} chars). The model "
            f"answered into the reasoning channel or hit the cap; it did not choose wrongly. "
            f"Fix the request shape before reading any score."
        )
    return content


def _paired(per_arm: dict[str, dict[str, str]], a: str, b: str, qs: list[str]):
    """Paired exact two-sided sign test on discordant pairs, plus a McNemar CI on the difference.

    Every arm answers the SAME queries, so the pairing is the evidence: comparing marginal totals
    throws it away and cannot tell "two queries flipped" from "thirty flipped each way".
    """
    ab = sum(1 for q in qs if per_arm[a][q] == q and per_arm[b][q] != q)
    ba = sum(1 for q in qs if per_arm[b][q] == q and per_arm[a][q] != q)
    nd, n = ab + ba, len(qs)
    k = min(ab, ba)
    pv = min(1.0, 2 * sum(comb(nd, i) for i in range(k + 1)) / 2**nd) if nd else 1.0
    diff = (ab - ba) / n
    se = sqrt(ab + ba - (ab - ba) ** 2 / n) / n if nd else 0.0
    return ab, ba, nd, pv, diff, diff - 1.96 * se, diff + 1.96 * se


def main(argv: list[str]) -> int:
    cat = entries()
    if argv and argv[0] == "--all":
        # Whole catalogue, no sampling -- removes the frozen draw's composition caveat entirely.
        sample = [n for n, _ in cat]
    else:
        sample = json.loads(Path("bench/results/choosability-sample.json").read_text(encoding="utf-8"))
        if argv:
            sample = sample[: int(argv[0])]
    queries = {n: q for n in sample if (q := query_for(n))}
    print(f"catalogue={len(cat)} skills   sample={len(sample)}   usable queries={len(queries)}")
    skipped = [n for n in sample if n not in queries]
    if skipped:
        print(f"  skipped (body too short to build a query): {skipped}")
    scores: dict[str, int] = {}
    per_arm: dict[str, dict[str, str]] = {}
    sizes: dict[str, int] = {}
    with httpx.Client() as client:
        for arm in ("FULL", "FIRST", "NAMES"):
            rendered = render(cat, arm)
            hits = 0
            misses = []
            answers: dict[str, str] = {}
            for name, q in queries.items():
                got = ask(client, rendered, q)
                answers[name] = got
                if got == name:
                    hits += 1
                else:
                    misses.append((name, got[:40]))
            scores[arm] = hits
            per_arm[arm] = answers
            sizes[arm] = len(rendered)
            print(f"{arm:6} {hits}/{len(queries)} = {hits / len(queries) * 100:.0f}%   "
                  f"catalogue {len(rendered):,} chars")
            for want, got in misses[:4]:
                print(f"       miss: wanted {want!r} got {got!r}")

    # Persist the full per-query matrix. The first run printed only misses[:4], which discards the
    # PAIRING -- every arm answers the SAME 30 queries, so the right test for "is FIRST worse than
    # FULL" is the discordant-pair count (queries FULL got and FIRST missed, vs the reverse), not a
    # comparison of marginal totals. Marginal totals of 15 vs 13 are consistent with anything from
    # "two queries flipped" to "thirteen flipped each way"; only the matrix separates those.
    out = Path("bench/results/choosability-matrix.json")
    prior = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    prior.append({"scores": scores, "catalogue_chars": sizes, "answers": per_arm})
    out.write_text(json.dumps(prior, indent=1), encoding="utf-8")
    print(f"\nwrote {out} (run {len(prior)})")
    # Run-to-run movement at temperature 0 is the instrument's own noise floor. A margin smaller
    # than it is not a result. (Reproduction is batch-local, ADR-0152.) Compare ONLY against runs
    # over the SAME population: this file accumulates 30-query and 145-query runs, and differencing
    # their raw hit counts yields a large, well-formed, entirely meaningless "drift" -- a number
    # that looks like a measurement because both operands are real.
    same_n = [r for r in prior[:-1] if len(r["answers"]["FULL"]) == len(queries)]
    if same_n:
        base = same_n[0]["scores"]
        print(f"  score drift vs the first n={len(queries)} run: "
              f"{ {a: scores[a] - base[a] for a in scores} }")
    else:
        print(f"  (no prior n={len(queries)} run to compare against -- noise floor unmeasured here)")
    for a, b in (("FULL", "FIRST"), ("FULL", "NAMES")):
        ab = sum(1 for n in queries if per_arm[a][n] == n and per_arm[b][n] != n)
        ba = sum(1 for n in queries if per_arm[b][n] == n and per_arm[a][n] != n)
        print(f"  discordant pairs {a}>{b}: {ab}   {b}>{a}: {ba}   (net {ab - ba:+d} of {ab + ba} that differ)")
    print("\n--- verdict against the pre-registered rules ---")
    n = len(queries)
    void_at = 0.25 if (argv and argv[0] == "--all") else 0.5
    if scores["FULL"] / n < void_at:
        print(f"VOID: the FULL control scored {scores['FULL']}/{n} (< {void_at:.0%}). The query construction is the")
        print("suspect, not the catalogue shapes. No arm comparison is meaningful.")
        return 3
    # The pre-registered tolerance ("within a few points of FULL") was written against the frozen
    # 30-skill sample. It is an ABSOLUTE band, so the `[n]` smoke-test path silently rescales it
    # into nonsense: at n=3 a NAMES score of 0/3 is "-2" and passes, printing "97% saving is
    # available" about an arm that chose correctly zero times. Refuse to render the verdict on a
    # sample the rule was not registered against rather than quietly reinterpreting it.
    if n < 20:
        print(f"scores only (n={n}): FULL={scores['FULL']} FIRST={scores['FIRST']} NAMES={scores['NAMES']}.")
        print("NO VERDICT: the pre-registered tolerance is an absolute band written for the full")
        print("30-skill sample. Applying it to a truncated run would rescale the rule after the")
        print("fact -- exactly what pre-registration exists to prevent. Run the full sample.")
        return 4
    print(f"control OK: FULL={scores['FULL']}/{n} = {scores['FULL'] / n:.1%} (VOID below {void_at:.0%}).")
    # The decision rule is the PAIRED EXACT SIGN TEST on discordant pairs, not a comparison of
    # marginal totals against an absolute band. The band (`FIRST >= FULL - 2`) was registered for
    # the 30-skill sample and stayed in this code when `--all` was added: at n=142 it is a +/-1.4pp
    # equivalence bar that almost nothing can clear, and it returned "must be RETRACTED" on a
    # difference the registered test does not resolve (p=0.1185). A decision rule that silently
    # stops matching its own pre-registration reports exactly like one that still matches it.
    for arm in ("FIRST", "NAMES"):
        ab, ba, nd, pv, diff, lo, hi = _paired(per_arm, "FULL", arm, list(queries))
        if pv < 0.05:
            call = f"FULL is ahead: {arm} loses {diff:.1%} (95% CI {lo:.1%} to {hi:.1%})"
        else:
            # Failing to reject is NOT evidence of equivalence -- this is a superiority test, and a
            # non-inferiority design with a stated margin would be the right instrument. Report the
            # interval so a reader can see the largest loss this run could still have missed.
            call = (f"NOT separable at p<0.05; this run cannot exclude a {hi:.1%} loss for {arm} "
                    f"(point estimate {diff:+.1%}, 95% CI {lo:.1%} to {hi:.1%})")
        print(f"FULL vs {arm:5}: discordant {ab}-{ba} (n={nd})  exact two-sided sign p={pv:.4f}")
        print(f"               {call}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
