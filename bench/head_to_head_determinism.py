#!/usr/bin/env python3
"""H2H-DETERMINISM (head-to-head on the DETERMINISM axis, 2026-09-13): per-task byte-determinism
of Claude Code (as-ships) vs zakcode -- the axis the parity instrument (head_to_head.py) cannot see.

WHY THIS EXISTS. head_to_head.py reads 12/12 PARITY on correctness (pass/fail): the parity signal
has SATURATED and can no longer discriminate the two agents. The campaign's beyond-parity directive
is "make zakcode start to get better over Claude Code by making it MORE DETERMINISTIC"
(DETERMINISM-REVIEW.md, ADR-0162). Determinism is a SEPARATE axis from correctness -- two agents can
both pass a task while one reproduces byte-identical output across runs and the other does not. This
instrument measures that axis from the SAME result files head_to_head.py reads, so a saturated
parity verdict is no longer the end of the comparison.

METRIC. Per arm, per task: the number of DISTINCT output-tree states across the arm's N runs, where
a state is the sorted (path, sha256) set over agent-visible files (``.pytest_cache`` /
``__pycache__`` excluded). 1 distinct state across N runs => byte-DETERMINISTIC; >1 => NON-DET.

ASYMMETRIC VERDICT (honest by construction). Non-determinism is PROVEN by a single counterexample
(>=2 distinct states observed); determinism is only ever "not yet observed to vary" and is weak at
small N. A DET verdict at N<3 is flagged with a trailing ``~``. An arm whose runs captured no
agent-visible output is ``?empty`` and never counts as deterministic.

CONFIG ASYMMETRY -- READ THIS BEFORE QUOTING A VERDICT. The zakcode arms were sampled pinworkspace +
temperature 0 + stable prompt identity (ADR-0157): zakcode's DETERMINISM CONFIGURATION, which a user
gets with ``ZAKCODE_TEMPERATURE=0 ZAKCODE_STABLE_PROMPT_IDENTITY=1`` in a repo path. Claude
Code was sampled AS-SHIPS (``pin=None``; it offers no determinism mode). So ``ZAKCODE-WINS`` is the
honest product claim -- "zakcode reproduces where Claude Code as-ships does
not" -- NOT "same sampling, zakcode more deterministic". The comparison is not matched sampling and
does not pretend to be, and it is CONFOUNDED BY MODEL too: Claude Code runs a Claude model
(Fable 5.1 in the H2H), zakcode runs zds-qwen3.6-35b, and the pod cannot serve Claude (no API
here; see run_claude_code.py). The verdict is a PRODUCT-level claim about the two stacks as they
run -- NEVER that the zakcode LOOP is inherently more deterministic. Under ``--no-pin`` (fresh
mkdtemp workspace per run) zakcode is itself
non-deterministic (06 census 2026-09-13: 6/6 distinct) because the absolute workspace path sits in
the system prompt (the ``Workspace root (cwd)`` line, plus a workspace-derived cache key), and the
bench randomizes it each run. That is a BENCH artifact, not a defect: a real re-run at a FIXED
workspace path gets a byte-identical prompt, and zakcode is then byte-deterministic (pin1 / ARM-B /
ARMC-06 all 3/3 identical, modulo a small provider residual) -- why the claim is scoped as above.

Usage:
    ./.venv/bin/python bench/head_to_head_determinism.py bench/results
"""

import json
import os
import sys
from collections import defaultdict

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


def _sig(run: dict) -> tuple[tuple[str, str], ...]:
    """Canonical output-tree signature: sorted (path, digest) over agent-visible files."""
    dig = run.get("digests") or run.get("py_digests") or {}
    return tuple(
        sorted(
            (f, h)
            for f, h in dig.items()
            if ".pytest_cache" not in f and "__pycache__" not in f
        )
    )


def determinism(runs: list[dict]) -> dict:
    """distinct output states, N, the files that vary, and whether any run captured no output."""
    sigs = [_sig(r) for r in runs]
    perfile: dict[str, set] = defaultdict(set)
    present: dict[str, int] = defaultdict(int)
    for s in sigs:
        for f, h in s:
            perfile[f].add(h)
            present[f] += 1
    varying = sorted(f for f in perfile if len(perfile[f]) > 1 or present[f] < len(runs))
    return {
        "distinct": len(set(sigs)),
        "n": len(runs),
        "varying": varying,
        "empty_any": any(len(s) == 0 for s in sigs),
    }


def _label(d: dict | None) -> str:
    if d is None:
        return "-"
    if d["empty_any"] and d["distinct"] == 1:
        return f"?empty({d['n']})"
    if d["distinct"] == 1:
        return f"DET({d['n']})" + ("~" if d["n"] < 3 else "")
    return f"NONDET({d['distinct']}/{d['n']})"


def verdict(cc: dict | None, zak: dict | None) -> str:
    if cc is None or zak is None:
        return "(pending)"
    cc_det = cc["distinct"] == 1 and not cc["empty_any"]
    zak_det = zak["distinct"] == 1 and not zak["empty_any"]
    cc_nondet = cc["distinct"] > 1
    zak_nondet = zak["distinct"] > 1
    if zak_det and cc_nondet:
        return "ZAKCODE-WINS"
    if cc_det and zak_nondet:
        return "CC-WINS"
    if zak_det and cc_det:
        return "TIE-DET"
    if zak_nondet and cc_nondet:
        return "TIE-NONDET"
    return "INCONCLUSIVE"


def _zak_path(d: str, task: str, arm: str) -> str:
    return os.path.join(d, f"determinism-zakcode-pinworkspace-temp0-{task}.H2H-{arm}-{task}.json")


def main(d: str) -> None:
    print("HEAD-TO-HEAD DETERMINISM  (byte-identical output across runs)")
    print("config: zakcode = pinworkspace+temp0+stable-identity (its determinism mode);")
    print("        Claude Code = as-ships (no determinism mode). NOT matched sampling.")
    print(f"{'task':24} {'CC':>12} | {'35B':>12} {'verdict':13} | {'27B':>12} {'verdict':13}")
    tally: dict[str, int] = defaultdict(int)
    for task in TASKS:
        cc_j = _load(os.path.join(d, f"determinism-claude-code-asships-{task}.H2H-CC-{task}.json"))
        cc = determinism(cc_j["runs"]) if cc_j else None
        cols, detail = [], []
        for arm in ZAK:
            zj = _load(_zak_path(d, task, arm))
            z = determinism(zj["runs"]) if zj else None
            v = verdict(cc, z)
            if arm == "35B" and v != "(pending)":
                tally[v] += 1
            cols.append(f"{_label(z):>12} {v:13}")
            if z and z["varying"]:
                detail.append(f"      {arm} varies: {', '.join(z['varying'])}")
        print(f"{task:24} {_label(cc):>12} | {' | '.join(cols)}")
        if cc and cc["varying"]:
            print(f"      CC varies: {', '.join(cc['varying'])}")
        for line in detail:
            print(line)
    print()
    print("35B-vs-CC tally:", ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bench/results")
