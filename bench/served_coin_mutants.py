"""Mutation proof of served sample 7's two offline instruments: does each check earn its place?

usage: python bench/served_coin_mutants.py reader [--work DIR]
       python bench/served_coin_mutants.py build <K-src-dir> --base DIR [--work DIR]

READER. Each mutant is a COPY of `served_coin.py` (with `served_stops.py` beside it) changed in
one place and run BY ITS OWN PATH, so the file that was changed is the file that ran.

BUILD. Each mutant is a COPY of build K's `src` tree (`served_builds/k.patch` on main) with one
change to the draw, and `served_coin_build_proof.py` runs against it through PYTHONPATH. The
proof prints which `zakcode` it imported; a run that did not import its own copy proves nothing
and is not counted. `--base` is where the proof puts its workspaces: OUTSIDE any repository, as
the veto-door bench requires.

A mutant counts as killed only by a NAMED failing check: a crash is a hole in the instrument,
not a catch. The unchanged copy must pass. Exit 0 only if every mutant is killed that way.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BENCH = Path(__file__).resolve().parent

#: name -> [(old, new) or (old, new, count)]; every `old` must occur exactly `count` times.
READER: dict[str, list[tuple]] = {
    "ladder-stop-counted-as-the-models": [
        (
            'answer = "harness_stop_refused" if tools else "words_refused"',
            'answer = "words_refused"',
        )
    ],
    "draw-read-after-the-request": [("for e in events[start:i]", "for e in events[i + 1 : end]")],
    "pairs-keyed-without-the-turn": [
        (
            'by_pair.setdefault((e["turn"], int(e["draw"]["pair"])), {})',
            'by_pair.setdefault(("", int(e["draw"]["pair"])), {})',
        )
    ],
    "witness-check-off": [
        (
            "violations = [w for w in map(_witness, drawn) if w] + _pair_faults(drawn)",
            "violations = _pair_faults(drawn)",
        )
    ],
    "pair-faults-off": [
        (
            "violations = [w for w in map(_witness, drawn) if w] + _pair_faults(drawn)",
            "violations = [w for w in map(_witness, drawn) if w]",
        )
    ],
    "ungoverned-draw-allowed": [('    if not episode["governed"]:\n', "    if False:\n")],
    "one-sided-sign-test": [("return hits / 2**k", "return hits / 2 ** (k + 1)")],
    "gain-ratio-strict": [
        ('if h <= RULE["gain_ratio"] * s and', 'if h < RULE["gain_ratio"] * s and')
    ],
    "flat-includes-20-points": [
        ('if abs(s - h) < RULE["points"]:', 'if abs(s - h) <= RULE["points"]:')
    ],
    "p-at-the-line-passes": [('p < RULE["p"]', 'p <= RULE["p"]', 2)],
    "leave-one-out-off": [
        (
            '    if rests_on:\n        whole["verdict"] = "MIXED"',
            '    if False:\n        whole["verdict"] = "MIXED"',
        )
    ],
    "leave-one-out-only-for-gain": [
        ('if whole["verdict"] in ("GAIN", "HARM"):', 'if whole["verdict"] in ("GAIN",):')
    ],
    "min-pairs-ignored": [('if len(pairs) < RULE["min_pairs"]:', "if len(pairs) < 0:")],
    "floor-ignored": [('if sent_share < RULE["floor"] and', "if sent_share < 0 and")],
    "floor-hides-a-harm": [
        (
            'if sent_share < RULE["floor"] and whole["verdict"] != "HARM":',
            'if sent_share < RULE["floor"]:',
        )
    ],
    "thin-run-kept": [
        (
            'elif len(w["complete_pairs"]) < RULE["min_pairs_a_run"]:',
            'elif len(w["complete_pairs"]) < 0:',
        )
    ],
    "contradicted-run-kept": [('elif w["witness_violations"]:', "elif False:")],
    "episodes-never-close": [
        (
            '            elif not finished and row["state"] != "rested":\n'
            "                current = None",
            '            elif not finished and row["state"] != "rested":\n                pass',
        )
    ],
    "a-rest-closes-the-episode": [
        ('elif not finished and row["state"] != "rested":', "elif not finished:")
    ],
    "governed-means-the-last-request-only": [("governed = governed or door", "governed = door")],
    "stop-index-off-by-one": [("own[0] + 1 if own else None", "own[0] if own else None")],
    "sides-swapped": [
        (
            's = Fraction(sum(p["sent_stopped"] for p in pairs), n)\n'
            '    h = Fraction(sum(p["silent_stopped"] for p in pairs), n)',
            's = Fraction(sum(p["silent_stopped"] for p in pairs), n)\n'
            '    h = Fraction(sum(p["sent_stopped"] for p in pairs), n)',
        )
    ],
    "discordant-counts-swapped": [
        (
            'plus = sum(p["sent_stopped"] and not p["silent_stopped"] for p in pairs)\n'
            '    minus = sum(p["silent_stopped"] and not p["sent_stopped"] for p in pairs)',
            'minus = sum(p["sent_stopped"] and not p["silent_stopped"] for p in pairs)\n'
            '    plus = sum(p["silent_stopped"] and not p["sent_stopped"] for p in pairs)',
        )
    ],
    "turn-ending-words-not-a-stop": [
        ('OWN_STOPS = ("words_refused", "turn_end")', 'OWN_STOPS = ("words_refused",)')
    ],
    "old-trace-accepted": [
        ('if usage and not any("rails" in (e.get("data") or {}) for e in usage):', "if False:")
    ],
    "unpaired-miscounted": [
        (
            '"unpaired_draws": len(drawn) - 2 * len(pairs),',
            '"unpaired_draws": len(drawn) - len(pairs),',
        )
    ],
}

_RESET = (
    "        # Build K: no pair spans two turns, so a new turn owes no side and numbers from one.\n"
    "        self._coin_open = False\n"
    "        self._coin_owed = None\n"
    "        self._coin_pair = 0\n"
)
BUILD: dict[str, list[tuple]] = {
    "no-turn-reset": [(_RESET, "", 2)],
    "draw-every-request": [
        ("            self._coin_open = True\n", "            self._coin_open = False\n")
    ],
    "second-slot-same-side": [
        (
            "slot, self._coin_silent, self._coin_owed = 2, self._coin_owed, None",
            "slot, self._coin_silent, self._coin_owed = 2, not self._coin_owed, None",
        )
    ],
    "scope-ignored": [
        (
            "        if not (self._hook_governs_turn_end and "
            "self.session.task_network.is_complete()):\n",
            "        if not self.session.task_network.is_complete():\n",
        )
    ],
    "draw-never-released": [
        (
            "            self._coin_open = False  # build K: that plan moved on; "
            "the next to finish draws anew\n",
            "            pass\n",
        )
    ],
    "always-sent": [("        return self._coin_silent\n", "        return False\n")],
    "coin-ignored": [("self._coin.random() < 0.5", "self._coin.random() < 2")],
}


def _mutate(path: Path, edits: list[tuple]) -> None:
    text = path.read_text()
    for edit in edits:
        old, new, count = (*edit, 1)[:3]
        found = text.count(old)
        if found != count:
            raise SystemExit(
                f"anchor found {found} times, not {count}, in {path.name}: {old[:70]!r}"
            )
        text = text.replace(old, new)
    path.write_text(text)


def _run(argv: list[str], extra_env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), **extra_env}
    return subprocess.run(argv, capture_output=True, text=True, timeout=1200, env=env, check=False)


def _report(name: str, done: subprocess.CompletedProcess[str], ran_own_copy: bool) -> bool:
    """One line per mutant; True if it died by a named check in its own copy (control: passed)."""
    fails = re.findall(r"^\s*FAIL (.+?)(?: -- |$)", done.stdout, re.M)
    crashed = "Traceback" in (done.stderr or "")
    print(
        f"{name:40s} rc={done.returncode} own_copy={ran_own_copy} "
        f"failed_checks={len(fails)} crashed={crashed}"
    )
    for fail in fails[:2]:
        print(f"      killed by: {fail[:100]}")
    if name == "control":
        return done.returncode == 0 and ran_own_copy and not fails and not crashed
    return done.returncode == 1 and ran_own_copy and bool(fails) and not crashed


def reader(work: Path) -> bool:
    good = True
    for name, edits in {"control": [], **READER}.items():
        box = work / name
        box.mkdir(parents=True)
        for file in ("served_coin.py", "served_stops.py"):
            shutil.copy(BENCH / file, box / file)
        _mutate(box / "served_coin.py", edits)
        done = _run([sys.executable, str(box / "served_coin.py"), "--selftest"], {})
        good &= _report(name, done, "selftest:" in done.stdout)  # run by path: its own copy
    print(f"reader: {len(READER)} mutants")
    return good


def build(work: Path, k_src: Path, base: Path) -> bool:
    good = True
    for name, edits in {"control": [], **BUILD}.items():
        tree = work / name / "src"
        shutil.copytree(k_src, tree)
        _mutate(tree / "zakcode" / "agent" / "loop.py", edits)
        done = _run(
            [sys.executable, str(BENCH / "served_coin_build_proof.py"), "--base", str(base)],
            {"PYTHONPATH": str(tree)},
        )
        own = f"'{tree}/zakcode/__init__.py'" in done.stdout
        good &= _report(name, done, own)
    print(f"build: {len(BUILD)} mutants")
    return good


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("what", choices=("reader", "build"))
    parser.add_argument("k_src", nargs="?", help="build: K's src directory")
    parser.add_argument("--base", help="build: where the proof puts workspaces (outside any repo)")
    parser.add_argument("--work", help="where the mutant copies go (default: a temp directory)")
    args = parser.parse_args()
    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="coin-mutants-"))
    work = work / args.what
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    if args.what == "reader":
        good = reader(work)
    else:
        if not args.k_src or not args.base:
            parser.error("build needs <K-src-dir> and --base")
        good = build(work, Path(args.k_src).resolve(), Path(args.base))
    print(
        "MUTATION PROOF:",
        "every mutant killed by a named check; the control passes" if good else "NOT PROVEN",
    )
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
