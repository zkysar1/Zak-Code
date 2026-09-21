"""Mutation proof of `served_ladder.py`: does each of its known-answer checks earn its place?

usage: python bench/served_ladder_mutants.py [--work DIR]

Each mutant is a COPY of `served_ladder.py` changed in one place and run BY ITS OWN PATH, so the
file that was changed is the file that ran. It imports the product's tracker from wherever the
unchanged reader does; the product is not what is mutated here.

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

_LAP = 'lap = bool(it["refused_stops"]) or own_load'
_OWN = "own_load = _body_delivered(it, only=loop_skill)"
_SHIPPED = '{"lap": laps, "work": work} if rule == "shipped"'  # the replay of the product's count
_WRITTEN = '{"lap": lap, "work": work} if product == "shipped"'  # ...and the synthetic product's
_EPOCH_LOOP = (
    '            for call in it["calls"]:'
    "  # the loop hands the tracker the epoch AFTER the batch ran\n"
)

#: name -> [(old, new)]; every `old` must occur exactly once.
MUTANTS: dict[str, list[tuple[str, str]]] = {
    # ── the replay ──
    "one-tracker-for-every-turn": [
        (
            "    for iterations in turns:\n"
            "        tracker = StuckTracker(uncounted_outcome_tools=UNCOUNTED)"
            "  # one per turn, as the loop\n",
            "    tracker = StuckTracker(uncounted_outcome_tools=UNCOUNTED)\n"
            "    for iterations in turns:\n",
        )
    ],
    "the-tracker-is-never-reset": [('if it["wordy_completions_before"] or stopped:', "if False:")],
    "a-refused-stop-rung-resets-nothing": [
        ('if it["wordy_completions_before"] or stopped:', 'if it["wordy_completions_before"]:')
    ],
    "a-file-edit-opens-no-epoch": [
        ("                    epoch += 1\n", "                    epoch += 0\n")
    ],
    "the-epoch-is-read-before-the-batch": [
        (_EPOCH_LOOP, "            for call in []:\n"),
        (
            "            action = tracker.next_action()\n            narrowed = ",
            "            action = tracker.next_action()\n"
            '            for call in it["calls"]:\n'
            "                result = by_id.get(call.id)\n"
            "                if call.name in EDIT_TOOLS and result is not None"
            " and not result.is_error:\n"
            "                    epoch += 1\n"
            "            narrowed = ",
        ),
    ],
    "read-only-rung-forgotten": [
        ("narrowed = action is StuckAction.NARROW", "narrowed = action is StuckAction.STOP")
    ],
    "window-never-forgets": [("while len(recent) > window:", "while len(recent) > window * 1000:")],
    "several-sessions-are-read-as-one": [
        ("    if len(files) != 1:  #", "    if not files:  #"),
        ("    read = read_transcript(files[0])\n", "    read = read_transcript(files[-1])\n"),
    ],
    "turn-split-ignored": [("if sizes and sum(sizes) == calls:", "if sizes and sum(sizes) == -1:")],
    # ── what a lap is ──
    "a-pointer-counts-as-a-body": [
        ("if result.output.lstrip().startswith(POINTER_HEAD):", "if False:")
    ],
    "a-body-of-any-skill-is-a-lap": [(_OWN, "own_load = _body_delivered(it)")],
    "an-unnamed-loop-matches-any-skill": [
        (
            "if only is None or (only and _skill_asked(call) == only):",
            "if only is None or not only or _skill_asked(call) == only:",
        )
    ],
    "refused-stops-are-no-lap": [(_LAP, "lap = own_load")],
    "the-loop-skill-is-never-learned": [
        ('loop_skill = it["refused_stops"][-1]', 'loop_skill = ""')
    ],
    "any-harness-delivery-is-a-refused-stop": [
        ('if skill is not None and VETO_HEAD in said.split("\\n", 1)[0]:', "if skill is not None:")
    ],
    "lap-rule-never-starts-again": [('(rule == "lap" and lap)', '(rule == "lap" and not lap)')],
    "body-rule-never-starts-again": [
        ('(rule == "body" and body)', '(rule == "body" and not body)')
    ],
    "reentry-rule-never-starts-again": [
        ('(rule == "reentry" and calls_skill)', '(rule == "reentry" and not calls_skill)')
    ],
    # ── what is said of a rung ──
    "the-rung-is-about-the-most-sighted-call": [
        ("key=lambda s: tracker._outcome_counts[s]", "key=lambda s: len(seen[s])")
    ],
    "gaps-taken-over-every-sighting": [
        ('where = seen[worst][-int(row["repeats"] or 1) :]', "where = seen[worst]")
    ],
    "laps-never-counted": [("            laps += lap\n", "            laps += 0\n")],
    "own-loads-never-counted": [
        ("            own_loads += own_load\n", "            own_loads += 0\n")
    ],
    "refused-stops-between-never-counted": [
        ('refusals += len(it["refused_stops"])', "refusals += 0")
    ],
    "own-loads-counted-as-refused-stops": [
        ('refusals += len(it["refused_stops"])', "refusals += lap")
    ],
    "only-refused-stops-forgets-the-own-loads": [
        (' > 0 and not r.get("own_loads_between")', " > 0")
    ],
    "bodies-never-counted": [("            bodies += body\n", "            bodies += 0\n")],
    "error-flag-never-read": [
        (
            'output, failed = _text(block.get("content")), bool(block.get("is_error"))',
            'output, failed = _text(block.get("content")), False',
        )
    ],
    "errors-after-a-read-only-rung-not-counted": [
        (
            'rungs[-1]["errors_in_the_next_iteration"] = sum(r.is_error for r in it["results"])',
            'rungs[-1]["errors_in_the_next_iteration"] = 0',
        )
    ],
    "was-an-error-asks-the-whole-batch": [
        (
            "if c.id in by_id and _keyed(rule, c.name, by_id[c.id], epoch, work) == worst\n",
            "if c.id in by_id\n",
        )
    ],
    "summary-counts-rungs-of-any-signal": [
        (
            'repeated = [r for r in rungs if SIG_REPEATED_OUTCOME in str(r["signals"])]',
            "repeated = list(rungs)",
        )
    ],
    # ── the control ──
    "control-always-passes": [('"passes": notes_match and stops_match,', '"passes": True,')],
    "control-passes-on-an-empty-trace": [
        ("notes_match = Fraction(matched,", "notes_match = not noted or Fraction(matched,")
    ],
    "control-forgives-rungs-never-noted": [("max(len(noted), len(mine), 1)", "max(len(noted), 1)")],
    "control-forgives-notes-not-reproduced": [
        ("max(len(noted), len(mine), 1)", "max(len(mine), 1)")
    ],
    "control-threshold-strict": [
        ('len(mine), 1)) >= RULE["control_match"]', 'len(mine), 1)) > RULE["control_match"]')
    ],
    "matches-need-an-unbroken-run": [
        ("if a == b else max(table[i - 1][j], table[i][j - 1])", "if a == b else 0")
    ],
    "trace-notes-of-any-signal-counted": [
        (
            'if data.get("kind") == "stuck" and SIG_REPEATED_OUTCOME in str(data.get("signals")):',
            'if data.get("kind") == "stuck":',
        )
    ],
    "replayed-rungs-of-any-signal-matched": [
        (
            '        for r in whole_turn["detail"]\n'
            '        if SIG_REPEATED_OUTCOME in str(r["signals"])\n',
            '        for r in whole_turn["detail"]\n',
        )
    ],
    "refused-stops-never-compared": [
        ('"passes": notes_match and stops_match,', '"passes": notes_match,')
    ],
    "refused-stops-must-agree-exactly": [
        ('<= RULE["refused_stops_may_differ_by"]', '< RULE["refused_stops_may_differ_by"]')
    ],
    "refused-stops-may-differ-by-two": [
        ('"refused_stops_may_differ_by": 1,', '"refused_stops_may_differ_by": 2,')
    ],
    "undelivered-refusals-counted": [
        (
            'if data.get("kind") == "turn_end_skill" and not data.get("refused"):',
            'if data.get("kind") == "turn_end_skill":',
        )
    ],
    # ── the reading ──
    "half-the-worlds-is-not-enough": [
        (
            "if len(read) * 2 < len(worlds) or not read:",
            "if len(read) * 2 <= len(worlds) or not read:",
        )
    ],
    "unbelieved-worlds-are-read": [
        ('        elif not w["control"]["passes"]:', "        elif False:")
    ],
    "min-rungs-ignored": [('if whole_turn < RULE["min_rungs"]:', "if whole_turn < 0:")],
    "min-rungs-strict": [
        ('if whole_turn < RULE["min_rungs"]:', 'if whole_turn <= RULE["min_rungs"]:')
    ],
    "regular-threshold-strict": [
        ('if removed >= RULE["regular"]:', 'if removed > RULE["regular"]:')
    ],
    "circling-threshold-strict": [
        ('if removed <= RULE["circling"]:', 'if removed < RULE["circling"]:')
    ],
    "share-taken-of-the-lap-rule": [
        (
            'removed = Fraction(whole_turn - by_rule["lap"], whole_turn)',
            'removed = Fraction(by_rule["lap"], whole_turn)',
        )
    ],
    "reading-uses-the-body-rule": [
        (
            'removed = Fraction(whole_turn - by_rule["lap"], whole_turn)',
            'removed = Fraction(whole_turn - by_rule["body"], whole_turn)',
        )
    ],
    "reading-uses-the-reentry-rule": [
        (
            'removed = Fraction(whole_turn - by_rule["lap"], whole_turn)',
            'removed = Fraction(whole_turn - by_rule["reentry"], whole_turn)',
        )
    ],
    # ── the product's count since ADR-0209: what the `shipped` replay hands the tracker ──
    "the-shipped-replay-hands-over-no-lap": [(_SHIPPED, '{"work": work} if rule == "shipped"')],
    "the-shipped-replay-hands-over-no-work": [(_SHIPPED, '{"lap": laps} if rule == "shipped"')],
    "a-plan-call-is-work-to-the-replay": [
        ("                if call.name not in NOT_WORK\n", "                if call.name\n")
    ],
    "failed-work-is-work-to-the-replay": [
        (
            "                and (done := by_id.get(call.id)) is not None\n"
            "                and not done.is_error\n",
            "                and by_id.get(call.id) is not None\n",
        )
    ],
    "the-receipt-flag-is-never-put-back": [
        ("        data={RECEIPT_OF_CHANGE: True} if receipt else None,\n", "        data=None,\n")
    ],
    "every-plan-result-is-a-receipt-of-change": [
        (" and not failed and output.lstrip().startswith(RECEIPT_HEAD)\n", " and not failed\n")
    ],
    "a-receipt-is-described-over-every-sighting-of-its-words": [
        ("epoch, work=work if receipt else None)\n\n\ndef _matches", "epoch)\n\n\ndef _matches")
    ],
    "the-replay-never-says-a-rung-was-on-a-receipt": [
        ('"receipt": bool(evidence.get("receipt")),', '"receipt": False,')
    ],
    "the-trace-notes-on-a-receipt-are-never-counted": [
        ('on_a_receipt += bool(data.get("receipt"))', "on_a_receipt += 0")
    ],
    # ── ...and what the synthetic product since ADR-0209 hands ITS tracker ──
    "the-synthetic-product-never-tells-the-lap": [
        (_WRITTEN, '{"work": work} if product == "shipped"')
    ],
    "the-synthetic-product-never-tells-the-work": [
        (_WRITTEN, '{"lap": lap} if product == "shipped"')
    ],
    "a-refused-stop-is-no-lap-to-the-synthetic-product": [
        (
            "            lap, loop_skill = lap + 1, skill  #",
            "            lap, loop_skill = lap, skill  #",
        )
    ],
    "an-own-load-is-no-lap-to-the-synthetic-product": [
        ("                        and bool(loop_skill)\n", "                        and False\n")
    ],
    "the-synthetic-product-takes-a-pointer-for-a-body": [
        ('                        and not (data or {}).get("pointer")\n', "")
    ],
    "a-plan-call-is-work-to-the-synthetic-product": [
        ("work += tool not in NOT_WORK and not err", "work += not err")
    ],
    "failed-work-is-work-to-the-synthetic-product": [
        ("work += tool not in NOT_WORK and not err", "work += tool not in NOT_WORK")
    ],
    # ── which count wrote a world ──
    "written-under-always-says-either": [
        (
            'told[0] if len(told) == 1 else ("either" if told else "neither")',
            '"either" if told else "neither"',
        )
    ],
    "no-note-and-no-rung-is-not-agreement": [
        (
            '"reproduces_the_notes": longer == 0\n            or Fraction(',
            '"reproduces_the_notes": longer > 0\n            and Fraction(',
        )
    ],
    "by-count-threshold-strict": [
        (
            'or Fraction(agreed, longer) >= RULE["control_match"]',
            'or Fraction(agreed, longer) > RULE["control_match"]',
        )
    ],
    "by-count-rungs-of-any-signal-matched": [
        (
            '[RULES.index(count)]["detail"]\n'
            '            if SIG_REPEATED_OUTCOME in str(r["signals"])\n',
            '[RULES.index(count)]["detail"]\n',
        )
    ],
    "by-count-reads-the-whole-turn-replay-twice": [
        ('out["replays"][RULES.index(count)]["detail"]', 'out["replays"][0]["detail"]')
    ],
}


def _mutate(path: Path, edits: list[tuple[str, str]]) -> None:
    text = path.read_text()
    for old, new in edits:
        found = text.count(old)
        if found != 1:
            raise SystemExit(f"anchor found {found} times, not once, in {path.name}: {old[:70]!r}")
        text = text.replace(old, new)
    path.write_text(text)


def _report(name: str, done: subprocess.CompletedProcess[str]) -> bool:
    """One line per mutant; True if it died by a named check (the control: passed)."""
    fails = re.findall(r"^\s*FAIL (.+?)(?: -- |$)", done.stdout, re.M)
    crashed = "Traceback" in (done.stderr or "")
    ended = re.search(r"^\d+ ok, \d+ failed$", done.stdout, re.M)  # the selftest reached its end
    print(
        f"{name:42s} rc={done.returncode} finished={bool(ended)} "
        f"failed_checks={len(fails)} crashed={crashed}"
    )
    for fail in fails[:2]:
        print(f"      killed by: {fail[:100]}")
    if done.returncode not in (0, 1) or not ended:
        print(f"      {(done.stderr or done.stdout).strip().splitlines()[-1][:160]}")
    if name == "control":
        return done.returncode == 0 and bool(ended) and not fails and not crashed
    return done.returncode == 1 and bool(ended) and bool(fails) and not crashed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--work", help="where the mutant copies go (default: a temp directory)")
    args = parser.parse_args()
    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="ladder-mutants-"))
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    if os.environ.get("PYTHONPATH"):
        env["PYTHONPATH"] = os.environ["PYTHONPATH"]  # which product tree the reader imports
    good = True
    for name, edits in {"control": [], **MUTANTS}.items():
        box = work / name
        box.mkdir()
        shutil.copy(BENCH / "served_ladder.py", box / "served_ladder.py")
        try:
            _mutate(box / "served_ladder.py", edits)
        except SystemExit as stale:  # an anchor the reader no longer holds: say which, go on
            print(f"{name:42s} NOT APPLIED: {stale}")
            good = False
            continue
        done = subprocess.run(
            [sys.executable, str(box / "served_ladder.py"), "--selftest"],
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
            check=False,
        )
        good &= _report(name, done)
    print(f"{len(MUTANTS)} mutants")
    print(
        "MUTATION PROOF:",
        "every mutant killed by a named check; the control passes" if good else "NOT PROVEN",
    )
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
