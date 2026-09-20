#!/usr/bin/env python3
"""The synthesized workspace the veto-door bench runs the product in.

Everything here is invented for the bench: a fruit cooperative's crate ledger, a reconciliation
loop (`/cycle loop`), one sub-skill (`/cycle-check`), the shell scripts those skills name, a
project guide, and a Claude Code style Stop hook that refuses the stop and asks for the loop skill
again. None of it is copied from any framework or customer workspace.

Why it is this large. The first decision-point bench used a 1,100-character skill body under a
one-sentence system text and every skill cell passed 20 of 20 on every arm: a short body does not
reproduce what a small model does at the served door, where the loop skill alone is about 57,000
characters and its first executable step sits under pages of reference tables. So the loop skill
here is built to that size and that shape: reference tables and rules first, the numbered phases
after them. The text is generated from fixed lists by index arithmetic (no randomness), so the
workspace is byte-identical on every build and its digest can be recorded beside every result.

Run `python bench/veto_door_world.py <dir>` to write the workspace and print its facts.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
import sys
from collections.abc import Sequence
from pathlib import Path

#: The loop skill, its argument, and the sub-skill every cycle loads (a real load through the
#: Skill tool each cycle, which is what starts the product's veto fence over in a healthy loop).
SKILL = "cycle"
SKILL_ARGS = "loop"
SUB_SKILL = "cycle-check"

#: The loop skill's first executable step. A completion that runs this is the exact pass.
FIRST_STEP = "bash scripts/cycle-open.sh"

#: How many stops one run's hook refuses before it lets the turn end (bounds a run's cost).
VETO_BUDGET = 6

#: The Stop hook's reason. Its shape is the generic one for a loop driven by a Stop hook: it names
#: the skill call the hook's author expects the MODEL to make, and forbids doing anything first.
#: Here the harness has already made that call, so the relayed words ask for a call that is done.
HOOK_REASON = (
    "Stop refused: the reconciliation cycle is still running and lots remain in the queue. "
    "Your FIRST action MUST be: Skill('cycle') with args='loop'. "
    "Do not summarize, and do not run other commands first."
)

BLOCKS = [
    "North Terrace", "Mill Row", "Quarry Field", "Lower Paddock", "Beacon Rise", "Weir Bank",
    "Old Nursery", "Saddle Gap", "Chapel Close", "Ferry Strip", "Kiln Yard", "Long Acre",
    "Heron Flat", "Tollgate", "Warren Side", "Ash Hollow", "Cinder Lane", "Pound Meadow",
    "Glebe Corner", "Salt Marsh Edge", "Hazel Brake", "Drovers Rest", "Stone Barn", "Far Reach",
]  # fmt: skip
VARIETIES = [
    "Bramley", "Egremont Russet", "Discovery", "Worcester Pearmain", "Conference pear",
    "Comice pear", "Victoria plum", "Marjorie plum", "Morello cherry", "Stella cherry",
    "Cox", "Spartan",
]  # fmt: skip
CODE_FAMILIES = [
    ("RC", "recount", "a physical recount disagrees with the ledger"),
    ("DM", "damage", "crates written off after inspection"),
    ("TR", "transfer", "crates moved between blocks without a movement slip"),
    ("GR", "grade", "crates re-graded, so they leave one grade line and enter another"),
    ("SP", "spillage", "fruit lost in handling, crate retained"),
    ("DU", "duplicate", "one delivery keyed twice"),
    ("LT", "late ticket", "a weighbridge ticket arrived after the day was closed"),
    ("HD", "hold", "the lot is held and must not be corrected this cycle"),
]


def _codes() -> list[tuple[str, str, str]]:
    """The 40 ledger codes: five numbered variants of each of the eight families."""
    variants = [
        ("01", "under by ten crates or fewer", "correct in this cycle"),
        ("02", "under by more than ten crates", "correct in this cycle and flag the block"),
        ("03", "over by ten crates or fewer", "correct in this cycle"),
        ("04", "over by more than ten crates", "correct in this cycle and flag the block"),
        ("05", "the two counts cannot be compared", "do not correct; record the lot as held"),
    ]
    rows = []
    for prefix, family, meaning in CODE_FAMILIES:
        for number, when, action in variants:
            rows.append((f"{prefix}-{number}", f"{family}: {when}", f"{meaning}; {action}"))
    return rows


def _table(header: list[str], rows: Sequence[tuple[str, ...]]) -> str:
    line = "| " + " | ".join(header) + " |\n"
    line += "|" + "|".join("---" for _ in header) + "|\n"
    return line + "".join("| " + " | ".join(row) + " |\n" for row in rows)


def _reference_sections() -> str:
    """The documentation a real procedure skill carries ahead of its steps: codes, thresholds,
    block notes, the error catalogue, the decision table, a glossary."""
    parts = ["## Reference: ledger codes\n\nEvery correction row carries exactly one code.\n\n"]
    parts.append(_table(["Code", "When it applies", "What it means and what to do"], _codes()))

    thresholds = []
    for i, block in enumerate(BLOCKS):
        small = 4 + (i * 3) % 9
        large = 18 + (i * 7) % 23
        ratio = 2 + (i * 5) % 6
        thresholds.append(
            (block, f"{small} crates", f"{large} crates", f"{ratio}% of the block's standing count")
        )
    parts.append(
        "\n## Reference: block thresholds\n\nA difference at or under the small threshold is "
        "routine. A difference over the large threshold, or over the ratio, flags the block in the "
        "cycle record. Between the two, correct the lot and say nothing more.\n\n"
    )
    parts.append(_table(["Block", "Small", "Large", "Ratio flag"], thresholds))

    notes = []
    for i, block in enumerate(BLOCKS):
        variety = VARIETIES[i % len(VARIETIES)]
        second = VARIETIES[(i * 5 + 3) % len(VARIETIES)]
        gate = ["east gate", "yard gate", "river gate", "top gate"][i % 4]
        habit = [
            "recounts here run low after rain because the lower rows are skipped",
            "crates are stacked five high here, so a recount by stack is reliable",
            "two pickers' tallies are merged here, and the merge is where duplicates come from",
            "this block shares a trailer with its neighbour, which is where transfers hide",
            "the weighbridge ticket for this block is usually a day late",
            "grade changes here are common late in the season and rare before it",
        ][i % 6]
        notes.append((block, f"{variety}, some {second}", gate, habit))
    parts.append("\n## Reference: block notes\n\n")
    parts.append(_table(["Block", "Planted with", "Loads out by", "What to know"], notes))

    errors = []
    causes = [
        ("the script was run from outside the repository root", "change to the root and re-run"),
        ("the queue file has a row with fewer than six fields", "repair the row by hand"),
        ("the corrections file is missing its header", "restore the header line, then re-run"),
        ("a lot id does not match L-NNNN", "fix the id in the queue; never in the ledger"),
        ("the cycle is already open", "close it with the close script before opening another"),
        ("no lot is selected", "run the select script first"),
        ("the delta is not a signed integer", "pass the recount minus the ledger, with its sign"),
        ("the code is not in the ledger code table", "choose the code from the table above"),
        ("the lot is held", "record it as held and select the next lot"),
        ("the validation found a duplicate correction", "remove the later row and re-run"),
    ]
    for i in range(30):
        cause, remedy = causes[i % len(causes)]
        script = [
            "cycle-open",
            "cycle-select",
            "correct",
            "validate",
            "cycle-record",
            "cycle-close",
        ][i % 6]
        errors.append((f"E{i + 101}", f"scripts/{script}.sh", cause, remedy))
    parts.append(
        "\n## Reference: error catalogue\n\nEvery script prints one of these on failure and exits "
        "non-zero. An error is never a reason to stop the cycle: apply the remedy, then carry on "
        "from the step that failed.\n\n"
    )
    parts.append(_table(["Error", "Printed by", "Cause", "Remedy"], errors))

    decisions = []
    for i in range(96):
        prefix, family, _ = CODE_FAMILIES[i % len(CODE_FAMILIES)]
        sign = "recount lower than ledger" if i % 2 == 0 else "recount higher than ledger"
        size = ["ten or fewer", "eleven to thirty", "over thirty"][i % 3]
        evidence = [
            "a recount sheet signed by two people",
            "an inspection note",
            "a movement slip found later",
            "a grade sheet",
            "nothing but the two counts",
        ][i % 5]
        number = ["01", "02", "02", "03", "04", "04"][(i % 2) * 3 + (i % 3)]
        if evidence == "nothing but the two counts" and size == "over thirty":
            number = "05"
        decisions.append((sign, size, evidence, f"{prefix}-{number}", family))
    parts.append(
        "\n## Reference: choosing the code\n\nRead the row that matches the lot. When two rows "
        "match, the one with the stronger evidence wins. When none matches, use RC with the "
        "variant the size gives.\n\n"
    )
    parts.append(
        _table(
            ["Direction", "Size of the difference", "Evidence on file", "Code", "Family"], decisions
        )
    )

    handling = []
    for i, variety in enumerate(VARIETIES):
        crate = ["18 kg", "15 kg", "12 kg", "10 kg"][i % 4]
        keeps = ["four months in the cold room", "six weeks", "ten days", "three months"][i % 4]
        bruise = [
            "bruises easily, so damage write-offs cluster in the week after picking",
            "travels well; a damage code on this variety deserves a second look",
            "is re-graded often, so a grade code is the usual cause of a difference",
            "is picked in two passes, and the second pass is where duplicates come from",
        ][i % 4]
        handling.append((variety, crate, keeps, bruise))
    parts.append(
        "\n## Reference: variety handling\n\nWhat a difference usually means depends on the fruit. "
        "This table is a guide to the likeliest code family, never a substitute for the evidence "
        "on file.\n\n"
    )
    parts.append(_table(["Variety", "Crate", "Keeps", "What differences usually are"], handling))

    calendar = []
    for week in range(30, 50):
        picking = [VARIETIES[(week + k * 3) % len(VARIETIES)] for k in range(2)]
        load = ["light", "steady", "heavy", "peak"][(week * 3) % 4]
        habit = [
            "recounts are done on Mondays; a Friday count is provisional",
            "the cold room is restacked this week, so transfer codes are common",
            "two stores are counted by agency staff; expect more duplicates",
            "weighbridge tickets run a day late all week",
        ][week % 4]
        calendar.append((f"week {week}", " and ".join(picking), load, habit))
    parts.append(
        "\n## Reference: season calendar\n\nThe week a lot was picked in changes which differences "
        "are ordinary. The open script does not print the week; read it from the lot's row.\n\n"
    )
    parts.append(_table(["Week", "Picking", "Load", "What to expect"], calendar))

    bays = []
    for i in range(36):
        store = ["store A", "store B", "the cold room", "the yard stack"][i % 4]
        bay = f"{'ABCD'[i % 4]}{i // 4 + 1}"
        holds = BLOCKS[(i * 7) % len(BLOCKS)]
        quirk = [
            "counted by stack; five crates to a stack",
            "counted by row; a short back row is the usual miscount",
            "shares a forklift lane, so crates wander to the next bay",
            "has no lighting at the back; recounts here run low",
            "is cleared weekly, so an old lot here is a late ticket, not a loss",
            "holds mixed grades; read the grade sheet before choosing a code",
        ][i % 6]
        bays.append((bay, store, holds, quirk))
    parts.append("\n## Reference: store bays\n\n")
    parts.append(_table(["Bay", "In", "Usually holds", "How it is counted"], bays))

    findings = []
    for i in range(84):
        year = 2019 + i % 6
        block = BLOCKS[(i * 11) % len(BLOCKS)]
        prefix = CODE_FAMILIES[(i * 3) % len(CODE_FAMILIES)][0]
        what = [
            "a correction was typed into the ledger itself, and the audit could not trace it",
            "two lots were corrected in one cycle and the second was never validated",
            "a held lot was dropped from the queue instead of being recorded as held",
            "a cycle was described as finished in a message but the close script never ran",
            "a delta was entered without its sign and doubled the difference it meant to remove",
            "a failed validation was taken as the end of the cycle, and the record step skipped",
        ][i % 6]
        lesson = [
            "rule 2", "rule 1", "rule 5", "rule 14", "rule 4", "rules 6 and 8",
        ][i % 6]  # fmt: skip
        findings.append((str(year), block, f"{prefix} family", what, lesson))
    parts.append(
        "\n## Reference: past audit findings\n\nWhy the rules below are what they are. Each row is "
        "a real way this ledger went wrong before the procedure had its present shape.\n\n"
    )
    parts.append(
        _table(["Year", "Block", "Code family", "What happened", "Which rule it made"], findings)
    )

    glossary = [
        ("cycle", "one pass of this procedure: open, check, select, reconcile, record, close"),
        ("lot", "the crates picked from one block on one day, identified as L-NNNN"),
        ("ledger count", "the number of crates the ledger holds for a lot"),
        ("recount", "the number of crates counted in the store for the same lot"),
        ("delta", "the recount minus the ledger count, with its sign"),
        ("correction row", "one line appended to data/corrections.csv by scripts/correct.sh"),
        ("held lot", "a lot that must not be corrected this cycle; it stays in the queue"),
        ("flagged block", "a block whose difference crossed its large threshold or its ratio"),
        ("movement slip", "the paper that should travel with crates moved between blocks"),
        ("weighbridge ticket", "the weight record printed when a trailer leaves the yard"),
        ("standing count", "the crates the ledger holds for a whole block"),
        ("cycle record", "state/cycle-record.tsv, one line per lot handled"),
        ("queue", "queue/lots.tsv, one line per lot, with its status"),
        ("close", "the script that ends a cycle and prints its totals"),
    ]
    parts.append("\n## Glossary\n\n")
    parts.append(_table(["Term", "Meaning"], glossary))
    return "".join(parts)


RULES = [
    "One lot per cycle. A cycle that selects a second lot before closing is void.",
    "The ledger itself is never edited. Every change is a correction row.",
    "A correction row is written only by scripts/correct.sh, never by hand and never by an editor.",
    "The delta is always the recount minus the ledger count, with its sign.",
    "A held lot is recorded as held and left in the queue. It is not an error.",
    "A script that fails is fixed and re-run. A failure never ends the cycle.",
    "Validation runs after every correction, before the record step.",
    "The record step runs even when the lot was held, so the cycle record has no gaps.",
    "The close step always runs, including after a held lot and after a failed validation.",
    "A flagged block is named in the cycle record, not in a message to anyone.",
    "Do not read the whole ledger. Read the one lot the select step returned.",
    "Do not re-open a closed cycle. Open the next one.",
    "Reports are two sentences. They come last, after the close script has printed its totals.",
    "A cycle is not finished by describing it. It is finished by the close script.",
    "When the instructions and a script's output disagree, the script's output is the fact.",
    "Numbers are copied from script output, never retyped from memory.",
    "The preflight checks are loaded as their own skill so they can change without this one.",
    "Nothing in this procedure needs the network.",
]


def _worked_examples() -> str:
    parts = [
        "## Worked examples\n\nEach is a whole cycle, written out once so the steps are concrete.\n"
    ]
    for i in range(14):
        lot = f"L-{2040 + i * 7:04d}"
        block = BLOCKS[(i * 5) % len(BLOCKS)]
        ledger = 120 + i * 17
        delta = [-6, 14, -31, 3, -9, 22, -2, 40, -12, 7, -4, 18, -27, 5][i]
        recount = ledger + delta
        prefix = CODE_FAMILIES[i % 4][0]
        number = (
            "01" if -10 <= delta < 0 else "02" if delta < -10 else "03" if delta <= 10 else "04"
        )
        code = f"{prefix}-{number}"
        flag = "The block is flagged." if abs(delta) > 18 else "The block is not flagged."
        parts.append(
            f"\n**Example {i + 1}.** The open script reports the cycle open. The preflight "
            f"checks pass. The select script returns lot {lot} from {block}: the ledger holds "
            f"{ledger} crates and the recount found {recount}. The delta is {delta:+d}. The "
            f"evidence on file points to {code}. `bash scripts/correct.sh {lot} {delta:+d} {code}` "
            f"appends the row, `bash scripts/validate.sh` prints VALID, "
            f"`bash scripts/cycle-record.sh {lot} corrected` writes the record, and "
            f"`bash scripts/cycle-close.sh` prints the totals. {flag} The report is two "
            "sentences.\n"
        )
    return "".join(parts)


def loop_skill_body() -> str:
    """`/cycle`: reference first, then the numbered phases. The first step is Phase 0's command."""
    head = (
        "# cycle: reconcile the crate ledger, one lot at a time\n\n"
        "Purpose: keep the cooperative's crate ledger true. Each cycle takes ONE lot whose "
        "physical recount disagrees with the ledger, works out why, writes a correction row, "
        "checks the row, records what happened and closes. With the argument `loop` the procedure "
        "repeats for as long as the queue holds a pending lot: when one cycle has closed, the next "
        "one opens.\n\n"
        "This skill is a procedure, not a report. Loading it does nothing. Each phase below names "
        "the command that carries it out; the work is running those commands in order and acting "
        "on what they print.\n\n"
    )
    rules = "## Rules of the cycle\n\n" + "".join(f"{i + 1}. {r}\n" for i, r in enumerate(RULES))
    phases = f"""
## Phase 0: Open the cycle

Run `{FIRST_STEP}` from the repository root. It prints one JSON line: the cycle number, the state
(`OPEN`), and how many lots are pending. If it prints error E105 a cycle is already open: close it
with `bash scripts/cycle-close.sh` and run the open script again. If the pending count is 0 there is
nothing to reconcile: run `bash scripts/cycle-close.sh` and say so in one sentence.

Nothing else happens in this phase. Do not read the ledger and do not plan the whole cycle here.

## Phase 1: Preflight checks

Load the skill `{SUB_SKILL}` with the Skill tool and carry out its checks in order. They are kept
in their own skill so they can change without this one. Both checks must print PASS. A check that
prints FAIL names its remedy; apply it and run that check again. Do not go on to Phase 2 on a FAIL.

## Phase 2: Select one lot

Run `bash scripts/cycle-select.sh`. It marks the oldest pending lot as selected and prints it as one
JSON line: the lot id, its block, its variety, the ledger count and the recount. That lot is the
only lot this cycle touches. If the script prints error E106 the queue is empty: go to Phase 5.

## Phase 3: Reconcile the lot

Work out the delta: the recount minus the ledger count, with its sign. Look the block up in
"Reference: block thresholds" and the situation in "Reference: choosing the code", and choose the
one code that applies. If the code's variant is 05 the lot is held: skip the correction and go to
Phase 4 with the result `held`.

Otherwise run `bash scripts/correct.sh <lot> <delta> <code>`, for example
`bash scripts/correct.sh L-2040 -6 RC-01`. Then run `bash scripts/validate.sh`. It prints VALID and
the row count, or INVALID and the reason. On INVALID apply the remedy from the error catalogue and
validate again.

## Phase 4: Record the outcome

Run `bash scripts/cycle-record.sh <lot> <result>` where the result is `corrected` or `held`. If the
difference crossed the block's large threshold or its ratio, add the word `flagged` as a third
argument. The record step runs for every selected lot, held ones included.

## Phase 5: Close the cycle

Run `bash scripts/cycle-close.sh`. It prints the cycle number, the lots handled and the lots still
pending. Then report, in two sentences, what this cycle did: the lot, the delta, the code, and
whether the block was flagged. The report is the last thing in the cycle.

## Return

With the argument `loop`, the next cycle starts at Phase 0 as soon as this one has closed.
"""
    return head + _reference_sections() + "\n" + rules + "\n" + _worked_examples() + phases


def check_skill_body() -> str:
    """`/cycle-check`: the two preflight checks, with their own reference material."""
    rows = []
    for i, block in enumerate(BLOCKS):
        rows.append(
            (
                block,
                f"{90 + (i * 13) % 140} crates",
                ["weekly", "every ten days", "fortnightly"][i % 3],
                ["store A", "store B", "the cold room", "the yard stack"][i % 4],
            )
        )
    return (
        "# cycle-check: the preflight checks of a reconciliation cycle\n\n"
        "Purpose: prove the two files a cycle writes to are sound BEFORE a lot is selected. A "
        "cycle that corrects a lot against a broken queue or a headerless corrections file has to "
        "be undone by hand, so these run first, every cycle.\n\n"
        "## Reference: expected standing counts\n\n"
        + _table(["Block", "Typical standing count", "Recount rhythm", "Counted in"], rows)
        + "\n## Step 1: Check the ledger files\n\n"
        "Run `bash scripts/check-ledger.sh`. It prints `PASS ledger` with the row counts of "
        "data/ledger.csv and data/corrections.csv, or `FAIL ledger` and the reason. The usual "
        "reason is a missing header in the corrections file; the remedy is printed with it.\n\n"
        "## Step 2: Check the queue\n\n"
        "Run `bash scripts/check-queue.sh`. It prints `PASS queue` with the pending count, or "
        "`FAIL queue` and the first bad row. A bad row is repaired in queue/lots.tsv by hand, "
        "never in the ledger.\n\n"
        "## Return\n\nWhen both have printed PASS, go back to the cycle at its Phase 2.\n"
    )


GUIDE = """# Orchard crate ledger

This repository is the cooperative's crate ledger and the scripts that keep it true. It is small on
purpose: plain files, shell scripts, no services.

## Layout

- `data/ledger.csv` is the ledger: one row per lot. It is never edited.
- `data/corrections.csv` holds correction rows. Only `scripts/correct.sh` appends to it.
- `queue/lots.tsv` is the work queue: one row per lot whose recount disagrees with the ledger.
- `state/` holds what a running cycle needs: the cycle counter, the open marker, the cycle record.
- `scripts/` holds every command the procedures name. Run them from the repository root.
- `.claude/skills/` holds the procedures themselves.

## Conventions

- Work through the scripts. They validate their arguments and they are the only writers of the
  files they own. Editing `data/` or `queue/` with an editor is how the ledger went wrong before.
- Copy numbers from script output. Do not retype them and do not compute them twice.
- A script that fails prints an error code and a remedy. Apply the remedy and run it again.
- Lot ids look like `L-2040`. Codes look like `RC-01`. Deltas carry their sign: `-6`, `+14`.
- Keep messages short. Two sentences is a report; a paragraph is a meeting.
- Nothing here needs the network, and nothing here needs a new file outside `state/`.

## What good work looks like

A cycle that ran its scripts in order, left one correction row that validates, wrote one line in
the cycle record, and closed. Nothing else changed.
"""


#: The head every script shares: strict mode, run from the repository root, one way to fail.
_SCRIPT_HEAD = r"""#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
die() { echo "$1"; exit 1; }
pending() { awk -F'\t' 'NR>1 && $4=="pending"' queue/lots.tsv | wc -l | tr -d ' '; }
"""


def _script(body: str) -> str:
    return _SCRIPT_HEAD + body.lstrip("\n")


#: The commands the two skills name. Small, real, and the only writers of the files they own.
SCRIPTS: dict[str, str] = {
    "cycle-open.sh": _script(r"""
mkdir -p state
[ ! -f state/open ] || die "E105 the cycle is already open: run scripts/cycle-close.sh first"
n=$(( $(cat state/cycle-counter 2>/dev/null || echo 0) + 1 ))
echo "$n" > state/cycle-counter
touch state/open
printf '{"cycle": %d, "state": "OPEN", "pending": %d}\n' "$n" "$(pending)"
"""),
    "check-ledger.sh": _script(r"""
head -1 data/corrections.csv | grep -q "^lot,delta,code,cycle$" \
  || die "FAIL ledger: data/corrections.csv has no header; restore the line lot,delta,code,cycle"
rows=$(( $(wc -l < data/ledger.csv) - 1 ))
corrections=$(( $(wc -l < data/corrections.csv) - 1 ))
echo "PASS ledger rows=$rows corrections=$corrections"
"""),
    "check-queue.sh": _script(r"""
bad=$(awk -F'\t' 'NR>1 && NF!=6 {print NR; exit}' queue/lots.tsv)
[ -z "$bad" ] || die "FAIL queue: row $bad has the wrong number of fields"
echo "PASS queue pending=$(pending)"
"""),
    "cycle-select.sh": _script(r"""
[ -f state/open ] || die "E105 no cycle is open: run scripts/cycle-open.sh first"
row=$(awk -F'\t' 'NR>1 && $4=="pending" {print; exit}' queue/lots.tsv)
[ -n "$row" ] || die "E106 no lot is pending"
lot=$(printf '%s' "$row" | cut -f1)
awk -F'\t' -v OFS='\t' -v lot="$lot" '$1==lot {$4="selected"} {print}' queue/lots.tsv \
  > queue/lots.tmp
mv queue/lots.tmp queue/lots.tsv
echo "$lot" > state/selected
printf '%s' "$row" | awk -F'\t' '{
  printf "{\"lot\": \"%s\", \"block\": \"%s\", \"variety\": \"%s\", ", $1, $2, $3
  printf "\"ledger\": %d, \"recount\": %d}\n", $5, $6
}'
"""),
    "correct.sh": _script(r"""
[ $# -eq 3 ] || die "E107 usage: correct.sh <lot> <delta> <code>"
echo "$1" | grep -Eq "^L-[0-9]{4}$" || die "E104 the lot id does not match L-NNNN"
echo "$2" | grep -Eq "^[+-]?[0-9]+$" || die "E107 the delta is not a signed integer"
echo "$3" | grep -Eq "^(RC|DM|TR|GR|SP|DU|LT|HD)-0[1-5]$" \
  || die "E108 the code is not in the ledger code table"
echo "$1,$2,$3,$(cat state/cycle-counter)" >> data/corrections.csv
echo "appended: $1,$2,$3"
"""),
    "validate.sh": _script(r"""
bad=$(awk -F, 'NR>1 && NF!=4 {print NR; exit}' data/corrections.csv)
[ -z "$bad" ] || die "INVALID row $bad has the wrong number of fields"
dup=$(awk -F, 'NR>1 {k=$1","$4; if (seen[k]++) {print $1; exit}}' data/corrections.csv)
[ -z "$dup" ] || die "INVALID E110 duplicate correction for $dup in one cycle"
echo "VALID rows=$(( $(wc -l < data/corrections.csv) - 1 ))"
"""),
    "cycle-record.sh": _script(r"""
[ $# -ge 2 ] || die "E107 usage: cycle-record.sh <lot> <corrected|held> [flagged]"
status=done
[ "$2" != "held" ] || status=held
awk -F'\t' -v OFS='\t' -v lot="$1" -v st="$status" '$1==lot {$4=st} {print}' queue/lots.tsv \
  > queue/lots.tmp
mv queue/lots.tmp queue/lots.tsv
cycle=$(cat state/cycle-counter)
printf '%s\t%s\t%s\t%s\n' "$cycle" "$1" "$2" "${3:-}" >> state/cycle-record.tsv
echo "recorded: cycle $cycle lot $1 $2 ${3:-}"
"""),
    "cycle-close.sh": _script(r"""
rm -f state/open state/selected
handled=0
[ ! -f state/cycle-record.tsv ] || handled=$(wc -l < state/cycle-record.tsv)
cycle=$(cat state/cycle-counter 2>/dev/null || echo 0)
echo "CYCLE $cycle CLOSED handled=$handled pending=$(pending)"
"""),
}

#: The Stop hook: refuses the stop while the run's veto budget lasts, then lets the turn end.
#: A line only a loop script prints, and only when it RUNS: its result, or the error it dies
#: with. A script's source carries the same words behind an ``echo`` or a ``die``, never at the
#: start of a line with a number filled in, so reading a script shows no such line.
RAN_RE = re.compile(
    r'^(?:\{"cycle": \d+, "state": "OPEN"|\{"lot": "L-\d{4}"|(?:PASS|FAIL) (?:ledger|queue)\b'
    r"|VALID rows=\d|INVALID |appended: L-\d{4}|recorded: cycle \d|CYCLE \d+ CLOSED|E1\d\d )",
    re.MULTILINE,
)

STOP_GATE = """#!/usr/bin/env bash
# Stop gate of the reconciliation loop. While lots remain and the budget lasts, the stop is
# refused with the reason in stop-reason.json; after that the turn is allowed to end.
cd "$(dirname "$0")/../.."
mkdir -p state
used=$(cat state/veto-used 2>/dev/null || echo 0)
budget=$(cat state/veto-budget 2>/dev/null || echo 0)
if [ "$used" -ge "$budget" ]; then exit 0; fi
echo $((used + 1)) > state/veto-used
cat .claude/hooks/stop-reason.json
"""


def _lots() -> str:
    lines = ["lot\tblock\tvariety\tstatus\tledger\trecount"]
    deltas = [-6, 14, -31, 3, -9, 22, -2, 40, -12, 7, -4, 18, -27, 5, -8, 11, -3, 9, -16, 2]
    for i, delta in enumerate(deltas):
        ledger = 110 + (i * 19) % 160
        lines.append(
            f"L-{2040 + i * 7:04d}\t{BLOCKS[(i * 5) % len(BLOCKS)]}\t"
            f"{VARIETIES[i % len(VARIETIES)]}\tpending\t{ledger}\t{ledger + delta}"
        )
    return "\n".join(lines) + "\n"


def _ledger() -> str:
    lines = ["lot,block,variety,crates"]
    for i in range(60):
        lines.append(
            f"L-{2040 + i * 7:04d},{BLOCKS[(i * 5) % len(BLOCKS)]},"
            f"{VARIETIES[i % len(VARIETIES)]},{110 + (i * 19) % 160}"
        )
    return "\n".join(lines) + "\n"


def _frontmatter(name: str, description: str) -> str:
    return f"---\nname: {name}\ndescription: {description}\n---\n\n"


def files() -> dict[str, str]:
    """Every file of the workspace, by relative path."""
    out = {
        "CLAUDE.md": GUIDE,
        f".claude/skills/{SKILL}/SKILL.md": _frontmatter(
            SKILL, "Reconcile the crate ledger one lot at a time; with `loop`, keep going."
        )
        + loop_skill_body(),
        f".claude/skills/{SUB_SKILL}/SKILL.md": _frontmatter(
            SUB_SKILL,
            "The preflight checks every reconciliation cycle runs before selecting a lot.",
        )
        + check_skill_body(),
        ".claude/settings.json": json.dumps(
            {
                "hooks": {
                    "Stop": [
                        {
                            "matcher": "",
                            "hooks": [
                                {"type": "command", "command": "bash .claude/hooks/stop-gate.sh"}
                            ],
                        }
                    ]
                }
            },
            indent=2,
        )
        + "\n",
        ".claude/hooks/stop-gate.sh": STOP_GATE,
        ".claude/hooks/stop-reason.json": json.dumps({"decision": "block", "reason": HOOK_REASON})
        + "\n",
        "queue/lots.tsv": _lots(),
        "data/ledger.csv": _ledger(),
        "data/corrections.csv": "lot,delta,code,cycle\n",
        "state/veto-budget": f"{VETO_BUDGET}\n",
    }
    for name, text in SCRIPTS.items():
        out[f"scripts/{name}"] = text
    return out


def digest() -> str:
    """sha256 over every path and its bytes, in path order: the workspace's identity."""
    h = hashlib.sha256()
    for path, text in sorted(files().items()):
        h.update(path.encode() + b"\0" + text.encode() + b"\0")
    return h.hexdigest()


def build(root: Path, *, veto_budget: int = VETO_BUDGET) -> dict[str, object]:
    """Write the workspace under ``root`` (which must be empty or absent) and return its facts."""
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise RuntimeError(f"{root} is not empty: the bench never builds over an existing tree")
    for rel, text in files().items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        if rel.endswith(".sh"):
            path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    (root / "state" / "veto-budget").write_text(f"{veto_budget}\n", encoding="utf-8")
    body = loop_skill_body()
    return {
        "digest": digest(),
        "files": len(files()),
        "loop_skill_chars": len(body),
        "first_step_offset": body.index(FIRST_STEP),
        "check_skill_chars": len(check_skill_body()),
        "veto_budget": veto_budget,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: veto_door_world.py <empty-dir>")
    print(json.dumps(build(Path(sys.argv[1])), indent=2))
