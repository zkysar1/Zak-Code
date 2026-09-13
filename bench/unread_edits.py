#!/usr/bin/env python3
"""How often does the model edit a file it has not read this run? (review lever L3, measure first)

Lever L3 would have ``edit_file`` refuse a path the session has not read (a ``write_file`` of the
same path counts as knowing it). Worth building only if such edits happen and fail: an unread edit
that succeeds costs nothing, and a refusal costs a turn. This counts both halves over the provider
request dumps (``ZBENCH_DUMP_REQUESTS``: ``<root>/<cell>/run-N/wire-NNNN.json``).

Per run the LARGEST wire dump is the conversation (rb-10853: the turn-end side requests are a few
KB and often sort last). Tool calls are walked in order; a path is KNOWN once a ``read_file``,
``write_file`` or ``edit_file`` named it. Each ``edit_file`` is scored known/unread and by the
result text the model saw. Counts only, never content; the positive control is the number of
``edit_file`` results matched to their call.

    python3 unread_edits.py /root/zb-dumps
"""

from __future__ import annotations

import collections
import glob
import json
import os
import posixpath
import sys

KNOWING_TOOLS = {"read_file", "write_file", "edit_file"}
#: ``--reads-dont-count``: positive control for the unread branch -- score edits as if reading a
#: file taught the model nothing, so any edit that followed only a read must count as unread.
CONTROL_TOOLS = {"write_file", "edit_file"}
OUTCOMES = (
    ("not_found", "'old_string' not found in"),
    ("multiple", "occurrences of 'old_string'"),
    ("no_change", "No change needed"),
    ("file_missing", "File not found:"),
    ("not_applied", "NOT applied"),
)


def conversation_dump(run_dir: str) -> str | None:
    wires = glob.glob(os.path.join(run_dir, "wire-*.json"))
    return max(wires, key=os.path.getsize) if wires else None


def norm(path: str) -> str:
    return posixpath.normpath(path.replace("\\", "/")).lstrip("./") or path


def known(path: str, seen: set[str]) -> bool:
    p = norm(path)
    return any(p == s or p.endswith("/" + s) or s.endswith("/" + p) for s in seen)


def outcome_of(text: str) -> str:
    for name, marker in OUTCOMES:
        if marker in text:
            return name
    return "ok"


def main(root: str, knowing: set[str] = KNOWING_TOOLS) -> int:
    totals: collections.Counter = collections.Counter()
    by_state: dict[str, collections.Counter] = {
        "known": collections.Counter(),
        "unread": collections.Counter(),
    }
    rows = []
    for cell in sorted(os.listdir(root)):
        cell_dir = os.path.join(root, cell)
        if not os.path.isdir(cell_dir):
            continue
        c: collections.Counter = collections.Counter()
        for run_dir in sorted(glob.glob(os.path.join(cell_dir, "run-*"))):
            dump = conversation_dump(run_dir)
            if dump is None:
                continue
            with open(dump, encoding="utf-8") as fh:
                messages = json.load(fh).get("messages") or []
            c["runs"] += 1
            results = {
                m.get("tool_call_id"): m.get("content")
                for m in messages
                if m.get("role") == "tool" and isinstance(m.get("content"), str)
            }
            seen: set[str] = set()
            run_unread = 0
            for m in messages:
                if m.get("role") != "assistant":
                    continue
                for call in m.get("tool_calls") or []:
                    fn = call.get("function") or {}
                    name = fn.get("name")
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except ValueError:
                        continue
                    path = args.get("path") or args.get("file_path")
                    if name not in knowing or not isinstance(path, str):
                        continue
                    if name == "edit_file":
                        c["edits"] += 1
                        state = "known" if known(path, seen) else "unread"
                        text = results.get(call.get("id"))
                        if text is None:
                            c["edits_unmatched"] += 1
                            continue
                        c["edits_matched"] += 1
                        by_state[state][outcome_of(text)] += 1
                        if state == "unread":
                            c["unread_edits"] += 1
                            run_unread += 1
                    seen.add(norm(path))
            c["runs_with_unread_edit"] += bool(run_unread)
        rows.append((cell, c))
        totals.update(c)

    print(f"{'cell':36} {'runs':>4} {'edits':>5} {'unread':>6} {'runs+':>5}")
    for cell, c in rows:
        flag = "  <-- unread edits" if c["unread_edits"] else ""
        print(
            f"{cell:36} {c['runs']:>4} {c['edits']:>5} {c['unread_edits']:>6} "
            f"{c['runs_with_unread_edit']:>5}{flag}"
        )
    print(
        f"\nruns {totals['runs']}  edit_file calls {totals['edits']}"
        f"  unread {totals['unread_edits']}"
        f"  runs with an unread edit {totals['runs_with_unread_edit']}"
    )
    for state, counter in by_state.items():
        n = sum(counter.values())
        failed = n - counter["ok"]
        rate = f"{failed / n:.0%}" if n else "n/a"
        print(f"{state:6} edits {n:>4}  failed {failed:>3} ({rate})  {dict(counter)}")
    print(
        f"positive control: {totals['edits_matched']} edit results matched to their call,"
        f" {totals['edits_unmatched']} unmatched"
    )
    return 0


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    control = "--reads-dont-count" in sys.argv
    sys.exit(
        main(argv[0] if argv else "/root/zb-dumps", CONTROL_TOOLS if control else KNOWING_TOOLS)
    )
