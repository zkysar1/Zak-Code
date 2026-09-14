#!/usr/bin/env python3
"""Verify 15i-realguide-mind-beliefid: app.ids.next_belief_id() must allocate ids in the project's convention —
belief records are `bel-NNN` — stated ONLY in the guide (a real 47K CLAUDE.md, "### ID Formats", under
"## Universal Conventions"). The prompt says only "the id string for the next belief record"; with NO ids assigned
yet the prefix must come from the convention (a prefix-copying implementation has nothing to copy), so the
empty-list case is the discriminator. Padded (`bel-001`) and unpadded (`bel-1`) forms both pass; the number must
be the next in sequence. Run from the task workspace root (cwd == workspace).
"""
import importlib
import re
import sys
from pathlib import Path

CASES = [
    ([], 1),
    (["bel-001", "bel-002"], 3),
    ([f"bel-{i:03d}" for i in range(1, 42)], 42),
]


def main() -> int:
    ws = Path.cwd()
    sys.path.insert(0, str(ws))
    for mod in ("app.ids", "app"):
        sys.modules.pop(mod, None)
    try:
        mod = importlib.import_module("app.ids")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: could not import app.ids ({type(e).__name__}: {e})")
        return 1
    fn = getattr(mod, "next_belief_id", None)
    if not callable(fn):
        print("FAIL: app.ids.next_belief_id is missing or not callable")
        return 1
    for existing, expected_n in CASES:
        try:
            got = fn(list(existing))
        except NotImplementedError:
            print("FAIL: next_belief_id() still raises NotImplementedError (stub not implemented)")
            return 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL: next_belief_id({existing[:3]}{'...' if len(existing) > 3 else ''}) raised {type(e).__name__}: {e}")
            return 1
        if not isinstance(got, str):
            print(f"FAIL: next_belief_id() returned {type(got).__name__}, expected str")
            return 1
        s = got.strip()
        m = re.match(r"^bel-(\d+)$", s)
        if not m:
            print(f"FAIL: {s!r} (for {len(existing)} existing ids) is not the project's belief-id format bel-NNN")
            return 1
        if int(m.group(1)) != expected_n:
            print(f"FAIL: {s!r} for {len(existing)} existing ids; expected the next sequential id, number {expected_n}")
            return 1
    print("PASS: next_belief_id() allocates sequential ids in the project's bel-NNN format")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
