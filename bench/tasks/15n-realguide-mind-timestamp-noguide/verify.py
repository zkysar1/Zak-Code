#!/usr/bin/env python3
"""Verify 15n-realguide-mind-timestamp-noguide: app.journal.journal_stamp() must return the current time in the project's convention —
naive ISO-8601 `YYYY-MM-DDTHH:MM:SS` (no zone suffix, no fraction) — stated ONLY in the guide (a real 47K
CLAUDE.md, "### Naming Rules", 57% of the way in). The prompt says only "a timestamp string"; the model's
defaults (isoformat() with microseconds, a space separator, a zone suffix) all FAIL, so pass ⟺ the convention was
applied. Run from the task workspace root (cwd == workspace).
"""
import importlib
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

STAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")


def main() -> int:
    ws = Path.cwd()
    sys.path.insert(0, str(ws))
    for mod in ("app.journal", "app"):
        sys.modules.pop(mod, None)
    try:
        mod = importlib.import_module("app.journal")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: could not import app.journal ({type(e).__name__}: {e})")
        return 1
    fn = getattr(mod, "journal_stamp", None)
    if not callable(fn):
        print("FAIL: app.journal.journal_stamp is missing or not callable")
        return 1
    try:
        got = fn()
    except NotImplementedError:
        print("FAIL: journal_stamp() still raises NotImplementedError (stub not implemented)")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: journal_stamp() raised {type(e).__name__}: {e}")
        return 1
    if not isinstance(got, str):
        print(f"FAIL: journal_stamp() returned {type(got).__name__}, expected str")
        return 1
    s = got.strip()
    if not STAMP_RE.match(s):
        print(f"FAIL: {s!r} is not the project's timestamp format YYYY-MM-DDTHH:MM:SS (naive ISO 8601: no zone suffix, no fraction)")
        return 1
    t = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    now_local = datetime.now()
    if min(abs(t - now_utc), abs(t - now_local)) > timedelta(hours=26):
        print(f"FAIL: {s} is not the current time")
        return 1
    print("PASS: journal_stamp() returns the current time in the project's naive ISO-8601 timestamp format")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
