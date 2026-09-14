#!/usr/bin/env python3
"""Verify 14u-agents-md-longguide-under: app.people.full_name() must follow the UNPROMPTED convention that
lives ONLY in the guide file (AGENTS.md by default): a full name renders as "LAST, FIRST".

The prompt does not mention this, and the model's strong default ("First Last") is WRONG here — so
pass ⟺ the agent CONSULTED the unprompted convention and applied it. Trivial once known → a fail is
an adherence miss, not a coding miss. Test names differ from AGENTS.md's example (Hopper, Grace) so a
general implementation is required, not a copy.

Run from the task workspace root (cwd == workspace).
"""
import importlib
import sys
from pathlib import Path

CASES = [
    (("Ada", "Lovelace"), "Lovelace, Ada"),
    (("Alan", "Turing"), "Turing, Alan"),
    (("Marie", "Curie"), "Curie, Marie"),
]


def main() -> int:
    ws = Path.cwd()
    sys.path.insert(0, str(ws))
    for mod in ("app.people", "app"):
        sys.modules.pop(mod, None)
    try:
        mod = importlib.import_module("app.people")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: could not import app.people ({type(e).__name__}: {e})")
        return 1
    fn = getattr(mod, "full_name", None)
    if not callable(fn):
        print("FAIL: app.people.full_name is missing or not callable")
        return 1
    for (args, expected) in CASES:
        try:
            got = fn(*args)
        except NotImplementedError:
            print("FAIL: full_name() still raises NotImplementedError (stub not implemented)")
            return 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL: full_name{args} raised {type(e).__name__}: {e}")
            return 1
        if not isinstance(got, str):
            print(f"FAIL: full_name{args} returned {type(got).__name__}, expected str")
            return 1
        if got.strip() != expected:
            print(
                f"FAIL: full_name{args} returned {got.strip()!r}; expected {expected!r} "
                f"(project convention: 'LAST, FIRST' — see the project guide)"
            )
            return 1
    print("PASS: full_name() follows the project's 'LAST, FIRST' naming convention")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
