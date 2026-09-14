#!/usr/bin/env python3
"""Verify 12-agents-md-convention: app.versioning.version_string() must follow the UNPROMPTED
project convention that lives ONLY in the guide file (AGENTS.md by default): version strings are
`v<MAJOR>.<MINOR>` with the patch component OMITTED.

The task prompt does NOT mention this format, and the model's strong default (full semver
"MAJOR.MINOR.PATCH") is WRONG here — so pass ⟺ the agent CONSULTED the unprompted convention and
applied it. Trivial to apply once known (a one-line format), so a fail is an adherence miss, not a
coding-skill miss.

Triples below deliberately DIFFER from AGENTS.md's worked example (2,5,3 -> v2.5) so a general
implementation is required, not a copy of the example.

Run from the task workspace root (cwd == workspace), as the bench harnesses do.
"""
import importlib
import sys
from pathlib import Path

# (major, minor, patch) -> expected adherent output
CASES = [
    ((1, 0, 0), "v1.0"),
    ((10, 11, 7), "v10.11"),
    ((3, 4, 9), "v3.4"),
    ((0, 9, 2), "v0.9"),
]


def main() -> int:
    ws = Path.cwd()
    sys.path.insert(0, str(ws))
    for mod in ("app.versioning", "app"):
        sys.modules.pop(mod, None)
    try:
        mod = importlib.import_module("app.versioning")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: could not import app.versioning ({type(e).__name__}: {e})")
        return 1
    fn = getattr(mod, "version_string", None)
    if not callable(fn):
        print("FAIL: app.versioning.version_string is missing or not callable")
        return 1
    for (args, expected) in CASES:
        try:
            got = fn(*args)
        except NotImplementedError:
            print("FAIL: version_string() still raises NotImplementedError (stub not implemented)")
            return 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL: version_string{args} raised {type(e).__name__}: {e}")
            return 1
        if not isinstance(got, str):
            print(f"FAIL: version_string{args} returned {type(got).__name__}, expected str")
            return 1
        if got.strip() != expected:
            print(
                f"FAIL: version_string{args} returned {got.strip()!r}; expected {expected!r} "
                f"(project convention: v<MAJOR>.<MINOR>, patch omitted — see the project guide)"
            )
            return 1
    print("PASS: version_string() follows the project's v<MAJOR>.<MINOR> convention (patch omitted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
