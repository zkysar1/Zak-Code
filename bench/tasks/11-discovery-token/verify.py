#!/usr/bin/env python3
"""Verify 11-discovery-token: app.meta.release_tag() must return the exact token defined ONLY in
ops/release-registry.md. pass ⟺ the agent discovered and read that file; there is no other way to
obtain the un-guessable value, and compliance (return the string) is trivial once found. So a
verify_rc==0 is a clean DISCOVERY hit and a verify_rc==1 is a discovery miss — no adherence or
serialization residual is in the path.

Run from the task workspace root (cwd == workspace), as the bench harnesses do.
"""
import importlib
import sys
from pathlib import Path

TOKEN = "ZK-9F3A-2231-RELEASE"


def main() -> int:
    ws = Path.cwd()
    # import the agent's app/meta.py freshly from the workspace
    sys.path.insert(0, str(ws))
    for mod in ("app.meta", "app"):
        sys.modules.pop(mod, None)
    try:
        meta = importlib.import_module("app.meta")
    except Exception as e:  # noqa: BLE001 — any import failure is a fail with a clear message
        print(f"FAIL: could not import app.meta ({type(e).__name__}: {e})")
        return 1
    fn = getattr(meta, "release_tag", None)
    if not callable(fn):
        print("FAIL: app.meta.release_tag is missing or not callable")
        return 1
    try:
        got = fn()
    except NotImplementedError:
        print("FAIL: release_tag() still raises NotImplementedError (stub not implemented)")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: release_tag() raised {type(e).__name__}: {e}")
        return 1
    if not isinstance(got, str):
        print(f"FAIL: release_tag() returned {type(got).__name__}, expected str")
        return 1
    if got.strip() != TOKEN:
        # do NOT echo the expected token in the FAIL message content beyond what's needed;
        # the value is not secret, but keep the diff tight.
        print(f"FAIL: release_tag() returned {got.strip()!r}; expected the official tag from the repo docs")
        return 1
    print("PASS: release_tag() returns the official release tag defined in ops/release-registry.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
