#!/usr/bin/env python3
"""Build the 17n-realguide-envserver-deltaassert-noguide workspace: cell 17's repo with NO guide file — the
DEFAULT-RATE control for thrust 26 (the same task, the same verify; the rule is stated nowhere)."""
import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BASE = _HERE.parent / "17-realguide-envserver-deltaassert" / "build_workspace.py"
_spec = importlib.util.spec_from_file_location("cell17_builder", _BASE)
_cell17 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_cell17)
_cell17.MODE = "noguide"

if __name__ == "__main__":
    args = sys.argv[1:]
    _cell17.main(*args) if args else _cell17.main(_HERE / "workspace")
