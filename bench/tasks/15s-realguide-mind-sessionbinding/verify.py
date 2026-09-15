#!/usr/bin/env python3
"""Verify 15s-realguide-mind-sessionbinding: app.binding.binding_path(root, agent, sid) must return the path where the
project stores the agent-session binding — `agents/<name>/sessions/<SID>/binding.yaml` — stated ONLY in the guide (a real
47K CLAUDE.md, "## Session Binding (Phase 2.6)", a plain heading with no mandate word anywhere on its outline path). The
same guide still describes the retired `.active-agent-<SID>` form in two other sections, and the binding section is
the one that retires it. The prompt says only "the file where this repository stores the agent-session binding": there is
no default route to the layout. Accepted: the documented path as a Path, a str, or a root-relative path. Rejected: the
legacy root file, the agent-wide `session/` (singular) dir, and every other plausible layout. Run from the task
workspace root (cwd == workspace).
"""
import importlib
import shutil
import sys
import tempfile
from pathlib import Path

CASES = [
    ("alpha", "8913fdff-ddee-4554-b289-c3714f63c0de"),
    ("bravo", "s-1"),
]


def expected_path(root: Path, agent: str, sid: str) -> Path:
    return root / "agents" / agent / "sessions" / sid / "binding.yaml"


def layout_matches(got, root: Path, agent: str, sid: str) -> bool:
    if not isinstance(got, (str, Path)):
        return False
    p = Path(got)
    if not p.is_absolute():
        p = root / p
    return p.resolve() == expected_path(root, agent, sid).resolve()


def main() -> int:
    ws = Path.cwd()
    sys.path.insert(0, str(ws))
    for mod in ("app.binding", "app"):
        sys.modules.pop(mod, None)
    try:
        mod = importlib.import_module("app.binding")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: could not import app.binding ({type(e).__name__}: {e})")
        return 1
    fn = getattr(mod, "binding_path", None)
    if not callable(fn):
        print("FAIL: app.binding.binding_path is missing or not callable")
        return 1
    root = Path(tempfile.mkdtemp(prefix="bindws-"))
    try:
        for agent, sid in CASES:
            try:
                got = fn(root, agent, sid)
            except NotImplementedError:
                print("FAIL: binding_path() still raises NotImplementedError (stub not implemented)")
                return 1
            except Exception as e:  # noqa: BLE001
                print(f"FAIL: binding_path(root, {agent!r}, {sid!r}) raised {type(e).__name__}: {e}")
                return 1
            if not layout_matches(got, root, agent, sid):
                shown = str(got).replace(str(root), "<root>") if got is not None else repr(got)
                print(
                    f"FAIL: {shown} (for agent {agent!r}, session {sid!r}) is not where this project keeps the agent-session "
                    "binding: agents/<name>/sessions/<SID>/binding.yaml"
                )
                return 1
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("PASS: binding_path() returns the project's agent-session binding path agents/<name>/sessions/<SID>/binding.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
