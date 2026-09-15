#!/usr/bin/env python3
"""Verify 16p-realguide-envproc-aspirationkeys: app.aspirations.make_aspiration(task_key, target_key) must build the
record with the project's key names — `ayoTaskKey` for the task reference and `targetAyoKey` for the object
target — stated ONLY in the guide (a real 14K CLAUDE.md, under "## Constraints", past the fold cap). The prompt
says only "the aspiration record for a task reference and an object target"; any other names (taskKey, task_key,
task, targetKey, target, …) FAIL, so pass ⟺ the convention was applied. Extra keys are allowed. Run from the task
workspace root (cwd == workspace).
"""
import importlib
import sys
from pathlib import Path

CASES = [("task-7f3a", "ayo-1c9e"), ("moveTo:door", "ayo-0042")]


def main() -> int:
    ws = Path.cwd()
    sys.path.insert(0, str(ws))
    for mod in ("app.aspirations", "app"):
        sys.modules.pop(mod, None)
    try:
        mod = importlib.import_module("app.aspirations")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: could not import app.aspirations ({type(e).__name__}: {e})")
        return 1
    fn = getattr(mod, "make_aspiration", None)
    if not callable(fn):
        print("FAIL: app.aspirations.make_aspiration is missing or not callable")
        return 1
    for task_key, target_key in CASES:
        try:
            got = fn(task_key, target_key)
        except NotImplementedError:
            print("FAIL: make_aspiration() still raises NotImplementedError (stub not implemented)")
            return 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL: make_aspiration({task_key!r}, {target_key!r}) raised {type(e).__name__}: {e}")
            return 1
        if not isinstance(got, dict):
            print(f"FAIL: make_aspiration() returned {type(got).__name__}, expected dict")
            return 1
        keys = sorted(got.keys())
        if got.get("ayoTaskKey") != task_key or got.get("targetAyoKey") != target_key:
            print(f"FAIL: record keys {keys} do not carry the task reference as 'ayoTaskKey' and the object target as 'targetAyoKey' (the project's convention)")
            return 1
    print("PASS: make_aspiration() uses the project's ayoTaskKey / targetAyoKey names")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
