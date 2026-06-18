#!/usr/bin/env python
"""Held-out oracle for task 04-todo-cli. Run with cwd = the agent's workspace.

Exercises BOTH layers against the agent's own code, independent of the tests it wrote:
  1. store.TodoStore data layer: incrementing+non-reused ids, list filtering, missing-id.
  2. cli.py CLI contract via subprocess: add/list/--all/done/rm output + error path.
Exits 0 on success, 1 on the first failed expectation.
"""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

WS = Path.cwd()


def fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def check_store() -> int | None:
    if not (WS / "store.py").is_file():
        return fail("store.py missing")
    sys.path.insert(0, str(WS))
    try:
        import store

        importlib.reload(store)
        TodoStore = store.TodoStore
    except Exception as e:  # noqa: BLE001
        return fail(f"could not import store.TodoStore: {type(e).__name__}: {e}")

    path = Path(tempfile.mkdtemp()) / "t.json"
    try:
        s = TodoStore(str(path))
        i1 = s.add("alpha")
        i2 = s.add("beta")
        if [i1, i2] != [1, 2]:
            return fail(f"add ids = {[i1, i2]}, expected [1, 2] (auto-increment from 1)")
        if len(s.list()) != 2:
            return fail(f"list() len = {len(s.list())}, expected 2")
        if s.complete(999) is not False:
            return fail("complete(missing) must return False")
        if s.remove(999) is not False:
            return fail("remove(missing) must return False")
        if s.complete(i1) is not True:
            return fail("complete(existing) must return True")
        open_only = s.list(include_done=False)
        if any(t.get("id") == i1 for t in open_only):
            return fail("list(include_done=False) still shows a completed todo")
        # id non-reuse after removal: remove beta(2), add gamma -> must be 3, not 2
        if s.remove(i2) is not True:
            return fail("remove(existing) must return True")
        i3 = s.add("gamma")
        if i3 != 3:
            return fail(f"id reused: add after remove gave {i3}, expected 3 (no reuse)")
        # persistence: a fresh store on the same path sees prior state
        s2 = TodoStore(str(path))
        ids = sorted(t.get("id") for t in s2.list())
        if ids != [1, 3]:
            return fail(f"persistence broken: reloaded ids = {ids}, expected [1, 3]")
    except Exception as e:  # noqa: BLE001
        return fail(f"TodoStore raised during the sequence: {type(e).__name__}: {e}")
    return None


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "cli.py", *args], cwd=WS, capture_output=True, text=True, timeout=30
    )


def check_cli() -> int | None:
    if not (WS / "cli.py").is_file():
        return fail("cli.py missing")
    # Run inside a clean cwd so todos.json starts empty: use a temp copy of cli.py + store.py.
    tmp = Path(tempfile.mkdtemp())
    for name in ("cli.py", "store.py"):
        (tmp / name).write_bytes((WS / name).read_bytes())

    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "cli.py", *args], cwd=tmp, capture_output=True, text=True, timeout=30
        )

    r = run("add", "buy", "milk")
    if r.returncode != 0 or "#1" not in r.stdout:
        return fail(f"`add buy milk`: rc={r.returncode} out={r.stdout!r} (expected '#1')")
    run("add", "walk dog")
    r = run("done", "1")
    if r.returncode != 0 or "#1" not in r.stdout:
        return fail(f"`done 1`: rc={r.returncode} out={r.stdout!r}")
    # open list hides the done item (#1), shows #2
    r = run("list")
    if "#1" in r.stdout or "#2" not in r.stdout:
        return fail(f"`list` should hide done #1, show #2: out={r.stdout!r}")
    # --all shows the done item, prefixed 'x '
    r = run("list", "--all")
    if "#1" not in r.stdout or "x " not in r.stdout:
        return fail(f"`list --all` should show done #1 with 'x ' prefix: out={r.stdout!r}")
    # missing-id error path: nonzero exit, message on stderr
    r = run("done", "999")
    if r.returncode == 0 or "no such todo" not in (r.stdout + r.stderr):
        return fail(f"`done 999` should fail with 'no such todo': rc={r.returncode}")
    # persisted state is real JSON
    todos = json.loads((tmp / "todos.json").read_text(encoding="utf-8"))
    if not isinstance(todos, (list, dict)):
        return fail("todos.json is not valid JSON list/dict")
    return None


def main() -> int:
    for check in (check_store, check_cli):
        rc = check()
        if rc is not None:
            return rc
    print("PASS: TodoStore data layer + cli.py contract correct on the held-out sequence")
    return 0


if __name__ == "__main__":
    sys.exit(main())
