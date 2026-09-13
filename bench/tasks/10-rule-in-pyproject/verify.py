#!/usr/bin/env python
"""Held-out oracle for 10-rule-in-pyproject. Run with cwd = the agent's workspace.

WHAT THIS TASK IS FOR. The determinism review (ADR-0163) ranked "more convention filenames" (L5:
fold `pyproject.toml [tool.*]`, `Makefile` targets, `.editorconfig` into the discovered context)
as a model-free lever it could not measure: no task's rule lived in such a file. Here the
functional contract is fully stated in the prompt and is easy; the house style -- 79 columns,
annotations on every function, docstrings on every public symbol, sorted imports -- lives only in
`pyproject.toml` (`[tool.ruff]`) and is what `make lint` runs in CI. The sibling module obeys it.
An agent that reads the config, the Makefile or the sibling, or runs `make lint`, ships clean;
one that writes in its own default style fails the lint that CI would fail.

Exits 0 on success, 1 on the first failure, with the reason.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

WS = Path.cwd()


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    # The style check needs ruff. Without it this oracle cannot verify the rule the task exists
    # to measure -- say so rather than pass silently (ADR-0145: a dead instrument reads as clean).
    probe = subprocess.run([sys.executable, "-m", "ruff", "--version"], capture_output=True, text=True)
    if probe.returncode != 0:
        fail("VERIFIER BROKEN: ruff is not runnable from this interpreter, so the pyproject rule "
             "cannot be checked. Instrument failure, not an agent failure -- do not score this run.")

    if not (WS / "utils" / "duration.py").is_file():
        fail("utils/duration.py does not exist")
    sys.path.insert(0, str(WS))
    for mod in list(sys.modules):
        if mod == "utils" or mod.startswith("utils."):
            del sys.modules[mod]
    try:
        pkg = importlib.import_module("utils")
    except Exception as exc:
        fail(f"the utils package does not import: {type(exc).__name__}: {exc}")
    fn = getattr(pkg, "parse_duration", None)
    if fn is None:
        fail("parse_duration is not exported from the utils package (the existing helper is)")

    # 1. the functional contract from the prompt
    for text, want in [("1h30m", 5400), ("45s", 45), ("2h", 7200), ("90m", 5400),
                       ("1h1m1s", 3661), ("0s", 0), ("10h", 36000), ("1m", 60)]:
        try:
            got = fn(text)
        except Exception as exc:
            fail(f"parse_duration({text!r}) raised {type(exc).__name__}: {exc}")
        if got != want or isinstance(got, bool) or not isinstance(got, int):
            fail(f"parse_duration({text!r}) = {got!r}, expected {want}")
    for bad in ["", "30", "1.5h", "30m1h", "-1h", "1h 30m", "1x", "h", "1h30"]:
        try:
            got = fn(bad)
        except ValueError:
            continue
        except Exception as exc:
            fail(f"parse_duration({bad!r}) raised {type(exc).__name__}, expected ValueError")
        fail(f"parse_duration({bad!r}) returned {got!r}, expected ValueError")

    # 2. the rule that lives in pyproject.toml: CI's `make lint` must be clean
    lint = subprocess.run([sys.executable, "-m", "ruff", "check", "--config", "pyproject.toml",
                           "--no-cache", "utils", "tests"],
                          cwd=WS, capture_output=True, text=True, timeout=60)
    if lint.returncode != 0:
        tail = (lint.stdout + lint.stderr).strip().splitlines()
        fail("`make lint` (ruff, configured in pyproject.toml) fails:\n" + "\n".join(tail[:12]))

    # 3. tests were added and the suite passes in the project's own runner
    test_files = list((WS / "tests").glob("test_*.py"))
    if not any("parse_duration" in p.read_text(encoding="utf-8", errors="replace") for p in test_files):
        fail("no test under tests/ mentions parse_duration")
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                          cwd=WS, capture_output=True, text=True, timeout=90)
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-6:]
        fail("the project's tests do not pass:\n" + "\n".join(tail))

    print("PASS: parse_duration meets the contract, is exported, is tested, and the package "
          "passes the lint pyproject.toml configures")


if __name__ == "__main__":
    main()
