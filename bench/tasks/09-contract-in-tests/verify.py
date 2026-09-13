#!/usr/bin/env python
"""Held-out oracle for 09-contract-in-tests. Run with cwd = the agent's workspace.

WHAT THIS TASK IS FOR. The determinism review (ADR-0163) could not measure levers L4 (auto-derived
verify command) and L6 (test-file hint on edit) because no task's contract lived ONLY in its tests:
every prompt stated the full contract, or a convention file did. Here the prompt says "10% off for
bulk orders" and nothing more. The threshold (a dozen), the below-threshold behaviour (full price,
no error) and the rounding rule are stated only by tests/test_discounts.py, whose four BULK10
tests fail on the seed. An agent that reads or runs the tests learns the contract; one that
implements "10% off" from the prompt alone passes three of the four and fails the fourth.

The tests ARE the contract, so editing them is failing the task: the file's sha256 is pinned.

Exits 0 on success, 1 on the first failure, with the reason.
"""
from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

WS = Path.cwd()
TESTS_SHA256 = "344636de4c3c7dde437b7699a7e6a2a060cc8f9d67634861b4015b1039928290"


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    tf = WS / "tests" / "test_discounts.py"
    if not tf.is_file():
        fail("tests/test_discounts.py is missing -- the tests are the contract")
    if hashlib.sha256(tf.read_bytes()).hexdigest() != TESTS_SHA256:
        fail("tests/test_discounts.py was modified -- the tests are the contract; the implementation "
             "has to satisfy them, not the other way round")

    sys.path.insert(0, str(WS))
    for mod in ("pricing", "pricing.discounts"):
        sys.modules.pop(mod, None)
    try:
        d = importlib.import_module("pricing.discounts")
    except Exception as exc:
        fail(f"pricing.discounts does not import: {type(exc).__name__}: {exc}")

    # 1. listed like the existing code
    try:
        codes = d.list_codes()
    except Exception as exc:
        fail(f"list_codes() raised {type(exc).__name__}: {exc}")
    if "BULK10" not in codes:
        fail(f"'BULK10' is not in list_codes() (currently {codes})")

    # 2. the contract the tests state, plus cases the tests do not spell out
    cases = [
        (("4.00", 12, "BULK10"), Decimal("43.20"), "10% off from a dozen units"),
        (("4.00", 11, "BULK10"), Decimal("44.00"), "below a dozen: full price, not an error"),
        (("4.00", 1, "bulk10"), Decimal("4.00"), "one unit, lowercase code: full price"),
        (("4.00", 50, "BULK10"), Decimal("180.00"), "well above the threshold"),
        (("0.35", 13, "BULK10"), Decimal("4.10"), "half-up rounding of 4.095"),
        (("10.00", 1, "WELCOME5"), Decimal("9.50"), "the existing code is unchanged"),
        (("2.50", 4, None), Decimal("10.00"), "no code is unchanged"),
    ]
    for args, want, why in cases:
        try:
            got = d.apply_discount(*args)
        except Exception as exc:
            fail(f"apply_discount{args!r} raised {type(exc).__name__}: {exc} ({why})")
        if not isinstance(got, Decimal) or got != want:
            fail(f"apply_discount{args!r} = {got!r}, expected Decimal('{want}') ({why})")
    try:
        d.apply_discount("1.00", 20, "NOPE")
    except d.UnknownCode:
        pass
    except Exception as exc:
        fail(f"an unknown code raised {type(exc).__name__} instead of UnknownCode")
    else:
        fail("an unknown code no longer raises UnknownCode")

    # 3. the project's own tests pass, in the project's own runner
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
                          cwd=WS, capture_output=True, text=True, timeout=90)
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-6:]
        fail("the project's tests do not pass:\n" + "\n".join(tail))

    print("PASS: BULK10 is listed, applies from a dozen units, is full price below it, rounds "
          "half up, leaves the existing codes alone, and the untouched tests pass")


if __name__ == "__main__":
    main()
