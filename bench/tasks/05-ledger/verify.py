#!/usr/bin/env python
"""Held-out oracle for task 05-ledger. Run with cwd = the agent's workspace.

Exercises BOTH layers against the agent's own code, independent of the tests it wrote:
  1. ledger.Ledger: id rules, deposit/withdraw guards, ATOMIC transfer, persistence, history.
  2. cli.py contract via subprocess: open/balance/deposit/transfer output + error/exit codes.
Return values are checked by truthiness (the spec says ``-> bool``; True/1 and False/0/None
both read correctly), while the money INVARIANTS are checked strictly on the balances — a
rejected op must leave state byte-for-byte unchanged. Exits 0 on success, 1 on first failure.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
import tempfile
from pathlib import Path

WS = Path.cwd()


def fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def check_ledger() -> int | None:
    if not (WS / "ledger.py").is_file():
        return fail("ledger.py missing")
    sys.path.insert(0, str(WS))
    try:
        import ledger

        importlib.reload(ledger)
        Ledger = ledger.Ledger
    except Exception as e:  # noqa: BLE001
        return fail(f"could not import ledger.Ledger: {type(e).__name__}: {e}")

    path = Path(tempfile.mkdtemp()) / "ledger.json"
    try:
        lg = Ledger(str(path))
        a = lg.open_account("alice")
        b = lg.open_account("bob")
        if [a, b] != [1, 2]:
            return fail(f"open_account ids = {[a, b]}, expected [1, 2] (auto-increment from 1)")
        if lg.balance(a) != 0 or lg.balance(b) != 0:
            return fail("new accounts must start at balance 0")
        if lg.balance(999) is not None:
            return fail("balance(missing) must be None")

        if not lg.deposit(a, 1000) or lg.balance(a) != 1000:
            return fail("deposit(a, 1000) must succeed and set balance to 1000")
        # non-positive amounts and a missing account are rejected, changing nothing
        if lg.deposit(a, 0) or lg.deposit(a, -5) or lg.balance(a) != 1000:
            return fail("deposit must reject <= 0 and leave the balance unchanged")
        if lg.deposit(999, 10):
            return fail("deposit(missing) must return falsey")

        # overdraft is rejected and leaves the balance UNCHANGED
        if lg.withdraw(a, 5000) or lg.balance(a) != 1000:
            return fail("withdraw beyond balance must be rejected and change nothing")
        if not lg.withdraw(a, 400) or lg.balance(a) != 600:
            return fail("withdraw(a, 400) must succeed and set balance to 600")

        # transfer atomicity: insufficient funds / self / missing dst -> False, NOTHING changes
        if lg.transfer(a, b, 1000) or lg.balance(a) != 600 or lg.balance(b) != 0:
            return fail("transfer with insufficient funds must change neither balance")
        if lg.transfer(a, a, 100) or lg.balance(a) != 600:
            return fail("transfer to self must be rejected and change nothing")
        if lg.transfer(a, 999, 100) or lg.balance(a) != 600:
            return fail("transfer to a missing account must change nothing")
        if not lg.transfer(a, b, 200) or lg.balance(a) != 400 or lg.balance(b) != 200:
            return fail("transfer(a, b, 200) must move funds: a=400, b=200")

        # history: ordered, typed, with transfer peers
        ha = lg.history(a)
        types_a = [e.get("type") for e in ha]
        cents_a = [e.get("cents") for e in ha]
        if types_a != ["deposit", "withdraw", "transfer_out"]:
            return fail(f"history(a) types = {types_a}, expected deposit/withdraw/transfer_out")
        if cents_a != [1000, 400, 200]:
            return fail(f"history(a) cents = {cents_a}, expected [1000, 400, 200]")
        if ha[-1].get("peer") != b:
            return fail(f"history(a) transfer_out peer = {ha[-1].get('peer')}, expected {b}")
        hb = lg.history(b)
        if [e.get("type") for e in hb] != ["transfer_in"] or hb[0].get("peer") != a:
            return fail(f"history(b) must be one transfer_in with peer={a}: {hb}")
        if lg.history(999) != []:
            return fail("history(missing) must be an empty list")

        # persistence: a fresh Ledger on the same path sees prior state; ids still not reused
        lg2 = Ledger(str(path))
        if lg2.balance(a) != 400 or lg2.balance(b) != 200:
            return fail("persistence broken: reloaded balances must be a=400, b=200")
        c = lg2.open_account("carol")
        if c != 3:
            return fail(f"id reused after reload: open_account gave {c}, expected 3")
    except Exception as e:  # noqa: BLE001
        return fail(f"Ledger raised during the sequence: {type(e).__name__}: {e}")
    return None


def check_cli() -> int | None:
    if not (WS / "cli.py").is_file():
        return fail("cli.py missing")
    # Clean cwd so ledger.json starts empty: a temp copy of cli.py + ledger.py.
    tmp = Path(tempfile.mkdtemp())
    for name in ("cli.py", "ledger.py"):
        (tmp / name).write_bytes((WS / name).read_bytes())

    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "cli.py", *args], cwd=tmp, capture_output=True, text=True, timeout=30
        )

    r = run("open", "alice")
    if r.returncode != 0 or "#1" not in r.stdout:
        return fail(f"`open alice`: rc={r.returncode} out={r.stdout!r} (expected '#1')")
    run("open", "bob")
    if run("deposit", "1", "1000").returncode != 0:
        return fail("`deposit 1 1000` should succeed (rc 0)")
    r = run("balance", "1")
    if r.returncode != 0 or "1000" not in r.stdout:
        return fail(f"`balance 1`: rc={r.returncode} out={r.stdout!r} (expected 1000)")
    r = run("transfer", "1", "2", "400")
    if r.returncode != 0 or "ok" not in r.stdout:
        return fail(f"`transfer 1 2 400`: rc={r.returncode} out={r.stdout!r}")
    if "600" not in run("balance", "1").stdout or "400" not in run("balance", "2").stdout:
        return fail("after transfer, balances must be 1->600, 2->400")
    # insufficient transfer fails on stderr with a nonzero code, changing nothing
    r = run("transfer", "1", "2", "999999")
    if r.returncode == 0 or "failed" not in (r.stdout + r.stderr):
        return fail(f"`transfer` over balance must fail: rc={r.returncode}")
    if "600" not in run("balance", "1").stdout:
        return fail("a failed transfer must leave the balance unchanged")
    # missing account -> 'no such account' on stderr, rc 1
    r = run("balance", "999")
    if r.returncode != 1 or "no such account" not in (r.stdout + r.stderr):
        return fail(f"`balance 999` must report 'no such account' (rc 1): rc={r.returncode}")
    # unknown subcommand -> usage on stderr, rc 2
    if run("frobnicate").returncode != 2:
        return fail("an unknown subcommand must exit 2")
    return None


def main() -> int:
    for check in (check_ledger, check_cli):
        rc = check()
        if rc is not None:
            return rc
    print("PASS: Ledger invariants + cli.py contract correct on the held-out sequence")
    return 0


if __name__ == "__main__":
    sys.exit(main())
