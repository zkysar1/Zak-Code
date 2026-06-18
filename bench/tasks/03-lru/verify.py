#!/usr/bin/env python
"""Held-out oracle for task 03-lru. Run with cwd = the agent's workspace.

Exercises the canonical LRU eviction sequence directly against the agent's LRUCache,
independent of whatever tests it wrote. Exits 0 on success, 1 on failure.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

WS = Path.cwd()


def fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def main() -> int:
    if not (WS / "lru.py").is_file():
        return fail("lru.py missing")
    sys.path.insert(0, str(WS))
    try:
        import lru

        importlib.reload(lru)
        LRUCache = lru.LRUCache
    except Exception as e:  # noqa: BLE001
        return fail(f"could not import lru.LRUCache: {type(e).__name__}: {e}")

    try:
        c = LRUCache(2)
        c.put(1, 1)
        c.put(2, 2)
        if c.get(1) != 1:
            return fail(f"get(1) after put(1,1)/put(2,2) = {c.get(1)}, expected 1")
        c.put(3, 3)  # evicts key 2 (LRU, since get(1) made 1 recent)
        if c.get(2) != -1:
            return fail(f"get(2) = {c.get(2)}, expected -1 (2 should be evicted)")
        c.put(4, 4)  # evicts key 1
        if c.get(1) != -1:
            return fail(f"get(1) = {c.get(1)}, expected -1 (1 should be evicted)")
        if c.get(3) != 3:
            return fail(f"get(3) = {c.get(3)}, expected 3")
        if c.get(4) != 4:
            return fail(f"get(4) = {c.get(4)}, expected 4")
        # update-as-use: updating an existing key must refresh recency
        c2 = LRUCache(2)
        c2.put(1, 1)
        c2.put(2, 2)
        c2.put(1, 10)  # update 1 -> now 2 is LRU
        c2.put(3, 3)  # evicts 2
        if c2.get(2) != -1:
            return fail(f"update-as-use failed: get(2) = {c2.get(2)}, expected -1")
        if c2.get(1) != 10:
            return fail(f"get(1) after update = {c2.get(1)}, expected 10")
    except Exception as e:  # noqa: BLE001
        return fail(f"LRUCache raised during the sequence: {type(e).__name__}: {e}")

    print("PASS: LRU eviction + update-as-use correct on the held-out sequence")
    return 0


if __name__ == "__main__":
    sys.exit(main())
