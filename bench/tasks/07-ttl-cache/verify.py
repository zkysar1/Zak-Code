#!/usr/bin/env python
"""Held-out oracle for 07-ttl-cache. Run with cwd = the agent's workspace.

WHY THIS TASK EXISTS. ADR-0151's addendum measured that difficulty on the DISCOVERABILITY axis
(hidden-but-findable requirements) costs turns and not correctness -- Claude Code passed
06-plugin-conventions at 2.7x the effort. So this task moves to the other axis: every rule is
stated explicitly in the prompt, and the difficulty is that the rules INTERACT. The naive
implementation satisfies each rule read alone and violates them read together, and the workspace's
visible tests are deliberately basic enough to pass for such an implementation.

The four interaction traps, each independently checked below:
  T1  a `get` on an EXPIRED entry must not bump recency (rule 2 vs rules 3+5)
  T2  expired entries must be dropped BEFORE evicting a live LRU (rule 4 vs rule 5)
  T3  `put` on an existing key must RESET the TTL, not keep the original expiry (rule 1)
  T4  `len()` must count live entries only, not the internal dict (rule 6)

Exits 0 on success, 1 on the first failure.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

WS = Path.cwd()


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, d):
        self.t += d


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    if not (WS / "cache.py").is_file():
        fail("cache.py does not exist")
    sys.path.insert(0, str(WS))
    sys.modules.pop("cache", None)
    Cache = importlib.import_module("cache").Cache

    # --- baseline: the visible contract must still hold -------------------------------------
    clk = Clock()
    c = Cache(capacity=2, ttl_seconds=10, now=clk)
    c.put("a", 1)
    if c.get("a") != 1:
        fail("baseline: put/get does not round-trip")
    if c.get("nope") is not None:
        fail("baseline: get of a missing key must be None")

    # --- T3: put on an existing key RESETS the TTL ------------------------------------------
    clk = Clock()
    c = Cache(capacity=4, ttl_seconds=10, now=clk)
    c.put("a", 1)
    clk.advance(8)
    c.put("a", 2)          # refresh: TTL restarts here
    clk.advance(5)         # 13s since first write, 5s since the refresh -> still LIVE
    if c.get("a") != 2:
        fail("T3: put() on an existing key must RESET the TTL clock; the entry expired 13s after "
             "the FIRST write even though it was rewritten 5s ago (rule 1)")

    # --- T4: len() counts LIVE entries only -------------------------------------------------
    clk = Clock()
    c = Cache(capacity=4, ttl_seconds=10, now=clk)
    c.put("a", 1)
    c.put("b", 2)
    clk.advance(11)        # both expired, neither touched since
    if len(c) != 0:
        fail(f"T4: len() returned {len(c)} with every entry expired; it must count LIVE entries "
             f"only, not the internal dict (rule 6)")
    c.put("c", 3)
    if len(c) != 1:
        fail(f"T4: len() returned {len(c)} with one live entry after the expired ones (rule 6)")

    # --- T1: a get on an EXPIRED entry must not bump recency --------------------------------
    # a and b live; c expires. Touching c must not make b the LRU.
    clk = Clock()
    c = Cache(capacity=3, ttl_seconds=10, now=clk)
    c.put("x", 1)          # oldest
    clk.advance(1)
    c.put("y", 2)
    clk.advance(1)
    c.put("z", 3)
    clk.advance(9)         # x is now 11s old -> EXPIRED; y is 10s -> EXPIRED; z is 9s -> live
    if c.get("x") is not None:
        fail("T1 setup: x should be expired")
    # x and y are gone; z alone is live. Insert two more; nothing live should be evicted.
    c.put("p", 4)
    c.put("q", 5)
    if c.get("z") != 3:
        fail("T1/T2: 'z' was live and within capacity once the expired entries were dropped, "
             "but it was evicted -- expired entries must be reclaimed before a live entry is "
             "evicted (rules 2, 4, 5)")

    # --- T2: expired entries are dropped BEFORE a live LRU is evicted -----------------------
    clk = Clock()
    c = Cache(capacity=3, ttl_seconds=10, now=clk)
    c.put("old", 1)        # will expire
    clk.advance(9)
    c.put("mid", 2)        # live
    c.put("new", 3)        # live
    clk.advance(2)         # old is 11s -> expired; mid/new are 2s -> live
    c.put("extra", 4)      # needs room: must drop 'old', NOT evict 'mid'
    if c.get("mid") != 2:
        fail("T2: inserting into a full cache evicted the live LRU ('mid') while an EXPIRED entry "
             "('old') was still occupying capacity; rule 4 requires dropping expired entries "
             "first")
    if c.get("new") != 3 or c.get("extra") != 4:
        fail("T2: 'new' and 'extra' should both be present after the expired entry was reclaimed")

    # --- rule 5 + rule 7: eviction order and keys() ------------------------------------------
    clk = Clock()
    c = Cache(capacity=3, ttl_seconds=1000, now=clk)
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)
    c.get("a")             # a becomes MRU -> LRU order is b, c, a
    if list(c.keys()) != ["b", "c", "a"]:
        fail(f"rule 7: keys() returned {list(c.keys())}; expected ['b','c','a'] "
             f"(least-recently-used first, and a live get() makes an entry most-recent)")
    c.put("d", 4)          # evicts b, the live LRU
    if c.get("b") is not None:
        fail("rule 5: 'b' was the least-recently-used live entry and should have been evicted")
    if c.get("c") != 3 or c.get("a") != 1 or c.get("d") != 4:
        fail("rule 5: the wrong entry was evicted")

    print("PASS: TTL reset on update, live-only len(), expired gets do not bump recency, "
          "expired entries reclaimed before live eviction, and LRU order correct")


if __name__ == "__main__":
    main()
