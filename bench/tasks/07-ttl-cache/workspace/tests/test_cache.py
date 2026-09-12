"""Basic coverage. These pass for a naive implementation -- they are not the whole contract."""
import pytest

from cache import Cache


class Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t
    def advance(self, d): self.t += d


def test_put_get():
    c = Cache(capacity=2, ttl_seconds=10, now=Clock())
    c.put("a", 1)
    assert c.get("a") == 1
    assert c.get("missing") is None


def test_evicts_when_full():
    c = Cache(capacity=2, ttl_seconds=100, now=Clock())
    c.put("a", 1); c.put("b", 2); c.put("c", 3)
    assert c.get("a") is None
    assert c.get("b") == 2 and c.get("c") == 3


def test_expiry():
    clk = Clock()
    c = Cache(capacity=2, ttl_seconds=10, now=clk)
    c.put("a", 1)
    clk.advance(11)
    assert c.get("a") is None


def test_update_value():
    c = Cache(capacity=2, ttl_seconds=100, now=Clock())
    c.put("a", 1); c.put("a", 2)
    assert c.get("a") == 2
    assert len(c) == 1
