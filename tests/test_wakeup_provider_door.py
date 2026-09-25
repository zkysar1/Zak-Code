"""ADR-0250, the door half: a due autonomous-loop sentinel is held, not fired, when the
provider would refuse the turn it opens.

Measured 2026-09-25 (three worker Bodies, a 12-hour pod outage): each sentinel that fired into
the powered-off pod bought a compaction whose summarizer call failed, four 900-second retry
budgets and a ``veto_stall`` -- and the repeat guard then cancelled the net on the second
identical cycle. The door now asks the provider one cheap ``GET /models`` first and, when the
base does not answer, puts the sentinel back at the provider backoff and says so in one line.

Two halves, each with its positive control: the door function with a fake provider (held vs
fired vs no probe at all), and ``LiteLLMProvider.unreachable`` against a closed loopback port,
a live loopback listing, no base, a named-provider model, and a base ``local_only`` refuses.
Hermetic: loopback only, no clock dependence.
"""

from __future__ import annotations

import io
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
from typing import Any

from rich.console import Console

from zakcode.cli import _hold_sentinel_if_provider_down
from zakcode.providers.litellm_provider import LiteLLMProvider
from zakcode.session.store import Session
from zakcode.wakeup import LOOP_SENTINEL, WakeupSlot

MODEL = "fake/scripted"
_NO_KEY = "offline-test-key"


# ── the door function ────────────────────────────────────────────────────────────


class _Probe:
    def __init__(self, reason: str | None) -> None:
        self.reason = reason
        self.asked = 0

    def unreachable(self) -> str | None:
        self.asked += 1
        return self.reason


def _door(provider: Any) -> tuple[Session, WakeupSlot, Any, Console]:
    """A session whose due sentinel the door has just taken, an agent around ``provider``."""
    session = Session(cwd=".", model=MODEL)
    slot = WakeupSlot(session)
    slot.arm(LOOP_SENTINEL, 60)
    assert slot.take_due_prompt(now=time.time() + 120) == LOOP_SENTINEL
    agent = SimpleNamespace(loop=SimpleNamespace(provider=provider), session=session)
    return session, slot, agent, Console(file=io.StringIO(), width=200)


def test_a_due_sentinel_is_held_when_the_provider_does_not_answer() -> None:
    session, slot, agent, console = _door(
        _Probe("http://pod:9090/v1 did not answer GET /models (URLError)")
    )
    assert _hold_sentinel_if_provider_down(console, agent, slot) is True
    held = slot.pending()
    assert held is not None and held.prompt == LOOP_SENTINEL and held.delay_seconds == 600
    assert (
        session.sentinel_turn_open is False
    )  # no turn opened, so none is marked as a sentinel turn
    assert session.sentinel_provider_repeats == 1
    out = console.file.getvalue()  # type: ignore[attr-defined]
    assert "wake-up held" in out and "600s" in out


def test_a_due_sentinel_fires_when_the_provider_answers() -> None:
    """THE POSITIVE CONTROL: the same door, a base that answers -- nothing is held, the turn
    opens as a sentinel turn exactly as before this ADR."""
    session, slot, agent, console = _door(_Probe(None))
    assert _hold_sentinel_if_provider_down(console, agent, slot) is False
    assert slot.pending() is None
    assert session.sentinel_turn_open is True
    assert console.file.getvalue() == ""  # type: ignore[attr-defined]


def test_a_provider_without_a_probe_lets_the_sentinel_fire() -> None:
    """A stand-in provider (or an older one) has no ``unreachable``; the door does not guess."""
    session, slot, agent, console = _door(object())
    assert _hold_sentinel_if_provider_down(console, agent, slot) is False
    assert slot.pending() is None and session.sentinel_turn_open is True


def test_the_hold_backs_off_across_consecutive_held_firings() -> None:
    session, slot, agent, console = _door(_Probe("dead"))
    delays = []
    for _ in range(4):
        assert _hold_sentinel_if_provider_down(console, agent, slot) is True
        held = slot.pending()
        assert held is not None
        delays.append(held.delay_seconds)
        slot.take_due_prompt(now=time.time() + 4000)  # the next firing, taken by the door again
    assert delays == [600, 1200, 2400, 3600]


# ── the provider probe ───────────────────────────────────────────────────────────


class _Listing(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 — http.server's name
        body = json.dumps({"data": [{"id": "stub"}]}).encode()
        self.send_response(200 if self.path.endswith("/models") else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:  # quiet
        return


def _closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_a_closed_port_reads_as_unreachable_and_a_live_listing_as_reachable() -> None:
    dead = LiteLLMProvider(
        model="openai/stub",
        api_base=f"http://127.0.0.1:{_closed_port()}/v1",
        api_key=_NO_KEY,
        context_window=8192,
    )
    reason = dead.unreachable(timeout=2.0)
    assert reason is not None and "did not answer GET /models" in reason

    server = HTTPServer(("127.0.0.1", 0), _Listing)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        live = LiteLLMProvider(
            model="openai/stub",
            api_base=f"http://127.0.0.1:{server.server_port}/v1",
            api_key=_NO_KEY,
            context_window=8192,
        )
        assert live.unreachable(timeout=2.0) is None  # the positive control
    finally:
        server.shutdown()
        server.server_close()


def test_nothing_to_ask_is_not_unreachable() -> None:
    """No base, a named provider's model, or a base ``local_only`` would refuse: the probe has
    no standing and says nothing -- a hold here would park a loop whose calls are fine."""
    no_base = LiteLLMProvider(model="openai/stub", api_key=_NO_KEY, context_window=8192)
    assert no_base.unreachable(timeout=1.0) is None

    named = LiteLLMProvider(
        model="anthropic/stub",
        api_base=f"http://127.0.0.1:{_closed_port()}/v1",
        api_key=_NO_KEY,
        context_window=8192,
    )
    assert named.unreachable(timeout=1.0) is None

    refused = LiteLLMProvider(
        model="openai/stub",
        api_base=f"http://127.0.0.1:{_closed_port()}/v1",
        api_key=_NO_KEY,
        context_window=8192,
        local_only=True,
        local_api_bases=["http://10.0.0.1:9090/v1"],
    )
    assert refused.unreachable(timeout=1.0) is None
