"""The per-Mind spend gate: a Mind that has spent its cap sends no model call.

What is pinned here, each with its positive control:

* the provider asks the gate before BOTH dispatch paths (``acomplete`` and ``astream``), so
  at the cap the wire is never reached, and below it the same fake is reached exactly once;
* an unarmed gate (no budget pushed) changes nothing, which is what lets Zak-Code ship to
  every sidecar before any vessel pushes a budget;
* a call's cost is counted from what the provider recorded, an unpriced one is counted as
  unpriced and never guessed, and the soft trip at ``cap - reserve`` fires exactly once;
* the ledger survives a new process, and a ledger that cannot be read fails CLOSED;
* ``POST /budget`` is bearer-gated, refuses bad values, arms the gate, shows in
  ``/sidecar/health``, and a soft trip starts the run's ending with ``mind_budget_cap``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import zakcode.providers.litellm_provider as lp
from zakcode.messages import Message
from zakcode.providers.base import QuotaExhausted
from zakcode.providers.litellm_provider import LiteLLMProvider
from zakcode.providers.spend_gate import (
    STOP_REASON,
    SpendCapReached,
    SpendGate,
    get_gate,
    reset_gate_for_tests,
)
from zakcode.usage import Usage

ROUTE = "openai/route-a"
GATEWAY = "https://gateway.invalid/v1"
_MSGS = [Message.user("hello")]
TOKEN = "spend-gate-test-token"


@pytest.fixture(autouse=True)
def _fresh_gate(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Every test starts with a new, unarmed process-wide gate and no request pacing."""
    monkeypatch.setattr(lp, "_MIN_REQUEST_INTERVAL_S", 0.0)
    reset_gate_for_tests(SpendGate())
    yield
    reset_gate_for_tests(None)


def _provider() -> LiteLLMProvider:
    return LiteLLMProvider(model=ROUTE, api_base=GATEWAY, api_key="sk-test", context_window=131_072)


def _response(cost: float | None) -> SimpleNamespace:
    """One buffered completion; ``cost`` is what litellm priced it at (None: not priced)."""
    return SimpleNamespace(
        model="route-a",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="ok", reasoning_content=None, tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1, total_tokens=4),
        _hidden_params={"response_cost": cost},
    )


class _Stream:
    """A two-chunk stream that reports no usage, as some gateways do."""

    _hidden_params: dict[str, Any] = {}

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._chunks()

    async def _chunks(self) -> AsyncIterator[Any]:
        for content, finish in (("o", None), ("k", "stop")):
            delta = SimpleNamespace(content=content, tool_calls=None)
            yield SimpleNamespace(
                model="route-a",
                choices=[SimpleNamespace(delta=delta, finish_reason=finish)],
                usage=None,
            )


class _Wire:
    """A counting fake of ``litellm.acompletion``: how many requests reached the wire."""

    def __init__(self, answer: Any) -> None:
        self.answer = answer
        self.calls = 0

    async def __call__(self, **_kwargs: Any) -> Any:
        self.calls += 1
        return self.answer


# ── the provider asks the gate before each dispatch ──────────────────────────────────────


async def test_at_the_cap_acomplete_refuses_before_the_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wire = _Wire(_response(0.01))
    monkeypatch.setattr(lp.litellm, "acompletion", wire)
    get_gate().set_budget(cap_usd=1.0, reserve_usd=0.0, floor_usd=1.0)

    with pytest.raises(SpendCapReached) as refused:
        await _provider().acomplete(_MSGS)

    assert wire.calls == 0
    assert isinstance(refused.value, QuotaExhausted), "the loop ends the turn on a QuotaExhausted"
    assert get_gate().snapshot()["refused_calls"] == 1


async def test_below_the_cap_acomplete_reaches_the_wire_once_and_is_charged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The positive control for the refusal above: the same fake, with room under the cap.
    wire = _Wire(_response(0.25))
    monkeypatch.setattr(lp.litellm, "acompletion", wire)
    get_gate().set_budget(cap_usd=10.0, reserve_usd=0.0, floor_usd=1.0)

    await _provider().acomplete(_MSGS)

    assert wire.calls == 1
    assert get_gate().snapshot()["spent_usd"] == pytest.approx(1.25)


async def test_at_the_cap_astream_refuses_before_the_wire(monkeypatch: pytest.MonkeyPatch) -> None:
    wire = _Wire(_Stream())
    monkeypatch.setattr(lp.litellm, "acompletion", wire)
    get_gate().set_budget(cap_usd=1.0, reserve_usd=0.0, floor_usd=1.5)

    events = _provider().astream(_MSGS)
    with pytest.raises(SpendCapReached):
        await events.__anext__()

    assert wire.calls == 0


async def test_below_the_cap_astream_reaches_the_wire_and_an_unreported_usage_is_unpriced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wire = _Wire(_Stream())
    monkeypatch.setattr(lp.litellm, "acompletion", wire)
    get_gate().set_budget(cap_usd=10.0, reserve_usd=0.0, floor_usd=0.0)

    async for _ in _provider().astream(_MSGS):
        pass

    assert wire.calls == 1
    snap = get_gate().snapshot()
    assert snap["unpriced_calls"] == 1, "a stream with no usage is counted, never guessed"
    assert snap["spent_usd"] == 0.0


async def test_an_unarmed_gate_changes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    wire = _Wire(_response(99.0))
    monkeypatch.setattr(lp.litellm, "acompletion", wire)

    for _ in range(3):
        await _provider().acomplete(_MSGS)

    assert wire.calls == 3
    snap = get_gate().snapshot()
    assert snap["armed"] is False and snap["spent_usd"] == 0.0 and snap["refused_calls"] == 0


# ── counting, the soft trip, and the ledger ──────────────────────────────────────────────


def test_soft_trip_fires_once_at_cap_less_reserve_and_the_hard_cap_refuses() -> None:
    gate = SpendGate()
    fired: list[int] = []
    gate.set_soft_trip_hook(lambda: fired.append(1))
    gate.set_budget(cap_usd=1.0, reserve_usd=0.2, floor_usd=0.0)

    gate.charge(Usage(cost_usd=0.79))
    assert fired == [] and gate.snapshot()["soft_tripped"] is False
    gate.charge(Usage(cost_usd=0.01))
    assert fired == [1], "spend reached cap - reserve: the ending starts"
    gate.admit()  # inside the reserve band the ending may still call the model
    gate.charge(Usage(cost_usd=0.15))
    assert fired == [1], "the soft trip fires once"
    gate.charge(Usage(cost_usd=0.05))
    with pytest.raises(SpendCapReached):
        gate.admit()
    assert gate.snapshot()["tripped"] is True


def test_a_bad_or_unpriced_cost_is_counted_as_unpriced_and_never_guessed() -> None:
    gate = SpendGate()
    gate.set_budget(cap_usd=5.0, reserve_usd=0.0, floor_usd=0.0)
    gate.charge(Usage(cost_usd=0.0, cost_unpriced=True))
    gate.charge(SimpleNamespace(cost_usd=float("nan"), cost_unpriced=False))
    gate.charge(SimpleNamespace(cost_usd=-1.0, cost_unpriced=False))
    gate.charge(None)
    gate.charge(Usage(cost_usd=0.5))
    snap = gate.snapshot()
    assert snap["unpriced_calls"] == 4
    assert snap["spent_usd"] == pytest.approx(0.5)


def test_the_floor_only_raises_spend_and_a_lower_floor_is_a_new_period() -> None:
    gate = SpendGate()
    gate.set_budget(cap_usd=10.0, reserve_usd=0.0, floor_usd=2.0)
    gate.charge(Usage(cost_usd=1.0))
    assert gate.snapshot()["spent_usd"] == pytest.approx(3.0)
    gate.set_budget(cap_usd=10.0, reserve_usd=0.0, floor_usd=2.5)  # the row under-reads
    assert gate.snapshot()["spent_usd"] == pytest.approx(3.0)
    gate.set_budget(cap_usd=10.0, reserve_usd=0.0, floor_usd=4.0)  # the row caught up and more
    assert gate.snapshot()["spent_usd"] == pytest.approx(4.0)
    gate.set_budget(cap_usd=10.0, reserve_usd=0.0, floor_usd=0.5)  # the row's period reset
    assert gate.snapshot()["spent_usd"] == pytest.approx(0.5)


def test_set_budget_refuses_values_that_are_not_money() -> None:
    gate = SpendGate()
    for cap, reserve, floor in (
        (-1, 0, 0),
        (1, 2, 0),
        (float("nan"), 0, 0),
        (1, 0, "x"),
        (True, 0, 0),
    ):
        with pytest.raises(ValueError):
            gate.set_budget(cap_usd=cap, reserve_usd=reserve, floor_usd=floor)
    assert gate.snapshot()["armed"] is False


def test_the_ledger_survives_a_new_process(tmp_path: Path) -> None:
    ledger = tmp_path / "run" / "spend-ledger.json"
    first = SpendGate(ledger)
    first.set_budget(cap_usd=2.0, reserve_usd=0.5, floor_usd=1.0)
    first.charge(Usage(cost_usd=0.6))

    second = SpendGate(ledger)
    snap = second.snapshot()
    assert snap["armed"] is True
    assert snap["spent_usd"] == pytest.approx(1.6)
    assert snap["soft_tripped"] is True
    second.charge(Usage(cost_usd=0.4))
    with pytest.raises(SpendCapReached):
        second.admit()


def test_an_unreadable_ledger_fails_closed_until_a_push_rearms(tmp_path: Path) -> None:
    ledger = tmp_path / "spend-ledger.json"
    ledger.write_text("{not json", encoding="utf-8")

    gate = SpendGate(ledger)
    with pytest.raises(SpendCapReached):
        gate.admit()
    snap = gate.snapshot()
    assert snap["ledger_ok"] is False and snap["tripped"] is True

    gate.set_budget(cap_usd=3.0, reserve_usd=0.0, floor_usd=1.0)
    gate.admit()  # re-armed from the vessel's push: room again
    assert gate.snapshot()["ledger_ok"] is True


def test_no_ledger_file_means_an_unarmed_gate(tmp_path: Path) -> None:
    gate = SpendGate(tmp_path / "absent.json")
    gate.admit()
    assert gate.snapshot()["armed"] is False


# ── POST /budget and /sidecar/health ─────────────────────────────────────────────────────


def _app(tmp_path: Path) -> Any:
    pytest.importorskip("fastapi")
    from zakcode.config import Settings
    from zakcode.server.app import create_app
    from zakcode.session.store import SessionStore

    settings = Settings(
        default_model="scripted/test",
        context_window=8192,
        workspace_root=tmp_path,
        auth_token=TOKEN,
        run_stop_message="Time to stop: write your digest.",
    )
    return create_app(
        settings=settings,
        store=SessionStore(base_dir=tmp_path / "sessions"),
        agent_factory=lambda s, m, p: None,
    )


def _client(tmp_path: Path) -> Any:
    from fastapi.testclient import TestClient

    return TestClient(_app(tmp_path), raise_server_exceptions=False)


async def _in_loop_stop_reason_after(tmp_path: Path, budget: dict[str, float]) -> Any:
    """Push ``budget``, let any task it started run, then ask the run to stop.

    In-loop on purpose: the soft trip starts the ending as a task on the server's own loop, as
    it does under uvicorn, so the test drives the app on the test's loop (httpx's ASGI
    transport) and yields to it before the second request.
    """
    import asyncio

    import httpx

    transport = httpx.ASGITransport(app=_app(tmp_path))
    async with httpx.AsyncClient(transport=transport, base_url="http://sidecar.test") as client:
        pushed = await client.post("/budget", json=budget, headers=_auth())
        assert pushed.status_code == 200
        for _ in range(5):
            await asyncio.sleep(0)
        stop = await client.post("/run/stop", json={"reason": "stopped"}, headers=_auth())
        return stop.json()


def _auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def test_post_budget_is_bearer_gated(tmp_path: Path) -> None:
    client = _client(tmp_path)
    body = {"cap_usd": 5.0, "reserve_usd": 1.0, "floor_usd": 0.0}
    assert client.post("/budget", json=body).status_code == 401
    assert (
        client.post("/budget", json=body, headers={"Authorization": "Bearer wrong"}).status_code
        == 401
    )
    assert get_gate().snapshot()["armed"] is False

    ok = client.post("/budget", json=body, headers=_auth())
    assert ok.status_code == 200
    assert ok.json()["spend_gate"]["armed"] is True


def test_post_budget_refuses_a_reserve_above_the_cap(tmp_path: Path) -> None:
    client = _client(tmp_path)
    bad = client.post("/budget", json={"cap_usd": 1.0, "reserve_usd": 2.0}, headers=_auth())
    assert bad.status_code == 400
    assert get_gate().snapshot()["armed"] is False


def test_sidecar_health_reports_the_gate(tmp_path: Path) -> None:
    client = _client(tmp_path)
    before = client.get("/sidecar/health", headers=_auth()).json()["spend_gate"]
    assert before["armed"] is False
    client.post(
        "/budget", json={"cap_usd": 4.0, "reserve_usd": 1.0, "floor_usd": 1.5}, headers=_auth()
    )
    after = client.get("/sidecar/health", headers=_auth()).json()["spend_gate"]
    assert after["armed"] is True and after["cap_usd"] == 4.0 and after["spent_usd"] == 1.5


async def test_a_soft_trip_starts_the_runs_ending_with_mind_budget_cap(tmp_path: Path) -> None:
    # A push whose floor is already past cap - reserve trips the soft stop at once; a later
    # stop request then reports the reason the run is already ending with.
    later = await _in_loop_stop_reason_after(
        tmp_path, {"cap_usd": 4.0, "reserve_usd": 1.0, "floor_usd": 3.5}
    )
    assert later == {"stopping": True, "reason": STOP_REASON}


async def test_without_a_soft_trip_a_stop_keeps_its_own_reason(tmp_path: Path) -> None:
    # The positive control for the test above: the same sequence, with spend below the band.
    later = await _in_loop_stop_reason_after(
        tmp_path, {"cap_usd": 4.0, "reserve_usd": 1.0, "floor_usd": 0.5}
    )
    assert later == {"stopping": True, "reason": "stopped"}
