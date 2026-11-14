"""The per-Mind spend gate: a Mind that has spent its cap sends no model call.

What is pinned here, each with its positive control:

* the provider asks the gate before BOTH dispatch paths (``acomplete`` and ``astream``), so
  at the cap the wire is never reached, and below it the same fake is reached exactly once;
* an unarmed gate (no budget pushed) changes nothing, which is what lets Zak-Code ship to
  every sidecar before any vessel pushes a budget;
* a call's cost is counted from what the provider recorded, an unpriced one is counted as
  unpriced and never guessed, and the soft trip at ``cap - reserve`` fires exactly once;
* the ledger survives a new process, and a ledger that cannot be read fails CLOSED;
* a refusal against a cap only the ledger holds waits, off the event loop and under one bound,
  for the vessel's first push and is then decided against the cap the gate holds, on both
  dispatch paths; every other call (unarmed, under the cap, already pushed to, unreadable ledger)
  is decided at once;
* ``POST /budget`` is bearer-gated, refuses bad values, arms the gate, shows in
  ``/sidecar/health``, and a soft trip starts the run's ending with ``mind_budget_cap``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import zakcode.providers.litellm_provider as lp
import zakcode.providers.spend_gate as sg
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


def test_an_armed_ledger_at_an_old_cap_takes_a_push_that_raises_it(tmp_path: Path) -> None:
    """A Mind's ledger from an earlier run is armed at that run's cap, and the row was raised since.

    The gate boots armed, refuses at the old cap, and says ``armed`` in ``/sidecar/health``, so a
    vessel that pushes only a gate that says unarmed never reaches it. This pins the contract such
    a vessel relies on: a push to an armed gate is accepted, the vessel's push of the row's cap and
    floor is what admits it again, and the new cap is in the ledger for the next process.
    """
    ledger = tmp_path / "run" / "spend-ledger.json"
    earlier = SpendGate(ledger)
    earlier.set_budget(cap_usd=15.0, reserve_usd=4.32, floor_usd=15.95)  # accrued past its cap

    gate = SpendGate(ledger)  # a new process: it boots from what the earlier run left
    snap = gate.snapshot()
    assert snap["armed"] is True and snap["ledger_ok"] is True
    assert snap["cap_usd"] == 15.0 and snap["tripped"] is True
    with pytest.raises(SpendCapReached):
        gate.admit()  # refused at the old cap, whatever the row says now

    pushed = gate.set_budget(cap_usd=21.0, reserve_usd=4.32, floor_usd=15.95)  # the row, as raised
    assert pushed["cap_usd"] == 21.0 and pushed["tripped"] is False
    assert pushed["spent_usd"] == pytest.approx(15.95)
    gate.admit()  # room again
    assert SpendGate(ledger).snapshot()["cap_usd"] == 21.0


def test_a_push_that_raises_the_cap_past_spend_starts_the_ending_at_the_new_line(
    tmp_path: Path,
) -> None:
    """The soft trip an earlier cap left in the ledger must not silence the new cap's.

    The earlier run ended past its cap, so its ledger holds a set soft trip. The row is raised
    and pushed to the gate that booted from that ledger. Spend is now short of the new
    ``cap - reserve``, so the trip is cleared (and the clear is written to the ledger), and the
    ending starts once, at the new line.
    """
    ledger = tmp_path / "run" / "spend-ledger.json"
    earlier = SpendGate(ledger)
    earlier.set_budget(cap_usd=15.0, reserve_usd=4.32, floor_usd=15.95)
    assert earlier.snapshot()["soft_tripped"] is True

    gate = SpendGate(ledger)  # a new process: it boots holding the earlier cap's trip
    fired: list[int] = []
    gate.set_soft_trip_hook(lambda: fired.append(1))
    assert gate.snapshot()["soft_tripped"] is True

    pushed = gate.set_budget(cap_usd=22.0, reserve_usd=4.32, floor_usd=15.95)  # the row, as raised
    assert pushed["soft_tripped"] is False and pushed["tripped"] is False
    assert fired == [], "a push that raises the cap past spend starts no ending"
    assert SpendGate(ledger).snapshot()["soft_tripped"] is False, "the clear is in the ledger"

    gate.charge(Usage(cost_usd=1.0))  # 16.95: short of the new line, 22 - 4.32 = 17.68
    assert fired == []
    gate.charge(Usage(cost_usd=0.75))  # 17.70: past it
    assert fired == [1] and gate.snapshot()["soft_tripped"] is True
    gate.charge(Usage(cost_usd=0.1))
    assert fired == [1], "the ending starts once"


def test_a_push_that_leaves_spend_past_the_soft_line_keeps_the_trip() -> None:
    """The clear is by value: a push that does not put spend back under the line changes nothing.

    The vessel pushes the row on every flush, and a raise of the cap can be smaller than the spend
    past the line. Neither is a new trip, and neither clears the one that fired.
    """
    gate = SpendGate()
    fired: list[int] = []
    gate.set_soft_trip_hook(lambda: fired.append(1))
    gate.set_budget(cap_usd=15.0, reserve_usd=4.32, floor_usd=15.95)  # past the line: it fires
    assert fired == [1]

    gate.set_budget(cap_usd=15.0, reserve_usd=4.32, floor_usd=15.95)  # the same row, pushed again
    pushed = gate.set_budget(cap_usd=20.0, reserve_usd=4.32, floor_usd=15.95)  # line 15.68 < 15.95
    assert pushed["soft_tripped"] is True
    gate.charge(Usage(cost_usd=1.0))
    assert fired == [1], "spend is still past the line, so the trip stays set and fires once"


def test_no_ledger_file_means_an_unarmed_gate(tmp_path: Path) -> None:
    gate = SpendGate(tmp_path / "absent.json")
    gate.admit()
    assert gate.snapshot()["armed"] is False


# ── the first push: a refusal against a cap only the ledger holds waits for it ───────────

# The row as raised since an earlier run wrote its ledger at a $15.00 cap with $15.95 spent.
_RAISED = {"cap_usd": 22.0, "reserve_usd": 4.32, "floor_usd": 15.95}
_SPEND_LOG = sg.__name__


def _ledger_gate(tmp_path: Path, *, floor: float = 15.95) -> SpendGate:
    """A gate a new process booted from the ledger an earlier run left: armed at a $15.00 cap.

    With the default floor spend is past the cap, so ``admit`` would refuse; with a lower one
    there is room. No push has reached this gate yet, which is what makes its cap possibly stale.
    """
    ledger = tmp_path / "run" / "spend-ledger.json"
    SpendGate(ledger).set_budget(cap_usd=15.0, reserve_usd=4.32, floor_usd=floor)
    return SpendGate(ledger)


def _spy_on_the_wait(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Swap the first-push wait for an instant one that records each entry into it.

    For a test that asserts a call does NOT wait: with the real wait out of the way, the mutation
    that sends the call into it goes red at once, instead of sitting out the bound.
    """
    entered: list[float] = []

    async def instant(_gate: SpendGate, bound_s: float) -> None:
        entered.append(bound_s)

    monkeypatch.setattr(SpendGate, "_wait_for_first_push", instant)
    return entered


def _spend_log(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == _SPEND_LOG]


@pytest.mark.parametrize("route", ["on the loop", "from another thread"])
async def test_a_refusal_against_a_ledger_only_cap_waits_for_the_first_push_and_is_admitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    route: str,
) -> None:
    """The incident at unit scale: the ledger holds $15.00, the row is $22.00, no push yet.

    The call this gate would refuse is held, not refused. The push admits it, taken on the loop
    (as ``POST /budget`` is) or from another thread, and a push the gate itself refuses (bad
    values) is not a push, so it releases nothing. The bound is far longer than the test: a call
    the push does not release sits out the 2 s allowance below and fails.
    """
    monkeypatch.setattr(sg, "FIRST_PUSH_WAIT_S", 60.0)
    caplog.set_level(logging.INFO, logger=_SPEND_LOG)
    gate = _ledger_gate(tmp_path)

    call = asyncio.create_task(gate.admit_async())
    await asyncio.sleep(0.3)
    assert not call.done(), "a refusal against a cap only the ledger holds waits for the push"
    assert gate.snapshot()["refused_calls"] == 0, "a call still waiting is not a refusal yet"

    with pytest.raises(ValueError):
        gate.set_budget(cap_usd=-1, reserve_usd=0, floor_usd=0)
    await asyncio.sleep(0.3)
    assert not call.done(), "a push the gate refuses does not release the wait"

    if route == "from another thread":
        await asyncio.to_thread(gate.set_budget, **_RAISED)
    else:
        gate.set_budget(**_RAISED)
    await asyncio.wait_for(call, timeout=2.0)  # admitted: it returns, it does not raise

    snap = gate.snapshot()
    assert snap["cap_usd"] == 22.0 and snap["refused_calls"] == 0
    lines = _spend_log(caplog)
    assert sum("holding a model call up to 60 s" in m for m in lines) == 1, lines
    assert sum("first push arrived after" in m for m in lines) == 1, lines


async def test_no_push_within_the_bound_refuses_as_before_and_counts_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(sg, "FIRST_PUSH_WAIT_S", 0.4)
    caplog.set_level(logging.INFO, logger=_SPEND_LOG)
    gate = _ledger_gate(tmp_path)

    started = time.monotonic()
    with pytest.raises(SpendCapReached) as refused:
        await gate.admit_async()
    waited = time.monotonic() - started

    assert 0.3 <= waited < 5.0, "it waited out the bound, and no longer, before it refused"
    assert "spent $15.95 of its $15.00 cap" in str(refused.value), "the refusal is admit()'s own"
    assert gate.snapshot()["refused_calls"] == 1
    lines = _spend_log(caplog)
    assert sum("no first push from the vessel in" in m for m in lines) == 1, lines


async def test_a_push_that_confirms_the_cap_ends_the_wait_with_a_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The push ends the wait; it does not decide the call. The gate holds a cap now, and a row
    that really is spent refuses the call exactly as ``admit`` always did."""
    monkeypatch.setattr(sg, "FIRST_PUSH_WAIT_S", 60.0)
    caplog.set_level(logging.INFO, logger=_SPEND_LOG)
    gate = _ledger_gate(tmp_path)

    call = asyncio.create_task(gate.admit_async())
    await asyncio.sleep(0.3)
    assert not call.done(), "the refusal is held for the push"

    gate.set_budget(cap_usd=15.0, reserve_usd=4.32, floor_usd=15.95)  # the row still says $15.00
    with pytest.raises(SpendCapReached) as refused:
        await asyncio.wait_for(call, timeout=2.0)

    assert "spent $15.95 of its $15.00 cap" in str(refused.value), "the refusal is admit()'s own"
    assert gate.snapshot()["refused_calls"] == 1
    assert sum("first push arrived after" in m for m in _spend_log(caplog)) == 1


async def test_a_gate_the_vessel_has_pushed_to_refuses_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sg, "FIRST_PUSH_WAIT_S", 60.0)
    entered = _spy_on_the_wait(monkeypatch)
    gate = _ledger_gate(tmp_path)
    gate.set_budget(cap_usd=15.0, reserve_usd=4.32, floor_usd=15.95)  # a push: the cap stands

    with pytest.raises(SpendCapReached):
        await asyncio.wait_for(gate.admit_async(), timeout=1.0)

    assert entered == [], "a gate this process has taken a push on does not wait"
    assert gate.snapshot()["refused_calls"] == 1


async def test_an_unarmed_gate_admits_at_once_without_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sg, "FIRST_PUSH_WAIT_S", 60.0)
    entered = _spy_on_the_wait(monkeypatch)

    await asyncio.wait_for(SpendGate().admit_async(), timeout=1.0)
    await asyncio.wait_for(SpendGate(tmp_path / "absent.json").admit_async(), timeout=1.0)

    assert entered == []


async def test_a_call_under_the_restored_cap_is_not_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sg, "FIRST_PUSH_WAIT_S", 60.0)
    entered = _spy_on_the_wait(monkeypatch)
    gate = _ledger_gate(tmp_path, floor=5.0)  # restored at $15.00 with $5.00 spent: room

    await asyncio.wait_for(gate.admit_async(), timeout=1.0)

    assert entered == [], "only a refusal waits"
    assert gate.snapshot()["refused_calls"] == 0


async def test_an_unreadable_ledger_refuses_at_once_without_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sg, "FIRST_PUSH_WAIT_S", 60.0)
    entered = _spy_on_the_wait(monkeypatch)
    ledger = tmp_path / "spend-ledger.json"
    ledger.write_text("{not json", encoding="utf-8")
    gate = SpendGate(ledger)

    with pytest.raises(SpendCapReached):
        await asyncio.wait_for(gate.admit_async(), timeout=1.0)

    assert entered == [], "a ledger that cannot be read refuses as before, with no wait"
    assert gate.snapshot()["refused_calls"] == 1


async def test_the_bound_is_one_window_so_a_later_call_is_refused_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A vessel that never pushes costs one window, not one window per call."""
    monkeypatch.setattr(sg, "FIRST_PUSH_WAIT_S", 0.3)
    gate = _ledger_gate(tmp_path)
    with pytest.raises(SpendCapReached):
        await gate.admit_async()  # opens the window and waits it out

    entered = _spy_on_the_wait(monkeypatch)
    with pytest.raises(SpendCapReached):
        await gate.admit_async()  # the window is closed: it does not open another

    assert entered == []
    assert gate.snapshot()["refused_calls"] == 2


async def test_a_call_that_joins_an_open_window_waits_only_what_is_left_of_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sg, "FIRST_PUSH_WAIT_S", 0.8)
    gate = _ledger_gate(tmp_path)
    first = asyncio.create_task(gate.admit_async())  # opens the window
    await asyncio.sleep(0.4)  # half of it has gone

    started = time.monotonic()
    with pytest.raises(SpendCapReached):
        await gate.admit_async()  # joins it
    joined_for = time.monotonic() - started

    assert joined_for < 0.7, f"a call that joined waited {joined_for:.2f} s of a 0.8 s window"
    with pytest.raises(SpendCapReached):
        await first


async def test_the_event_loop_keeps_running_while_a_call_waits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The push is taken by a handler on this same loop, so a wait that held the loop would never
    see it: every call would sit out the whole bound."""
    monkeypatch.setattr(sg, "FIRST_PUSH_WAIT_S", 0.6)
    gate = _ledger_gate(tmp_path)
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    other_work = asyncio.create_task(ticker())
    try:
        with pytest.raises(SpendCapReached):
            await gate.admit_async()
    finally:
        other_work.cancel()
        await asyncio.gather(other_work, return_exceptions=True)

    assert ticks >= 5, f"the loop ran {ticks} ticks of other work during a 0.6 s wait"


async def test_a_cancelled_wait_ends_at_once_and_holds_no_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wait parked in an executor thread cannot be cancelled: the thread outlives the call and
    a process exit waits for it. This one is polled, so cancelling it leaves nothing behind."""
    monkeypatch.setattr(sg, "FIRST_PUSH_WAIT_S", 60.0)
    gate = _ledger_gate(tmp_path)

    def pool_threads() -> list[str]:
        return [t.name for t in threading.enumerate() if t.name.startswith("asyncio")]

    before = pool_threads()
    call = asyncio.create_task(gate.admit_async())
    await asyncio.sleep(0.3)
    call.cancel()
    await asyncio.gather(call, return_exceptions=True)

    assert call.cancelled()
    assert len(pool_threads()) <= len(before), f"the cancelled wait left a thread: {pool_threads()}"
    assert gate.snapshot()["refused_calls"] == 0, "a cancelled call is not a refusal"


# The provider's two dispatch paths each ask the gate, so each must hold a refusal for the push.


async def _dispatch(kind: str) -> None:
    """One model call down the named path of the provider."""
    if kind == "acomplete":
        await _provider().acomplete(_MSGS)
    else:
        async for _ in _provider().astream(_MSGS):
            pass


def _wire_for(kind: str) -> _Wire:
    return _Wire(_response(0.25) if kind == "acomplete" else _Stream())


@pytest.mark.parametrize("kind", ["acomplete", "astream"])
async def test_both_dispatch_paths_hold_a_refusal_for_the_first_push(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    monkeypatch.setattr(sg, "FIRST_PUSH_WAIT_S", 60.0)
    wire = _wire_for(kind)
    monkeypatch.setattr(lp.litellm, "acompletion", wire)
    gate = _ledger_gate(tmp_path)
    reset_gate_for_tests(gate)

    call = asyncio.create_task(_dispatch(kind))
    await asyncio.sleep(0.3)
    assert not call.done(), "the call is held, not refused"
    assert wire.calls == 0, "and held before the wire"

    gate.set_budget(**_RAISED)
    await asyncio.wait_for(call, timeout=2.0)

    assert wire.calls == 1


@pytest.mark.parametrize("kind", ["acomplete", "astream"])
async def test_both_dispatch_paths_refuse_before_the_wire_when_no_push_comes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    monkeypatch.setattr(sg, "FIRST_PUSH_WAIT_S", 0.3)
    wire = _wire_for(kind)
    monkeypatch.setattr(lp.litellm, "acompletion", wire)
    gate = _ledger_gate(tmp_path)
    reset_gate_for_tests(gate)

    with pytest.raises(SpendCapReached):
        await _dispatch(kind)

    assert wire.calls == 0
    assert gate.snapshot()["refused_calls"] == 1


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


async def test_post_budget_on_the_servers_own_loop_releases_a_waiting_model_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The seam itself: the push is a real ``POST /budget`` handled on the loop the call waits on.

    A model call is held on a gate booted from a stale ledger, and the app's own handler (httpx's
    ASGI transport drives it on this loop, as uvicorn does) takes the push that admits it.
    """
    pytest.importorskip("fastapi")
    import httpx

    monkeypatch.setattr(sg, "FIRST_PUSH_WAIT_S", 60.0)
    wire = _Wire(_response(0.25))
    monkeypatch.setattr(lp.litellm, "acompletion", wire)
    reset_gate_for_tests(_ledger_gate(tmp_path))
    transport = httpx.ASGITransport(app=_app(tmp_path))  # after the reset: it serves this gate

    async with httpx.AsyncClient(transport=transport, base_url="http://sidecar.test") as client:
        call = asyncio.create_task(_provider().acomplete(_MSGS))
        await asyncio.sleep(0.3)
        assert not call.done() and wire.calls == 0, "the call is held before the wire"

        pushed = await client.post("/budget", json=_RAISED, headers=_auth())
        assert pushed.status_code == 200
        await asyncio.wait_for(call, timeout=2.0)

    assert wire.calls == 1
    assert get_gate().snapshot()["cap_usd"] == 22.0
