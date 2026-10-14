"""A cached prompt a gateway reports only in ``total_tokens`` (ADR-0294).

Through a LiteLLM proxy in front of Bedrock, a request that carries a cache breakpoint comes
back with a ``usage`` whose ``prompt_tokens`` is the UNCACHED part alone and no cache field at
all: on the later turns of a 49,445-token conversation ``prompt_tokens`` read 3, where the same
prompt sent without a breakpoint read 49,445. The cached part appears in ``total_tokens`` and
nowhere else. ``_extract_usage`` took ``prompt_tokens`` as it came, so the prompt anchor
(ADR-0077) saw 3 tokens, ``/cost`` showed a prompt of 3, and the cost was priced on 3.

These tests pin the reading that replaced it: for a request that carried a breakpoint and got
no cache field back, the excess of ``total_tokens`` over prompt plus completion is a cached
prompt, counted in ``prompt_tokens`` (so the anchor sees it) and in ``cache_unsplit_tokens``
(so a display can say "cached, read or write unknown"), and priced at the cache-WRITE rate so
the cost never under-reports. Hermetic: no network, no credentials (``conftest`` points
``ZAKCODE_HOME`` at a temporary folder, so no ``~/.zakcode/.env`` is read).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import zakcode.providers.litellm_provider as lp
from tests.test_provider_stream import _chunk
from zakcode.agent.compact import Compactor
from zakcode.agent.loop import AgentLoop
from zakcode.agent.prompt import DYNAMIC_BOUNDARY
from zakcode.config import load_settings
from zakcode.messages import Message
from zakcode.providers.base import StreamUsage
from zakcode.providers.litellm_provider import LiteLLMProvider
from zakcode.session.store import Session
from zakcode.tools.base import ToolRegistry

#: The counts measured on a later turn of one conversation through the gateway.
UNCACHED, CACHED, COMPLETION = 3, 49_445, 120
#: The cached part is in the total and nowhere else.
TOTAL = UNCACHED + CACHED + COMPLETION

#: Priced, with a cache-write rate, in litellm's bundled map (an offline box has the same one).
MODEL = "claude-sonnet-4-5-20250929"

#: A system prompt with a stable half and a dynamic half: a Claude route puts its breakpoint
#: on the stable half. Without the boundary there is nothing to split and no breakpoint.
SPLIT_SYSTEM = f"stable identity\n\n{DYNAMIC_BOUNDARY}\n\ndynamic tail"
PLAIN_SYSTEM = "a bare system prompt with no boundary marker"


@pytest.fixture(autouse=True)
def _no_request_pacing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The provider spaces request starts a second apart; that sleep is not under test."""
    monkeypatch.setattr(lp, "_MIN_REQUEST_INTERVAL_S", 0.0)


def _gateway_usage(**reported: Any) -> SimpleNamespace:
    """The usage the gateway sends: no cache field, the cached part only in the total."""
    return SimpleNamespace(
        prompt_tokens=UNCACHED, completion_tokens=COMPLETION, total_tokens=TOTAL, **reported
    )


def _response(
    usage: Any, *, model: str = MODEL, hidden: dict[str, Any] | None = None
) -> SimpleNamespace:
    message = SimpleNamespace(content="ok", tool_calls=None, reasoning_content=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=usage,
        model=model,
        _hidden_params={} if hidden is None else hidden,
    )


def _extract(
    usage: Any,
    *,
    carried: bool,
    model: str = MODEL,
    hidden: dict[str, Any] | None = None,
) -> Any:
    response = _response(usage, model=model, hidden=hidden)
    return LiteLLMProvider._extract_usage(response, carried_breakpoint=carried)


def _rates(model: str) -> dict[str, Any]:
    return dict(lp.litellm.get_model_info(model))


# ── the excess is a cached prompt, when a breakpoint went out and nothing was split ──────────


def test_the_excess_of_the_total_over_the_parts_is_a_cached_prompt() -> None:
    usage = _extract(_gateway_usage(), carried=True)
    assert usage.prompt_tokens == UNCACHED + CACHED
    assert usage.cache_unsplit_tokens == CACHED
    assert usage.completion_tokens == COMPLETION
    assert usage.total_tokens == TOTAL
    # Read or written is not known, so neither count is claimed.
    assert (usage.cache_read_tokens, usage.cache_creation_tokens) == (0, 0)
    # The prompt side and the completion now make the whole total.
    assert usage.prompt_tokens + usage.completion_tokens == usage.total_tokens


@pytest.mark.parametrize(
    "usage",
    [
        pytest.param(_gateway_usage(), id="the-gateway-shape"),
        pytest.param(
            SimpleNamespace(prompt_tokens=100, completion_tokens=40, total_tokens=190),
            id="thinking-tokens-in-the-total",
        ),
    ],
)
def test_without_a_breakpoint_the_excess_is_not_a_cached_prompt(usage: Any) -> None:
    """Some providers' totals include thinking tokens that ``completion_tokens`` leaves out.
    With no breakpoint on the request there is no reason to read the excess as a cache."""
    got = _extract(usage, carried=False)
    assert (got.prompt_tokens, got.cache_unsplit_tokens) == (usage.prompt_tokens, 0)
    # The default is the old reading, so a caller that says nothing changes nothing.
    got = LiteLLMProvider._extract_usage(_response(usage))
    assert (got.prompt_tokens, got.cache_unsplit_tokens) == (usage.prompt_tokens, 0)


@pytest.mark.parametrize(
    "reported",
    [
        pytest.param({"cache_read_input_tokens": 40_000}, id="anthropic-read"),
        pytest.param({"cache_creation_input_tokens": CACHED}, id="anthropic-write"),
        pytest.param(
            {"prompt_tokens_details": SimpleNamespace(cached_tokens=40_000)}, id="openai-cached"
        ),
    ],
)
def test_a_reply_that_names_its_cache_tokens_is_taken_as_it_says(reported: dict[str, Any]) -> None:
    """A backend that splits the cache itself is believed; the excess is not read on top."""
    usage = _extract(_gateway_usage(**reported), carried=True)
    assert usage.prompt_tokens == UNCACHED
    assert usage.cache_unsplit_tokens == 0


def test_parts_that_add_up_leave_nothing_to_count() -> None:
    exact = SimpleNamespace(prompt_tokens=100, completion_tokens=40, total_tokens=140)
    assert _extract(exact, carried=True).cache_unsplit_tokens == 0
    # A total below the parts is a backend's rounding, not a negative cache.
    short = SimpleNamespace(prompt_tokens=100, completion_tokens=40, total_tokens=120)
    got = _extract(short, carried=True)
    assert (got.prompt_tokens, got.cache_unsplit_tokens) == (100, 0)
    # An omitted total is derived from the parts, so it has no excess either.
    omitted = SimpleNamespace(prompt_tokens=100, completion_tokens=40, total_tokens=0)
    got = _extract(omitted, carried=True)
    assert (got.prompt_tokens, got.total_tokens, got.cache_unsplit_tokens) == (100, 140, 0)


# ── priced at the write rate, so the cost never under-reports ────────────────────────────────


def _full_price_at_the_write_rate() -> float:
    rates = _rates(MODEL)
    return (
        UNCACHED * rates["input_cost_per_token"]
        + CACHED * rates["cache_creation_input_token_cost"]
        + COMPLETION * rates["output_cost_per_token"]
    )


def test_the_cached_excess_is_priced_at_the_cache_write_rate() -> None:
    rates = _rates(MODEL)
    assert rates.get("cache_creation_input_token_cost") is not None, "premise: write rate mapped"
    usage = _extract(_gateway_usage(), carried=True)
    assert usage.cost_usd == pytest.approx(_full_price_at_the_write_rate())
    assert not usage.cost_unpriced
    # The write rate is the dearer of the two cache rates: if the excess was a read, the figure
    # is above the truth, and it is never below it.
    as_reads = (
        UNCACHED * rates["input_cost_per_token"]
        + CACHED * rates["cache_read_input_token_cost"]
        + COMPLETION * rates["output_cost_per_token"]
    )
    assert usage.cost_usd > as_reads


def test_a_cost_priced_on_the_short_prompt_is_raised_to_the_cached_prompt() -> None:
    """litellm priced the whole response on the 3 tokens the usage named. The cached part is
    added at the write rate rather than left out."""
    rates = _rates(MODEL)
    short = UNCACHED * rates["input_cost_per_token"] + COMPLETION * rates["output_cost_per_token"]
    usage = _extract(_gateway_usage(), carried=True, hidden={"response_cost": short})
    assert usage.cost_usd > short
    assert usage.cost_usd == pytest.approx(_full_price_at_the_write_rate())


def test_a_higher_cost_from_the_backend_is_kept() -> None:
    usage = _extract(_gateway_usage(), carried=True, hidden={"response_cost": 5.0})
    assert usage.cost_usd == 5.0


def test_an_unpriced_route_counts_its_tokens_and_stays_unpriced() -> None:
    assert LiteLLMProvider._litellm_token_cost("Claude4.5", 10, 10) is None, "premise: unmapped"
    usage = _extract(_gateway_usage(), carried=True, model="Claude4.5")
    assert (usage.prompt_tokens, usage.cache_unsplit_tokens) == (UNCACHED + CACHED, CACHED)
    assert usage.cost_usd == 0.0
    assert usage.cost_unpriced


# ── the flag is the request that was SENT: both call paths ───────────────────────────────────


class Gateway:
    """``litellm.acompletion`` pointed at a gateway that reports a cached prompt only in the
    total. ``requests`` records the kwargs of every call."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        if not kwargs.get("stream"):
            return _response(_gateway_usage())

        async def _stream() -> AsyncIterator[Any]:
            yield _chunk(content="ok")
            yield _chunk(content="", finish_reason="stop")
            # The include_usage chunk: empty choices, the usage the gateway sent.
            yield SimpleNamespace(
                choices=[], usage=_gateway_usage(), model=MODEL, _hidden_params={}
            )

        return _stream()


def _breakpoints(request: dict[str, Any]) -> int:
    """How many cache breakpoints a request carried, counted without the code under test."""
    return sum(
        1
        for message in request["messages"]
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict) and "cache_control" in block
    )


def _provider() -> LiteLLMProvider:
    return LiteLLMProvider(
        model="anthropic/claude-sonnet-4-6",
        api_key="sk-test",
        api_base="https://gateway.invalid",
        temperature=0.0,
    )


async def test_a_buffered_call_that_carried_a_breakpoint_reports_the_cached_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = Gateway()
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)

    result = await _provider().acomplete([Message.user("hi")], system=SPLIT_SYSTEM)

    assert _breakpoints(gateway.requests[0]) == 1, "premise: the request carried a breakpoint"
    assert result.usage.prompt_tokens == UNCACHED + CACHED
    assert result.usage.cache_unsplit_tokens == CACHED


async def test_a_buffered_call_without_a_breakpoint_is_left_as_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = Gateway()
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)

    result = await _provider().acomplete([Message.user("hi")], system=PLAIN_SYSTEM)

    assert _breakpoints(gateway.requests[0]) == 0, "premise: no breakpoint went out"
    assert result.usage.prompt_tokens == UNCACHED
    assert result.usage.cache_unsplit_tokens == 0


async def test_a_streamed_call_that_carried_a_breakpoint_reports_the_cached_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = Gateway()
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)

    events = [e async for e in _provider().astream([Message.user("hi")], system=SPLIT_SYSTEM)]

    assert _breakpoints(gateway.requests[0]) == 1, "premise: the request carried a breakpoint"
    [streamed] = [e.usage for e in events if isinstance(e, StreamUsage)]
    assert streamed.prompt_tokens == UNCACHED + CACHED
    assert streamed.cache_unsplit_tokens == CACHED


async def test_a_streamed_call_without_a_breakpoint_is_left_as_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = Gateway()
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)

    events = [e async for e in _provider().astream([Message.user("hi")], system=PLAIN_SYSTEM)]

    assert _breakpoints(gateway.requests[0]) == 0, "premise: no breakpoint went out"
    [streamed] = [e.usage for e in events if isinstance(e, StreamUsage)]
    assert (streamed.prompt_tokens, streamed.cache_unsplit_tokens) == (UNCACHED, 0)


# ── what the loop does with it: the prompt anchor (ADR-0077) ─────────────────────────────────


def _loop(tmp_path: Path) -> AgentLoop:
    session = Session(cwd=str(tmp_path), model="anthropic/claude-sonnet-4-6")
    return AgentLoop(
        _provider(),
        ToolRegistry(),
        session,
        settings=load_settings(workspace_root=tmp_path),
        compactor=Compactor(),
    )


async def test_the_prompt_anchor_sees_the_cached_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The anchor floors the compaction check with what the provider measured. Fed the 3
    tokens the gateway names, it floored nothing and the turn died at the window."""
    gateway = Gateway()
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)
    loop = _loop(tmp_path)

    result = await loop.arun_turn("continue")

    assert result.stop_reason == "completed", result.error
    assert _breakpoints(gateway.requests[0]) == 1, "premise: the loop's own prompt is split"
    assert loop.session.prompt_anchor_tokens == UNCACHED + CACHED
    assert result.usage.cache_unsplit_tokens == CACHED


async def test_streaming_the_prompt_anchor_sees_the_cached_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gateway = Gateway()
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)
    loop = _loop(tmp_path)

    events = [event async for event in loop.astream_turn("continue")]

    assert events[-1].stop_reason == "completed", events[-1].error
    assert _breakpoints(gateway.requests[0]) == 1, "premise: the loop's own prompt is split"
    assert loop.session.prompt_anchor_tokens == UNCACHED + CACHED
