"""A 200 the route's content filter emptied is a provider failure, not an empty answer.

An OpenAI-compatible gateway in front of a Claude model occasionally answered HTTP 200 with no
content and ``finish_reason="content_filter"`` on long text (reported 2026-10-13, cause
unconfirmed). The loop read that as the model going silent: it nudged a model that had said
nothing and reported "the model went silent", and a turn that had already spoken ended
``completed`` on it, so the operator got an empty answer and no cause. The provider now raises
:class:`ContentFiltered` for exactly that shape. The loop samples it again ONCE, at the
configured temperature, and when that is withheld too ends the turn ``provider_error`` carrying
the filter's name. Each withheld call was billed though it raised, so it is recorded: in the
session, in the budget's token and cost ceilings and in the turn's usage. Hermetic:
``litellm.acompletion`` is faked.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import zakcode.agent.loop as loop_module
import zakcode.providers.litellm_provider as lp
from zakcode.agent.budget import IterationBudget
from zakcode.agent.loop import AgentLoop
from zakcode.config import load_settings
from zakcode.events import AgentUsage
from zakcode.messages import Message
from zakcode.providers.base import (
    Capabilities,
    ContentFiltered,
    LLMResult,
    ModelOutputRejected,
    Provider,
    ProviderError,
    ProviderStreamEvent,
    RateLimited,
    RequestFailed,
    StreamDone,
    StreamTextDelta,
    StreamThinkingDelta,
    StreamToolCallDelta,
    StreamUsage,
)
from zakcode.providers.litellm_provider import LiteLLMProvider
from zakcode.session.store import Session
from zakcode.tools.base import ToolRegistry
from zakcode.usage import Usage

POD = "http://pod.local:8080/v1"

#: What every scripted call reports having been billed for, so a total reads as a call count.
PROMPT_TOKENS = 1000


def _provider(**kw: Any) -> LiteLLMProvider:
    return LiteLLMProvider(
        model="openai/zds-qwen3.8-27b", api_base=POD, context_window=131072, **kw
    )


def _response(
    text: str = "",
    *,
    finish: str = "content_filter",
    tool_calls: list[Any] | None = None,
    reasoning: str | None = None,
    cost: float = 0.0,
) -> SimpleNamespace:
    """A buffered completion the way litellm hands it back."""
    message = SimpleNamespace(content=text, tool_calls=tool_calls, reasoning_content=reasoning)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish)],
        usage=SimpleNamespace(
            prompt_tokens=PROMPT_TOKENS, completion_tokens=0, total_tokens=PROMPT_TOKENS
        ),
        _hidden_params={"response_cost": cost},
    )


def _chunk(
    text: str | None = None, *, finish: str | None = None, reasoning: str | None = None
) -> SimpleNamespace:
    """One streamed chunk: a text or reasoning delta, a finish marker, or both."""
    delta = SimpleNamespace(content=text, tool_calls=None, reasoning_content=reasoning)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish)], usage=None)


def _tool_call_chunk() -> SimpleNamespace:
    """One streamed chunk that opens a complete tool call."""
    call = SimpleNamespace(
        index=0, id="c1", function=SimpleNamespace(name="read_file", arguments="{}")
    )
    delta = SimpleNamespace(content=None, tool_calls=[call], reasoning_content=None)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=None)], usage=None)


def _usage_chunk(prompt_tokens: int = PROMPT_TOKENS) -> SimpleNamespace:
    """The last chunk of every stream this provider opens: it asks for ``include_usage``, so a
    usage-only chunk (no choices) follows the finish marker and becomes a ``StreamUsage``."""
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=0, total_tokens=prompt_tokens
        ),
    )


class _Litellm:
    """A scripted ``litellm.acompletion``: each call takes the next script (the last one repeats)
    and answers it as a buffered response or, for a streaming call, as a chunk stream."""

    def __init__(self, *scripts: Any) -> None:
        self.scripts = list(scripts)
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        script = self.scripts.pop(0) if len(self.scripts) > 1 else self.scripts[0]
        return self._stream(script) if kwargs.get("stream") else script

    @staticmethod
    async def _stream(chunks: list[Any]) -> AsyncIterator[Any]:
        for chunk in chunks:
            yield chunk


# ── the provider ──────────────────────────────────────────────────────────────


async def test_an_empty_answer_ending_in_content_filter_is_raised_not_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lp.litellm, "acompletion", _Litellm(_response()))
    provider = _provider()
    with pytest.raises(ContentFiltered) as ei:
        await provider.acomplete([Message.user("hi")])
    assert isinstance(ei.value, ModelOutputRejected)  # rides the loop's rejection retry
    assert isinstance(ei.value, RateLimited)
    assert "content_filter" in str(ei.value) and provider.model in str(ei.value)
    # The 200 was billed. It raises instead of returning, so it carries its own usage.
    assert ei.value.usage is not None and ei.value.usage.prompt_tokens == PROMPT_TOKENS


async def test_an_empty_answer_that_was_not_filtered_is_left_to_the_silence_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lp.litellm, "acompletion", _Litellm(_response(finish="stop")))
    result = await _provider().acomplete([Message.user("hi")])
    assert result.text == "" and result.finish_reason == "stop"


async def test_a_filtered_answer_that_delivered_text_comes_back_as_it_came(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lp.litellm, "acompletion", _Litellm(_response("a partial answer")))
    result = await _provider().acomplete([Message.user("hi")])
    assert result.text == "a partial answer" and result.finish_reason == "content_filter"


async def test_a_filtered_answer_that_delivered_a_tool_call_comes_back_as_it_came(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call = SimpleNamespace(
        id="c1", type="function", function=SimpleNamespace(name="read_file", arguments="{}")
    )
    monkeypatch.setattr(lp.litellm, "acompletion", _Litellm(_response(tool_calls=[call])))
    result = await _provider().acomplete([Message.user("hi")])
    assert [c.name for c in result.tool_calls] == ["read_file"]


async def test_a_filtered_answer_that_only_thought_is_still_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Reasoning is not an answer: the model thought, then the route withheld what it said.
    monkeypatch.setattr(lp.litellm, "acompletion", _Litellm(_response(reasoning="let me see")))
    with pytest.raises(ContentFiltered):
        await _provider().acomplete([Message.user("hi")])


async def test_a_stream_that_ends_filtered_with_nothing_delivered_is_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The shape a real stream has: the finish marker, then the usage chunk the provider asked
    # for. That usage chunk becomes an event, and it is not an answer.
    chunks = [_chunk(None, finish="content_filter"), _usage_chunk()]
    monkeypatch.setattr(lp.litellm, "acompletion", _Litellm(chunks))
    provider = _provider()
    events: list[Any] = []
    with pytest.raises(ContentFiltered) as ei:
        async for ev in provider.astream([Message.user("hi")]):
            events.append(ev)
    assert [type(ev) for ev in events] == [StreamUsage]  # nothing but usage; no StreamDone
    assert ei.value.usage is not None and ei.value.usage.prompt_tokens == PROMPT_TOKENS
    # The raw sample survives the raise: that is what a filtered empty is diagnosed from.
    assert provider.last_stream_sample is not None
    assert provider.last_stream_sample["finish_reason"] == "content_filter"


async def test_a_stream_reporting_its_usage_in_parts_is_billed_for_the_sum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The loop sums every usage event of an attempt; the exception carries the same sum.
    chunks = [_chunk(None, finish="content_filter"), _usage_chunk(600), _usage_chunk(400)]
    monkeypatch.setattr(lp.litellm, "acompletion", _Litellm(chunks))
    with pytest.raises(ContentFiltered) as ei:
        _ = [ev async for ev in _provider().astream([Message.user("hi")])]
    assert ei.value.usage is not None and ei.value.usage.prompt_tokens == PROMPT_TOKENS


async def test_a_stream_that_only_thought_before_the_filter_is_still_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [_chunk(reasoning="let me see"), _chunk(None, finish="content_filter"), _usage_chunk()]
    monkeypatch.setattr(lp.litellm, "acompletion", _Litellm(chunks))
    events: list[Any] = []
    with pytest.raises(ContentFiltered):
        async for ev in _provider().astream([Message.user("hi")]):
            events.append(ev)
    assert [type(ev) for ev in events] == [StreamThinkingDelta, StreamUsage]


async def test_a_stream_that_ends_filtered_after_text_is_a_normal_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [_chunk("par"), _chunk(None, finish="content_filter"), _usage_chunk()]
    monkeypatch.setattr(lp.litellm, "acompletion", _Litellm(chunks))
    events = [ev async for ev in _provider().astream([Message.user("hi")])]
    assert [type(ev) for ev in events] == [StreamTextDelta, StreamUsage, StreamDone]
    assert isinstance(events[-1], StreamDone) and events[-1].finish_reason == "content_filter"


async def test_a_stream_that_ends_filtered_after_a_tool_call_is_a_normal_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A tool call is an answer: discarding it would throw away the model's work.
    chunks = [_tool_call_chunk(), _chunk(None, finish="content_filter"), _usage_chunk()]
    monkeypatch.setattr(lp.litellm, "acompletion", _Litellm(chunks))
    events = [ev async for ev in _provider().astream([Message.user("hi")])]
    assert [type(ev) for ev in events] == [StreamToolCallDelta, StreamUsage, StreamDone]


# ── the loop, with the real provider under it ─────────────────────────────────


def _loop(
    provider: Provider,
    *,
    budget: IterationBudget | None = None,
    failover: Callable[[ProviderError], tuple[Provider, str] | None] | None = None,
) -> AgentLoop:
    settings = load_settings(workspace_root=Path.cwd())
    return AgentLoop(
        provider,
        ToolRegistry(),
        Session(cwd="/tmp/work", model="test/model"),
        settings=settings,
        budget=budget,
        model_failover=failover,
    )


@pytest.fixture(autouse=True)
def _no_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loop sizes its prompt with litellm's token counter; keep that off any tokenizer."""
    monkeypatch.setattr(lp.litellm, "token_counter", lambda **kw: 10)


def test_the_loop_samples_a_filtered_answer_again_once_and_says_why(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake = _Litellm(_response(), _response("the answer", finish="stop"))
    monkeypatch.setattr(lp.litellm, "acompletion", fake)
    with caplog.at_level(logging.WARNING, logger="zakcode.agent.loop"):
        result = asyncio.run(_loop(_provider()).arun_turn("hi"))
    assert result.stop_reason == "completed" and not result.degraded
    assert result.assistant_messages[0].text == "the answer"
    assert len(fake.calls) == 2
    retry_logs = [r.getMessage() for r in caplog.records if "retrying" in r.getMessage()]
    assert any("content filter withheld the answer" in m for m in retry_logs)
    assert any("(1/1)" in m for m in retry_logs)  # the resample's own budget, not the interrupt one
    assert not any("malformed tool call" in m for m in retry_logs)


def test_the_resample_keeps_the_temperature_nobody_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Nothing set means nothing sent: a raised temperature the operator never chose can be
    # refused by a model that fixes its own (a thinking mode), and that 400 would hide the filter.
    fake = _Litellm(_response(), _response("the answer", finish="stop"))
    monkeypatch.setattr(lp.litellm, "acompletion", fake)
    asyncio.run(_loop(_provider()).arun_turn("hi"))
    assert "temperature" not in fake.calls[0] and "temperature" not in fake.calls[1]


def test_the_resample_keeps_the_configured_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _Litellm(_response(), _response("the answer", finish="stop"))
    monkeypatch.setattr(lp.litellm, "acompletion", fake)
    asyncio.run(_loop(_provider(temperature=0.2)).arun_turn("hi"))
    assert [call["temperature"] for call in fake.calls] == [0.2, 0.2]


def test_a_route_that_keeps_filtering_ends_the_turn_naming_the_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _Litellm(_response())
    monkeypatch.setattr(lp.litellm, "acompletion", fake)
    result = asyncio.run(_loop(_provider()).arun_turn("hi"))
    assert result.stop_reason == "provider_error" and result.degraded
    assert "content_filter" in result.error  # the cause the operator reads, not "went silent"
    assert "safety refusal" in result.error  # litellm files a model's refusal under this reason
    # One resample, not the interrupt bound: a refusal repeats, and each call is a long prompt.
    assert loop_module._MAX_FILTER_RETRIES == 1 and len(fake.calls) == 2


def test_the_streaming_loop_samples_a_filtered_answer_again_once_and_says_why(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _Litellm(
        [_chunk(None, finish="content_filter"), _usage_chunk()],
        [_chunk("done"), _chunk(finish="stop"), _usage_chunk()],
    )
    monkeypatch.setattr(lp.litellm, "acompletion", fake)

    async def collect() -> list[Any]:
        return [ev async for ev in _loop(_provider(temperature=0.2)).astream_turn("hi")]

    events = asyncio.run(collect())
    assert events[-1].stop_reason == "completed"
    assert [call["temperature"] for call in fake.calls] == [0.2, 0.2]
    statuses = [getattr(ev, "message", "") for ev in events]
    assert any("content filter withheld the answer" in s for s in statuses)
    assert any("(1/1)" in s for s in statuses)
    assert not any("malformed tool call" in s for s in statuses)


def test_a_streaming_route_that_keeps_filtering_ends_the_turn_after_one_resample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _Litellm([_chunk(None, finish="content_filter"), _usage_chunk()])
    monkeypatch.setattr(lp.litellm, "acompletion", fake)

    async def collect() -> list[Any]:
        return [ev async for ev in _loop(_provider()).astream_turn("hi")]

    events = asyncio.run(collect())
    assert events[-1].stop_reason == "provider_error" and "content_filter" in events[-1].error
    assert len(fake.calls) == 2


# ── every billed call is recorded ─────────────────────────────────────────────
#
# With the real provider under the loop at PROMPT_TOKENS per call, the withheld call raised
# before the loop recorded any usage: a filtered-then-answered turn recorded 1000 for 2 calls,
# and an always-filtered one recorded 0 for 4 (main, which read the same 200 as a silent
# model, recorded 3000 for 3). Max_tokens and max_cost_usd never saw the withheld calls.


def _assert_both_calls_recorded(loop: AgentLoop, budget: IterationBudget, turn: Usage) -> None:
    spent = loop.session.cumulative_usage()
    assert spent.prompt_tokens == spent.total_tokens == 2 * PROMPT_TOKENS
    assert turn.prompt_tokens == 2 * PROMPT_TOKENS
    assert budget.tokens_spent == 2 * PROMPT_TOKENS
    # The ceiling sees them: 1500 is under two calls and over either one.
    assert budget.tokens_exhausted() and budget.over_budget()


def test_a_filtered_call_and_the_answer_after_it_are_both_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _Litellm(_response(cost=0.25), _response("the answer", finish="stop", cost=0.25))
    monkeypatch.setattr(lp.litellm, "acompletion", fake)
    budget = IterationBudget(10, max_tokens=1500, max_cost_usd=0.4)
    loop = _loop(_provider(), budget=budget)
    result = asyncio.run(loop.arun_turn("hi"))
    # The answer came back, and the ceilings now stop the turn after it: two calls are over
    # 1500, though either alone is under (it ended "completed" when only one was counted).
    assert result.assistant_messages[0].text == "the answer" and len(fake.calls) == 2
    assert result.stop_reason == "budget_exhausted"
    _assert_both_calls_recorded(loop, budget, result.usage)
    # The cost ceiling too: 0.4 is under two calls at 0.25 and over either one.
    assert loop.session.cumulative_usage().cost_usd == pytest.approx(0.5)
    assert budget.cost_spent == pytest.approx(0.5) and budget.cost_exhausted()


def test_a_route_that_keeps_filtering_still_has_both_its_calls_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _Litellm(_response())
    monkeypatch.setattr(lp.litellm, "acompletion", fake)
    budget = IterationBudget(10, max_tokens=1500)
    loop = _loop(_provider(), budget=budget)
    result = asyncio.run(loop.arun_turn("hi"))
    assert result.stop_reason == "provider_error" and len(fake.calls) == 2
    _assert_both_calls_recorded(loop, budget, result.usage)


def test_the_streaming_loop_records_a_filtered_call_and_the_answer_after_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _Litellm(
        [_chunk(None, finish="content_filter"), _usage_chunk()],
        [_chunk("done"), _chunk(finish="stop"), _usage_chunk()],
    )
    monkeypatch.setattr(lp.litellm, "acompletion", fake)
    budget = IterationBudget(10, max_tokens=1500)
    loop = _loop(_provider(), budget=budget)

    async def collect() -> list[Any]:
        return [ev async for ev in loop.astream_turn("hi")]

    events = asyncio.run(collect())
    # As the buffered twin: the answer came back, and the token ceiling stops the turn after it.
    assert any(getattr(ev, "text", "") == "done" for ev in events) and len(fake.calls) == 2
    assert events[-1].stop_reason == "budget_exhausted"
    _assert_both_calls_recorded(loop, budget, events[-1].usage)
    reported = [ev.usage for ev in events if isinstance(ev, AgentUsage)]
    assert [u.prompt_tokens for u in reported] == [2 * PROMPT_TOKENS]


def test_a_streaming_route_that_keeps_filtering_still_has_both_its_calls_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _Litellm([_chunk(None, finish="content_filter"), _usage_chunk()])
    monkeypatch.setattr(lp.litellm, "acompletion", fake)
    budget = IterationBudget(10, max_tokens=1500)
    loop = _loop(_provider(), budget=budget)

    async def collect() -> list[Any]:
        return [ev async for ev in loop.astream_turn("hi")]

    events = asyncio.run(collect())
    assert events[-1].stop_reason == "provider_error" and len(fake.calls) == 2
    _assert_both_calls_recorded(loop, budget, events[-1].usage)


def test_a_withheld_call_is_part_of_its_own_turn_and_no_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _Litellm(_response(), _response("one", finish="stop"), _response("two", finish="stop"))
    monkeypatch.setattr(lp.litellm, "acompletion", fake)
    loop = _loop(_provider())

    async def two_turns() -> tuple[Usage, Usage]:
        first = await loop.arun_turn("hi")
        second = await loop.arun_turn("again")
        return first.usage, second.usage

    first, second = asyncio.run(two_turns())
    assert first.prompt_tokens == 2 * PROMPT_TOKENS  # the withheld call and the answer
    assert second.prompt_tokens == PROMPT_TOKENS  # a clean turn owes nothing to the last
    assert loop.session.cumulative_usage().prompt_tokens == 3 * PROMPT_TOKENS


def test_the_streaming_loop_keeps_a_withheld_call_in_its_own_turn_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _Litellm(
        [_chunk(None, finish="content_filter"), _usage_chunk()],
        [_chunk("one"), _chunk(finish="stop"), _usage_chunk()],
        [_chunk("two"), _chunk(finish="stop"), _usage_chunk()],
    )
    monkeypatch.setattr(lp.litellm, "acompletion", fake)
    loop = _loop(_provider())

    async def two_turns() -> list[int]:
        done: list[int] = []
        for text in ("hi", "again"):
            events = [ev async for ev in loop.astream_turn(text)]
            done.append(events[-1].usage.prompt_tokens)
        return done

    assert asyncio.run(two_turns()) == [2 * PROMPT_TOKENS, PROMPT_TOKENS]


def test_a_withheld_call_is_counted_by_whoever_the_caller_names() -> None:
    # A side call passes its own accounting; the main conversation's must not also fire.
    withheld = Usage(prompt_tokens=PROMPT_TOKENS, total_tokens=PROMPT_TOKENS)
    attempts = iter([ContentFiltered("withheld", usage=withheld), None])
    answered = LLMResult(text="ok", usage=withheld, finish_reason="stop")

    async def complete(call_kw: dict[str, Any]) -> LLMResult:
        failure = next(attempts)
        if failure is not None:
            raise failure
        return answered

    counted: list[Usage | None] = []
    loop = _loop(_provider())
    assert asyncio.run(loop._complete_with_retry(complete, on_withheld=counted.append)) is answered
    assert counted == [withheld]
    assert loop.session.cumulative_usage().total_tokens == 0

    # Without one, the main conversation's accounting counts it.
    attempts = iter([ContentFiltered("withheld", usage=withheld), None])
    asyncio.run(loop._complete_with_retry(complete))
    assert loop.session.cumulative_usage().total_tokens == PROMPT_TOKENS


class _Scripted(Provider):
    """Plays one scripted call per ``astream``: its events, and an exception item raises there."""

    def __init__(self, *calls: list[ProviderStreamEvent | Exception]) -> None:
        self.script = list(calls)
        self.calls = 0

    async def acomplete(  # type: ignore[no-untyped-def,override]
        self, messages, *, system=None, tools=None, **kw
    ) -> LLMResult:
        raise AssertionError("a scripted provider streams")

    async def astream(  # type: ignore[no-untyped-def,override]
        self, messages, *, system=None, tools=None, response_format=None, **kw
    ) -> AsyncIterator[ProviderStreamEvent]:
        self.calls += 1
        for item in self.script.pop(0):
            if isinstance(item, Exception):
                raise item
            yield item

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 100

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=8192)


def _billed() -> Usage:
    return Usage(prompt_tokens=PROMPT_TOKENS, total_tokens=PROMPT_TOKENS)


def _answers(text: str) -> list[ProviderStreamEvent | Exception]:
    return [
        StreamTextDelta(text=text),
        StreamUsage(usage=_billed()),
        StreamDone(finish_reason="stop"),
    ]


def test_a_summarizer_call_a_filter_withheld_is_billed_as_a_summarizer_call(
    tmp_path: Path,
) -> None:
    provider = _Scripted(
        [StreamUsage(usage=_billed()), ContentFiltered("withheld", usage=_billed())],
        _answers("<summary>the summary</summary>"),
    )
    loop = AgentLoop(
        provider,
        ToolRegistry(),
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
    )
    history = [Message.user("question " + "x" * 400), Message.assistant_text("answer")]
    assert asyncio.run(loop._summarize_for_compaction(history)) == "the summary"
    assert provider.calls == 2
    spent = loop.session.usages
    assert [u.side_call for u in spent] == ["summarizer", "summarizer"]
    assert sum(u.prompt_tokens for u in spent) == 2 * PROMPT_TOKENS
    assert loop._compaction_cost["summarizer_prompt_tokens"] == 2 * PROMPT_TOKENS
    assert loop._turn_withheld_usage.total_tokens == 0  # a side call is not part of a turn


def test_a_withheld_call_that_reports_no_usage_adds_nothing() -> None:
    # A provider may raise ContentFiltered without having measured the call.
    loop = _loop(_provider(), budget=IterationBudget(10, max_tokens=1500))
    loop._account_withheld_call(None)
    assert loop.session.cumulative_usage().total_tokens == 0
    assert loop._turn_withheld_usage.total_tokens == 0
    assert loop.budget is not None and loop.budget.tokens_spent == 0


def test_a_replacement_route_gets_its_own_resample() -> None:
    # The failover hands the new route a fresh retry budget, as it does for every other bound.
    primary = _Scripted([ContentFiltered("withheld")], [RequestFailed("primary down")])
    backup = _Scripted([ContentFiltered("withheld")], _answers("recovered"))
    loop = _loop(primary, failover=lambda exc: (backup, "primary -> backup"))

    async def collect() -> list[Any]:
        return [ev async for ev in loop.astream_turn("hi")]

    events = asyncio.run(collect())
    assert events[-1].stop_reason == "completed"
    assert any(getattr(ev, "text", "") == "recovered" for ev in events)
    assert (primary.calls, backup.calls) == (2, 2)
