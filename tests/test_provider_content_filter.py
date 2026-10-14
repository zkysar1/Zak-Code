"""A 200 the route's content filter emptied is a provider failure, not an empty answer.

An OpenAI-compatible gateway in front of a Claude model occasionally answered HTTP 200 with no
content and ``finish_reason="content_filter"`` on long text (reported 2026-10-13, cause
unconfirmed). The loop read that as the model going silent: it nudged a model that had said
nothing and reported "the model went silent", and a turn that had already spoken ended
``completed`` on it, so the operator got an empty answer and no cause. The provider now raises
:class:`ContentFiltered` for exactly that shape, which rides the loop's existing bounded retry
(a raised temperature, so the retry is a fresh sample) and, once the retries are spent, ends the
turn ``provider_error`` carrying the filter's name. Hermetic: ``litellm.acompletion`` is faked.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import zakcode.agent.loop as loop_module
import zakcode.providers.litellm_provider as lp
from zakcode.agent.loop import AgentLoop
from zakcode.config import load_settings
from zakcode.messages import Message
from zakcode.providers.base import (
    ContentFiltered,
    ModelOutputRejected,
    RateLimited,
    StreamDone,
    StreamTextDelta,
    StreamThinkingDelta,
    StreamUsage,
)
from zakcode.providers.litellm_provider import LiteLLMProvider
from zakcode.session.store import Session
from zakcode.tools.base import ToolRegistry

POD = "http://pod.local:8080/v1"


def _provider() -> LiteLLMProvider:
    return LiteLLMProvider(model="openai/zds-qwen3.8-27b", api_base=POD, context_window=131072)


def _response(
    text: str = "",
    *,
    finish: str = "content_filter",
    tool_calls: list[Any] | None = None,
    reasoning: str | None = None,
) -> SimpleNamespace:
    """A buffered completion the way litellm hands it back."""
    message = SimpleNamespace(content=text, tool_calls=tool_calls, reasoning_content=reasoning)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=7, completion_tokens=0, total_tokens=7),
        _hidden_params={"response_cost": 0.0},
    )


def _chunk(
    text: str | None = None, *, finish: str | None = None, reasoning: str | None = None
) -> SimpleNamespace:
    """One streamed chunk: a text or reasoning delta, a finish marker, or both."""
    delta = SimpleNamespace(content=text, tool_calls=None, reasoning_content=reasoning)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish)], usage=None)


def _usage_chunk() -> SimpleNamespace:
    """The last chunk of every stream this provider opens: it asks for ``include_usage``, so a
    usage-only chunk (no choices) follows the finish marker and becomes a ``StreamUsage``."""
    return SimpleNamespace(
        choices=[], usage=SimpleNamespace(prompt_tokens=7, completion_tokens=0, total_tokens=7)
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
    with pytest.raises(ContentFiltered):
        async for ev in provider.astream([Message.user("hi")]):
            events.append(ev)
    assert [type(ev) for ev in events] == [StreamUsage]  # nothing but usage; no StreamDone
    # The raw sample survives the raise: that is what a filtered empty is diagnosed from.
    assert provider.last_stream_sample is not None
    assert provider.last_stream_sample["finish_reason"] == "content_filter"


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


# ── the loop, with the real provider under it ─────────────────────────────────


def _loop(provider: LiteLLMProvider) -> AgentLoop:
    settings = load_settings(workspace_root=Path.cwd())
    return AgentLoop(
        provider, ToolRegistry(), Session(cwd="/tmp/work", model="test/model"), settings=settings
    )


@pytest.fixture(autouse=True)
def _no_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loop sizes its prompt with litellm's token counter; keep that off any tokenizer."""
    monkeypatch.setattr(lp.litellm, "token_counter", lambda **kw: 10)


def test_the_loop_retries_a_filtered_answer_as_a_fresh_sample_and_says_why(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake = _Litellm(_response(), _response("the answer", finish="stop"))
    monkeypatch.setattr(lp.litellm, "acompletion", fake)
    with caplog.at_level(logging.WARNING, logger="zakcode.agent.loop"):
        result = asyncio.run(_loop(_provider()).arun_turn("hi"))
    assert result.stop_reason == "completed" and not result.degraded
    assert result.assistant_messages[0].text == "the answer"
    assert len(fake.calls) == 2
    assert fake.calls[1]["temperature"] == loop_module._REJECTION_RETRY_TEMP_FLOOR  # a new sample
    retry_logs = [r.getMessage() for r in caplog.records if "retrying" in r.getMessage()]
    assert any("content filter withheld the answer" in m for m in retry_logs)
    assert not any("malformed tool call" in m for m in retry_logs)


def test_a_route_that_keeps_filtering_ends_the_turn_naming_the_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _Litellm(_response())
    monkeypatch.setattr(lp.litellm, "acompletion", fake)
    result = asyncio.run(_loop(_provider()).arun_turn("hi"))
    assert result.stop_reason == "provider_error" and result.degraded
    assert "content_filter" in result.error  # the cause the operator reads, not "went silent"
    assert len(fake.calls) == 1 + loop_module._MAX_INTERRUPT_RETRIES


def test_the_streaming_loop_retries_a_filtered_answer_and_says_why(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _Litellm(
        [_chunk(None, finish="content_filter"), _usage_chunk()],
        [_chunk("done"), _chunk(finish="stop"), _usage_chunk()],
    )
    monkeypatch.setattr(lp.litellm, "acompletion", fake)

    async def collect() -> list[Any]:
        return [ev async for ev in _loop(_provider()).astream_turn("hi")]

    events = asyncio.run(collect())
    assert events[-1].stop_reason == "completed"
    assert len(fake.calls) == 2
    statuses = [getattr(ev, "message", "") for ev in events]
    assert any("content filter withheld the answer" in s for s in statuses)
    assert not any("malformed tool call" in s for s in statuses)
