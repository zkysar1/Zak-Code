"""A reasoning-overflow retry gets more room to answer, and a refusal of its switch is not the end
(ADR-0307).

Reported 2026-10-14 against a LiteLLM 1.9x gateway in front of Bedrock/Vertex. Every call asks for
the 8,192-token output cap of ADR-0018, and on Bedrock a reasoning model's thinking counts against
it, so a hard turn can end ``finish_reason=length`` with nothing visible. The overflow retry
(ADR-0056) asks only for thinking off, in llama.cpp's spelling: three of six routes took the switch
and showed no sign of it, two refused it by name, which the provider learns, and one refused it
naming nothing ("The model returned the following errors: Validation Error"), which cannot be
learned, so the retry failed as ``RequestFailed`` and ended the turn. A route that always overflows
cost three 8,192-token calls, and the give-up said the model went silent.

Hermetic: the loop meets scripted routes, and the wire tests fake ``litellm.acompletion``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import zakcode.providers.litellm_provider as lp
from zakcode.agent.loop import _MAX_EMPTY_RETRIES, AgentLoop
from zakcode.config import load_settings
from zakcode.messages import Message
from zakcode.providers.base import (
    AuthError,
    Capabilities,
    LLMResult,
    Provider,
    RequestFailed,
    ToolCall,
)
from zakcode.providers.litellm_provider import LiteLLMProvider
from zakcode.providers.routing import thinking_extra_body
from zakcode.providers.thinking import THINKING_SWITCH_KEY
from zakcode.session.store import Session
from zakcode.tools.base import Tool, ToolContext, ToolRegistry, ToolResult, ToolSpec
from zakcode.usage import Usage

#: The cap every call is sent unless it asks for another (ADR-0018).
DEFAULT_CAP = lp._MAX_COMPLETION_TOKENS
THINKING_OFF = thinking_extra_body(False)
#: The route's refusal of the switch, Bedrock's own words: it names no field.
UNNAMED = "The model returned the following errors: Validation Error"
#: Reasoning that no cap ends: the route always overflows.
FOREVER = 10**9


@pytest.fixture(autouse=True)
def _no_request_pacing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The provider spaces request starts a second apart; that sleep is not under test."""
    monkeypatch.setattr(lp, "_MIN_REQUEST_INTERVAL_S", 0.0)


class Route(Provider):
    """A route to a reasoning model as the loop meets it, every call's kwargs recorded.

    Each scripted int is a completion that needs that many output tokens of reasoning before it
    answers: under the cap the call asked for (the default when it asked for none) it ends on
    ``length`` with nothing visible, as the report measured. A call carrying an override the
    route ``refuses`` (the thinking switch, or a cap of its own; ``any`` refuses every call) is
    refused with ``refusal`` and consumes nothing, as a 400 that names no field reaches the loop.
    """

    def __init__(
        self,
        script: list[int | LLMResult],
        *,
        refuses: frozenset[str] = frozenset(),
        refusal: type[Exception] = RequestFailed,
        window: int = 131_072,
        max_output: int | None = None,
        prompt_tokens: int = 0,
    ) -> None:
        self.script = list(script)
        self.refuses = refuses
        self.refusal = refusal
        self.window = window
        self.max_output = max_output
        self.prompt_tokens = prompt_tokens
        self.calls: list[dict[str, Any]] = []

    async def acomplete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        self.calls.append(dict(kwargs))
        if (
            "any" in self.refuses
            or ("switch" in self.refuses and "extra_body" in kwargs)
            or ("cap" in self.refuses and "max_tokens" in kwargs)
        ):
            raise self.refusal(UNNAMED)
        if not self.script:
            raise AssertionError("the route ran out of scripted completions")
        step = self.script.pop(0)
        if isinstance(step, LLMResult):
            return step
        cap = kwargs.get("max_tokens", DEFAULT_CAP)
        spent = min(cap, step)
        usage = Usage(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=spent,
            total_tokens=self.prompt_tokens + spent,
        )
        if cap < step:
            return LLMResult(finish_reason="length", usage=usage)
        return LLMResult(text="the answer", finish_reason="stop", usage=usage)

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(context_window=self.window, max_output=self.max_output)


class EchoTool(Tool):
    spec = ToolSpec(name="echo", description="Echo back the provided text.")

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(output=str(args.get("text", "")))


def _make_loop(provider: Provider, tmp_path: Path) -> AgentLoop:
    registry = ToolRegistry()
    registry.register(EchoTool())
    return AgentLoop(
        provider,
        registry,
        Session(cwd=str(tmp_path), model="openai/route-b"),
        settings=load_settings(workspace_root=tmp_path),
        max_iterations=20,
    )


def _caps(calls: list[dict[str, Any]]) -> list[int | None]:
    """The output cap each call asked for; None is the default."""
    return [call.get("max_tokens") for call in calls]


def _notes(loop: AgentLoop, kind: str) -> list[str]:
    return [e.detail for e in loop._trace.of_kind("intervention") if e.data.get("kind") == kind]


def _stop_reason(events: list[Any]) -> str | None:
    done = [ev for ev in events if getattr(ev, "stop_reason", None) is not None]
    return done[-1].stop_reason if done else None


def _statuses(events: list[Any]) -> list[str]:
    return [getattr(ev, "message", "") for ev in events]


# ── the cap of the retry ─────────────────────────────────────────────────────────────────────


async def test_a_route_that_always_overflows_is_asked_for_more_on_each_retry(
    tmp_path: Path,
) -> None:
    route = Route([FOREVER] * (1 + _MAX_EMPTY_RETRIES))
    loop = _make_loop(route, tmp_path)
    result = await loop.arun_turn("question")
    assert result.stop_reason == "gave_up"
    # The first call is sent the default; each overflow retry asks for twice it.
    assert _caps(route.calls) == [None, 2 * DEFAULT_CAP, 2 * DEFAULT_CAP]
    assert [call.get("extra_body") for call in route.calls] == [None, THINKING_OFF, THINKING_OFF]


async def test_a_retry_with_the_room_it_needed_answers(tmp_path: Path) -> None:
    route = Route([12_000, 12_000])
    loop = _make_loop(route, tmp_path)
    result = await loop.arun_turn("question")
    assert result.stop_reason == "completed"
    assert result.assistant_messages[-1].text == "the answer"
    assert _caps(route.calls) == [None, 2 * DEFAULT_CAP]


async def test_the_raise_is_for_that_one_call(tmp_path: Path) -> None:
    step = LLMResult(tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "a"})])
    route = Route([FOREVER, step, LLMResult(text="the answer")])
    loop = _make_loop(route, tmp_path)
    result = await loop.arun_turn("question")
    assert result.stop_reason == "completed"
    assert _caps(route.calls) == [None, 2 * DEFAULT_CAP, None]
    assert "extra_body" not in route.calls[2]


async def test_the_raise_is_held_to_the_models_declared_output_cap(tmp_path: Path) -> None:
    route = Route([FOREVER, 12_000], max_output=12_000)
    loop = _make_loop(route, tmp_path)
    result = await loop.arun_turn("question")
    assert result.stop_reason == "completed"
    assert _caps(route.calls) == [None, 12_000]


async def test_a_declared_cap_no_higher_than_the_default_leaves_the_retry_at_the_default(
    tmp_path: Path,
) -> None:
    route = Route([FOREVER, 4_000], max_output=DEFAULT_CAP)
    loop = _make_loop(route, tmp_path)
    result = await loop.arun_turn("question")
    assert result.stop_reason == "completed"
    assert _caps(route.calls) == [None, None]
    assert route.calls[1]["extra_body"] == THINKING_OFF


async def test_the_raise_is_held_to_the_room_the_window_has_left(tmp_path: Path) -> None:
    # A 32,768-token window, the 18,000-token prompt the overflowed call measured, and the
    # count's worst error, 5% of the window (1,638): 32,768 - 1,638 - 18,000 = 13,130 tokens.
    route = Route([FOREVER, 12_000], window=32_768, prompt_tokens=18_000)
    loop = _make_loop(route, tmp_path)
    result = await loop.arun_turn("question")
    assert result.stop_reason == "completed"
    assert _caps(route.calls) == [None, 13_130]


async def test_a_window_with_no_room_past_the_default_leaves_the_retry_at_the_default(
    tmp_path: Path,
) -> None:
    route = Route([FOREVER, 4_000], window=32_768, prompt_tokens=24_000)
    loop = _make_loop(route, tmp_path)
    result = await loop.arun_turn("question")
    assert result.stop_reason == "completed"
    assert _caps(route.calls) == [None, None]


async def test_stream_a_route_that_always_overflows_is_asked_for_more(tmp_path: Path) -> None:
    route = Route([FOREVER] * (1 + _MAX_EMPTY_RETRIES))
    loop = _make_loop(route, tmp_path)
    events = [ev async for ev in loop.astream_turn("question")]
    assert _stop_reason(events) == "gave_up"
    assert _caps(route.calls) == [None, 2 * DEFAULT_CAP, 2 * DEFAULT_CAP]


# ── what the give-up says ────────────────────────────────────────────────────────────────────


async def test_the_give_up_says_reasoning_exhausted_the_output_cap(tmp_path: Path) -> None:
    route = Route([FOREVER] * (1 + _MAX_EMPTY_RETRIES))
    loop = _make_loop(route, tmp_path)
    await loop.arun_turn("question")
    [note] = _notes(loop, "gave_up")
    assert "output cap" in note and "reasoning" in note
    assert "silent" not in note


async def test_stream_the_give_up_says_reasoning_exhausted_the_output_cap(tmp_path: Path) -> None:
    route = Route([FOREVER] * (1 + _MAX_EMPTY_RETRIES))
    loop = _make_loop(route, tmp_path)
    events = [ev async for ev in loop.astream_turn("question")]
    [note] = _notes(loop, "gave_up")
    assert "output cap" in note and "reasoning" in note
    stopping = [s for s in _statuses(events) if s.startswith("stopping:")]
    assert stopping and "output cap" in stopping[-1] and "reasoning" in stopping[-1]
    assert not any("went silent" in s for s in _statuses(events))


async def test_reasoning_that_stopped_on_its_own_is_not_called_the_cap(tmp_path: Path) -> None:
    thought = LLMResult(thinking="Here's a thinking process: 1.", finish_reason="stop")
    route = Route([thought] * (1 + _MAX_EMPTY_RETRIES))
    loop = _make_loop(route, tmp_path)
    await loop.arun_turn("question")
    [note] = _notes(loop, "gave_up")
    assert "reason" in note
    assert "output cap" not in note and "silent" not in note


async def test_plain_silence_still_reads_as_silence(tmp_path: Path) -> None:
    route = Route([LLMResult()] * (1 + _MAX_EMPTY_RETRIES))
    loop = _make_loop(route, tmp_path)
    result = await loop.arun_turn("question")
    assert result.stop_reason == "gave_up"
    assert _notes(loop, "gave_up") == ["model went silent — repeated empty completions"]
    assert _caps(route.calls) == [None, None, None]


# ── a refusal of the switch that names nothing ───────────────────────────────────────────────


async def test_an_unnamed_refusal_of_the_switch_does_not_end_the_turn(tmp_path: Path) -> None:
    route = Route([FOREVER, 12_000], refuses=frozenset({"switch"}))
    loop = _make_loop(route, tmp_path)
    result = await loop.arun_turn("question")
    assert result.stop_reason == "completed"
    assert result.assistant_messages[-1].text == "the answer"
    # Refused with the switch, then re-issued without it and still with the room.
    assert [call.get("extra_body") for call in route.calls] == [None, THINKING_OFF, None]
    assert _caps(route.calls) == [None, 2 * DEFAULT_CAP, 2 * DEFAULT_CAP]
    assert len(_notes(loop, "overflow_retry_refused")) == 1


async def test_a_route_that_refuses_the_raised_cap_too_gets_the_plain_call(
    tmp_path: Path,
) -> None:
    route = Route([FOREVER, 4_000], refuses=frozenset({"switch", "cap"}))
    loop = _make_loop(route, tmp_path)
    result = await loop.arun_turn("question")
    assert result.stop_reason == "completed"
    assert [call.get("extra_body") for call in route.calls] == [None, THINKING_OFF, None, None]
    assert _caps(route.calls) == [None, 2 * DEFAULT_CAP, 2 * DEFAULT_CAP, None]


async def test_stream_an_unnamed_refusal_of_the_switch_does_not_end_the_turn(
    tmp_path: Path,
) -> None:
    route = Route([FOREVER, 12_000], refuses=frozenset({"switch"}))
    loop = _make_loop(route, tmp_path)
    events = [ev async for ev in loop.astream_turn("question")]
    assert _stop_reason(events) == "completed"
    assert [call.get("extra_body") for call in route.calls] == [None, THINKING_OFF, None]
    assert _caps(route.calls) == [None, 2 * DEFAULT_CAP, 2 * DEFAULT_CAP]
    assert any("thinking switch" in s for s in _statuses(events))


async def test_a_refused_call_with_nothing_of_the_retry_on_it_still_ends_the_turn(
    tmp_path: Path,
) -> None:
    route = Route([], refuses=frozenset({"any"}))
    loop = _make_loop(route, tmp_path)
    result = await loop.arun_turn("question")
    assert result.stop_reason == "provider_error"
    assert len(route.calls) == 1


async def test_another_failure_of_the_retry_is_not_taken_for_a_refusal(tmp_path: Path) -> None:
    route = Route([FOREVER], refuses=frozenset({"switch"}), refusal=AuthError)
    loop = _make_loop(route, tmp_path)
    result = await loop.arun_turn("question")
    assert result.stop_reason == "provider_error"
    assert len(route.calls) == 2


# ── on the wire, through the provider ────────────────────────────────────────────────────────


class BadRequestError(Exception):
    """Named like litellm's: the provider matches the class by MRO name, plus the status."""

    status_code = 400


def _completion(cap: int, needs: int) -> SimpleNamespace:
    answered = cap >= needs
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="the answer" if answered else "", tool_calls=None),
                finish_reason="stop" if answered else "length",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=1_000, completion_tokens=min(cap, needs), total_tokens=1_000 + cap
        ),
        _hidden_params={"response_cost": 0.0},
    )


async def _chunks(cap: int, needs: int) -> AsyncIterator[Any]:
    answered = cap >= needs
    yield SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content="the answer" if answered else None, tool_calls=None),
                finish_reason="stop" if answered else "length",
            )
        ],
        usage=None,
    )


class Gateway:
    """A scripted ``litellm.acompletion`` for one route, every request recorded: it answers once
    the cap sent leaves room for ``needs`` tokens of reasoning, and may refuse the switch."""

    def __init__(self, *, needs: int, refuses_switch: bool = False) -> None:
        self.needs = needs
        self.refuses_switch = refuses_switch
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.refuses_switch and THINKING_SWITCH_KEY in (kwargs.get("extra_body") or {}):
            raise BadRequestError(f"litellm.BadRequestError: OpenAIException - {UNNAMED}")
        if kwargs.get("stream"):
            return _chunks(kwargs["max_tokens"], self.needs)
        return _completion(kwargs["max_tokens"], self.needs)


def _wire_loop(tmp_path: Path) -> tuple[AgentLoop, LiteLLMProvider]:
    provider = LiteLLMProvider(
        model="openai/route-b",
        api_base="https://gateway.invalid/v1",
        api_key="sk-test",
        context_window=131_072,
    )
    loop = AgentLoop(
        provider,
        ToolRegistry(),
        Session(cwd=str(tmp_path), model="openai/route-b"),
        settings=load_settings(workspace_root=tmp_path),
    )
    return loop, provider


def _switch(call: dict[str, Any]) -> bool:
    return THINKING_SWITCH_KEY in (call.get("extra_body") or {})


@pytest.mark.parametrize("path", ["buffered", "streamed"])
async def test_on_the_wire_the_route_is_sent_8192_and_then_more(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
) -> None:
    gateway = Gateway(needs=FOREVER)
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)
    loop, _ = _wire_loop(tmp_path)
    if path == "buffered":
        assert (await loop.arun_turn("question")).stop_reason == "gave_up"
    else:
        assert _stop_reason([ev async for ev in loop.astream_turn("question")]) == "gave_up"
    assert [call["max_tokens"] for call in gateway.calls] == [8_192, 16_384, 16_384]
    assert [_switch(call) for call in gateway.calls] == [False, True, True]


@pytest.mark.parametrize("path", ["buffered", "streamed"])
async def test_on_the_wire_an_unnamed_refusal_of_the_switch_is_re_issued_without_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
) -> None:
    gateway = Gateway(needs=12_000, refuses_switch=True)
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)
    loop, provider = _wire_loop(tmp_path)
    if path == "buffered":
        assert (await loop.arun_turn("question")).stop_reason == "completed"
    else:
        assert _stop_reason([ev async for ev in loop.astream_turn("question")]) == "completed"
    assert [call["max_tokens"] for call in gateway.calls] == [8_192, 16_384, 16_384]
    assert [_switch(call) for call in gateway.calls] == [False, True, False]
    # The provider could not learn it: nothing in the refusal names a field.
    assert provider.rejected_request_fields == []
