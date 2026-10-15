"""A route that refuses ``temperature`` or tool use is answered without it (ADR-0303).

Reported 2026-10-14 against a LiteLLM 1.9x gateway in front of Bedrock/Vertex, as the sequel to
the ``stop`` refusal of ADR-0296. Two more 400s, relayed by the gateway in Bedrock's own words,
ended every call that met them: one request per call, nothing learned, no re-issue.

* ``This model doesn't support the temperature field. Remove temperature and try again.``
* ``This model doesn't support tool use.``

``temperature`` is a sampling field the provider adds (the configured value, or the 0.0 the side
calls pass), so a refusal of it is learned like the stop field: left out of every later request,
and the call re-issued. Tool use is not a field to strip: the call needs its tools, so a refusal
of them switches the provider to the text protocol for the session, which carries the tools in
the prompt, and the call is answered that way. A route can refuse more than one of these, so a
call is repaired once per distinct refusal, at most three times, where ADR-0296 allowed one.

Hermetic: no network, no credentials (``conftest`` points ``ZAKCODE_HOME`` at a temporary folder,
so no ``~/.zakcode/.env`` is read). The two sentences are the report's, byte for byte. The gateway
that answers them is scripted from the wording and was not captured; the wire tests at the end
send the same sentences through litellm's own HTTP client, so the exception the provider reads is
the one litellm raises.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest

import zakcode.providers.litellm_provider as lp
from tests.test_provider_stream import _chunk
from tests.test_stop_refusal import (
    LIMIT_SENTENCE,
    NO_STOP_SENTENCE,
    PROBE_REPLY,
    SENTINELS,
    TOOL_REPLY,
    TOOLS,
    _answer,
    _Probe,
    limit_sentence,
    refusal,
)
from zakcode.agent.compact import Compactor
from zakcode.agent.loop import AgentLoop
from zakcode.config import load_settings
from zakcode.messages import Message
from zakcode.providers import base
from zakcode.providers.base import RequestFailed, StreamTextDelta, StreamToolCallDelta
from zakcode.providers.litellm_provider import LiteLLMProvider
from zakcode.providers.text_tools import TextToolCallingProvider
from zakcode.session.store import Session
from zakcode.tools.base import ToolRegistry

GATEWAY = "https://gateway.invalid/v1"
WINDOW = 131_072
ROUTE = "openai/route-a"

#: The two refusals of the report, the gateway's sentences byte for byte.
TEMPERATURE_SENTENCE = (
    "This model doesn't support the temperature field. Remove temperature and try again."
)
TOOL_USE_SENTENCE = "This model doesn't support tool use."


@pytest.fixture(autouse=True)
def _no_request_pacing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The provider spaces request starts a second apart; that sleep is not under test."""
    monkeypatch.setattr(lp, "_MIN_REQUEST_INTERVAL_S", 0.0)


def _provider(temperature: float | None = 0.2) -> LiteLLMProvider:
    return LiteLLMProvider(
        model=ROUTE,
        api_base=GATEWAY,
        api_key="sk-test",
        context_window=WINDOW,
        temperature=temperature,
    )


def _wrapped(mode: str, temperature: float | None = None) -> TextToolCallingProvider:
    return TextToolCallingProvider(_provider(temperature), mode=mode, single_tool_per_turn=True)


class Gateway:
    """``litellm.acompletion`` pointed at a route that refuses what it is told to.

    Each request is checked in the order ``tools``, ``stop``, ``temperature``: the first of
    those it carries that the route refuses is answered with that refusal, so a route refusing
    several refuses them one per request. ``limits``: the stop-length limit this route quotes
    for each successive request carrying a stop, the last repeating. ``always``: every request
    is answered with this sentence, whatever it carries. Anything else is answered from
    ``replies`` in order, the last one repeating. ``requests`` records every call's kwargs.
    """

    def __init__(
        self,
        *,
        tools: bool = False,
        stop: bool = False,
        temperature: bool = False,
        limits: Sequence[int] = (),
        always: str | None = None,
        replies: Sequence[str] = ("ok",),
    ) -> None:
        self.refuses = {"tools": tools, "stop": stop, "temperature": temperature}
        self.limits = list(limits)
        self.always = always
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        self.answered = 0

    def _sentence(self, kwargs: dict[str, Any]) -> str | None:
        if self.always is not None:
            return self.always
        if self.refuses["tools"] and kwargs.get("tools"):
            return TOOL_USE_SENTENCE
        if self.refuses["stop"] and kwargs.get("stop"):
            return NO_STOP_SENTENCE
        if self.limits and kwargs.get("stop"):
            asked = sum(1 for request in self.requests[:-1] if request.get("stop"))
            limit = self.limits[min(asked, len(self.limits) - 1)]
            for index, sequence in enumerate(kwargs["stop"]):
                if len(sequence) >= limit:
                    return limit_sentence(limit, index)
        if self.refuses["temperature"] and kwargs.get("temperature") is not None:
            return TEMPERATURE_SENTENCE
        return None

    async def __call__(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        sentence = self._sentence(kwargs)
        if sentence is not None:
            raise refusal(sentence)
        text = self.replies[min(self.answered, len(self.replies) - 1)]
        self.answered += 1
        if not kwargs.get("stream"):
            return _answer(text)

        async def _stream() -> AsyncIterator[Any]:
            yield _chunk(content=text)
            yield _chunk(content="", finish_reason="stop")

        return _stream()

    def sent(self, field: str) -> list[Any]:
        """The ``field`` each request carried; ``None`` where it carried none."""
        return [request.get(field) for request in self.requests]


HOWS = ["acomplete", "astream"]


async def _run(provider: Any, how: str, **kw: Any) -> tuple[str, list[tuple[str, Any]]]:
    """One call down the buffered path or the streamed one: the text and the tool calls."""
    if how == "acomplete":
        result = await provider.acomplete([Message.user("hi")], system="S", **kw)
        return result.text, [(call.name, call.arguments) for call in result.tool_calls]
    events = [e async for e in provider.astream([Message.user("hi")], system="S", **kw)]
    text = "".join(e.text for e in events if isinstance(e, StreamTextDelta))
    calls = [
        (e.name, json.loads(e.arguments_delta))
        for e in events
        if isinstance(e, StreamToolCallDelta)
    ]
    return text, calls


def _system(request: dict[str, Any]) -> str:
    return next(m["content"] for m in request["messages"] if m["role"] == "system")


# ── the classifier ───────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        TEMPERATURE_SENTENCE,
        f"litellm.BadRequestError: OpenAIException - {TEMPERATURE_SENTENCE}",
        json.dumps({"message": TEMPERATURE_SENTENCE}),
        "This model doesn’t support the temperature field. Remove temperature.",
        "This model does not support the temperature field.",
    ],
)
def test_the_temperature_sentence_is_read_as_a_refusal_of_the_field(text: str) -> None:
    assert base.temperature_field_unsupported(text) is True
    assert base.tool_use_unsupported(text) is False
    assert base.stop_field_unsupported(text) is False


@pytest.mark.parametrize(
    "text",
    [
        TOOL_USE_SENTENCE,
        f"litellm.BadRequestError: OpenAIException - {TOOL_USE_SENTENCE}",
        json.dumps({"message": TOOL_USE_SENTENCE}),
        "This model doesn’t support tool use.",
        "This model does not support tool use.",
    ],
)
def test_the_tool_use_sentence_is_read_as_a_refusal_of_the_tools(text: str) -> None:
    assert base.tool_use_unsupported(text) is True
    assert base.temperature_field_unsupported(text) is False


@pytest.mark.parametrize(
    "text",
    [
        NO_STOP_SENTENCE,
        LIMIT_SENTENCE,
        "temperature must be between 0 and 2",
        "This model doesn't support the stopSequences field.",
        "Tool use is required for this request.",
    ],
)
def test_other_complaints_are_neither(text: str) -> None:
    assert base.temperature_field_unsupported(text) is False
    assert base.tool_use_unsupported(text) is False


def test_the_classifiers_read_an_exception_too() -> None:
    assert base.temperature_field_unsupported(refusal(TEMPERATURE_SENTENCE)) is True
    assert base.tool_use_unsupported(refusal(TOOL_USE_SENTENCE)) is True


# ── temperature: learned like the stop field ─────────────────────────────────────────────────


@pytest.mark.parametrize("how", HOWS)
async def test_an_ordinary_route_gets_the_temperature_it_was_given(
    monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    gateway = Gateway()
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)

    assert (await _run(_provider(), how))[0] == "ok"
    assert gateway.sent("temperature") == [0.2]


@pytest.mark.parametrize("how", HOWS)
async def test_a_refused_temperature_is_left_out_and_the_call_re_issued(
    monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    gateway = Gateway(temperature=True)
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)
    provider = _provider()

    assert (await _run(provider, how))[0] == "ok"
    assert gateway.sent("temperature") == [0.2, None]
    assert provider.rejected_request_fields == ["temperature"]


@pytest.mark.parametrize("how", HOWS)
async def test_a_refused_temperature_is_not_asked_again(
    monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    """One wasted request per provider instance: the second call goes out without it."""
    gateway = Gateway(temperature=True)
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)
    provider = _provider()

    await _run(provider, how)
    await _run(provider, how)

    assert gateway.sent("temperature") == [0.2, None, None]


@pytest.mark.parametrize("how", HOWS)
async def test_a_side_calls_temperature_of_zero_is_refused_and_learned_too(
    monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    """The side calls pass ``temperature=0.0`` per call to a provider configured with none. A
    zero is a temperature the request carried, not an absent one."""
    gateway = Gateway(temperature=True)
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)
    provider = _provider(temperature=None)

    assert (await _run(provider, how, temperature=0.0))[0] == "ok"
    assert gateway.sent("temperature") == [0.0, None]
    assert provider.rejected_request_fields == ["temperature"]


@pytest.mark.parametrize("how", HOWS)
async def test_a_refusal_of_a_temperature_we_never_sent_is_surfaced_not_repaired(
    monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    gateway = Gateway(always=TEMPERATURE_SENTENCE)
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)
    provider = _provider(temperature=None)

    with pytest.raises(RequestFailed, match="temperature field"):
        await _run(provider, how)

    assert len(gateway.requests) == 1
    assert provider.rejected_request_fields == []


async def test_a_5xx_quoting_the_temperature_sentence_is_not_a_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []

    async def gateway(**kwargs: Any) -> Any:
        requests.append(kwargs)
        raise lp.litellm.InternalServerError(
            message=f"OpenAIException - {TEMPERATURE_SENTENCE}",
            model="route-a",
            llm_provider="openai",
        )

    monkeypatch.setattr(lp.litellm, "acompletion", gateway)
    provider = _provider()

    with pytest.raises(base.ProviderError):
        await provider.acomplete([Message.user("hi")], system="S")

    assert provider.rejected_request_fields == []


# ── the repair: once per distinct refusal, at most three ─────────────────────────────────────


@pytest.mark.parametrize("how", HOWS)
async def test_a_call_with_stop_and_temperature_to_a_route_refusing_both_completes(
    monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    """Refused for its stop, re-issued, refused for its temperature, re-issued, answered. With
    one repair per call the second refusal ended the call."""
    gateway = Gateway(stop=True, temperature=True)
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)
    provider = _provider()

    assert (await _run(provider, how, stop=SENTINELS))[0] == "ok"
    assert gateway.sent("stop") == [SENTINELS, None, None]
    assert gateway.sent("temperature") == [0.2, 0.2, None]
    assert provider.rejected_request_fields == ["stop", "temperature"]


@pytest.mark.parametrize("how", HOWS)
async def test_the_repair_stops_after_three(monkeypatch: pytest.MonkeyPatch, how: str) -> None:
    """A route that teaches something new on every request is still given four requests, not
    more: the fourth refusal is reported as it is."""
    stops = ["a" * 20, "b" * 15, "c" * 12, "d" * 10, "e" * 5]
    gateway = Gateway(limits=[20, 15, 12, 10])
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)

    with pytest.raises(RequestFailed, match="limit of 10"):
        await _run(_provider(), how, stop=stops)

    assert gateway.sent("stop") == [stops, stops[1:], stops[2:], stops[3:]]


@pytest.mark.parametrize("how", HOWS)
async def test_a_repeated_refusal_that_teaches_nothing_is_reported_as_it_is(
    monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    """The bound is not a retry count. A route that quotes the same limit again once nothing it
    was sent reaches that length would be sent the request it just refused, twice more."""
    gateway = Gateway(always=limit_sentence(12))
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)
    provider = _provider()

    with pytest.raises(RequestFailed, match="limit of 12"):
        await _run(provider, how, stop=SENTINELS)

    assert gateway.sent("stop") == [SENTINELS, ["<|im_end|>"]]
    assert provider.max_stop_chars == 12


# ── tool use: the session moves to the text protocol ─────────────────────────────────────────


@pytest.mark.parametrize("how", HOWS)
@pytest.mark.parametrize("mode", ["native", "auto"])
async def test_a_route_that_refuses_tool_use_is_answered_through_the_text_protocol(
    monkeypatch: pytest.MonkeyPatch, mode: str, how: str
) -> None:
    gateway = Gateway(tools=True, replies=[TOOL_REPLY])
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)
    provider = _wrapped(mode)
    # Positive control: before the refusal this provider answers tools natively.
    assert provider._use_text_mode(TOOLS) is False

    text, calls = await _run(provider, how, tools=TOOLS)

    assert calls == [("read_file", {"path": "a.py"})]
    assert text == ""
    first, second = gateway.requests
    assert first["tools"] == TOOLS
    assert "tools" not in second
    assert "<tool_call>" in _system(second)
    assert second["stop"] == SENTINELS


@pytest.mark.parametrize("how", HOWS)
async def test_the_text_protocol_is_kept_for_the_session(
    monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    """One wasted request per provider instance: the next call goes out in the text protocol."""
    gateway = Gateway(tools=True, replies=[TOOL_REPLY])
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)
    provider = _wrapped("native")

    await _run(provider, how, tools=TOOLS)
    await _run(provider, how, tools=TOOLS)

    assert [("tools" in request) for request in gateway.requests] == [True, False, False]


async def test_a_new_provider_instance_asks_for_native_tools_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway = Gateway(tools=True, replies=[TOOL_REPLY])
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)

    await _run(_wrapped("native"), "acomplete", tools=TOOLS)
    await _run(_wrapped("native"), "acomplete", tools=TOOLS)

    assert [("tools" in request) for request in gateway.requests] == [True, False, True, False]


@pytest.mark.parametrize("how", HOWS)
async def test_a_tool_use_refusal_of_a_call_without_tools_is_surfaced(
    monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    gateway = Gateway(always=TOOL_USE_SENTENCE)
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)
    provider = _wrapped("native")

    with pytest.raises(RequestFailed, match="tool use") as raised:
        await _run(provider, how)

    # The plain failure it always was: only a call that sent tools is answered as refusing them.
    assert not isinstance(raised.value, base.ToolUseRefused)
    assert len(gateway.requests) == 1
    assert provider._use_text_mode(TOOLS) is False


@pytest.mark.parametrize("how", HOWS)
async def test_a_route_refusing_tools_stop_and_temperature_completes(
    monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    """One wasted request per refusal, three in all, and the next call wastes none."""
    gateway = Gateway(tools=True, stop=True, temperature=True, replies=[TOOL_REPLY])
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)
    provider = _wrapped("native", temperature=0.2)

    assert (await _run(provider, how, tools=TOOLS))[1] == [("read_file", {"path": "a.py"})]
    assert (await _run(provider, how, tools=TOOLS))[1] == [("read_file", {"path": "a.py"})]

    shapes = [
        ("tools" in r, r.get("stop") is not None, r.get("temperature") is not None)
        for r in gateway.requests
    ]
    assert shapes == [
        (True, False, True),  # refused: tool use
        (False, True, True),  # the text protocol, refused: stop
        (False, False, True),  # refused: temperature
        (False, False, False),  # answered
        (False, False, False),  # the next call, answered at once
    ]


# ── the agent loop, whose turn the refusal used to end ───────────────────────────────────────


def _loop(tmp_path: Path, probe: _Probe) -> AgentLoop:
    registry = ToolRegistry()
    registry.register(probe)
    return AgentLoop(
        _wrapped("native"),
        registry,
        Session(cwd=str(tmp_path), model=ROUTE),
        settings=load_settings(workspace_root=tmp_path),
        compactor=Compactor(),
    )


async def test_a_turn_on_a_route_that_refuses_tool_use_goes_on(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gateway = Gateway(tools=True, replies=[PROBE_REPLY, "all done"])
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)
    probe = _Probe()

    result = await _loop(tmp_path, probe).arun_turn("use the probe")

    assert result.stop_reason == "completed", result.error
    assert probe.runs == 1
    assert [("tools" in request) for request in gateway.requests] == [True, False, False]


async def test_a_streamed_turn_on_a_route_that_refuses_tool_use_goes_on(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gateway = Gateway(tools=True, replies=[PROBE_REPLY, "all done"])
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)
    probe = _Probe()

    events = [e async for e in _loop(tmp_path, probe).astream_turn("use the probe")]

    assert events[-1].stop_reason == "completed", events[-1].error
    assert probe.runs == 1
    assert [("tools" in request) for request in gateway.requests] == [True, False, False]


# ── what litellm raises for the real HTTP answer ─────────────────────────────────────────────

_CHAT_BODY: dict[str, Any] = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1,
    "model": "route-a",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ready"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
}


class Wire:
    """Every request litellm sends, answered from a canned body: a 400 ``{"message": ...}`` for
    a request the ``sentence_for`` rule objects to, a 200 otherwise."""

    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []
        self.sentence_for: Any = None


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> Wire:
    seen = Wire()

    async def send(self: httpx.AsyncClient, request: httpx.Request, **kw: Any) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen.bodies.append(body)
        sentence = seen.sentence_for(body) if seen.sentence_for else None
        if sentence:
            return httpx.Response(400, json={"message": sentence}, request=request)
        return httpx.Response(200, json=_CHAT_BODY, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    for name in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
        monkeypatch.delenv(name, raising=False)
    return seen


def _temperature_rule(body: dict[str, Any]) -> str | None:
    return TEMPERATURE_SENTENCE if body.get("temperature") is not None else None


def _tools_rule(body: dict[str, Any]) -> str | None:
    return TOOL_USE_SENTENCE if body.get("tools") else None


@pytest.mark.parametrize(
    "rule, sentence, tools",
    [
        pytest.param(_temperature_rule, TEMPERATURE_SENTENCE, None, id="temperature"),
        pytest.param(_tools_rule, TOOL_USE_SENTENCE, TOOLS, id="tool-use"),
    ],
)
async def test_litellm_raises_the_exception_the_scripted_gateway_raises(
    wire: Wire, rule: Any, sentence: str, tools: Any
) -> None:
    """The fake gateway above raises ``refusal(sentence)``. This is the real thing: the same
    400 body through litellm's own client gives an exception with the same text and a 4xx
    status, so what the tests above prove is about the exception the provider will read."""
    wire.sentence_for = rule
    kwargs = _provider()._build_kwargs([{"role": "user", "content": "hi"}], tools)

    with pytest.raises(lp.litellm.BadRequestError) as raised:
        await lp.litellm.acompletion(**kwargs)

    assert str(raised.value) == str(refusal(sentence))
    assert raised.value.status_code == 400


async def test_the_temperature_on_the_wire_is_left_out_after_the_refusal(wire: Wire) -> None:
    wire.sentence_for = _temperature_rule

    result = await _provider().acomplete([Message.user("hi")], system="S")

    assert result.text == "ready"
    assert [body.get("temperature") for body in wire.bodies] == [0.2, None]


async def test_the_tools_on_the_wire_become_the_text_protocol_after_the_refusal(
    wire: Wire,
) -> None:
    wire.sentence_for = _tools_rule

    result = await _wrapped("native").acomplete([Message.user("hi")], system="S", tools=TOOLS)

    assert result.text == "ready"
    assert [("tools" in body) for body in wire.bodies] == [True, False]
