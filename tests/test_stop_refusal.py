"""A backend that refuses the ``stop`` it is sent is answered once, without it (ADR-0296).

Reported 2026-10-14 against an OpenAI-compatible gateway in front of Bedrock. In the text tool
protocol every request carries a ``stop`` list (``</tool_call>`` and ``<tool_result>`` first, then
chat-template tokens), and two of the open models behind the gateway answer that list with a 400
that ended the turn at its first iteration, with no tokens spent:

* GPT-oss: ``This model doesn't support the stopSequences field. Remove stopSequences and try
  again.``
* DeepSeek-R1: ``The stop sequence you provided at index 0 is longer than the model limit of 10
  characters. Try again with a stop sequence that is shorter than 10 characters.``

``stop`` is a top-level argument and neither sentence names it the way ``prompt_cache_key`` was
named, so the ADR-0181 repair did not see either. The provider now reads both, remembers what it
learned for the session, and re-issues the call once with the stop list the backend will take.

Hermetic: no network, no credentials (``conftest`` points ``ZAKCODE_HOME`` at a temporary folder,
so no ``~/.zakcode/.env`` is read). The two sentences are the report's, byte for byte. The gateway
that answers them is scripted from the wording and was not captured; the wire tests at the end
send the same sentences through litellm's own HTTP client so the exception the provider reads is
the one litellm raises.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import zakcode.providers.litellm_provider as lp
from tests.test_provider_stream import _chunk
from zakcode.agent.compact import Compactor
from zakcode.agent.loop import AgentLoop
from zakcode.config import PermissionTier, load_settings
from zakcode.messages import Message
from zakcode.providers import base
from zakcode.providers.base import (
    ProviderError,
    RequestFailed,
    StreamTextDelta,
    StreamToolCallDelta,
)
from zakcode.providers.litellm_provider import LiteLLMProvider
from zakcode.providers.text_tools import TextToolCallingProvider
from zakcode.session.store import Session
from zakcode.tools.base import (
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)

GATEWAY = "https://gateway.invalid/v1"
WINDOW = 131_072
MODEL = "openai/gpt-oss-120b"

#: The two refusals of the report, the gateway's sentences byte for byte.
NO_STOP_SENTENCE = (
    "This model doesn't support the stopSequences field. Remove stopSequences and try again."
)
LIMIT_SENTENCE = (
    "The stop sequence you provided at index 0 is longer than the model limit of 10 characters. "
    "Try again with a stop sequence that is shorter than 10 characters."
)


def limit_sentence(chars: int, index: int = 0) -> str:
    """The report's second sentence for another limit, and for another offending index."""
    return LIMIT_SENTENCE.replace("10 characters", f"{chars} characters").replace(
        "index 0", f"index {index}"
    )


def refusal(sentence: str) -> Exception:
    """What litellm raises for a gateway's 400 whose body is ``{"message": sentence}``; the
    wire tests below measure that this text is the one litellm produces."""
    return lp.litellm.BadRequestError(
        message=f"OpenAIException - {sentence}", model="gpt-oss", llm_provider="openai"
    )


#: The stop list the text protocol sends on an ``openai/`` route: the single-tool close tag, the
#: protocol's own sentinel, two chat-template tokens, cut to the route's cap of four.
SENTINELS = ["</tool_call>", "<tool_result>", "<|im_start|>", "<|im_end|>"]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]
TOOL_REPLY = '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>'


@pytest.fixture(autouse=True)
def _no_request_pacing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The provider spaces request starts a second apart; that sleep is not under test."""
    monkeypatch.setattr(lp, "_MIN_REQUEST_INTERVAL_S", 0.0)


def _provider() -> LiteLLMProvider:
    return LiteLLMProvider(model=MODEL, api_base=GATEWAY, api_key="sk-test", context_window=WINDOW)


def _answer(text: str) -> SimpleNamespace:
    message = SimpleNamespace(content=text, tool_calls=None, reasoning_content=None)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=usage,
        model="gpt-oss",
        _hidden_params={},
    )


class StopGateway:
    """``litellm.acompletion`` pointed at the gateway of the report.

    ``no_stop``: any request that carries a ``stop`` is the first sentence. ``max_chars``: a
    request with a stop sequence of that many characters or more is the second sentence, naming
    the first offender's index. Strictly shorter is the only length it is scripted to take,
    because the second sentence says both "longer than the model limit" and "shorter than".
    ``always_refuse``: every request is the first sentence, stop or not. Anything else is
    answered from ``replies`` in order, the last one repeating. ``requests`` records the kwargs
    of every call.
    """

    def __init__(
        self,
        *,
        no_stop: bool = False,
        max_chars: int | None = None,
        always_refuse: bool = False,
        replies: Sequence[str] = ("ok",),
    ) -> None:
        self.no_stop = no_stop
        self.max_chars = max_chars
        self.always_refuse = always_refuse
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        self.answered = 0

    def _sentence(self, kwargs: dict[str, Any]) -> str | None:
        stop = kwargs.get("stop")
        sequences = [stop] if isinstance(stop, str) else list(stop or [])
        if self.always_refuse or (sequences and self.no_stop):
            return NO_STOP_SENTENCE
        if self.max_chars is not None:
            for index, sequence in enumerate(sequences):
                if len(sequence) >= self.max_chars:
                    return limit_sentence(self.max_chars, index)
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

    def stops(self) -> list[Any]:
        """The ``stop`` each request carried; ``None`` where it carried none."""
        return [request.get("stop") for request in self.requests]


HOWS = ["acomplete", "astream"]


async def _run(provider: LiteLLMProvider, how: str, **kw: Any) -> str:
    """One call down the buffered path or the streamed one; the text the caller got."""
    if how == "acomplete":
        return (await provider.acomplete([Message.user("hi")], system="S", **kw)).text
    events = [e async for e in provider.astream([Message.user("hi")], system="S", **kw)]
    return "".join(e.text for e in events if isinstance(e, StreamTextDelta))


# ── the classifier ───────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        NO_STOP_SENTENCE,
        f"litellm.BadRequestError: OpenAIException - {NO_STOP_SENTENCE}",
        json.dumps({"message": NO_STOP_SENTENCE}),
        "This model doesn’t support the stopSequences field. Remove stopSequences.",
        "This model does not support the stopSequences field.",
    ],
)
def test_the_first_sentence_is_read_as_a_refusal_of_the_whole_field(text: str) -> None:
    assert base.stop_field_unsupported(text) is True
    assert base.stop_length_limit(text) is None


@pytest.mark.parametrize(
    "text, expected",
    [
        (LIMIT_SENTENCE, 10),
        (f"litellm.BadRequestError: OpenAIException - {LIMIT_SENTENCE}", 10),
        (json.dumps({"message": LIMIT_SENTENCE}), 10),
        (limit_sentence(24, index=3), 24),
    ],
)
def test_the_second_sentence_is_read_as_a_character_limit(text: str, expected: int) -> None:
    assert base.stop_length_limit(text) == expected
    assert base.stop_field_unsupported(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Rate limit exceeded, retry later.",
        "This model doesn't support the temperature field.",
        # A limit on something else is not a limit on a stop sequence.
        "The tool name you provided is longer than the model limit of 64 characters.",
        "The stop sequence you provided is fine, thank you.",
    ],
)
def test_other_complaints_are_not_a_stop_refusal(text: str) -> None:
    assert base.stop_field_unsupported(text) is False
    assert base.stop_length_limit(text) is None


def test_the_classifier_reads_an_exception_too() -> None:
    assert base.stop_field_unsupported(refusal(NO_STOP_SENTENCE)) is True
    assert base.stop_length_limit(refusal(LIMIT_SENTENCE)) == 10


# ── the provider: both call paths ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("how", HOWS)
async def test_an_ordinary_gateway_gets_the_stop_it_was_given(
    monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    """Control: without a refusal the list goes out as it came, in its order, once."""
    gateway = StopGateway()
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)

    assert await _run(_provider(), how, stop=SENTINELS) == "ok"

    assert gateway.stops() == [SENTINELS]


@pytest.mark.parametrize("how", HOWS)
async def test_a_gateway_that_takes_no_stop_is_asked_once_more_without_one(
    monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    gateway = StopGateway(no_stop=True)
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)
    provider = _provider()

    assert await _run(provider, how, stop=SENTINELS) == "ok"

    assert gateway.stops() == [SENTINELS, None]
    assert "stop" in provider.rejected_request_fields


@pytest.mark.parametrize("how", HOWS)
async def test_what_the_gateway_refused_is_not_asked_again(
    monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    """The first call spends the refusal; the second sends no stop and is not refused."""
    gateway = StopGateway(no_stop=True)
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)
    provider = _provider()

    await _run(provider, how, stop=SENTINELS)
    await _run(provider, how, stop=SENTINELS)

    assert gateway.stops() == [SENTINELS, None, None]


@pytest.mark.parametrize("how", HOWS)
async def test_a_stop_over_the_limit_is_cut_to_the_sequences_that_fit(
    monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    gateway = StopGateway(max_chars=10)
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)
    provider = _provider()

    assert await _run(provider, how, stop=["</tool_call>", "END", "<tool_result>", "###"]) == "ok"

    assert gateway.stops() == [["</tool_call>", "END", "<tool_result>", "###"], ["END", "###"]]
    assert provider.max_stop_chars == 10
    assert "stop" not in provider.rejected_request_fields


@pytest.mark.parametrize("how", HOWS)
async def test_a_sequence_of_exactly_the_limit_is_not_sent(
    monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    """The sentence says "longer than the model limit of 10" and asks for a sequence "shorter
    than 10". They differ at exactly ten characters, and only "shorter" is safe under both
    readings: a sequence of ten that is refused costs the turn this repair exists to save."""
    ten, nine = "<|im_end|>", "<im_end|>"
    assert (len(ten), len(nine)) == (10, 9)
    gateway = StopGateway(max_chars=10)
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)

    await _run(_provider(), how, stop=[ten, nine])

    assert gateway.stops() == [[ten, nine], [nine]]


@pytest.mark.parametrize("how", HOWS)
async def test_when_nothing_fits_no_stop_is_sent_at_all(
    monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    """The text protocol's own sentinels are all ten characters or more: for a limit of ten
    none is left, and the request carries no ``stop`` key rather than an empty list."""
    gateway = StopGateway(max_chars=10)
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)

    assert await _run(_provider(), how, stop=SENTINELS) == "ok"

    assert gateway.stops() == [SENTINELS, None]
    assert "stop" not in gateway.requests[1]


async def test_a_larger_limit_keeps_what_fits_under_it(monkeypatch: pytest.MonkeyPatch) -> None:
    gateway = StopGateway(max_chars=12)
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)

    await _run(_provider(), "acomplete", stop=SENTINELS)

    assert gateway.stops() == [SENTINELS, ["<|im_end|>"]]


async def test_a_stricter_refusal_later_tightens_the_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The learned limit only goes down: a backend that turns out stricter than the first
    sentence said is learned again, and nothing sent afterwards loosens it."""
    gateway = StopGateway(max_chars=12)
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)
    provider = _provider()
    stop = ["</tool_call>", "<|im_end|>", "<im_end|>", "END"]
    assert [len(s) for s in stop] == [12, 10, 9, 3]

    await _run(provider, "acomplete", stop=stop)
    assert provider.max_stop_chars == 12
    gateway.max_chars = 9
    await _run(provider, "acomplete", stop=stop)

    assert provider.max_stop_chars == 9
    # Refused at twelve and cut to the three that fit; refused again at nine, and cut to the one.
    assert gateway.stops() == [stop, stop[1:], stop[1:], ["END"]]


def test_a_looser_refusal_arriving_late_does_not_loosen_the_limit() -> None:
    """Two calls in flight on one provider can be refused out of order: the stricter sentence
    is learned first and the looser one lands after it. The limit stays at the stricter, and
    the late call is still re-issued, now through the filter."""
    provider = _provider()
    sent = {"stop": ["x" * 20]}

    assert provider._refuse_stop(refusal(limit_sentence(9)), sent) is True
    assert provider._refuse_stop(refusal(limit_sentence(12)), sent) is True

    assert provider.max_stop_chars == 9


@pytest.mark.parametrize("how", HOWS)
async def test_a_refusal_of_a_stop_we_never_sent_is_surfaced_not_repaired(
    monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    gateway = StopGateway(always_refuse=True)
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)
    provider = _provider()

    with pytest.raises(RequestFailed, match="stopSequences"):
        await _run(provider, how)

    assert len(gateway.requests) == 1
    assert provider.rejected_request_fields == []
    assert provider.max_stop_chars is None


@pytest.mark.parametrize("how", HOWS)
async def test_the_repair_is_spent_once_per_call(monkeypatch: pytest.MonkeyPatch, how: str) -> None:
    """A gateway that refuses the re-issued request too is reported as is. The mechanism never
    loops."""
    gateway = StopGateway(always_refuse=True)
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)

    with pytest.raises(RequestFailed, match="stopSequences"):
        await _run(_provider(), how, stop=SENTINELS)

    assert gateway.stops() == [SENTINELS, None]


@pytest.mark.parametrize("how", HOWS)
async def test_a_limit_on_something_else_is_not_a_stop_refusal(
    monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    other = "The tool name you provided is longer than the model limit of 64 characters."

    async def gateway(**kwargs: Any) -> Any:
        requests.append(kwargs)
        raise refusal(other)

    requests: list[dict[str, Any]] = []
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)
    provider = _provider()

    with pytest.raises(RequestFailed, match="tool name"):
        await _run(provider, how, stop=SENTINELS)

    assert len(requests) == 1
    assert provider.max_stop_chars is None and provider.rejected_request_fields == []


async def test_a_5xx_quoting_the_sentence_is_not_a_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only a 4xx that rejects the request's shape can be a refusal of a field; a server error
    that happens to quote the sentence is the server's trouble."""

    async def gateway(**kwargs: Any) -> Any:
        requests.append(kwargs)
        raise lp.litellm.InternalServerError(
            message=f"OpenAIException - {NO_STOP_SENTENCE}", model="gpt-oss", llm_provider="openai"
        )

    requests: list[dict[str, Any]] = []
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)
    provider = _provider()

    with pytest.raises(ProviderError):
        await provider.acomplete([Message.user("hi")], system="S", stop=SENTINELS)

    assert len(requests) == 1
    assert provider.rejected_request_fields == []


# ── the text protocol over a refusing gateway ────────────────────────────────────────────────


def _text_mode(provider: LiteLLMProvider) -> TextToolCallingProvider:
    return TextToolCallingProvider(provider, mode="text", single_tool_per_turn=True)


@pytest.mark.parametrize(
    "gateway_kwargs",
    [
        pytest.param({"no_stop": True}, id="gpt-oss"),
        pytest.param({"max_chars": 10}, id="deepseek-r1"),
    ],
)
async def test_a_text_mode_call_parses_after_the_repair(
    monkeypatch: pytest.MonkeyPatch, gateway_kwargs: dict[str, Any]
) -> None:
    gateway = StopGateway(replies=[TOOL_REPLY], **gateway_kwargs)
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)

    result = await _text_mode(_provider()).acomplete(
        [Message.user("read it")], system="S", tools=TOOLS
    )

    assert gateway.stops() == [SENTINELS, None]
    [call] = result.tool_calls
    assert (call.name, call.arguments) == ("read_file", {"path": "a.py"})


@pytest.mark.parametrize(
    "gateway_kwargs",
    [
        pytest.param({"no_stop": True}, id="gpt-oss"),
        pytest.param({"max_chars": 10}, id="deepseek-r1"),
    ],
)
async def test_a_streamed_text_mode_call_parses_after_the_repair(
    monkeypatch: pytest.MonkeyPatch, gateway_kwargs: dict[str, Any]
) -> None:
    gateway = StopGateway(replies=[TOOL_REPLY], **gateway_kwargs)
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)

    events = [
        e
        async for e in _text_mode(_provider()).astream(
            [Message.user("read it")], system="S", tools=TOOLS
        )
    ]

    assert gateway.stops() == [SENTINELS, None]
    [call] = [e for e in events if isinstance(e, StreamToolCallDelta)]
    assert call.name == "read_file"
    assert json.loads(call.arguments_delta) == {"path": "a.py"}


# ── the agent loop, whose first iteration the refusal used to end ────────────────────────────


class _Probe(Tool):
    spec = ToolSpec(
        name="probe",
        description="a tool the model may call",
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    def __init__(self) -> None:
        self.runs = 0

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.runs += 1
        return ToolResult.ok("done")


PROBE_REPLY = '<tool_call>{"name": "probe", "arguments": {}}</tool_call>'


def _loop(tmp_path: Path, probe: _Probe) -> AgentLoop:
    registry = ToolRegistry()
    registry.register(probe)
    return AgentLoop(
        _text_mode(_provider()),
        registry,
        Session(cwd=str(tmp_path), model=MODEL),
        settings=load_settings(workspace_root=tmp_path),
        compactor=Compactor(),
    )


@pytest.mark.parametrize(
    "gateway_kwargs",
    [
        pytest.param({"no_stop": True}, id="gpt-oss"),
        pytest.param({"max_chars": 10}, id="deepseek-r1"),
    ],
)
async def test_a_turn_goes_on_past_its_first_iteration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, gateway_kwargs: dict[str, Any]
) -> None:
    gateway = StopGateway(replies=[PROBE_REPLY, "all done"], **gateway_kwargs)
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)
    probe = _Probe()

    result = await _loop(tmp_path, probe).arun_turn("use the probe")

    assert result.stop_reason == "completed", result.error
    assert probe.runs == 1
    # Refused, re-issued, then the second iteration: three requests and one stop list, the first.
    assert [("stop" in request) for request in gateway.requests] == [True, False, False]


@pytest.mark.parametrize(
    "gateway_kwargs",
    [
        pytest.param({"no_stop": True}, id="gpt-oss"),
        pytest.param({"max_chars": 10}, id="deepseek-r1"),
    ],
)
async def test_a_streamed_turn_goes_on_past_its_first_iteration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, gateway_kwargs: dict[str, Any]
) -> None:
    gateway = StopGateway(replies=[PROBE_REPLY, "all done"], **gateway_kwargs)
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)
    probe = _Probe()

    events = [e async for e in _loop(tmp_path, probe).astream_turn("use the probe")]

    assert events[-1].stop_reason == "completed", events[-1].error
    assert probe.runs == 1
    assert [("stop" in request) for request in gateway.requests] == [True, False, False]


# ── what litellm raises for the real HTTP answer ─────────────────────────────────────────────

_CHAT_BODY: dict[str, Any] = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-oss",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ready"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
}


class Wire:
    """Every request litellm sends, answered from a canned body: a 400 ``{"message": ...}`` for
    a request whose ``stop`` the ``sentence_for`` rule objects to, a 200 otherwise."""

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


def _no_stop_rule(body: dict[str, Any]) -> str | None:
    return NO_STOP_SENTENCE if body.get("stop") else None


def _limit_rule(body: dict[str, Any]) -> str | None:
    stop = body.get("stop") or []
    stop = [stop] if isinstance(stop, str) else stop
    for index, sequence in enumerate(stop):
        if len(sequence) >= 10:
            return limit_sentence(10, index)
    return None


@pytest.mark.parametrize(
    "rule, sentence",
    [
        pytest.param(_no_stop_rule, NO_STOP_SENTENCE, id="gpt-oss"),
        pytest.param(_limit_rule, LIMIT_SENTENCE, id="deepseek-r1"),
    ],
)
async def test_litellm_raises_the_exception_the_scripted_gateway_raises(
    wire: Wire, rule: Any, sentence: str
) -> None:
    """The fake gateway above raises ``refusal(sentence)``. This is the real thing: the same
    400 body through litellm's own client gives an exception with the same text and a 4xx
    status, so what the tests above prove is about the exception the provider will read."""
    wire.sentence_for = rule
    kwargs = _provider()._build_kwargs([{"role": "user", "content": "hi"}], None, stop=SENTINELS)

    with pytest.raises(lp.litellm.BadRequestError) as raised:
        await lp.litellm.acompletion(**kwargs)

    assert str(raised.value) == str(refusal(sentence))
    assert raised.value.status_code == 400


@pytest.mark.parametrize(
    "rule",
    [
        pytest.param(_no_stop_rule, id="gpt-oss"),
        pytest.param(_limit_rule, id="deepseek-r1"),
    ],
)
async def test_the_request_on_the_wire_is_repaired(wire: Wire, rule: Any) -> None:
    wire.sentence_for = rule

    result = await _provider().acomplete([Message.user("hi")], system="S", stop=SENTINELS)

    assert result.text == "ready"
    assert [body.get("stop") for body in wire.bodies] == [SENTINELS, None]
