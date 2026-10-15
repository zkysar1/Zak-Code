"""A refusal is read from the provider's own message, not from text it only quotes (ADR-0311).

Found on 2026-10-15 by a hostile review of ADR-0303. The tool-use, temperature and stop
detectors searched the whole text of a 4xx for their sentence, gated only on the request having
carried the field. A 400 about something else that merely QUOTES one of the sentences, as a
validation error that prints the offending input does, read as the refusal: the provider moved
the session to the text protocol, or dropped a configured temperature of 0, or the stop, for the
rest of the session. Such an echo is realistic: this repository's own docs and tests carry every
one of the sentences, and an agent sends files like those to the model.

A refusal is now read where a backend puts it: at the start of the provider's own message, once
litellm's wrapper, a LiteLLM proxy's wrapper and the vendor's JSON envelope are peeled off. The
live bodies are ADR-0303's and ADR-0296's sentences, byte for byte, in the two shapes a LiteLLM
1.9x gateway in front of Bedrock/Vertex answers with: the sentence as the body's message, and a
LiteLLM proxy's own wrapper around Bedrock's JSON with the router's trailer.

Hermetic: no network, no credentials. Each exception is the one litellm's own client raises for
the 400 body, built against a patched transport on the reserved ``.invalid`` host; the call-path
tests then raise that exception from a scripted gateway, so the text the provider reads is the
text litellm produces.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest

import zakcode.providers.litellm_provider as lp
from tests.test_provider_stream import _chunk
from tests.test_stop_refusal import (
    LIMIT_SENTENCE,
    NO_STOP_SENTENCE,
    SENTINELS,
    TOOL_REPLY,
    TOOLS,
    _answer,
)
from tests.test_temperature_tools_refusal import (
    GATEWAY,
    HOWS,
    ROUTE,
    TEMPERATURE_SENTENCE,
    TOOL_USE_SENTENCE,
    _provider,
    _run,
    _wrapped,
)
from zakcode.providers import base
from zakcode.providers.base import (
    RequestFailed,
    ToolUseRefused,
    stop_field_unsupported,
    stop_length_limit,
    temperature_field_unsupported,
    tool_use_unsupported,
)

#: The wording a backend uses for tool use on the streamed path; it must keep reading as one.
TOOL_USE_STREAMED_SENTENCE = "This model doesn't support tool use in streaming mode."


@pytest.fixture(autouse=True)
def _no_request_pacing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The provider spaces request starts a second apart; that sleep is not under test."""
    monkeypatch.setattr(lp, "_MIN_REQUEST_INTERVAL_S", 0.0)


# ── the 400 bodies ───────────────────────────────────────────────────────────────────────────


def bare(sentence: str) -> dict[str, Any]:
    """The gateway's 400 with the backend's sentence as its message."""
    return {"message": sentence}


def proxied(message: str) -> dict[str, Any]:
    """A LiteLLM proxy's 400 for a Bedrock answer: its own wrapper, Bedrock's JSON, and the
    router's trailer, as the gateway sends them."""
    bedrock = json.dumps({"message": message}, separators=(",", ":"))
    return {
        "error": {
            "message": f"litellm.BadRequestError: BedrockException - {bedrock}. "
            "Received Model Group=route-a\nAvailable Model Group Fallbacks=None",
            "type": None,
            "param": None,
            "code": "400",
        }
    }


def pydantic_echo(echoed: str) -> dict[str, Any]:
    """A 400 about the request's shape whose validation error prints the offending input."""
    return {
        "error": {
            "message": "1 validation error for ChatCompletionRequest\nmessages.1.content\n"
            "  Input should be a valid string [type=string_type, "
            f"input_value={{'text': {echoed!r}}}, input_type=dict]",
            "type": "invalid_request_error",
            "param": None,
            "code": "400",
        }
    }


def bedrock_echo(echoed: str) -> dict[str, Any]:
    """Bedrock's request validator quoting the value it refused, relayed by the proxy."""
    return proxied(
        f"1 validation error detected: Value '{echoed}' at "
        "'messages.1.member.content.1.member.text' failed to satisfy constraint: "
        "Member must have length less than or equal to 64"
    )


def detail_echo(echoed: str) -> dict[str, Any]:
    """A FastAPI-style 400 whose detail carries the input it refused."""
    return {
        "detail": [
            {
                "type": "string_type",
                "loc": ["body", "messages", 1, "content"],
                "msg": "Input should be a valid string",
                "input": {"text": echoed},
            }
        ]
    }


def short_quoted_echo(echoed: str) -> dict[str, Any]:
    """A validator's short message, the quoted value only a word in, relayed by the proxy."""
    return proxied(f"Value '{echoed}' at 'messages.1.member.content' failed to satisfy constraint")


def unquoted_echo(echoed: str) -> dict[str, Any]:
    """A gateway's own message that carries the refused content without quoting it."""
    return {
        "error": {
            "message": f"The input filter refused the content of messages.1: {echoed}",
            "type": "invalid_request_error",
            "param": None,
            "code": "400",
        }
    }


ECHOES = [
    pytest.param(pydantic_echo, id="pydantic"),
    pytest.param(bedrock_echo, id="bedrock-validator"),
    pytest.param(detail_echo, id="fastapi-detail"),
    pytest.param(short_quoted_echo, id="short-quoted"),
    pytest.param(unquoted_echo, id="unquoted"),
]
LIVE = [pytest.param(bare, id="bare"), pytest.param(proxied, id="proxied")]
#: Bodies litellm finds no message in, which it prints as the openai SDK's repr of the body.
REPR_BODIES = [
    pytest.param(lambda sentence: {"error": sentence}, id="error-string"),
    pytest.param(lambda sentence: {"detail": sentence}, id="detail-string"),
]
#: Every refusal sentence, and the one reader that must read it.
REFUSALS = [
    pytest.param(TOOL_USE_SENTENCE, "tool_use", id="tool-use"),
    pytest.param(TOOL_USE_STREAMED_SENTENCE, "tool_use", id="tool-use-streamed"),
    pytest.param(TEMPERATURE_SENTENCE, "temperature", id="temperature"),
    pytest.param(NO_STOP_SENTENCE, "stop_field", id="no-stop"),
    pytest.param(LIMIT_SENTENCE, "stop_length", id="stop-limit"),
]


async def raised_for(monkeypatch: pytest.MonkeyPatch, body: dict[str, Any]) -> Exception:
    """The exception litellm's own client raises when the gateway answers 400 with ``body``."""

    async def send(self: httpx.AsyncClient, request: httpx.Request, **kw: Any) -> httpx.Response:
        return httpx.Response(400, json=body, request=request)

    with monkeypatch.context() as patch:
        patch.setattr(httpx.AsyncClient, "send", send)
        for name in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
            patch.delenv(name, raising=False)
        try:
            await lp.litellm.acompletion(
                model=ROUTE,
                api_base=GATEWAY,
                api_key="sk-test",
                messages=[{"role": "user", "content": "hi"}],
            )
        except Exception as exc:  # noqa: BLE001 -- the exception is what is under test
            return exc
    raise AssertionError("litellm raised nothing for a 400")


def readers(text: str | BaseException) -> list[str]:
    """The names of the refusal readers that read ``text`` as their refusal."""
    read = {
        "stop_field": stop_field_unsupported(text),
        "stop_length": stop_length_limit(text) is not None,
        "temperature": temperature_field_unsupported(text),
        "tool_use": tool_use_unsupported(text),
    }
    return [name for name, hit in read.items() if hit]


# ── the readers ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("sentence, reader", REFUSALS)
@pytest.mark.parametrize("shape", LIVE)
async def test_the_live_bodies_are_read_as_their_refusal(
    monkeypatch: pytest.MonkeyPatch,
    shape: Callable[[str], dict[str, Any]],
    sentence: str,
    reader: str,
) -> None:
    exc = await raised_for(monkeypatch, shape(sentence))

    assert readers(exc) == [reader]
    if reader == "stop_length":
        assert stop_length_limit(exc) == 10


@pytest.mark.parametrize("sentence, reader", REFUSALS)
@pytest.mark.parametrize("echo", ECHOES)
@pytest.mark.parametrize("lead", ["", "notes: "], ids=["as-is", "in-notes"])
async def test_a_400_that_only_quotes_a_refusal_is_read_as_none(
    monkeypatch: pytest.MonkeyPatch,
    echo: Callable[[str], dict[str, Any]],
    lead: str,
    sentence: str,
    reader: str,
) -> None:
    exc = await raised_for(monkeypatch, echo(f"{lead}{sentence}"))

    # The premise: the sentence is in the text a substring search reads.
    assert sentence in str(exc)
    assert readers(exc) == []


@pytest.mark.parametrize("sentence, reader", REFUSALS)
@pytest.mark.parametrize("shape", REPR_BODIES)
async def test_a_refusal_in_a_body_printed_as_a_repr_is_read(
    monkeypatch: pytest.MonkeyPatch,
    shape: Callable[[str], dict[str, Any]],
    sentence: str,
    reader: str,
) -> None:
    exc = await raised_for(monkeypatch, shape(sentence))

    assert "Error code: 400 - {" in str(exc)
    assert readers(exc) == [reader]


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"message": TOOL_USE_SENTENCE}, id="message"),
        pytest.param({"error": {"message": TOOL_USE_SENTENCE}}, id="error-message"),
    ],
)
def test_a_refusal_in_a_raw_json_body_is_read(body: dict[str, Any]) -> None:
    assert readers(json.dumps(body)) == ["tool_use"]


def test_a_refusal_buried_deep_in_a_long_text_is_not_read() -> None:
    text = "word " * 40_000 + TOOL_USE_SENTENCE
    assert readers(text) == []


def test_a_message_that_opens_with_the_refusal_is_read_however_long() -> None:
    """The control: a refusal followed by more text is still the refusal."""
    assert readers(f"{TOOL_USE_SENTENCE} {'word ' * 40_000}") == ["tool_use"]


async def test_the_provider_message_is_the_innermost_one_and_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exc = await raised_for(monkeypatch, proxied(TOOL_USE_SENTENCE))

    assert base.provider_message(exc) == TOOL_USE_SENTENCE
    assert len(base.provider_message("word " * 40_000)) == base.PROVIDER_MESSAGE_MAX_CHARS


# ── the provider: an echo changes nothing on either call path ────────────────────────────────


class Scripted:
    """``litellm.acompletion`` answering each request in turn from ``answers`` (the last one
    repeating): an exception is raised, a string is the reply, buffered or streamed as asked.
    ``requests`` records every call's kwargs."""

    def __init__(self, *answers: Exception | str) -> None:
        self.answers = list(answers)
        self.requests: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        answer = self.answers[min(len(self.requests), len(self.answers)) - 1]
        if isinstance(answer, Exception):
            raise answer
        if not kwargs.get("stream"):
            return _answer(answer)

        async def _stream() -> AsyncIterator[Any]:
            yield _chunk(content=answer)
            yield _chunk(content="", finish_reason="stop")

        return _stream()

    def sent(self, field: str) -> list[Any]:
        """The ``field`` each request carried; ``None`` where it carried none."""
        return [request.get(field) for request in self.requests]


@pytest.mark.parametrize("echo", ECHOES)
@pytest.mark.parametrize("how", HOWS)
async def test_an_echo_of_the_tool_use_sentence_keeps_native_tools(
    monkeypatch: pytest.MonkeyPatch, how: str, echo: Callable[[str], dict[str, Any]]
) -> None:
    exc = await raised_for(monkeypatch, echo(f"notes: {TOOL_USE_SENTENCE}"))
    gateway = Scripted(exc, "ok")
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)
    provider = _wrapped("native")

    with pytest.raises(RequestFailed) as raised:
        await _run(provider, how, tools=TOOLS)
    await _run(provider, how, tools=TOOLS)

    # The 400 is surfaced as the failure it is, and the next call still sends its tools.
    assert not isinstance(raised.value, ToolUseRefused)
    assert [("tools" in request) for request in gateway.requests] == [True, True]
    assert provider._use_text_mode(TOOLS) is False


@pytest.mark.parametrize("echo", ECHOES)
@pytest.mark.parametrize("how", HOWS)
async def test_an_echo_of_the_temperature_sentence_keeps_a_temperature_of_zero(
    monkeypatch: pytest.MonkeyPatch, how: str, echo: Callable[[str], dict[str, Any]]
) -> None:
    exc = await raised_for(monkeypatch, echo(f"notes: {TEMPERATURE_SENTENCE}"))
    gateway = Scripted(exc, "ok")
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)
    provider = _provider(temperature=0.0)

    with pytest.raises(RequestFailed):
        await _run(provider, how)
    assert (await _run(provider, how))[0] == "ok"

    assert gateway.sent("temperature") == [0.0, 0.0]
    assert provider.rejected_request_fields == []


@pytest.mark.parametrize("sentence", [NO_STOP_SENTENCE, LIMIT_SENTENCE], ids=["no-stop", "limit"])
@pytest.mark.parametrize("echo", ECHOES)
@pytest.mark.parametrize("how", HOWS)
async def test_an_echo_of_a_stop_sentence_keeps_the_stop(
    monkeypatch: pytest.MonkeyPatch,
    how: str,
    echo: Callable[[str], dict[str, Any]],
    sentence: str,
) -> None:
    exc = await raised_for(monkeypatch, echo(f"notes: {sentence}"))
    gateway = Scripted(exc, "ok")
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)
    provider = _provider()

    with pytest.raises(RequestFailed):
        await _run(provider, how, stop=SENTINELS)
    assert (await _run(provider, how, stop=SENTINELS))[0] == "ok"

    assert gateway.sent("stop") == [SENTINELS, SENTINELS]
    assert provider.rejected_request_fields == []
    assert provider.max_stop_chars is None


# ── the provider: the live bodies still teach it, on either call path ────────────────────────


@pytest.mark.parametrize(
    "sentence", [TOOL_USE_SENTENCE, TOOL_USE_STREAMED_SENTENCE], ids=["tool-use", "streamed"]
)
@pytest.mark.parametrize("shape", LIVE)
@pytest.mark.parametrize("how", HOWS)
async def test_the_live_tool_use_refusal_still_moves_the_session_to_the_text_protocol(
    monkeypatch: pytest.MonkeyPatch,
    how: str,
    shape: Callable[[str], dict[str, Any]],
    sentence: str,
) -> None:
    exc = await raised_for(monkeypatch, shape(sentence))
    gateway = Scripted(exc, TOOL_REPLY)
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)
    provider = _wrapped("native")

    _, calls = await _run(provider, how, tools=TOOLS)

    assert calls == [("read_file", {"path": "a.py"})]
    assert [("tools" in request) for request in gateway.requests] == [True, False]
    assert provider._use_text_mode(TOOLS) is True


@pytest.mark.parametrize("shape", LIVE)
@pytest.mark.parametrize("how", HOWS)
async def test_the_live_temperature_refusal_still_drops_the_temperature(
    monkeypatch: pytest.MonkeyPatch, how: str, shape: Callable[[str], dict[str, Any]]
) -> None:
    exc = await raised_for(monkeypatch, shape(TEMPERATURE_SENTENCE))
    gateway = Scripted(exc, "ok")
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)
    provider = _provider(temperature=0.0)

    assert (await _run(provider, how))[0] == "ok"

    assert gateway.sent("temperature") == [0.0, None]
    assert provider.rejected_request_fields == ["temperature"]


@pytest.mark.parametrize("shape", LIVE)
@pytest.mark.parametrize("how", HOWS)
async def test_the_live_stop_refusals_still_teach_the_stop(
    monkeypatch: pytest.MonkeyPatch, how: str, shape: Callable[[str], dict[str, Any]]
) -> None:
    no_stop = await raised_for(monkeypatch, shape(NO_STOP_SENTENCE))
    limit = await raised_for(monkeypatch, shape(LIMIT_SENTENCE))
    refused = _provider()
    monkeypatch.setattr(lp.litellm, "acompletion", gateway := Scripted(no_stop, "ok"))

    assert (await _run(refused, how, stop=SENTINELS))[0] == "ok"
    assert gateway.sent("stop") == [SENTINELS, None]
    assert refused.rejected_request_fields == ["stop"]

    limited = _provider()
    monkeypatch.setattr(lp.litellm, "acompletion", gateway := Scripted(limit, "ok"))

    assert (await _run(limited, how, stop=SENTINELS))[0] == "ok"
    # Every sentinel is ten characters or more, so none is strictly shorter than the limit.
    assert gateway.sent("stop") == [SENTINELS, None]
    assert limited.max_stop_chars == 10
