"""A gateway that answers with another model says so, and the provider warns (ADR-0305).

Reported 2026-10-14 against a LiteLLM 1.9x gateway in front of Bedrock/Vertex: the gateway can
answer at HTTP 200 with a model other than the one asked for. A route asked for with native tools
was answered by another route, and a route that refused ``temperature`` by a sibling. On the
buffered path the only trace was ADR-0214's INFO line on the first sight of a name; on the
streamed path there was none, because litellm stamps the requested model on every chunk.

The gateway says so itself, in response headers that litellm keeps, on both paths, in
``_hidden_params['additional_headers']``, before the first chunk of a stream:

* ``llm_provider-x-litellm-attempted-fallbacks``: the fallbacks it attempted, ``0`` on a normal
  answer
* ``llm_provider-x-litellm-model-group``: the model group that answered
* ``llm_provider-x-litellm-model-name``: the deployment behind it

On any other backend the keys are absent. Hermetic: no network, no credentials. The header names
are litellm's; the values are neutral. The wire tests at the end send the headers through
litellm's own HTTP client, so the dictionary the provider reads is the one litellm builds.

A count above 0 with the very model group asked for, the model string less its provider prefix,
is no fallback: nothing answered instead, and nothing is warned (ADR-0316).
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import zakcode.providers.litellm_provider as lp
from zakcode.messages import Message
from zakcode.providers.litellm_provider import LiteLLMProvider

ROUTE = "openai/route-a"
GATEWAY = "https://gateway.invalid/v1"
_MSGS = [Message.user("hello")]

ATTEMPTS = "llm_provider-x-litellm-attempted-fallbacks"
GROUP = "llm_provider-x-litellm-model-group"
NAME = "llm_provider-x-litellm-model-name"
FELL_BACK = {ATTEMPTS: "1", GROUP: "route-b", NAME: "vendor/route-b-1"}


@pytest.fixture(autouse=True)
def _no_request_pacing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The provider spaces request starts a second apart; that sleep is not under test."""
    monkeypatch.setattr(lp, "_MIN_REQUEST_INTERVAL_S", 0.0)


def _provider() -> LiteLLMProvider:
    return LiteLLMProvider(model=ROUTE, api_base=GATEWAY, api_key="sk-test", context_window=131_072)


def _hidden(headers: dict[str, str] | None) -> dict[str, Any]:
    return {} if headers is None else {"additional_headers": dict(headers)}


def _response(headers: dict[str, str] | None, *, served: str = "route-a") -> SimpleNamespace:
    """One buffered completion, as litellm returns it, with the headers it kept."""
    return SimpleNamespace(
        model=served,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="ok", reasoning_content=None, tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1, total_tokens=4),
        _hidden_params=_hidden(headers),
    )


class _Stream:
    """What ``await litellm.acompletion(stream=True)`` returns: the chunks, and the headers
    litellm kept, readable before the first chunk. Every chunk carries the REQUESTED model,
    as litellm's stream wrapper stamps it. ``on_first`` runs as the first chunk is read."""

    def __init__(
        self, headers: dict[str, str] | None, on_first: Callable[[], None] | None = None
    ) -> None:
        self._hidden_params = _hidden(headers)
        self._on_first = on_first
        self.read = 0

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._chunks()

    async def _chunks(self) -> AsyncIterator[Any]:
        for content, finish in (("o", None), ("k", "stop")):
            if self.read == 0 and self._on_first is not None:
                self._on_first()
            self.read += 1
            delta = SimpleNamespace(content=content, tool_calls=None)
            choice = SimpleNamespace(delta=delta, finish_reason=finish)
            yield SimpleNamespace(model="route-a", choices=[choice], usage=None)


def _buffered(*answers: SimpleNamespace) -> Any:
    queue = list(answers)

    async def _acompletion(**_kwargs: Any) -> Any:
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return _acompletion


def _streamed(stream: _Stream) -> Any:
    async def _acompletion(**_kwargs: Any) -> Any:
        return stream

    return _acompletion


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


# ── the buffered path ────────────────────────────────────────────────────────────────────────


async def test_a_buffered_fallback_is_warned_naming_both_models(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="zakcode.providers")
    monkeypatch.setattr(
        lp.litellm, "acompletion", _buffered(_response(FELL_BACK, served="route-b"))
    )

    result = await _provider().acomplete(_MSGS)

    assert result.text == "ok"
    assert _warnings(caplog) == [
        "openai/route-a: the gateway answered with route-b (vendor/route-b-1) instead, "
        "after 1 attempted fallback(s)"
    ]


async def test_one_warning_per_pair_of_asked_and_answered(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The same fallback again is silent; a different answerer is warned in its turn."""
    caplog.set_level(logging.WARNING, logger="zakcode.providers")
    to_c = {**FELL_BACK, GROUP: "route-c", NAME: "vendor/route-c-1"}
    monkeypatch.setattr(
        lp.litellm,
        "acompletion",
        _buffered(_response(FELL_BACK), _response(FELL_BACK), _response(to_c)),
    )
    provider = _provider()

    for _ in range(3):
        await provider.acomplete(_MSGS)

    lines = _warnings(caplog)
    assert len(lines) == 2
    assert "route-b" in lines[0]
    assert "route-c" in lines[1]


@pytest.mark.parametrize(
    "headers, named",
    [
        pytest.param(
            {ATTEMPTS: "2", NAME: "vendor/route-b-1"}, "vendor/route-b-1", id="deployment"
        ),
        pytest.param({ATTEMPTS: "1"}, "a model it did not name", id="no-name"),
    ],
)
async def test_an_answerer_with_no_group_is_named_by_its_deployment_or_said_unnamed(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    headers: dict[str, str],
    named: str,
) -> None:
    caplog.set_level(logging.WARNING, logger="zakcode.providers")
    monkeypatch.setattr(lp.litellm, "acompletion", _buffered(_response(headers)))

    await _provider().acomplete(_MSGS)

    [line] = _warnings(caplog)
    assert named in line


# ── the streamed path ────────────────────────────────────────────────────────────────────────


async def test_a_streamed_fallback_is_warned_before_the_first_chunk(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The headers are there before any chunk, and the chunks only ever name the request."""
    caplog.set_level(logging.WARNING, logger="zakcode.providers")
    warned_at_first: list[int] = []
    stream = _Stream(FELL_BACK, on_first=lambda: warned_at_first.append(len(_warnings(caplog))))
    monkeypatch.setattr(lp.litellm, "acompletion", _streamed(stream))

    events = _provider().astream(_MSGS)
    await events.__anext__()

    assert warned_at_first == [1]
    assert stream.read == 1
    [line] = _warnings(caplog)
    assert ROUTE in line
    assert "route-b" in line
    async for _ in events:
        pass
    assert len(_warnings(caplog)) == 1


# ── no fallback, no warning ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param({**FELL_BACK, ATTEMPTS: "0"}, id="zero-attempts"),
        pytest.param({GROUP: "route-a", NAME: "vendor/route-a-1"}, id="no-attempts-header"),
        pytest.param({**FELL_BACK, ATTEMPTS: "many"}, id="unreadable-attempts"),
        pytest.param({}, id="no-gateway-headers"),
        pytest.param(None, id="no-additional-headers"),
    ],
)
@pytest.mark.parametrize("path", ["buffered", "streamed"])
async def test_no_warning_without_a_fallback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    headers: dict[str, str] | None,
    path: str,
) -> None:
    caplog.set_level(logging.WARNING, logger="zakcode.providers")
    if path == "buffered":
        monkeypatch.setattr(lp.litellm, "acompletion", _buffered(_response(headers)))
        await _provider().acomplete(_MSGS)
    else:
        monkeypatch.setattr(lp.litellm, "acompletion", _streamed(_Stream(headers)))
        async for _ in _provider().astream(_MSGS):
            pass

    assert _warnings(caplog) == []


@pytest.mark.parametrize(
    "model, group, warned",
    [
        pytest.param(ROUTE, "route-a", False, id="the-asked-alias"),
        pytest.param("openai/team/route-a", "team/route-a", False, id="an-alias-with-a-slash"),
        pytest.param("openai/team/route-a", "route-a", True, id="its-last-segment-only"),
    ],
)
@pytest.mark.parametrize("path", ["buffered", "streamed"])
async def test_a_fallback_answered_by_the_asked_model_group_is_not_warned(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    path: str,
    model: str,
    group: str,
    warned: bool,
) -> None:
    """ADR-0316: a gateway can count fallbacks and still answer from the group asked for, the
    model string less its provider prefix. Nothing answered instead, so nothing is warned."""
    caplog.set_level(logging.WARNING, logger="zakcode.providers")
    headers = {ATTEMPTS: "2", GROUP: group, NAME: "vendor/route-a-2"}
    provider = LiteLLMProvider(
        model=model, api_base=GATEWAY, api_key="sk-test", context_window=131_072
    )
    if path == "buffered":
        monkeypatch.setattr(lp.litellm, "acompletion", _buffered(_response(headers)))
        await provider.acomplete(_MSGS)
    else:
        monkeypatch.setattr(lp.litellm, "acompletion", _streamed(_Stream(headers)))
        async for _ in provider.astream(_MSGS):
            pass

    assert len(_warnings(caplog)) == (1 if warned else 0)


async def test_a_served_name_that_differs_is_not_a_fallback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A backend's own name for the model differs from the request in ordinary calls: ``x``
    answering ``openai/x``, or a server echoing the weights file it maps an alias to (ADR-0214).
    With no fallback header, that is the INFO line ADR-0214 prints, not a warning."""
    caplog.set_level(logging.INFO, logger="zakcode.providers")
    monkeypatch.setattr(lp.litellm, "acompletion", _buffered(_response(None, served="route-b")))

    await _provider().acomplete(_MSGS)

    assert _warnings(caplog) == []
    assert any("served by route-b" in r.getMessage() for r in caplog.records)


# ── through litellm's own client ─────────────────────────────────────────────────────────────


def _sse() -> bytes:
    chunk = {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "route-b"}
    frames = [
        {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}}]},
        {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    body = b"".join(b"data: " + json.dumps(frame).encode() + b"\n\n" for frame in frames)
    return body + b"data: [DONE]\n\n"


_CHAT_BODY: dict[str, Any] = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1,
    "model": "route-b",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
}


@pytest.fixture
def gateway(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Every request litellm's own client sends, answered with these response headers."""
    headers: dict[str, str] = {}

    async def send(self: httpx.AsyncClient, request: httpx.Request, **kw: Any) -> httpx.Response:
        stream = json.loads(request.content.decode("utf-8")).get("stream")
        if stream:
            return httpx.Response(
                200,
                content=_sse(),
                headers={"content-type": "text/event-stream", **headers},
                request=request,
            )
        return httpx.Response(200, json=_CHAT_BODY, headers=headers, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    for name in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
        monkeypatch.delenv(name, raising=False)
    return headers


@pytest.mark.parametrize("path", ["buffered", "streamed"])
@pytest.mark.parametrize("attempts, warned", [("1", 1), ("0", 0)])
async def test_the_gateways_headers_arrive_through_litellms_client(
    gateway: dict[str, str],
    caplog: pytest.LogCaptureFixture,
    path: str,
    attempts: str,
    warned: int,
) -> None:
    caplog.set_level(logging.WARNING, logger="zakcode.providers")
    gateway.update(
        {
            "x-litellm-attempted-fallbacks": attempts,
            "x-litellm-model-group": "route-b",
            "x-litellm-model-name": "vendor/route-b-1",
        }
    )
    provider = _provider()

    if path == "buffered":
        assert (await provider.acomplete(_MSGS)).text == "ok"
    else:
        async for _ in provider.astream(_MSGS):
            pass

    lines = _warnings(caplog)
    assert len(lines) == warned
    if warned:
        assert "route-b" in lines[0]
