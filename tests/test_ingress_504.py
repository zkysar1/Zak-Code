"""An ingress's HTML 504 is named as the server's timeout and reads without its tags (ADR-0306).

Reported 2026-10-14 against a LiteLLM 1.9x gateway in front of Bedrock/Vertex: the ingress in front
of the gateway cuts a long buffered call at 120 s with an HTML "504 Gateway Time-out" page, with no
JSON and no gateway headers. litellm raises its ``Timeout`` for it, which the provider mapped to
``TimedOut`` with the default bound, so the retry notice named ``ZAKCODE_REQUEST_TIMEOUT``, a
client ceiling that defaults to 600 s and had not expired, and the page's HTML reached the turn's
error verbatim.

Measured 2026-10-15 with litellm 1.86.2 against a loopback server sending that page: litellm raises
``Timeout`` carrying ``status_code`` 504 and the page in its message, where a client timeout's
``Timeout`` carries 408. Hermetic: no network, no credentials. The wire tests send the page
through litellm's own HTTP client.

Only the page says that an ingress cut the call (ADR-0316). The gateway answers a 504 of its own
as JSON, with no page, when its call to the backend timed out, and litellm raises the same
``Timeout`` carrying 504 for it (measured 2026-10-15, on both paths). That one is named as the
gateway's.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest

import zakcode.agent.loop as loop_module
import zakcode.providers.litellm_provider as lp
from zakcode.agent.loop import AgentLoop
from zakcode.config import load_settings
from zakcode.messages import Message
from zakcode.providers.base import AuthError, ProviderUnavailable, RequestFailed, TimedOut
from zakcode.providers.litellm_provider import LiteLLMProvider
from zakcode.session.store import Session
from zakcode.tools.base import ToolRegistry

INGRESS = "gateway ingress 504"
GATEWAY_504 = "gateway 504"
PAGE = (
    "<html>\r\n<head><title>504 Gateway Time-out</title></head>\r\n<body>\r\n"
    "<center><h1>504 Gateway Time-out</h1></center>\r\n</body>\r\n</html>\r\n"
)
_MSGS = [Message.user("hello")]


@pytest.fixture(autouse=True)
def _no_request_pacing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The provider spaces request starts a second apart; that sleep is not under test."""
    monkeypatch.setattr(lp, "_MIN_REQUEST_INTERVAL_S", 0.0)


def _vendor(name: str, message: str, status: int | None) -> Exception:
    """An exception shaped like litellm's: matched by class name, with an optional status."""
    return type(name, (Exception,), {"status_code": status})(message)


def _no_tags(text: str) -> bool:
    return "<" not in text and ">" not in text


# ── the mapping ──────────────────────────────────────────────────────────────────────────────


def test_an_html_504_is_a_timeout_the_ingress_imposed_with_the_page_text_and_no_tags() -> None:
    exc = _vendor("Timeout", f"litellm.Timeout: Timeout Error: OpenAIException - {PAGE}", 504)

    mapped = LiteLLMProvider._map_error(exc)

    assert isinstance(mapped, TimedOut)
    assert mapped.bound == INGRESS
    assert str(mapped) == "litellm.Timeout: Timeout Error: OpenAIException - 504 Gateway Time-out"


@pytest.mark.parametrize(
    "message, bound, text",
    [
        pytest.param(
            f"OpenAIException - {PAGE}",
            INGRESS,
            "OpenAIException - 504 Gateway Time-out",
            id="page",
        ),
        pytest.param(
            "upstream request timeout", GATEWAY_504, "upstream request timeout", id="no-page"
        ),
    ],
)
@pytest.mark.parametrize("name", ["Timeout", "APITimeoutError"])
def test_a_timeout_carrying_a_504_names_the_ingress_only_with_its_page(
    name: str, message: str, bound: str, text: str
) -> None:
    """ADR-0316: a 504 with no page is the gateway's own, not its ingress's."""
    mapped = LiteLLMProvider._map_error(_vendor(name, message, 504))

    assert isinstance(mapped, TimedOut)
    assert mapped.bound == bound
    assert str(mapped) == text


@pytest.mark.parametrize("name", ["InternalServerError", "APIError"])
def test_a_504_under_an_outage_class_stays_an_outage_and_loses_its_tags(name: str) -> None:
    """ADR-0224 retries a 5xx class as an outage, under the longer horizon a restart needs."""
    mapped = LiteLLMProvider._map_error(_vendor(name, f"OpenAIException - {PAGE}", 504))

    assert type(mapped) is ProviderUnavailable
    assert str(mapped) == "OpenAIException - 504 Gateway Time-out"


@pytest.mark.parametrize("status", [None, 408])
def test_a_client_timeout_still_names_the_request_timeout(status: int | None) -> None:
    """The control: the client's own ceiling expired, and the notice names that knob."""
    mapped = LiteLLMProvider._map_error(_vendor("Timeout", "Request timed out.", status))

    assert isinstance(mapped, TimedOut)
    assert mapped.bound == "ZAKCODE_REQUEST_TIMEOUT"
    assert str(mapped) == "Request timed out."


@pytest.mark.parametrize(
    "name, status, cls, text",
    [
        pytest.param("BadGatewayError", 502, ProviderUnavailable, "502 Bad Gateway", id="502"),
        pytest.param("PermissionDeniedError", 403, AuthError, "403 Forbidden", id="403"),
    ],
)
def test_a_page_with_another_status_keeps_its_class_and_loses_its_tags(
    name: str, status: int, cls: type[Exception], text: str
) -> None:
    """A page alone is not a timeout: an ingress answers other statuses with one too."""
    page = f"<html><head><title>{text}</title></head><body><h1>{text}</h1></body></html>"

    mapped = LiteLLMProvider._map_error(_vendor(name, f"OpenAIException - {page}", status))

    assert type(mapped) is cls
    assert str(mapped) == f"OpenAIException - {text}"


def test_a_page_whose_only_words_are_its_title_reads_as_its_title() -> None:
    page = "<html><head><title>504 Gateway Time-out</title></head><body></body></html>"

    mapped = LiteLLMProvider._map_error(_vendor("Timeout", f"OpenAIException - {page}", 504))

    assert str(mapped) == "OpenAIException - 504 Gateway Time-out"


def test_a_pages_scripts_styles_and_entities_do_not_reach_the_text() -> None:
    page = (
        '<!DOCTYPE html><html lang="en"><head><title>504</title>'
        "<style>body { color: red }</style></head><body><script>var retry = 1;</script>"
        "<h1>504 Gateway Time-out</h1><p>Try again &amp; wait</p></body></html>"
    )

    mapped = LiteLLMProvider._map_error(_vendor("Timeout", f"OpenAIException - {page}", 504))

    assert str(mapped) == "OpenAIException - 504 Gateway Time-out Try again & wait"


def test_angle_brackets_that_are_not_a_page_are_left_alone() -> None:
    message = "the model wrote <tool_call> where a JSON object was expected"

    mapped = LiteLLMProvider._map_error(_vendor("BadRequestError", message, 400))

    assert isinstance(mapped, RequestFailed)
    assert str(mapped) == message


# ── through litellm's own client ─────────────────────────────────────────────────────────────

_CHAT_BODY: dict[str, Any] = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1,
    "model": "route-a",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
}


#: The gateway's own 504 when its call to the backend timed out: JSON, no page. A LiteLLM proxy's
#: error object, and a bare message.
GATEWAY_504_BODIES = [
    pytest.param(
        {
            "error": {
                "message": "litellm.Timeout: Connection timed out after 120.0 seconds.",
                "type": None,
                "param": None,
                "code": "504",
            }
        },
        id="proxy-error",
    ),
    pytest.param({"message": "upstream request timeout"}, id="bare-message"),
]


@pytest.fixture
def ingress(monkeypatch: pytest.MonkeyPatch) -> list[int | dict[str, Any]]:
    """Every request litellm's own client sends is answered with the next scripted entry: 504
    with the ingress's page, 200 with a completion, or a body, which the gateway sends as its own
    JSON 504. The list is the script, consumed in order; the last entry repeats."""
    script: list[int | dict[str, Any]] = []

    async def send(self: httpx.AsyncClient, request: httpx.Request, **kw: Any) -> httpx.Response:
        status = script.pop(0) if len(script) > 1 else script[0]
        if isinstance(status, dict):
            return httpx.Response(504, json=status, request=request)
        if status == 504:
            return httpx.Response(
                504, content=PAGE.encode(), headers={"content-type": "text/html"}, request=request
            )
        return httpx.Response(200, json=_CHAT_BODY, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    for name in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
        monkeypatch.delenv(name, raising=False)
    return script


def _provider() -> LiteLLMProvider:
    return LiteLLMProvider(
        model="openai/route-a",
        api_base="https://gateway.invalid/v1",
        api_key="sk-test",
        context_window=131_072,
    )


@pytest.mark.parametrize("path", ["buffered", "streamed"])
async def test_the_ingress_page_through_litellms_client(ingress: list[int], path: str) -> None:
    ingress.append(504)
    provider = _provider()

    with pytest.raises(TimedOut) as caught:
        if path == "buffered":
            await provider.acomplete(_MSGS)
        else:
            async for _ in provider.astream(_MSGS):
                pass

    assert caught.value.bound == INGRESS
    assert _no_tags(str(caught.value))
    assert str(caught.value).count("504 Gateway Time-out") == 1


@pytest.mark.parametrize("body", GATEWAY_504_BODIES)
@pytest.mark.parametrize("path", ["buffered", "streamed"])
async def test_the_gateways_own_json_504_through_litellms_client_names_the_gateway(
    ingress: list[int | dict[str, Any]], path: str, body: dict[str, Any]
) -> None:
    """ADR-0316: litellm raises the same Timeout carrying 504 for it, and it is no ingress's."""
    ingress.append(body)
    provider = _provider()

    with pytest.raises(TimedOut) as caught:
        if path == "buffered":
            await provider.acomplete(_MSGS)
        else:
            async for _ in provider.astream(_MSGS):
                pass

    assert caught.value.bound == GATEWAY_504


@pytest.mark.parametrize("path", ["buffered", "streamed"])
async def test_a_client_read_timeout_through_litellms_client_names_the_request_timeout(
    monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    """The control on the wire: the client's own timeout reaches the provider as litellm's
    Timeout with its default status, 408, and keeps the client ceiling's name."""

    async def send(self: httpx.AsyncClient, request: httpx.Request, **kw: Any) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    provider = _provider()

    with pytest.raises(TimedOut) as caught:
        if path == "buffered":
            await provider.acomplete(_MSGS)
        else:
            async for _ in provider.astream(_MSGS):
                pass

    assert caught.value.bound == "ZAKCODE_REQUEST_TIMEOUT"


# ── what the operator reads ──────────────────────────────────────────────────────────────────


def _make_loop(provider: LiteLLMProvider) -> AgentLoop:
    settings = load_settings(workspace_root=Path.cwd())
    session = Session(cwd="/tmp/work", model="openai/route-a")
    return AgentLoop(provider, ToolRegistry(), session, settings=settings)


@pytest.fixture
def fast_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The loop's backoff is recorded, not waited."""
    recorded: list[float] = []

    async def _instant(delay: float) -> None:
        recorded.append(delay)

    monkeypatch.setattr(loop_module.asyncio, "sleep", _instant)
    return recorded


def test_the_retry_notice_names_the_ingress_and_the_turn_recovers(
    ingress: list[int], fast_sleep: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    """A buffered turn, as a sub-agent runs one: the ingress cuts the first call, the retry
    succeeds, and the notice says which side timed out."""
    ingress.extend([504, 200])
    loop = _make_loop(_provider())

    with caplog.at_level(logging.WARNING, logger="zakcode.agent.loop"):
        result = asyncio.run(loop.arun_turn("hi"))

    assert result.stop_reason == "completed"
    notices = [r.getMessage() for r in caplog.records if "retrying" in r.getMessage()]
    assert [n.split(";")[0] for n in notices] == [f"request timed out ({INGRESS})"]


@pytest.mark.parametrize("body", GATEWAY_504_BODIES)
def test_the_retry_notice_names_the_gateway_for_its_own_504(
    ingress: list[int | dict[str, Any]],
    fast_sleep: list[float],
    caplog: pytest.LogCaptureFixture,
    body: dict[str, Any],
) -> None:
    """ADR-0316: the gateway's own JSON 504 is not named as its ingress's in the notice."""
    ingress.extend([body, 200])
    loop = _make_loop(_provider())

    with caplog.at_level(logging.WARNING, logger="zakcode.agent.loop"):
        result = asyncio.run(loop.arun_turn("hi"))

    assert result.stop_reason == "completed"
    notices = [r.getMessage() for r in caplog.records if "retrying" in r.getMessage()]
    assert [n.split(";")[0] for n in notices] == [f"request timed out ({GATEWAY_504})"]


def test_a_turn_the_ingress_keeps_cutting_ends_with_the_page_text_and_no_tags(
    ingress: list[int], fast_sleep: list[float]
) -> None:
    ingress.append(504)
    loop = _make_loop(_provider())

    result = asyncio.run(loop.arun_turn("hi"))

    assert result.stop_reason == "provider_error"
    assert result.error is not None
    assert _no_tags(result.error)
    assert "504 Gateway Time-out" in result.error
