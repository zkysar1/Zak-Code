"""A credential inside an ingress's web page is redacted after the page is decoded (ADR-0312).

ADR-0306 reduces an HTML error page in a provider's exception to the page's words, unescaping
its entities. The redactor ran on the raw text before that, so a credential assignment written
with entities (``&quot;api_key&quot;: &quot;...&quot;``, or the numeric ``&#34;`` and ``&#x22;``)
was not seen as one, and the unescape then turned it into an ordinary assignment in the message
that reaches the turn's error. Measured on main before this change: the value was present for
all three entity forms and absent for the plain form.

Hermetic: no network, no credentials. The probe value is built at run time from a word and a
counter, and every check is a boolean.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest

import zakcode.agent.loop as loop_module
import zakcode.providers.litellm_provider as lp
from zakcode.agent.loop import AgentLoop
from zakcode.config import load_settings
from zakcode.messages import Message
from zakcode.providers.base import ProviderError, ProviderUnavailable, RequestFailed, TimedOut
from zakcode.providers.litellm_provider import LiteLLMProvider
from zakcode.session.store import Session
from zakcode.tools.base import ToolRegistry

_MSGS = [Message.user("hello")]


@pytest.fixture(autouse=True)
def _no_request_pacing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The provider spaces request starts a second apart; that sleep is not under test."""
    monkeypatch.setattr(lp, "_MIN_REQUEST_INTERVAL_S", 0.0)


def _probe_value() -> str:
    return "".join(f"probe{n}" for n in range(4))


#: The same assignment, its quotes written four ways. The first three are entities.
_QUOTES = {
    "named": "&quot;",
    "decimal": "&#34;",
    "hex": "&#x22;",
    "plain": '"',
}


def _page(quote: str, value: str, heading: str = "502 Bad Gateway") -> str:
    return (
        f"<html><head><title>{heading}</title></head><body>"
        f"<h1>{heading}</h1>"
        f"<p>{quote}api_key{quote}: {quote}{value}{quote}</p>"
        "</body></html>"
    )


def _vendor(name: str, message: str, status: int | None) -> Exception:
    """An exception shaped like litellm's: matched by class name, with a status."""
    return type(name, (Exception,), {"status_code": status})(message)


_CLASSES = [
    pytest.param("BadGatewayError", 502, ProviderUnavailable, id="502"),
    pytest.param("Timeout", 504, TimedOut, id="504"),
    pytest.param("BadRequestError", 400, RequestFailed, id="400"),
]


# ── the mapping ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("form", ["named", "decimal", "hex"])
@pytest.mark.parametrize("name, status, cls", _CLASSES)
def test_an_entity_escaped_credential_in_a_page_is_redacted(
    form: str, name: str, status: int, cls: type[ProviderError]
) -> None:
    value = _probe_value()
    exc = _vendor(name, f"OpenAIException - {_page(_QUOTES[form], value)}", status)

    mapped = LiteLLMProvider._map_error(exc)

    assert isinstance(mapped, cls)
    assert (value in str(mapped)) is False
    assert ("[REDACTED]" in str(mapped)) is True
    assert ("502 Bad Gateway" in str(mapped)) is True


@pytest.mark.parametrize("name, status, cls", _CLASSES)
def test_a_plain_credential_in_a_page_is_redacted_as_before(
    name: str, status: int, cls: type[ProviderError]
) -> None:
    """The control: the raw text's redaction already caught the plain form, and still does."""
    value = _probe_value()
    exc = _vendor(name, f"OpenAIException - {_page(_QUOTES['plain'], value)}", status)

    mapped = LiteLLMProvider._map_error(exc)

    assert isinstance(mapped, cls)
    assert (value in str(mapped)) is False
    assert ("[REDACTED]" in str(mapped)) is True


def test_a_credential_before_the_page_is_redacted_as_before() -> None:
    """The raw text's redaction is kept: a value outside the page never reaches the reduction."""
    value = _probe_value()
    page = "<html><body><h1>502 Bad Gateway</h1></body></html>"
    exc = _vendor("BadGatewayError", f"api_key={value} OpenAIException - {page}", 502)

    mapped = LiteLLMProvider._map_error(exc)

    assert (value in str(mapped)) is False
    assert ("502 Bad Gateway" in str(mapped)) is True


def test_a_page_with_no_credential_reads_as_before() -> None:
    page = "<html><body><h1>502 Bad Gateway</h1><p>Try again &amp; wait</p></body></html>"

    mapped = LiteLLMProvider._map_error(
        _vendor("BadGatewayError", f"OpenAIException - {page}", 502)
    )

    assert (str(mapped) == "OpenAIException - 502 Bad Gateway Try again & wait") is True


# ── through litellm's own client ─────────────────────────────────────────────────────────────


@pytest.fixture
def escaped_page(monkeypatch: pytest.MonkeyPatch) -> str:
    """Every request litellm's own client sends is answered with an ingress's 504 page that
    carries an entity-escaped credential assignment. Returns the probe value."""
    value = _probe_value()
    body = _page(_QUOTES["named"], value, "504 Gateway Time-out").encode()

    async def send(self: httpx.AsyncClient, request: httpx.Request, **kw: Any) -> httpx.Response:
        return httpx.Response(
            504, content=body, headers={"content-type": "text/html"}, request=request
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    for name in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
        monkeypatch.delenv(name, raising=False)
    return value


def _provider() -> LiteLLMProvider:
    return LiteLLMProvider(
        model="openai/route-a",
        api_base="https://gateway.invalid/v1",
        api_key="sk-test",
        context_window=131_072,
    )


@pytest.mark.parametrize("path", ["buffered", "streamed"])
async def test_an_escaped_credential_through_litellms_client_is_redacted(
    escaped_page: str, path: str
) -> None:
    provider = _provider()

    with pytest.raises(ProviderError) as caught:
        if path == "buffered":
            await provider.acomplete(_MSGS)
        else:
            async for _ in provider.astream(_MSGS):
                pass

    assert (escaped_page in str(caught.value)) is False
    assert ("504 Gateway Time-out" in str(caught.value)) is True


# ── what the operator reads ──────────────────────────────────────────────────────────────────


def test_the_turn_error_carries_no_escaped_credential(
    escaped_page: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def _instant(delay: float) -> None:
        return None

    monkeypatch.setattr(loop_module.asyncio, "sleep", _instant)
    settings = load_settings(workspace_root=Path.cwd())
    session = Session(cwd=str(tmp_path), model="openai/route-a")
    loop = AgentLoop(_provider(), ToolRegistry(), session, settings=settings)

    result = asyncio.run(loop.arun_turn("hi"))

    assert (result.stop_reason == "provider_error") is True
    assert result.error is not None
    assert (escaped_page in result.error) is False
