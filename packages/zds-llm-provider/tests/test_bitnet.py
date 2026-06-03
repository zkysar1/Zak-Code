"""Tests for BitNetProvider — the local OpenAI-compatible HTTP provider.

Every test injects a mock HTTP client. Zero network, zero API cost. ``httpx`` is imported
here only to construct its exception types for the transport-error mapping tests; the
provider's own httpx usage is fully mocked.
"""

from __future__ import annotations

import httpx
import pytest

from zds_llm_provider.bitnet import BitNetProvider, _translate_messages
from zds_llm_provider.messages import Message, ToolResultBlock
from zds_llm_provider.text_tools import TextToolCallingProvider
from zds_llm_provider.types import (
    AuthError,
    ContextWindowExceeded,
    RateLimited,
    RequestFailed,
)


class MockResponse:
    """Minimal httpx.Response stand-in."""

    def __init__(self, json_data: dict | None = None, status_code: int = 200) -> None:
        self._json = json_data or {}
        self.status_code = status_code
        self.text = __import__("json").dumps(self._json)

    def json(self) -> dict:
        return self._json


class MockClient:
    """Minimal httpx.AsyncClient stand-in: canned post/get responses or raised errors."""

    def __init__(
        self,
        *,
        post_response: MockResponse | None = None,
        post_exc: Exception | None = None,
        get_response: MockResponse | None = None,
        get_exc: Exception | None = None,
    ) -> None:
        self._post_response = post_response
        self._post_exc = post_exc
        self._get_response = get_response
        self._get_exc = get_exc

    async def post(self, url, *, json=None, headers=None) -> MockResponse:  # noqa: ANN001
        if self._post_exc is not None:
            raise self._post_exc
        assert self._post_response is not None
        return self._post_response

    async def get(self, url) -> MockResponse:  # noqa: ANN001
        if self._get_exc is not None:
            raise self._get_exc
        assert self._get_response is not None
        return self._get_response


def _chat_response(
    content: str = "hi there",
    tool_calls: list | None = None,
    usage: dict | None = None,
) -> dict:
    message: dict = {"content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "choices": [{"message": message, "finish_reason": "stop"}],
        "usage": usage or {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


async def test_acomplete_simple_text() -> None:
    client = MockClient(post_response=MockResponse(_chat_response("hello")))
    provider = BitNetProvider(client=client)
    result = await provider.acomplete([Message.user("hi")])
    assert result.text == "hello"
    assert result.usage.total_tokens == 8
    assert result.tool_calls == []


async def test_acomplete_with_tool_calls() -> None:
    tool_calls = [
        {
            "id": "call_x",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
        }
    ]
    client = MockClient(post_response=MockResponse(_chat_response("", tool_calls=tool_calls)))
    provider = BitNetProvider(client=client, supports_tools=True)
    result = await provider.acomplete([Message.user("read a.py")])
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "read_file"
    assert result.tool_calls[0].arguments == {"path": "a.py"}


async def test_acomplete_tool_args_unparseable() -> None:
    tool_calls = [
        {"id": "c", "type": "function", "function": {"name": "x", "arguments": "{not json"}}
    ]
    client = MockClient(post_response=MockResponse(_chat_response("", tool_calls=tool_calls)))
    provider = BitNetProvider(client=client)
    result = await provider.acomplete([Message.user("go")])
    assert result.tool_calls[0].arguments == {"_raw": "{not json"}


async def test_acomplete_connection_error() -> None:
    client = MockClient(post_exc=httpx.ConnectError("refused"))
    provider = BitNetProvider(client=client)
    with pytest.raises(RequestFailed):
        await provider.acomplete([Message.user("hi")])


async def test_acomplete_auth_error() -> None:
    client = MockClient(post_response=MockResponse({"error": "no key"}, status_code=401))
    provider = BitNetProvider(client=client)
    with pytest.raises(AuthError):
        await provider.acomplete([Message.user("hi")])


async def test_acomplete_rate_limited() -> None:
    client = MockClient(post_response=MockResponse({"error": "slow down"}, status_code=429))
    provider = BitNetProvider(client=client)
    with pytest.raises(RateLimited):
        await provider.acomplete([Message.user("hi")])


async def test_acomplete_context_exceeded() -> None:
    client = MockClient(
        post_response=MockResponse({"error": "context window exceeded"}, status_code=400)
    )
    provider = BitNetProvider(client=client)
    with pytest.raises(ContextWindowExceeded):
        await provider.acomplete([Message.user("hi")])


async def test_check_health_ok() -> None:
    client = MockClient(get_response=MockResponse({}, status_code=200))
    provider = BitNetProvider(client=client)
    assert await provider._check_health() is True


async def test_check_health_down() -> None:
    client = MockClient(get_response=MockResponse({}, status_code=503))
    provider = BitNetProvider(client=client)
    assert await provider._check_health() is False


async def test_check_health_connection_error() -> None:
    client = MockClient(get_exc=httpx.ConnectError("down"))
    provider = BitNetProvider(client=client)
    assert await provider._check_health() is False


def test_capabilities_defaults() -> None:
    caps = BitNetProvider(client=MockClient()).capabilities()
    assert caps.supports_tools is False
    assert caps.supports_vision is False
    assert caps.context_window == 8192


def test_capabilities_custom() -> None:
    provider = BitNetProvider(client=MockClient(), supports_tools=True, context_window=4096)
    caps = provider.capabilities()
    assert caps.supports_tools is True
    assert caps.context_window == 4096


def test_count_tokens_rough() -> None:
    provider = BitNetProvider(client=MockClient())
    n = provider.count_tokens([Message.user("a" * 16)], system="b" * 16)
    assert n == 8  # (16 + 16) // 4


def test_translate_messages_system() -> None:
    out = _translate_messages([Message.user("hi")], "be brief")
    assert out[0] == {"role": "system", "content": "be brief"}
    assert out[1]["role"] == "user"
    assert out[1]["content"] == "hi"


def test_translate_messages_tool_results() -> None:
    msg = Message.tool_results([ToolResultBlock(tool_use_id="call_0", output="ok")])
    out = _translate_messages([msg], None)
    assert out[0] == {"role": "tool", "tool_call_id": "call_0", "content": "ok"}


async def test_composition_with_text_tool_calling() -> None:
    """BitNet has no native tools; wrapping it adds text-protocol tool calling."""
    response_text = (
        '<tool_call>\n{"name": "read_file", "arguments": {"path": "z.py"}}\n</tool_call>'
    )
    client = MockClient(post_response=MockResponse(_chat_response(response_text)))
    provider = TextToolCallingProvider(BitNetProvider(client=client), mode="text")
    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]
    result = await provider.acomplete([Message.user("read z.py")], tools=tools)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "read_file"


def test_no_httpx_raises_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("zds_llm_provider.bitnet.httpx", None)
    with pytest.raises(ImportError):
        BitNetProvider()
