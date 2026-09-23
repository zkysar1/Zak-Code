"""ADR-0239: the HTTP clients litellm caches for a provider call use httpx, not aiohttp.

litellm caches one HTTP client per endpoint and event loop for an hour, then drops it
without closing it, on purpose: a request may still hold it. Under litellm's default aiohttp
transport, the collector reclaiming such a client makes aiohttp report "Unclosed client
session" through asyncio at ERROR, which a CLI prints in its own output (measured on a long
session: about one an hour). An httpx client is reclaimed without a word.

When the collector gets to a dropped client depends on what else still holds it (litellm's
own logging queue does, for a while), so these tests do not wait for it. They make one real
call through :class:`LiteLLMProvider` to a loopback stub and read which transport the client
litellm cached for that call carries. The control makes the same call with litellm's default
restored and must find aiohttp's transport, so a pass is the setting's doing and not an
instrument that cannot tell the two apart. It also fails the day litellm stops honouring the
setting's name.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import litellm
import pytest

from zakcode.messages import Message
from zakcode.providers.litellm_provider import LiteLLMProvider

#: The stub checks no credential; the SDK only needs a key to be present.
_NO_KEY = "offline-test-key"


class _Stub(BaseHTTPRequestHandler):
    """An OpenAI-compatible endpoint that answers every completion with "ok"."""

    protocol_version = "HTTP/1.1"  # keep the connection pooled, as a real endpoint does

    def do_POST(self) -> None:  # noqa: N802 - the http.server hook name
        self.rfile.read(int(self.headers.get("content-length", 0)))
        body = json.dumps(
            {
                "id": "stub",
                "object": "chat.completion",
                "created": 0,
                "model": "stub",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def endpoint() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()


def _transport_of(client: object) -> object:
    """The httpx transport behind an SDK client, a litellm handler, or a bare httpx client."""
    for holder in (getattr(client, "_client", None), getattr(client, "client", None), client):
        transport = getattr(holder, "_transport", None)
        if transport is not None:
            return transport
    return None


def _transports_cached_by_one_call(api_base: str) -> list[str]:
    """Make one provider call; name the transport of each client litellm cached for it."""

    async def run() -> list[str]:
        cache = litellm.in_memory_llm_clients_cache
        before = set(cache.cache_dict)
        provider = LiteLLMProvider(
            model="openai/stub", api_base=api_base, api_key=_NO_KEY, context_window=8192
        )
        result = await provider.acomplete([Message.user("hi")])
        assert result.text == "ok"
        added = [key for key in cache.cache_dict if key not in before]
        names = [type(_transport_of(cache.cache_dict[key])).__name__ for key in added]
        # Leave the cache as it was found, closing what this call opened on its own loop.
        for key in added:
            client = cache.cache_dict[key]
            close = getattr(client, "aclose", None) or getattr(client, "close", None)
            if close is not None:
                closing = close()
                if asyncio.iscoroutine(closing):
                    await closing
            cache._remove_key(key)
        await asyncio.sleep(0.2)  # let litellm's logging queue drain before the loop closes
        return names

    return asyncio.run(run())


def test_a_provider_call_caches_an_httpx_client(endpoint: str) -> None:
    names = _transports_cached_by_one_call(endpoint)
    assert names, "the call cached no client, so there is nothing to read"
    assert all(name == "AsyncHTTPTransport" for name in names), names


def test_control_litellms_default_caches_an_aiohttp_client(
    endpoint: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", False)
    assert "LiteLLMAiohttpTransport" in _transports_cached_by_one_call(endpoint)
