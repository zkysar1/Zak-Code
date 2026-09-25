"""ADR-0249: every socket litellm's httpx clients open carries TCP keepalive.

A backend that reboots mid-request leaves the client holding an ESTABLISHED socket with
nothing to send and nothing arriving. Without SO_KEEPALIVE the kernel never probes it, so a
buffered call sits until the whole ``request_timeout``. litellm has no socket-option hook on
its httpx path (the one ADR-0239 chose), so :mod:`zakcode.providers.litellm_provider` replaces
the transport builder its per-loop clients go through.

These tests read the option off the wire, not off a flag: one real call through
:class:`LiteLLMProvider` to a loopback stub, then the live socket behind the client litellm
cached for it. The control makes the same call with litellm's plain transport restored and
must find keepalive OFF — so a pass is the builder's doing and not an instrument that cannot
tell the two apart.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import litellm
import pytest
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

from zakcode.messages import Message
from zakcode.providers.litellm_provider import LiteLLMProvider, _keepalive_socket_options

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


def _keepalive_of_sockets_opened_by_one_call(api_base: str) -> list[int]:
    """Make one provider call; read SO_KEEPALIVE off every live socket litellm's client holds.

    The chain is httpx's: transport -> httpcore pool -> pooled connection -> HTTP/1.1
    connection -> network stream -> the raw socket. A link missing is a failure, not a skip:
    an instrument that cannot see must not report clean.
    """

    async def run() -> list[int]:
        cache = litellm.in_memory_llm_clients_cache
        before = set(cache.cache_dict)
        provider = LiteLLMProvider(
            model="openai/stub", api_base=api_base, api_key=_NO_KEY, context_window=8192
        )
        result = await provider.acomplete([Message.user("hi")])
        assert result.text == "ok"
        added = [key for key in cache.cache_dict if key not in before]
        assert added, "the call cached no client, so there is no socket to read"
        readings: list[int] = []
        for key in added:
            transport = _transport_of(cache.cache_dict[key])
            assert isinstance(transport, httpx.AsyncHTTPTransport), type(transport).__name__
            for pooled in transport._pool.connections:
                stream = pooled._connection._network_stream
                raw = stream.get_extra_info("socket")
                assert raw is not None, "the pooled connection exposes no raw socket"
                readings.append(raw.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE))
        assert readings, "the client holds no pooled connection to read"
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
        return readings

    return asyncio.run(run())


def test_the_transport_litellm_builds_carries_the_keepalive_options() -> None:
    transport = AsyncHTTPHandler._create_async_transport()
    assert isinstance(transport, httpx.AsyncHTTPTransport), type(transport).__name__
    options = list(transport._pool._socket_options or [])
    assert (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1) in options
    assert options == _keepalive_socket_options()


def test_the_schedule_names_the_idle_interval_and_count_this_platform_has() -> None:
    options = _keepalive_socket_options()
    assert options[0] == (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    by_name = {opt: val for _, opt, val in options[1:]}
    if hasattr(socket, "TCP_KEEPIDLE"):
        assert by_name[socket.TCP_KEEPIDLE] == 60
    elif hasattr(socket, "TCP_KEEPALIVE"):
        assert by_name[socket.TCP_KEEPALIVE] == 60
    if hasattr(socket, "TCP_KEEPINTVL"):
        assert by_name[socket.TCP_KEEPINTVL] == 30
    if hasattr(socket, "TCP_KEEPCNT"):
        assert by_name[socket.TCP_KEEPCNT] == 5


def test_a_provider_call_opens_keepalive_sockets(endpoint: str) -> None:
    readings = _keepalive_of_sockets_opened_by_one_call(endpoint)
    assert readings and all(value == 1 for value in readings), readings


def test_control_litellms_plain_transport_opens_sockets_without_keepalive(
    endpoint: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # litellm's own builder, as it stood before the replacement: no socket options.
    monkeypatch.setattr(
        AsyncHTTPHandler, "_create_httpx_transport", staticmethod(lambda: None), raising=True
    )
    readings = _keepalive_of_sockets_opened_by_one_call(endpoint)
    assert readings and all(value == 0 for value in readings), readings
