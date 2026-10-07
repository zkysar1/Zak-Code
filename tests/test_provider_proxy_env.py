"""ADR-0280: a process whose environment names a proxy reaches its provider through it.

httpx reads HTTP_PROXY / HTTPS_PROXY / ALL_PROXY / NO_PROXY only for a client built without an
explicit transport. ADR-0249 replaced the transport builder litellm's httpx clients go through
with one that always returned a transport, so a sandbox whose only route out is a proxy failed
every provider call with a connection error. These tests read what httpx built for the client
litellm constructs: the proxy mounts on it, and the transport that serves a given URL.

``tests/conftest.py`` starts every test from an environment that names no proxy.
"""

from __future__ import annotations

import httpcore
import httpx
import pytest
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

from zakcode.providers.litellm_provider import _keepalive_socket_options

_PROXY = "http://127.0.0.1:3128"


def _client() -> httpx.AsyncClient:
    """The client litellm builds for a provider call, through the builder ADR-0249 replaced."""
    return AsyncHTTPHandler(timeout=5).client


@pytest.mark.parametrize(
    ("variable", "pattern"),
    [("HTTPS_PROXY", "https://"), ("HTTP_PROXY", "http://"), ("ALL_PROXY", "all://")],
)
def test_a_proxy_in_the_environment_becomes_a_mount_on_the_client(
    monkeypatch: pytest.MonkeyPatch, variable: str, pattern: str
) -> None:
    monkeypatch.setenv(variable, _PROXY)
    assert pattern in {mount.pattern for mount in _client()._mounts}


def test_an_https_call_is_served_by_the_proxy_connection_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", _PROXY)
    transport = _client()._transport_for_url(httpx.URL("https://api.example.com/v1/chat"))
    assert isinstance(transport, httpx.AsyncHTTPTransport), type(transport).__name__
    assert isinstance(transport._pool, httpcore.AsyncHTTPProxy), type(transport._pool).__name__


def test_a_host_no_proxy_exempts_is_served_by_the_direct_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", _PROXY)
    monkeypatch.setenv("NO_PROXY", "localhost")
    client = _client()
    assert client._transport_for_url(httpx.URL("https://localhost/v1/chat")) is client._transport
    assert client._transport_for_url(httpx.URL("https://api.example.com/v1/chat")) is not (
        client._transport
    )


@pytest.mark.parametrize("environment", [{}, {"NO_PROXY": "localhost,127.0.0.1"}])
def test_without_a_proxy_the_client_goes_direct_with_keepalive(
    monkeypatch: pytest.MonkeyPatch, environment: dict[str, str]
) -> None:
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    client = _client()
    assert client._mounts == {}
    assert isinstance(client._transport, httpx.AsyncHTTPTransport)
    assert list(client._transport._pool._socket_options or []) == _keepalive_socket_options()
