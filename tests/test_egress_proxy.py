"""Tests for the network-egress sandbox: the domain-allowlisting proxy + its wiring.

Hermetic: the proxy is driven against a local upstream / raw sockets (no real internet), and the
subprocess-env injection is checked through the real ``run_capturing``.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

from zakcode._http import host_allowed
from zakcode.sandbox import EgressProxy
from zakcode.tools.builtins._proc import run_capturing

# ── shared host matcher ──────────────────────────────────────────────────────────


def test_host_allowed_matcher() -> None:
    assert host_allowed("example.com", ["example.com"])
    assert host_allowed("docs.example.com", ["example.com"])  # subdomain
    assert host_allowed("EXAMPLE.com.", ["example.com"])  # case + trailing dot
    assert not host_allowed("example.com", [])  # empty = no match (caller decides the meaning)
    assert not host_allowed("notexample.com", ["example.com"])
    assert not host_allowed("example.com.evil.test", ["example.com"])  # suffix trick
    assert not host_allowed("", ["example.com"])


# ── the proxy: blocking + forwarding ────────────────────────────────────────────


async def _raw_request(proxy_url: str, request: bytes) -> str:
    p = urlsplit(proxy_url)
    reader, writer = await asyncio.open_connection(p.hostname, p.port)
    writer.write(request)
    await writer.drain()
    data = await reader.read(4096)
    writer.close()
    return data.decode("latin-1", "replace")


async def test_proxy_blocks_unlisted_connect() -> None:
    async with EgressProxy(["example.com"]) as proxy:
        resp = await _raw_request(
            proxy.url, b"CONNECT evil.test:443 HTTP/1.1\r\nHost: evil.test\r\n\r\n"
        )
    assert "403" in resp


async def test_proxy_blocks_unlisted_http() -> None:
    async with EgressProxy(["example.com"]) as proxy:
        resp = await _raw_request(proxy.url, b"GET http://evil.test/x HTTP/1.1\r\n\r\n")
    assert "403" in resp


async def test_proxy_empty_allowlist_denies_everything() -> None:
    async with EgressProxy([]) as proxy:
        resp = await _raw_request(proxy.url, b"CONNECT example.com:443 HTTP/1.1\r\n\r\n")
    assert "403" in resp


async def test_proxy_blocks_nonstandard_port_connect() -> None:
    # Host is allowlisted, but CONNECT to SSH (:22) must be refused — only web ports (80/443).
    async with EgressProxy(["example.com"]) as proxy:
        resp = await _raw_request(proxy.url, b"CONNECT example.com:22 HTTP/1.1\r\n\r\n")
    assert "403" in resp


async def test_proxy_blocks_nonstandard_port_http() -> None:
    async with EgressProxy(["example.com"]) as proxy:
        resp = await _raw_request(proxy.url, b"GET http://example.com:8080/ HTTP/1.1\r\n\r\n")
    assert "403" in resp


async def test_proxy_blocks_non_http_scheme() -> None:
    # gopher:// (or ftp://, dict://) to an allowlisted host must be refused, not relayed.
    async with EgressProxy(["example.com"]) as proxy:
        resp = await _raw_request(proxy.url, b"GET gopher://example.com/_ HTTP/1.1\r\n\r\n")
    assert "403" in resp


async def test_proxy_stop_tears_down_the_listener() -> None:
    proxy = EgressProxy(["example.com"])
    await proxy.start()
    assert proxy.is_serving
    await proxy.stop()
    assert not proxy.is_serving


async def test_proxy_forwards_to_an_allowlisted_host() -> None:
    # A local upstream stands in for an "allowlisted" host (127.0.0.1 is on the list), so the
    # allow+forward path is exercised end-to-end with a real httpx client through the proxy.
    httpx = pytest.importorskip("httpx")

    async def _upstream(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.readuntil(b"\r\n\r\n")  # consume the request
        body = b"hello-through-proxy"
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s"
            % (len(body), body)
        )
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(_upstream, "127.0.0.1", 0)
    up_port = server.sockets[0].getsockname()[1]
    try:
        # allowed_ports=None so the upstream's ephemeral port is reachable (prod default is 80/443).
        async with (
            EgressProxy(["127.0.0.1"], allowed_ports=None) as proxy,
            httpx.AsyncClient(proxy=proxy.url, timeout=5.0) as client,
        ):
            resp = await client.get(f"http://127.0.0.1:{up_port}/")
        assert resp.status_code == 200
        assert resp.text == "hello-through-proxy"
    finally:
        server.close()
        await server.wait_closed()


async def test_proxy_subprocess_env_points_at_the_proxy() -> None:
    async with EgressProxy(["x.com"]) as proxy:
        env = proxy.subprocess_env()
        assert env["HTTP_PROXY"] == proxy.url and env["HTTPS_PROXY"] == proxy.url
        assert env["http_proxy"] == proxy.url  # lowercase too (some clients only read those)
        assert proxy.is_serving and proxy.port > 0


# ── subprocess env injection (the wiring _proc -> child) ─────────────────────────


async def test_run_capturing_injects_extra_env(tmp_path: Path) -> None:
    out, code = await run_capturing(
        argv=[
            sys.executable,
            "-c",
            "import os,sys; sys.stdout.write(os.environ.get('ZAK_TEST_PROXY',''))",
        ],
        cwd=str(tmp_path),
        timeout=30,
        extra_env={"ZAK_TEST_PROXY": "http://127.0.0.1:9999"},
    )
    assert code == 0 and out.strip() == "http://127.0.0.1:9999"


# ── Agent wiring: settings -> loop._egress_env -> ToolContext ────────────────────


class _StubProvider:
    """Minimal no-network provider for Agent construction."""

    async def acomplete(self, messages: Any, **kwargs: Any) -> Any:  # pragma: no cover - unused
        from zakcode.providers.base import LLMResult

        return LLMResult(text="")

    def count_tokens(self, messages: Any, *, system: Any = None) -> int:
        return 0

    def capabilities(self) -> Any:
        from zakcode.providers.base import Capabilities

        return Capabilities()


async def test_agent_egress_proxy_off_by_default(tmp_path: Path) -> None:
    import zakcode
    from zakcode.config import Settings

    agent = zakcode.Agent(
        provider=_StubProvider(), settings=Settings(default_model="x/y", workspace_root=tmp_path)
    )
    assert await agent.loop._egress_env() == {}  # no proxy, no env


async def test_agent_egress_proxy_starts_and_injects_env(tmp_path: Path) -> None:
    import zakcode
    from zakcode.config import Settings

    settings = Settings(
        default_model="x/y",
        workspace_root=tmp_path,
        egress_proxy=True,
        egress_allowed_domains=["example.com"],
    )
    agent = zakcode.Agent(provider=_StubProvider(), settings=settings)
    env = await agent.loop._egress_env()
    proxy = agent.loop._egress_proxy
    try:
        assert env["HTTP_PROXY"].startswith("http://127.0.0.1:")
        assert proxy is not None and proxy.is_serving
        # reused on the same loop (lazy-start caches it)
        assert await agent.loop._egress_env() == env
    finally:
        await agent.loop.aclose()
    # aclose() stops the listener and clears the slot (no leak on a long-lived loop).
    assert not proxy.is_serving and agent.loop._egress_proxy is None


def test_config_parses_egress_allowed_domains_csv() -> None:
    from zakcode.config import Settings

    s = Settings(default_model="x/y", egress_allowed_domains="example.com, pypi.org")  # type: ignore[arg-type]
    assert s.egress_allowed_domains == ["example.com", "pypi.org"]
