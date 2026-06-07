"""A localhost domain-allowlisting forward proxy — the network-egress half of the sandbox.

Opt-in (``ZAKCODE_EGRESS_PROXY``). When on, the agent points its **subprocess** tools (``bash`` /
``powershell``) at this proxy via ``HTTP(S)_PROXY`` env vars, so a child's outbound HTTP/HTTPS is
constrained to ``egress_allowed_domains`` (+ subdomains) on the standard web ports (80/443) — any
other host, port, or scheme gets ``403`` / refused. It supports HTTPS via ``CONNECT`` (a blind TLS
tunnel after a host+port check) and plain HTTP via absolute-form requests (scheme + host + port
checked).

**Scope + honesty (do not over-claim).** This enforces the allowlist for traffic that HONORS the
proxy env — curl, pip, git-over-https, wget, most HTTP libraries — which is the common subprocess
egress path, and is real defense-in-depth on top of the deny-first command gate. It is NOT a
kernel firewall: a process that ignores the proxy env (raw sockets, ``--noproxy``) can still
egress directly, and an **allowlisted domain that resolves to a private IP is reached** (the
allowlist is the policy — the proxy intentionally does not re-apply the web_fetch SSRF private-IP
block, so an operator can allowlist an internal host on purpose). True OS-level network isolation
(so bypass is impossible) is tracked backlog; **filesystem** confinement is already provided by
the workspace path guard (``resolve_path``).
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any
from urllib.parse import urlsplit

from zakcode._http import host_allowed

_RELAY_CHUNK = 64 * 1024
#: Ports the proxy will dial on an allowlisted host. Web egress is 80/443; restricting to these
#: refuses CONNECT-to-SSH (:22), SMTP (:25), DB/Redis ports, etc. ``None`` = any port (tests).
_DEFAULT_PORTS = frozenset({80, 443})
_CONNECT_TIMEOUT = 10.0  # bound the upstream connect so a black-holed host can't hang a turn
_IDLE_TIMEOUT = 120.0  # tear a tunnel down if neither side moves bytes for this long
_MAX_CONNECTIONS = 128  # backpressure ceiling on simultaneous in-flight handlers


class EgressProxy:
    """An asyncio localhost forward proxy that only reaches allowlisted domains on web ports.

    Empty allowlist = deny everything (the gate is on, nothing is permitted). Start it on a
    running loop, point children at :meth:`subprocess_env`, and call :meth:`stop` (or use it as an
    async context manager) to tear down the listener AND any in-flight tunnels.
    """

    def __init__(
        self,
        allowed_domains: list[str] | None,
        *,
        host: str = "127.0.0.1",
        allowed_ports: frozenset[int] | None = _DEFAULT_PORTS,
        connect_timeout: float = _CONNECT_TIMEOUT,
        idle_timeout: float = _IDLE_TIMEOUT,
        max_connections: int = _MAX_CONNECTIONS,
    ) -> None:
        self._allowed = [d for d in (allowed_domains or []) if d and d.strip()]
        self._host = host
        self._ports = allowed_ports
        self._connect_timeout = connect_timeout
        self._idle_timeout = idle_timeout
        self._max = max_connections
        self._server: asyncio.AbstractServer | None = None
        self._port = 0
        self.bound_loop: asyncio.AbstractEventLoop | None = None
        self._live: set[asyncio.Task[Any]] = set()

    @property
    def is_serving(self) -> bool:
        return self._server is not None and self._server.is_serving()

    @property
    def port(self) -> int:
        return self._port

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}"

    def subprocess_env(self) -> dict[str, str]:
        """The ``HTTP(S)_PROXY`` env vars to inject into a child so its egress is allowlisted."""
        u = self.url
        return {"HTTP_PROXY": u, "HTTPS_PROXY": u, "http_proxy": u, "https_proxy": u}

    def _allowed_port(self, port: int) -> bool:
        return self._ports is None or port in self._ports

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, self._host, 0)
        self._port = self._server.sockets[0].getsockname()[1]
        self.bound_loop = asyncio.get_running_loop()

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
            self._server = None
        # Cancel in-flight handlers too, so an explicit shutdown leaves no leaked tunnels.
        for task in list(self._live):
            task.cancel()
        self._live.clear()

    async def __aenter__(self) -> EgressProxy:
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.stop()

    # ── connection handling ────────────────────────────────────────────────────

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._live.add(task)
        try:
            if len(self._live) > self._max:  # backpressure: refuse beyond the ceiling
                await self._respond(writer, 503, "Too many connections")
                return
            request_line = await reader.readline()
            if not request_line:
                return
            fields = request_line.decode("latin-1", "replace").split()
            if len(fields) < 3:
                await self._respond(writer, 400, "Bad Request")
                return
            method, target, version = fields[0], fields[1], fields[2]
            if method.upper() == "CONNECT":
                await self._handle_connect(reader, writer, target)
            else:
                await self._handle_http(reader, writer, method, target, version)
        except Exception:  # noqa: BLE001 - a handler error must never crash the proxy server
            pass
        finally:
            if task is not None:
                self._live.discard(task)
            with contextlib.suppress(Exception):
                writer.close()

    async def _handle_connect(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, target: str
    ) -> None:
        host, _, port_s = target.rpartition(":")
        host = host or target
        port = int(port_s) if port_s.isdigit() else 443
        if not host_allowed(host, self._allowed) or not self._allowed_port(port):
            await self._respond(writer, 403, "Forbidden by egress allowlist")
            return
        if not await self._drain_headers(reader):  # malformed CONNECT preamble -> don't tunnel
            await self._respond(writer, 400, "Bad Request")
            return
        upstream = await self._connect(host, port)
        if upstream is None:
            await self._respond(writer, 502, "Bad Gateway")
            return
        up_reader, up_writer = upstream
        writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
        await writer.drain()
        await self._tunnel(reader, writer, up_reader, up_writer)

    async def _handle_http(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        method: str,
        target: str,
        version: str,
    ) -> None:
        parts = urlsplit(target)
        host = parts.hostname
        port = parts.port or (443 if parts.scheme == "https" else 80)
        if (
            parts.scheme not in ("http", "https")
            or not host
            or not host_allowed(host, self._allowed)
            or not self._allowed_port(port)
        ):
            await self._respond(writer, 403, "Forbidden by egress allowlist")
            return
        origin = parts.path or "/"
        if parts.query:
            origin += "?" + parts.query
        upstream = await self._connect(host, port)
        if upstream is None:
            await self._respond(writer, 502, "Bad Gateway")
            return
        up_reader, up_writer = upstream
        # Rewrite the request line to origin-form, then relay the client's headers/body to the
        # upstream and the response back (Connection: close semantics — fine for one-shot egress).
        up_writer.write(f"{method} {origin} {version}\r\n".encode("latin-1"))
        await up_writer.drain()
        await self._tunnel(reader, writer, up_reader, up_writer)

    async def _connect(
        self, host: str, port: int
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter] | None:
        """Open the upstream connection with a bounded connect timeout, or ``None`` on failure."""
        try:
            return await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=self._connect_timeout
            )
        except (OSError, TimeoutError):
            return None

    async def _tunnel(
        self,
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
        up_reader: asyncio.StreamReader,
        up_writer: asyncio.StreamWriter,
    ) -> None:
        # Co-terminating: when EITHER direction ends (EOF, idle timeout, broken pipe), cancel the
        # other and close both — so a half-open peer can never leave a tunnel (task + 2 sockets)
        # hanging forever.
        t1 = asyncio.create_task(self._pipe(client_reader, up_writer))
        t2 = asyncio.create_task(self._pipe(up_reader, client_writer))
        try:
            _, pending = await asyncio.wait({t1, t2}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        finally:
            for w in (up_writer, client_writer):
                with contextlib.suppress(Exception):
                    w.close()

    async def _pipe(self, src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
        try:
            while True:
                try:
                    data = await asyncio.wait_for(
                        src.read(_RELAY_CHUNK), timeout=self._idle_timeout
                    )
                except TimeoutError:
                    break  # idle too long — let the tunnel co-terminate
                if not data:
                    break
                dst.write(data)
                await dst.drain()
        except Exception:  # noqa: BLE001 - a broken pipe ends the relay, never crashes the server
            pass
        finally:
            with contextlib.suppress(Exception):
                dst.write_eof()  # half-close so the peer sees EOF; ignored if unsupported

    async def _drain_headers(self, reader: asyncio.StreamReader) -> bool:
        """Read+discard request headers up to the blank line. ``True`` if cleanly terminated."""
        for _ in range(100):
            line = await reader.readline()
            if line in (b"\r\n", b"\n"):
                return True
            if not line:  # EOF before the header block ended
                return False
        return False  # ran past the line cap without a terminator -> treat as malformed

    @staticmethod
    async def _respond(writer: asyncio.StreamWriter, code: int, message: str) -> None:
        body = message.encode("utf-8")
        writer.write(
            f"HTTP/1.1 {code} {message}\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Content-Type: text/plain\r\n"
            "Connection: close\r\n\r\n".encode("latin-1")
            + body
        )
        with contextlib.suppress(Exception):
            await writer.drain()


__all__ = ["EgressProxy"]
