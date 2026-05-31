"""Transports that carry JSON-RPC messages to and from an MCP server.

The :class:`Transport` protocol is the seam :class:`~zakcode.mcp.client.MCPClient`
depends on, so a server can be driven over a real subprocess
(:class:`StdioTransport`) in production or an in-memory fake in tests. Only stdio
(newline-delimited JSON) is implemented here; a streamable-HTTP transport can be
added as another :class:`Transport` without touching the client.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from zakcode.mcp.jsonrpc import MCPProtocolError

#: How long to wait for a child to exit after we ask it to terminate, before moving on.
_TERMINATE_GRACE_SECONDS = 5.0


@runtime_checkable
class Transport(Protocol):
    """A bidirectional channel for single JSON-RPC messages (plain dicts)."""

    async def start(self) -> None:
        """Establish the connection (e.g. spawn the subprocess). Idempotent."""
        ...

    async def send(self, message: dict[str, Any]) -> None:
        """Send one JSON-RPC message."""
        ...

    async def receive(self) -> dict[str, Any] | None:
        """Receive the next JSON-RPC message, or ``None`` at end-of-stream."""
        ...

    async def close(self) -> None:
        """Tear the connection down. Idempotent."""
        ...


class StdioTransport:
    """Speak newline-delimited JSON-RPC to a subprocess MCP server over stdio.

    Lazily spawns ``command`` (with ``args`` / ``env`` / ``cwd``) on :meth:`start`,
    writes each message as one ``<json>\\n`` line to the child's stdin, and reads one
    line at a time from its stdout. The child's stderr is continuously drained so a
    chatty server can never block on a full pipe.
    """

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        *,
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
    ) -> None:
        self._command = command
        self._args = list(args or [])
        self._env = env
        self._cwd = str(cwd) if cwd is not None else None
        self._proc: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._proc is not None:
            return
        # When a custom env is given, merge it onto the current environment so the
        # child still inherits PATH etc. (callers pass only the keys they want to set).
        full_env = {**os.environ, **self._env} if self._env is not None else None
        self._proc = await asyncio.create_subprocess_exec(
            self._command,
            *self._args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=full_env,
            cwd=self._cwd,
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def _drain_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        with contextlib.suppress(Exception):
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break

    async def send(self, message: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise MCPProtocolError("stdio transport is not started")
        data = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
        self._proc.stdin.write(data)
        await self._proc.stdin.drain()

    async def receive(self) -> dict[str, Any] | None:
        if self._proc is None or self._proc.stdout is None:
            raise MCPProtocolError("stdio transport is not started")
        while True:
            line = await self._proc.stdout.readline()
            if not line:
                return None  # EOF: the server closed stdout / exited
            text = line.decode("utf-8").strip()
            if not text:
                continue  # tolerate blank lines between messages
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise MCPProtocolError(f"invalid JSON from MCP server: {exc}") from exc
            if not isinstance(obj, dict):
                raise MCPProtocolError("MCP server sent a non-object JSON-RPC message")
            return obj

    async def close(self) -> None:
        proc = self._proc
        if proc is None:
            return
        self._proc = None
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            with contextlib.suppress(Exception):
                await self._stderr_task
            self._stderr_task = None
        with contextlib.suppress(Exception):
            if proc.stdin is not None:
                proc.stdin.close()
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=_TERMINATE_GRACE_SECONDS)
        except (TimeoutError, asyncio.TimeoutError):
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            with contextlib.suppress(Exception):
                await proc.wait()


__all__ = ["Transport", "StdioTransport"]
