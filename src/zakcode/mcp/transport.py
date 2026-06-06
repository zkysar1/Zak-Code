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

from zakcode._subprocess import new_group_kwargs, terminate_process_tree
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
            # Own process group/session: a launcher (npx/uvx) starts the real server as a
            # grandchild, so close() must be able to kill the whole tree. (audit4 #3)
            **new_group_kwargs(),
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
            # ``errors="replace"`` so a misbehaving server emitting non-UTF-8 bytes
            # degrades to a JSON-parse error (recoverable) rather than crashing the
            # transport — consistent with the file/shell tools' decode sites.
            text = line.decode("utf-8", errors="replace").strip()
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
        # Close stdin (signals the server to stop), then ask the process to exit.
        with contextlib.suppress(Exception):
            if proc.stdin is not None:
                proc.stdin.close()
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=_TERMINATE_GRACE_SECONDS)
        except TimeoutError:
            # The graceful stdin-close + terminate did not stop it in time; kill the whole
            # TREE (not just the parent), so a wrapper's grandchild server isn't orphaned
            # holding ports/locks/credentials. (audit4 #3)
            await terminate_process_tree(proc)
        # The process has now exited, so the child's stderr is at EOF and the drain
        # task has (or is about to) finish on its own. Cancel + await it to be sure,
        # suppressing CancelledError explicitly — contextlib.suppress(Exception) does
        # NOT catch it (asyncio.CancelledError is a BaseException in 3.11+).
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._stderr_task
            self._stderr_task = None


__all__ = ["Transport", "StdioTransport"]
