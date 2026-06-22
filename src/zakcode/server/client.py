"""A thin HTTP client for the Zak Code server.

This is the other half of the three-layer bet: a client (the CLI today, a web/IDE
client later) drives a **remote** engine over the *same* :data:`~zakcode.events.AgentEvent`
stream it would consume in-process. :meth:`ServerClient.astream_turn` mirrors
:meth:`zakcode.Agent.astream_turn` — an async iterator of ``AgentEvent`` — so the
exact same renderer can display a local or a remote turn. That symmetry is what the
parity test pins down.

The client is transport-injectable: pass an ``httpx.AsyncClient`` (e.g. one backed
by ``httpx.ASGITransport`` for hermetic tests, or a real one for ``zakcode serve``).
With none, it lazily creates a real client bound to ``base_url``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from zakcode.events import AgentEvent
from zakcode.server.wire import event_from_dict

DEFAULT_BASE_URL = "http://127.0.0.1:8000"


class ServerClient:
    """Drive a remote Zak Code server over HTTP + SSE."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        http_client: httpx.AsyncClient | None = None,
        auth_token: str | None = None,
    ) -> None:
        self._base_url = base_url
        self._client = http_client
        # We only own (and therefore close) a client we created ourselves; an
        # injected one is the caller's to manage.
        self._owns_client = http_client is None
        # When the server has ZAKCODE_AUTH_TOKEN set, every request needs a bearer token; without
        # this the remote-driver path 401s against an authenticated server (only the client we
        # create carries it -- an injected client's auth is the caller's to manage).
        self._auth_token = auth_token

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Authorization": f"Bearer {self._auth_token}"} if self._auth_token else None
            self._client = httpx.AsyncClient(base_url=self._base_url, timeout=None, headers=headers)
        return self._client

    async def health(self) -> dict[str, Any]:
        """Return the server's ``/health`` payload (raises on a non-2xx)."""
        resp = await self._http().get("/health")
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return data

    async def create_session(self) -> str:
        """Create a server-side session and return its id."""
        resp = await self._http().post("/sessions")
        resp.raise_for_status()
        return str(resp.json()["id"])

    async def astream_turn(
        self, message: str, session_id: str | None = None, model: str | None = None
    ) -> AsyncIterator[AgentEvent]:
        """Run one turn on the server, yielding each :data:`AgentEvent` as it arrives.

        Posts to ``/chat/stream`` and parses the Server-Sent-Events body, turning
        every ``data:`` frame back into a typed ``AgentEvent`` via the shared wire
        contract — so a malformed frame raises rather than being silently rendered.
        """
        payload = {"message": message, "session_id": session_id, "model": model}
        async with self._http().stream("POST", "/chat/stream", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data:"):
                    raw = line[len("data:") :].strip()
                    if raw:
                        yield event_from_dict(json.loads(raw))

    async def aclose(self) -> None:
        """Close the underlying HTTP client (only if this object created it)."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None


__all__ = ["ServerClient", "DEFAULT_BASE_URL"]
