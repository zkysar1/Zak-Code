"""HTTP API server (FastAPI) exposing the core engine over HTTP / SSE / WebSocket.

Installed via the ``server`` extra (``pip install 'zakcode[server]'``). This is the API
the CLI, a future web client, or any external application can drive — every surface emits
the **same** :data:`~zakcode.events.AgentEvent` stream the CLI consumes in-process, so a
client is a thin renderer, never a fork of agent logic.

* :mod:`zakcode.server.wire` — pure JSON shapes (event (de)serialization, request/response
  models, WebSocket control frames). Importable without the server extra.
* :mod:`zakcode.server.app` — :func:`create_app`, the FastAPI application (REST + SSE;
  WebSocket added in a later increment). Requires the server extra.

``create_app`` is re-exported here lazily so importing :mod:`zakcode.server` does not pull
in FastAPI unless the app is actually requested. See ``docs/ARCHITECTURE.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zakcode.server.app import create_app as create_app

__all__ = ["create_app"]


def __getattr__(name: str) -> Any:
    # Lazy re-export: only import app.py (and thus FastAPI) on first access, so
    # `import zakcode.server` stays cheap and dependency-light.
    if name == "create_app":
        from zakcode.server.app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
