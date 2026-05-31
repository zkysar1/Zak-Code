"""Minimal JSON-RPC 2.0 helpers for the MCP client (clean-room, no SDK).

MCP speaks JSON-RPC 2.0. Over stdio each message is a single JSON object on its own
line — newline-delimited UTF-8, **not** LSP-style ``Content-Length`` framing. This
module is pure: it builds request/notification objects and interprets responses.
Framing and I/O live in :mod:`zakcode.mcp.transport`.
"""

from __future__ import annotations

from typing import Any

JSONRPC_VERSION = "2.0"


class MCPError(Exception):
    """Base class for all MCP client errors."""


class JSONRPCError(MCPError):
    """A JSON-RPC error *response* returned by the server (carries a numeric code)."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"JSON-RPC error {code}: {message}")
        self.code = code
        self.message = message
        self.data = data


class MCPProtocolError(MCPError):
    """A malformed or unexpected message that violates the JSON-RPC/MCP contract."""


def make_request(
    request_id: int, method: str, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build a JSON-RPC request object (a message that expects a matching response)."""
    msg: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "id": request_id, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def make_notification(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a JSON-RPC notification object (no id; no response is expected)."""
    msg: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def is_response_for(message: dict[str, Any], request_id: int) -> bool:
    """Whether ``message`` is the JSON-RPC response to ``request_id``.

    A response carries the same ``id`` and exactly one of ``result`` / ``error``;
    notifications (no ``id``) and unrelated responses return ``False`` so the client
    can skip them while waiting for the one it asked for.
    """
    return (
        message.get("id") == request_id
        and "id" in message
        and ("result" in message or "error" in message)
    )


def result_or_raise(message: dict[str, Any]) -> dict[str, Any]:
    """Return a response's ``result`` object, or raise :class:`JSONRPCError`.

    A non-dict ``result`` (rare, but legal JSON-RPC) is wrapped as ``{"_result": x}``
    so the caller always gets a dict.
    """
    if "error" in message:
        err = message["error"]
        if isinstance(err, dict):
            raise JSONRPCError(
                int(err.get("code", -1)), str(err.get("message", "")), err.get("data")
            )
        raise JSONRPCError(-1, str(err))
    result = message.get("result")
    if not isinstance(result, dict):
        return {"_result": result}
    return result


__all__ = [
    "JSONRPC_VERSION",
    "MCPError",
    "JSONRPCError",
    "MCPProtocolError",
    "make_request",
    "make_notification",
    "is_response_for",
    "result_or_raise",
]
