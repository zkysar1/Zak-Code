"""SafeEventProjection — Layer 4 of the Pearl watch-surface security model.

The public watch stream (``GET /watch/{session_id}``) must never leak tool arguments,
tool output, filesystem paths, cost/token data, trace internals, or credential-shaped
strings to a browser. This module is the source-side filter that guarantees it, running
*inside* the type-safe pydantic layer before any event is serialized — not in an nginx
lua filter or a downstream Lambda (the most critical adversarial finding across proposals
was that content filtering downstream fails open).

**Whitelist by construction (the load-bearing property).** :meth:`SafeEventProjection.project`
does NOT strip fields off a raw event (a blacklist that fails open the moment the SDK adds a
new field or event type). It CONSTRUCTS a fresh :data:`SafeEvent` from an explicit allow-list
of fields, and returns ``None`` for any event type it does not recognize. A future zak-code SDK
upgrade that adds a field or an event type therefore cannot leak anything without a code change
here — the new shape simply projects to ``None`` or to its allow-listed subset. (Bravo audit
g-335-41 F2.)

Companion: :mod:`zakcode.secrets` ``redact_secrets`` (regex-shaped tokens, now incl. ``gsk_``/
``vin_``) is the base; :func:`redact_secrets_extended` layers env-value matching, workspace-path
stripping, and high-entropy detection on top, per spec sec 4 Layer 4.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel

from zakcode.secrets import redact_secrets

_REDACTED = "[redacted]"
_PATH_MARK = "[path]"

#: Env vars whose VALUES are treated as secrets and string-matched out of all text.
_SECRET_ENV_SUFFIXES = ("_KEY", "_TOKEN", "_SECRET", "_PASSWORD")
#: Env values shorter than this are ignored for value-matching (avoids redacting
#: trivial values like ``"1"`` or ``"true"`` that would mangle ordinary text).
_MIN_SECRET_VALUE_LEN = 8
#: High-entropy catch-all: only tokens at least this long are entropy-tested.
_ENTROPY_MIN_LEN = 25
#: Shannon entropy (bits/char) above which a long token is treated as a credential.
_ENTROPY_THRESHOLD = 4.5


# ── SafeEvent output models (each carries ONLY its allow-listed fields) ───────


class SafeText(BaseModel):
    """Assistant-visible text, secret-redacted."""

    event: Literal["text"] = "text"
    text: str


class SafeStatus(BaseModel):
    """A human-facing progress note (advisory), secret-redacted."""

    event: Literal["status"] = "status"
    message: str


class SafeToolSummary(BaseModel):
    """A tool call/result collapsed to name + status. Arguments and output are NEVER included."""

    event: Literal["tool_summary"] = "tool_summary"
    name: str = ""
    status: str = ""  # "running" | "completed" | "failed"


class SafeTaskUpdate(BaseModel):
    """The plan checklist reduced to per-task {description, status}. Notes/children dropped."""

    event: Literal["task_update"] = "task_update"
    tasks: list[dict[str, str]] = []


class SafeDone(BaseModel):
    """Terminal event reduced to its stop reason. error/trace/usage/degraded dropped."""

    event: Literal["done"] = "done"
    stop_reason: str


class SafeSessionRotated(BaseModel):
    """A driver session-rotation notice: the watched session was re-minted (e.g. after a daemon
    restart dropped it), so an observer should reconnect to the current session rather than treat
    the stream close as an error. A watch meta-event, NOT an AgentEvent — it carries only a
    redacted reason, no session ids or internals."""

    event: Literal["session_rotated"] = "session_rotated"
    reason: str = ""


class SafeUserMessage(BaseModel):
    """A user message consumed by the driver (the watch/talk unification): the question that
    became this turn's message, surfaced so every watcher sees it before the reply streams.
    A watch meta-event, NOT an AgentEvent. The text already crossed the gateway's
    sanitization boundary; it is redacted again here as defense in depth."""

    event: Literal["user_message"] = "user_message"
    text: str = ""


SafeEvent = (
    SafeText
    | SafeStatus
    | SafeToolSummary
    | SafeTaskUpdate
    | SafeDone
    | SafeSessionRotated
    | SafeUserMessage
)


# ── Extended secret redaction ────────────────────────────────────────────────


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in Counter(s).values())


def _redact_high_entropy(text: str) -> str:
    """Replace long, high-entropy alphanumeric tokens (credential-shaped) with a marker.

    Only tokens >= ``_ENTROPY_MIN_LEN`` chars with Shannon entropy > ``_ENTROPY_THRESHOLD``
    are replaced, so ordinary prose, snake_case identifiers, and git SHAs (entropy ~4.0)
    pass through untouched — the threshold is the known-safe allowlist by construction.
    """
    out: list[str] = []
    token: list[str] = []

    def flush() -> None:
        if not token:
            return
        tok = "".join(token)
        token.clear()
        if len(tok) >= _ENTROPY_MIN_LEN and _shannon_entropy(tok) > _ENTROPY_THRESHOLD:
            out.append(_REDACTED)
        else:
            out.append(tok)

    for ch in text:
        if ch.isalnum() or ch in "_-+/=":
            token.append(ch)
        else:
            flush()
            out.append(ch)
    flush()
    return "".join(out)


def redact_secrets_extended(
    text: str,
    secret_values: Iterable[str] = (),
    workspace_paths: Iterable[str] = (),
) -> str:
    """Extended source-side redaction for public event text (spec sec 4, Layer 4).

    Layers, in order: (1) :func:`zakcode.secrets.redact_secrets` — provider-prefixed
    tokens (now incl. ``gsk_``/``vin_``), PEM keys, URL credentials, ``key = value``
    assignments; (2) exact env secret VALUES; (3) workspace paths -> ``[path]``;
    (4) high-entropy catch-all. Never raises; returns ``""`` for falsy input.
    """
    if not text:
        return ""
    out = redact_secrets(text)[0]
    # (2) exact secret values from the environment — longest first so a value that is a
    # prefix of another does not partially-mask and strand a fragment.
    for val in sorted(
        (v for v in secret_values if v and len(v) >= _MIN_SECRET_VALUE_LEN), key=len, reverse=True
    ):
        if val in out:
            out = out.replace(val, _REDACTED)
    # (3) workspace paths (longest first, for the same reason).
    for path in sorted((p for p in workspace_paths if p and len(p) > 4), key=len, reverse=True):
        if path in out:
            out = out.replace(path, _PATH_MARK)
    # (4) high-entropy catch-all.
    return _redact_high_entropy(out)


def _load_env_secret_values(env: Mapping[str, str]) -> list[str]:
    """Collect credential VALUES from ``*_KEY``/``*_TOKEN``/``*_SECRET``/``*_PASSWORD`` vars."""
    return [
        v
        for k, v in env.items()
        if k.upper().endswith(_SECRET_ENV_SUFFIXES) and v and len(v) >= _MIN_SECRET_VALUE_LEN
    ]


def _expand_workspace_paths(workspace_root: str | None) -> list[str]:
    """The workspace root plus its immediate parent (guards against filesystem-structure leakage
    without over-redacting common prefixes like ``/opt``)."""
    if not workspace_root:
        return []
    # Match the path LITERALLY (no os.path.abspath): abspath("/ws") prepends a drive
    # letter on Windows, so the stored path would not match the posix path the server
    # emits in event text and the redaction would silently no-op cross-platform. Strip
    # trailing separators; os.path.dirname handles both / and \ for the parent.
    root = workspace_root.rstrip("/\\")
    parent = os.path.dirname(root)
    paths = [root]
    if len(parent) > 4 and parent != root:
        paths.append(parent)
    return paths


# ── The projection ────────────────────────────────────────────────────────────


class SafeEventProjection:
    """Projects raw ``AgentEvent`` frames to public :data:`SafeEvent` frames, or ``None``.

    Construct once per server (loads secret env VALUES + workspace paths at init); call
    :meth:`project` per event. Stateless w.r.t. the event (deterministic given the loaded
    config), so it is safe to share across sessions.
    """

    def __init__(
        self,
        env: Mapping[str, str] | None = None,
        workspace_root: str | None = None,
    ) -> None:
        self._secret_values = _load_env_secret_values(os.environ if env is None else env)
        self._workspace_paths = _expand_workspace_paths(workspace_root)

    def redact(self, text: str) -> str:
        return redact_secrets_extended(text, self._secret_values, self._workspace_paths)

    def project(self, event: Any) -> SafeEvent | None:
        """Return a fresh allow-listed SafeEvent, or ``None`` to DROP the event.

        Whitelist by construction: branch on the discriminator and read ONLY allow-listed
        attributes. Unknown discriminators (a future SDK event type, ``usage``,
        ``action_required``) return ``None`` — nothing is ever passed through by default.
        """
        kind = getattr(event, "event", None)
        if kind == "text":
            return SafeText(text=self.redact(str(getattr(event, "text", ""))))
        if kind == "status":
            return SafeStatus(message=self.redact(str(getattr(event, "message", ""))))
        if kind == "tool_call":
            # name only; arguments stripped entirely.
            return SafeToolSummary(
                name=self.redact(str(getattr(event, "name", ""))), status="running"
            )
        if kind == "tool_result":
            # output/data/artifacts stripped entirely; status derived from is_error.
            status = "failed" if getattr(event, "is_error", False) else "completed"
            return SafeToolSummary(name="", status=status)
        if kind == "task_update":
            safe_tasks: list[dict[str, str]] = []
            for t in getattr(event, "tasks", []) or []:
                if isinstance(t, Mapping):
                    safe_tasks.append(
                        {
                            "description": self.redact(
                                str(t.get("title", t.get("description", "")))
                            ),
                            "status": str(t.get("status", "")),
                        }
                    )
            return SafeTaskUpdate(tasks=safe_tasks)
        if kind == "done":
            # stop_reason only; error/trace/usage/degraded dropped.
            return SafeDone(stop_reason=str(getattr(event, "stop_reason", "")))
        if kind == "session_rotated":
            # A watch meta-event (not an AgentEvent): the driver rotated the watched session.
            # Only the redacted reason escapes — no session ids or internals.
            return SafeSessionRotated(reason=self.redact(str(getattr(event, "reason", ""))))
        if kind == "user_message":
            # A watch meta-event (not an AgentEvent): the question the driver just consumed.
            # Gateway-sanitized upstream; redacted again here as defense in depth.
            return SafeUserMessage(text=self.redact(str(getattr(event, "text", ""))))
        # "usage", "action_required", and any unrecognized/future event type: DROP.
        return None


__all__ = [
    "SafeText",
    "SafeStatus",
    "SafeToolSummary",
    "SafeTaskUpdate",
    "SafeDone",
    "SafeSessionRotated",
    "SafeUserMessage",
    "SafeEvent",
    "SafeEventProjection",
    "redact_secrets_extended",
]
