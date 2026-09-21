"""Project Zak Code's session messages into Claude Code's ``.jsonl`` transcript format.

A Claude-Code hook is handed a ``transcript_path`` and reads the conversation *from that
file*. Zak Code's :class:`~zakcode.messages.Message` history is the real source of truth for
what the model is sent; this module renders it in the exact line shape those Claude Code
consumers parse, so a CC framework's hook (e.g. a Stop-hook trailing-text detector, or an
audit that scans for ``tool_use`` events) works against us unchanged.

The FILE those lines go to is an **append-only record** (ADR-0206), as Claude Code's own is:
the loop writes each message once, in order, and never rewrites or truncates what is there.
A compaction shortens what the MODEL is sent; on disk it adds two lines and removes none —
see :func:`render_compaction_rows`. So a reader finds the whole session in the file, can
count its compactions, and can ask what happened after any one of them. This module only
renders text; the loop owns the file and the cursor that says what is already in it.

The shape is reverse-engineered from how it is *consumed* — the consumers' read pattern is
the contract, not any proprietary writer. Each transcript line is one JSON object:

* a ``type`` discriminator at the top level (``"assistant"`` or ``"user"``),
* a nested ``message`` object carrying ``role`` and a ``content`` list of typed blocks,
* an ISO-8601 ``timestamp`` plus the standard CC envelope ids (``sessionId``, ``uuid``,
  ``parentUuid``, ``cwd``) that readers tolerate but rarely require.

What the readers actually look at, and therefore what we guarantee:

* ``evt["type"] == "assistant"`` then ``evt["message"]["content"]`` is a list whose
  ``{"type": "text"}`` blocks expose their prose under ``.text`` — this is how a Stop-hook
  reads the last assistant turn.
* a ``content`` block with ``{"type": "tool_use", "name": ..., "input": {...}}`` — how an
  audit finds tool calls and their arguments.
* tool *results* ride on a ``user``-type line whose ``content`` holds ``tool_result``
  blocks (``tool_use_id`` / ``content`` / ``is_error``), mirroring the vendor wire format.
* a ``user`` line's ``content`` may also be a bare string — one audit reader accepts that,
  so plain user prose is emitted as a string rather than a one-element block list.

The mapping from Zak Code blocks:

* :class:`~zakcode.messages.TextBlock` → ``{"type": "text", "text": ...}``.
* :class:`~zakcode.messages.ToolUseBlock` → ``{"type": "tool_use", "id", "name", "input"}``.
* :class:`~zakcode.messages.ToolResultBlock` → ``{"type": "tool_result", "tool_use_id",
  "content", "is_error"}`` on a ``user`` line (CC carries tool output back as a user turn).
* :class:`~zakcode.messages.ThinkingBlock` and any ``system`` message → skipped: the
  transcript readers neither expect nor read them.

The function does **no file I/O** (the caller owns the write) and is **defensive**: an
odd, empty, or unexpected block is rendered as best it can be and never raises, because a
projection feeding a fail-open hook must never be the thing that breaks the turn.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import Any

from zakcode.messages import (
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)


def _iso_z(ts: datetime) -> str:
    """ISO-8601 with a trailing ``Z`` — the timestamp form CC transcript readers parse."""
    return ts.isoformat().replace("+00:00", "Z")


def _content_blocks(message: Message) -> list[dict[str, Any]]:
    """Project one message's blocks into Claude Code ``content`` block dicts.

    Skips blocks the transcript readers don't model (thinking) and tolerates anything
    unexpected — an attribute that isn't there simply isn't emitted.
    """
    out: list[dict[str, Any]] = []
    for block in message.blocks:
        if isinstance(block, TextBlock):
            out.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolUseBlock):
            out.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    # ``input`` is already a structured dict in our model; default to {}
                    # so a reader doing ``(item.get("input") or {}).get(...)`` is safe.
                    "input": block.input if isinstance(block.input, dict) else {},
                }
            )
        elif isinstance(block, ToolResultBlock):
            out.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.tool_use_id,
                    # CC names the payload ``content`` (our model calls it ``output``).
                    "content": block.output,
                    "is_error": bool(block.is_error),
                }
            )
        elif isinstance(block, ThinkingBlock):
            # Extended thinking has no place in the CC transcript readers — drop it.
            continue
        else:  # pragma: no cover - the union is closed, but never raise on a stray block.
            text = getattr(block, "text", None)
            if isinstance(text, str):
                out.append({"type": "text", "text": text})
    return out


def _line_for(
    message: Message, *, session_id: str, cwd: str, timestamp: str
) -> dict[str, Any] | None:
    """Build the full transcript line for one message, or ``None`` to skip it.

    ``assistant`` and ``user``/``tool`` roles map to CC ``"assistant"`` / ``"user"`` lines;
    ``system`` (and any unknown role) is skipped — the readers don't consume it.
    """
    role = message.role
    if role == "assistant":
        line_type = "assistant"
    elif role in ("user", "tool"):
        # CC carries tool results back as a ``user`` turn, so both map to "user".
        line_type = "user"
    else:
        return None

    content: Any = _content_blocks(message)
    # A plain user message (only text, no tool/structured blocks) is emitted with a string
    # ``content`` — the vendor shape for simple user turns, and a form one audit reader
    # explicitly accepts (``isinstance(content, str)``).
    if role == "user" and content and all(b.get("type") == "text" for b in content):
        content = "".join(b.get("text", "") for b in content)

    # ``message.role`` mirrors the line ``type`` ("assistant" or "user"); a tool turn is a
    # "user" line in CC, so it reports role "user" too.
    msg_role = "assistant" if line_type == "assistant" else "user"
    return {
        "type": line_type,
        "message": {"role": msg_role, "content": content},
        "timestamp": timestamp,
        "sessionId": session_id,
        "uuid": "",
        "parentUuid": None,
        "cwd": cwd,
    }


def render_claude_code_transcript(
    messages: Iterable[Message] | Sequence[Message],
    *,
    session_id: str = "",
    cwd: str = "",
    timestamp: datetime | None = None,
) -> str:
    """Render *messages* as Claude Code ``.jsonl`` transcript text.

    Returns one JSON object per line, newline-terminated (a trailing newline after the
    last line, matching how line-oriented readers ``for line in f`` / ``f.readlines()``
    consume the file). Returns ``""`` for no renderable messages.

    Defensive: never reads or writes a file, and never raises on an odd/empty
    message or block — a malformed turn is rendered as best it can be (worst case, an
    assistant line with empty ``content``) so a fail-open hook reading the projection is
    never broken by it. ``system`` messages and thinking blocks are intentionally omitted
    because the transcript consumers do not read them.
    """
    # Each line carries its message's EVENT time (``Message.created_at``, ADR-0049) — before
    # this, every line got render time and a whole history carried one timestamp, so a reader
    # could not date anything from the transcript. An explicit ``timestamp`` (deterministic —
    # for a caller that owns the clock, e.g. tests) still pins every line verbatim; a message
    # with no usable stamp falls back to the current UTC time, so a reader that filters by a
    # recent window (an audit dropping events older than N hours) keeps the projected line.
    pinned = _iso_z(timestamp) if timestamp is not None else None
    fallback = _iso_z(datetime.now(UTC))
    lines: list[str] = []
    for message in messages:
        created = getattr(message, "created_at", None)
        ts = pinned or (created.replace("+00:00", "Z") if created else fallback)
        try:
            line = _line_for(message, session_id=session_id, cwd=cwd, timestamp=ts)
        except Exception:  # noqa: BLE001 — a projection must never break its caller.
            continue
        if line is None:
            continue
        lines.append(json.dumps(line, ensure_ascii=False))
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


#: What Claude Code writes as the boundary row's ``content`` (read off a real transcript).
_COMPACT_BOUNDARY_TEXT = "Conversation compacted"


def render_compaction_rows(
    summary: str,
    *,
    trigger: str = "",
    messages_before: int = 0,
    messages_after: int = 0,
    pre_tokens: int = 0,
    session_id: str = "",
    cwd: str = "",
    timestamp: datetime | None = None,
) -> str:
    """The two lines a compaction adds to the transcript file, newline-terminated (ADR-0206).

    Claude Code's transcript is append-only, and a compaction shows in it as a pair of rows
    (shape read off a real Claude Code 2.1 transcript, keys only):

    * ``{"type": "system", "subtype": "compact_boundary", "content": "Conversation
      compacted", "compactMetadata": {"trigger": ..., "preTokens": ...}}`` — where the
      context was cut. A reader that asks "what was the first call AFTER the compaction"
      finds this row and walks forward from it.
    * ``{"type": "user", "isCompactSummary": true, "message": {"role": "user", "content":
      "<the summary>"}}`` — what replaced the cut region. A reader that counts a session's
      compactions counts these rows.

    Everything written before the pair stays in the file, the kept tail included: those
    messages were appended when they happened, so they sit ABOVE the boundary even though
    the model still sees them after it. That is Claude Code's layout too.

    ``compactMetadata`` carries only what was measured: ``trigger`` (``auto`` / ``manual`` /
    ``resume``), the live message counts on each side, and ``preTokens`` — the provider's
    last REPORTED prompt size (ADR-0077) — only when there is one. No estimate is written
    under a measured name. Never raises; ``summary`` is written as given.
    """
    ts = _iso_z(timestamp if timestamp is not None else datetime.now(UTC))
    envelope: dict[str, Any] = {
        "timestamp": ts,
        "sessionId": session_id,
        "uuid": "",
        "parentUuid": None,
        "cwd": cwd,
    }
    metadata: dict[str, Any] = {
        "trigger": trigger,
        "preMessages": int(messages_before),
        "postMessages": int(messages_after),
    }
    if pre_tokens > 0:
        metadata["preTokens"] = int(pre_tokens)
    boundary = {
        "type": "system",
        "subtype": "compact_boundary",
        "content": _COMPACT_BOUNDARY_TEXT,
        "level": "info",
        "compactMetadata": metadata,
        **envelope,
    }
    summary_row = {
        "type": "user",
        "message": {"role": "user", "content": str(summary)},
        "isCompactSummary": True,
        **envelope,
    }
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in (boundary, summary_row))


__all__ = ["render_claude_code_transcript", "render_compaction_rows"]
