"""Project Zak Code's session messages into Claude Code's ``.jsonl`` transcript format.

A Claude-Code hook is handed a ``transcript_path`` and reads the conversation *from that
file*. Zak Code's :class:`~zakcode.messages.Message` history is the real source of truth;
this module renders a **read-only projection** of it in the exact line shape those Claude
Code consumers parse, so a CC framework's hook (e.g. a Stop-hook trailing-text detector,
or an audit that scans for ``tool_use`` events) works against us unchanged.

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

The function is **pure** (no file I/O — the caller owns the write) and **defensive**: an
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

#: Fixed timestamp stamped on every projected line. Readers that filter by a time window
#: parse this field (``datetime.fromisoformat`` after trimming a trailing ``Z``); a single
#: deterministic value keeps the projection pure (no wall-clock) while staying parseable.
_PROJECTION_TS = datetime(1970, 1, 1, tzinfo=UTC).isoformat().replace("+00:00", "Z")


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


def _line_for(message: Message, *, session_id: str, cwd: str) -> dict[str, Any] | None:
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
        "timestamp": _PROJECTION_TS,
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
) -> str:
    """Render *messages* as Claude Code ``.jsonl`` transcript text.

    Returns one JSON object per line, newline-terminated (a trailing newline after the
    last line, matching how line-oriented readers ``for line in f`` / ``f.readlines()``
    consume the file). Returns ``""`` for no renderable messages.

    Pure and defensive: never reads or writes a file, and never raises on an odd/empty
    message or block — a malformed turn is rendered as best it can be (worst case, an
    assistant line with empty ``content``) so a fail-open hook reading the projection is
    never broken by it. ``system`` messages and thinking blocks are intentionally omitted
    because the transcript consumers do not read them.
    """
    lines: list[str] = []
    for message in messages:
        try:
            line = _line_for(message, session_id=session_id, cwd=cwd)
        except Exception:  # noqa: BLE001 — a projection must never break its caller.
            continue
        if line is None:
            continue
        lines.append(json.dumps(line, ensure_ascii=False))
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


__all__ = ["render_claude_code_transcript"]
