"""Canonical conversation model — the single message/content-block shape every layer
shares.

The whole engine speaks this vocabulary: the agent loop builds it, the session persists
it, the provider layer translates it to/from each vendor's wire format. Nothing here is
provider-specific. Keeping one canonical shape (rather than passing vendor dicts around)
is what lets Zak Code stay vendor-agnostic and losslessly persist/resume sessions.

Content is modeled as a discriminated union of typed blocks so tool calls keep structured
``dict`` input (never a re-parsed string) and tool results keep structured data — see
``docs/ARCHITECTURE.md`` (Structured everything, lossless).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from zakcode.artifacts import ArtifactRef

Role = Literal["system", "user", "assistant", "tool"]


def _now_iso() -> str:
    """Current UTC time, ISO-8601 — the event-time stamp every new message carries."""
    return datetime.now(UTC).isoformat()


class TextBlock(BaseModel):
    """A run of plain text authored by the user or the assistant."""

    type: Literal["text"] = "text"
    text: str


class ToolUseBlock(BaseModel):
    """The assistant's request to invoke a tool.

    ``input`` is a structured ``dict`` (the parsed tool arguments), never a JSON string.
    ``id`` correlates this call with its :class:`ToolResultBlock`.
    """

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(BaseModel):
    """The outcome of a tool invocation, carried back into the conversation.

    ``output`` is the human/model-readable text; ``data`` optionally preserves structured
    results (e.g. parsed JSON, match lists) without forcing them through a string.
    """

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    output: str = ""
    is_error: bool = False
    data: dict[str, Any] | None = None
    artifacts: list[ArtifactRef] = Field(default_factory=list)


class ThinkingBlock(BaseModel):
    """Extended-thinking content with its provider signature.

    Persisted so reasoning + tool use can be faithfully replayed where a provider supports
    it (M0 does not emit these; the type exists so history is forward-compatible).
    """

    type: Literal["thinking"] = "thinking"
    text: str
    signature: str | None = None


ContentBlock = Annotated[
    TextBlock | ToolUseBlock | ToolResultBlock | ThinkingBlock,
    Field(discriminator="type"),
]


class Message(BaseModel):
    """One turn in the conversation: a role plus an ordered list of content blocks."""

    role: Role
    blocks: list[ContentBlock] = Field(default_factory=list)
    #: Event time (UTC ISO-8601), stamped when the message is CREATED — not when a
    #: projection renders it (ADR-0049). Before this, the CC transcript stamped every
    #: line with render time, so a 270-record history carried ONE timestamp and a dead
    #: loop could not be dated from its own transcript (coach, zc-03, 2026-08-26 — the
    #: Mind's stop-hook log was the only clock). A document persisted by an older build
    #: loads with a load-time stamp (no worse than the render-time it had); equality of
    #: separately-constructed messages differs by this field, which is what event time
    #: means.
    created_at: str = Field(default_factory=_now_iso)

    # ── ergonomic constructors ──────────────────────────────────────────────
    @classmethod
    def user(cls, text: str) -> Message:
        """A user message containing a single text block."""
        return cls(role="user", blocks=[TextBlock(text=text)])

    @classmethod
    def system(cls, text: str) -> Message:
        """A system message containing a single text block."""
        return cls(role="system", blocks=[TextBlock(text=text)])

    @classmethod
    def assistant_text(cls, text: str) -> Message:
        """An assistant message containing a single text block."""
        return cls(role="assistant", blocks=[TextBlock(text=text)])

    @classmethod
    def tool_results(cls, results: list[ToolResultBlock]) -> Message:
        """A ``tool`` message bundling one or more tool results."""
        return cls(role="tool", blocks=list(results))

    # ── accessors ───────────────────────────────────────────────────────────
    @property
    def text(self) -> str:
        """Concatenated text of all :class:`TextBlock`s in this message."""
        return "".join(b.text for b in self.blocks if isinstance(b, TextBlock))

    @property
    def tool_uses(self) -> list[ToolUseBlock]:
        """All tool-call requests in this message (empty if none)."""
        return [b for b in self.blocks if isinstance(b, ToolUseBlock)]


__all__ = [
    "Role",
    "TextBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "ThinkingBlock",
    "ContentBlock",
    "Message",
]
