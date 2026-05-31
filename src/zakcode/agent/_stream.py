"""Streaming tool-call accumulator.

A provider's token stream delivers each tool call as a sequence of
:class:`~zakcode.providers.base.StreamToolCallDelta` fragments keyed by
``index``: the first fragment for an index carries ``id``/``name``, and every
fragment carries a slice of the JSON-encoded ``arguments`` string. The agent
loop folds those fragments into completed :class:`~zakcode.providers.base.ToolCall`
objects with :class:`ToolCallAccumulator`.

The accumulator mirrors the provider contract's parsing rule (see
``ToolCall.arguments``): the concatenated argument string is JSON-decoded into a
``dict``; on a decode failure it falls back to ``{"_raw": <string>}`` rather than
raising, so a malformed stream can never crash the loop. Blank/empty arguments
decode to ``{}``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from zakcode.providers.base import StreamToolCallDelta, ToolCall


@dataclass
class _PartialCall:
    """Mutable per-index accumulation state for one in-flight tool call."""

    index: int
    id: str | None = None
    name: str | None = None
    args_parts: list[str] = field(default_factory=list)

    def arguments_str(self) -> str:
        """The concatenation of every ``arguments_delta`` fragment seen so far."""
        return "".join(self.args_parts)


def parse_arguments(raw: str) -> dict[str, Any]:
    """Parse a concatenated tool-call argument string into a ``dict``.

    Empty/blank input yields ``{}``. A JSON value that decodes to something other
    than an object (or fails to decode at all) falls back to ``{"_raw": raw}`` so
    the caller always receives a ``dict`` and this never raises.
    """
    if not raw or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {"_raw": raw}
    if isinstance(parsed, dict):
        return parsed
    return {"_raw": raw}


class ToolCallAccumulator:
    """Folds a sequence of :class:`StreamToolCallDelta` into completed tool calls.

    Fragments are grouped by ``index``. The first fragment carrying an ``id`` or
    ``name`` for a given index sets that field (later non-``None`` values do not
    overwrite it); ``arguments_delta`` strings are concatenated in arrival order.
    :meth:`finalize` emits one :class:`ToolCall` per index, ordered by index.
    """

    def __init__(self) -> None:
        # Insertion order is preserved, but finalize() sorts by index regardless.
        self._calls: dict[int, _PartialCall] = {}

    def add(self, delta: StreamToolCallDelta) -> None:
        """Fold one streamed fragment into the accumulator."""
        partial = self._calls.get(delta.index)
        if partial is None:
            partial = _PartialCall(index=delta.index)
            self._calls[delta.index] = partial
        if delta.id is not None and partial.id is None:
            partial.id = delta.id
        if delta.name is not None and partial.name is None:
            partial.name = delta.name
        if delta.arguments_delta:
            partial.args_parts.append(delta.arguments_delta)

    def __len__(self) -> int:
        return len(self._calls)

    def __bool__(self) -> bool:
        return bool(self._calls)

    def finalize(self) -> list[ToolCall]:
        """Return the accumulated calls as :class:`ToolCall` objects, index-ordered.

        Arguments are parsed via :func:`parse_arguments` (never raises). A call
        whose ``id``/``name`` never arrived gets an empty-string fallback so the
        result is always a well-formed :class:`ToolCall`.
        """
        calls: list[ToolCall] = []
        for index in sorted(self._calls):
            partial = self._calls[index]
            calls.append(
                ToolCall(
                    id=partial.id or "",
                    name=partial.name or "",
                    arguments=parse_arguments(partial.arguments_str()),
                )
            )
        return calls


__all__ = ["ToolCallAccumulator", "parse_arguments"]
