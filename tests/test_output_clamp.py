"""Seam-level tool-output clamp (ADR-0023): no single result may swamp the window.

Field incident 2026-08-26 (131k local pod): a 2,776-line skill body landed whole in the
transcript — nothing between a tool and the session bounds output size for tools that
cannot know their own shape (bash, skill bodies, grep) — and the very next model call
overflowed the window. These tests pin the clamp's proportionality, the head+tail keep,
and that appended guidance (hook notes, rails) is never lost to the elision.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from zakcode.agent.loop import AgentLoop
from zakcode.config import PermissionTier
from zakcode.messages import Message, ToolResultBlock
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.store import Session
from zakcode.tools.base import (
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)


class _BigDumpTool(Tool):
    """Returns a scriptable payload (optionally with a hint rail)."""

    spec = ToolSpec(
        name="bigdump",
        description="test payload",
        parameters={"type": "object", "properties": {}},
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    def __init__(self, payload: str, *, hint: str | None = None) -> None:
        self._payload = payload
        self._hint = hint

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(self._payload, hint=self._hint)


class _ScriptProvider(Provider):
    def __init__(self, results: list[LLMResult], *, window: int) -> None:
        self._results = results
        self._window = window
        self.calls = 0

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:
        self.calls += 1
        return self._results[self.calls - 1]

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=self._window)


def _run(tool: _BigDumpTool, tmp_path: Path, *, window: int) -> ToolResultBlock:
    registry = ToolRegistry()
    registry.register(tool)
    results = [
        LLMResult(tool_calls=[ToolCall(id="c1", name="bigdump", arguments={})]),
        LLMResult(text="done"),
    ]
    loop = AgentLoop(
        _ScriptProvider(results, window=window),
        registry,
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        max_iterations=4,
    )
    result = asyncio.run(loop.arun_turn("dump it"))
    assert result.stop_reason == "completed"
    blocks = [b for m in loop.session.messages for b in m.blocks if isinstance(b, ToolResultBlock)]
    assert len(blocks) == 1
    return blocks[0]


def test_oversized_output_is_clamped_head_and_tail(tmp_path: Path) -> None:
    # window 8192 → max_chars = 8192 × 0.25 × 3 = 6144
    payload = "A" * 10_000 + "THE-MIDDLE-MARKER" + "B" * 10_000
    block = _run(_BigDumpTool(payload), tmp_path, window=8192)
    assert "[output clamped: 20,017 chars" in block.output
    assert block.output.startswith("A")
    assert block.output.rstrip().endswith("B")
    assert "THE-MIDDLE-MARKER" not in block.output
    assert len(block.output) < 6144 + 300  # payload bounded; only the note rides on top
    assert block.is_error is False


def test_small_output_passes_through_unchanged(tmp_path: Path) -> None:
    block = _run(_BigDumpTool("short and sweet"), tmp_path, window=8192)
    assert block.output == "short and sweet"


def test_clamp_scales_with_the_window(tmp_path: Path) -> None:
    # The same 20k payload fits comfortably under a 200k window (max 150,000 chars).
    payload = "A" * 10_000 + "THE-MIDDLE-MARKER" + "B" * 10_000
    block = _run(_BigDumpTool(payload), tmp_path, window=200_000)
    assert block.output == payload


def test_rails_survive_the_clamp(tmp_path: Path) -> None:
    payload = "C" * 30_000
    block = _run(_BigDumpTool(payload, hint="read the tail"), tmp_path, window=8192)
    assert "[output clamped:" in block.output
    assert block.output.rstrip().endswith("Hint: read the tail")


class _VerbatimDumpTool(_BigDumpTool):
    """Instructions, not data: a skill body or a rule (ADR-0065)."""

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(self._payload, hint=self._hint, verbatim=True)


def test_a_verbatim_result_is_never_clamped(tmp_path: Path) -> None:
    # 2026-08-28 (coach, zc-03): a 37,875-char /boot body clamped to 6 KB lost Steps 0–11.
    payload = (
        "## Step 0\n" + "A" * 10_000 + "\n## Step 5: THE-MIDDLE\n" + "B" * 10_000 + "\n## Step 12\n"
    )
    block = _run(_VerbatimDumpTool(payload, hint="follow it"), tmp_path, window=8192)
    assert "[output clamped:" not in block.output
    assert "## Step 5: THE-MIDDLE" in block.output
    assert block.output.rstrip().endswith("Hint: follow it")
