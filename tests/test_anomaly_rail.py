"""Anomaly rail (ADR-0020): a successful write to a path whose read failed this turn
carries the expected-to-exist question on the tool result.

Field incident 2026-08-26: an index said a knowledge node existed, the read of its file
failed, and the model silently wrote a fresh file — papering over an index/path-resolution
contradiction. The harness cannot tell an intentional create-if-missing from a pave-over,
so the write succeeds and the result asks. These tests pin exactly when the note appears
and when it must not.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from zakcode.agent.loop import _WRITE_AFTER_FAILED_READ_NOTE, AgentLoop
from zakcode.messages import Message, ToolResultBlock
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.store import Session
from zakcode.tools.base import ToolRegistry
from zakcode.tools.builtins.read_file import ReadFileTool
from zakcode.tools.builtins.write_file import WriteFileTool


class _ScriptProvider(Provider):
    """Returns one canned LLMResult per acomplete call, in order."""

    def __init__(self, results: list[LLMResult]) -> None:
        self._results = results
        self.calls = 0

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:
        self.calls += 1
        return self._results[self.calls - 1]

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=8192)


def _loop(results: list[LLMResult], tmp_path: Path) -> AgentLoop:
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    return AgentLoop(
        _ScriptProvider(results),
        registry,
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        max_iterations=8,
    )


def _read(path: str, call_id: str = "r1") -> LLMResult:
    return LLMResult(tool_calls=[ToolCall(id=call_id, name="read_file", arguments={"path": path})])


def _write(path: str, call_id: str = "w1") -> LLMResult:
    return LLMResult(
        tool_calls=[
            ToolCall(id=call_id, name="write_file", arguments={"path": path, "content": "# node\n"})
        ]
    )


def _tool_blocks(loop: AgentLoop) -> list[ToolResultBlock]:
    return [b for m in loop.session.messages for b in m.blocks if isinstance(b, ToolResultBlock)]


def test_write_after_failed_read_carries_the_note(tmp_path: Path) -> None:
    loop = _loop(
        [_read("world/drive.md"), _write("world/drive.md"), LLMResult(text="done")], tmp_path
    )
    result = asyncio.run(loop.arun_turn("recreate the node"))
    assert result.stop_reason == "completed"
    noted = [b for b in _tool_blocks(loop) if _WRITE_AFTER_FAILED_READ_NOTE in b.output]
    assert len(noted) == 1
    # the note rides a SUCCESS — it questions, it does not veto
    assert noted[0].is_error is False
    assert (tmp_path / "world" / "drive.md").read_text(encoding="utf-8") == "# node\n"


def test_plain_create_gets_no_note(tmp_path: Path) -> None:
    # The everyday create-a-new-file flow (no read attempt first) must stay silent.
    loop = _loop([_write("fresh.md"), LLMResult(text="done")], tmp_path)
    asyncio.run(loop.arun_turn("make a file"))
    assert not any(_WRITE_AFTER_FAILED_READ_NOTE in b.output for b in _tool_blocks(loop))


def test_write_to_a_different_path_gets_no_note(tmp_path: Path) -> None:
    loop = _loop([_read("missing-a.md"), _write("other-b.md"), LLMResult(text="done")], tmp_path)
    asyncio.run(loop.arun_turn("go"))
    assert not any(_WRITE_AFTER_FAILED_READ_NOTE in b.output for b in _tool_blocks(loop))


def test_successful_read_then_write_gets_no_note(tmp_path: Path) -> None:
    (tmp_path / "present.md").write_text("old\n", encoding="utf-8")
    loop = _loop([_read("present.md"), _write("present.md"), LLMResult(text="done")], tmp_path)
    asyncio.run(loop.arun_turn("update it"))
    assert not any(_WRITE_AFTER_FAILED_READ_NOTE in b.output for b in _tool_blocks(loop))


def test_note_fires_once_per_path(tmp_path: Path) -> None:
    loop = _loop(
        [
            _read("node.md"),
            _write("node.md", call_id="w1"),
            _write("node.md", call_id="w2"),
            LLMResult(text="done"),
        ],
        tmp_path,
    )
    asyncio.run(loop.arun_turn("go"))
    noted = [b for b in _tool_blocks(loop) if _WRITE_AFTER_FAILED_READ_NOTE in b.output]
    assert len(noted) == 1


def test_relative_and_absolute_spellings_collide(tmp_path: Path) -> None:
    # The model read the workspace-relative form and wrote the absolute form: same file
    # on disk, so the tripwire must key them identically.
    loop = _loop(
        [
            _read("sub/x.md"),
            _write(str(tmp_path / "sub" / "x.md")),
            LLMResult(text="done"),
        ],
        tmp_path,
    )
    asyncio.run(loop.arun_turn("go"))
    assert any(_WRITE_AFTER_FAILED_READ_NOTE in b.output for b in _tool_blocks(loop))


def test_memory_is_per_turn(tmp_path: Path) -> None:
    # A read failure in turn 1 must not haunt a write in turn 2 — the anomaly is only
    # an anomaly while the contradiction is live in the same turn's context.
    loop = _loop(
        [
            _read("later.md"),
            LLMResult(text="it is missing"),
            _write("later.md"),
            LLMResult(text="made it"),
        ],
        tmp_path,
    )
    asyncio.run(loop.arun_turn("check the file"))
    asyncio.run(loop.arun_turn("now create it"))
    assert not any(_WRITE_AFTER_FAILED_READ_NOTE in b.output for b in _tool_blocks(loop))
