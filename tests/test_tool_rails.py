"""Tests for tool-result rails: ``ToolResult.hint``/``fix``, the loop rendering them as
``Hint:``/``Fix:`` lines, the permission-denial remedy, and the builtin rails.

This is the "name the next action" lever applied to the tool->model boundary — the single
biggest help for a small model, which is weak at planning the next step and at recovering
from errors. Everything here is hermetic (scripted provider, fake tools, tmp workspace).
"""

from __future__ import annotations

from pathlib import Path

from zakcode.agent.loop import AgentLoop, _append_rail, _control_rail, _denial_remedy
from zakcode.config import PermissionTier
from zakcode.permissions import PermissionMode, PermissionPolicy
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
from zakcode.tools.builtins.edit import EditFileTool
from zakcode.tools.builtins.read_file import ReadFileTool
from zakcode.usage import Usage

# ── pure helpers ────────────────────────────────────────────────────────────────


def test_control_rail_carries_provenance_and_rail_word() -> None:
    # ADR-0021: a loop-INJECTED directive is a user-role message, so it carries the
    # [harness] provenance tag + the shared rail word; tool-result rails stay bare
    # (they ride inside a tool frame, already unambiguous).
    assert _control_rail("do the thing") == "[harness] Hint: do the thing"


def test_append_rail_fix_wins_over_hint() -> None:
    assert _append_rail("out", hint="h", fix="f") == "out\nFix: f"


def test_append_rail_hint_when_no_fix() -> None:
    assert _append_rail("out", hint="h", fix=None) == "out\nHint: h"


def test_append_rail_noop_when_neither() -> None:
    assert _append_rail("out", hint=None, fix=None) == "out"


def test_append_rail_empty_output_no_leading_newline() -> None:
    assert _append_rail("", hint="h", fix=None) == "Hint: h"


def test_denial_remedy_by_tier() -> None:
    ws = _denial_remedy(PermissionTier.WORKSPACE_WRITE)
    assert "acceptEdits" in ws and "workspace-write" in ws
    assert "allow" in _denial_remedy(PermissionTier.DANGER_FULL_ACCESS)
    assert _denial_remedy(None)  # non-empty generic fallback


# ── ToolResult fields ───────────────────────────────────────────────────────────


def test_toolresult_ok_carries_hint() -> None:
    r = ToolResult.ok("did", hint="next")
    assert r.hint == "next" and r.fix is None and r.is_error is False


def test_toolresult_error_carries_fix() -> None:
    r = ToolResult.error("nope", fix="try this")
    assert r.fix == "try this" and r.hint is None and r.is_error is True


# ── loop rendering ───────────────────────────────────────────────────────────────


class _Hinter(Tool):
    spec = ToolSpec(
        name="hinter", description="returns a hint", required_permission=PermissionTier.READ_ONLY
    )

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        return ToolResult.ok("did the thing", hint="do X next")


class _Failer(Tool):
    spec = ToolSpec(
        name="failer", description="fails with a fix", required_permission=PermissionTier.READ_ONLY
    )

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        return ToolResult.error("it broke", fix="try Y")


class _Writer(Tool):
    spec = ToolSpec(
        name="writer",
        description="write tool",
        required_permission=PermissionTier.WORKSPACE_WRITE,
        concurrency=ConcurrencyClass.NEVER_PARALLEL,
    )

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        return ToolResult.ok("wrote")


class _CallThenDone(Provider):
    """Requests one named tool call on the first turn, then completes."""

    def __init__(self, name: str, args: dict) -> None:
        self.name = name
        self.args = args
        self.calls = 0

    async def acomplete(self, messages, *, system=None, tools=None, **kw) -> LLMResult:
        self.calls += 1
        if self.calls == 1:
            return LLMResult(
                text="",
                tool_calls=[ToolCall(id="c1", name=self.name, arguments=self.args)],
                usage=Usage(total_tokens=1),
            )
        return LLMResult(text="done", tool_calls=[], usage=Usage(total_tokens=1))

    def count_tokens(self, messages, *, system=None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities()


def _loop(tmp_path: Path, provider: Provider, registry: ToolRegistry, *, policy=None) -> AgentLoop:
    return AgentLoop(
        provider,
        registry,
        Session(cwd=str(tmp_path), model="scripted/test"),
        workspace_root=tmp_path,
        permission_policy=policy,
        max_iterations=5,
    )


async def test_hint_renders_into_result_block(tmp_path: Path) -> None:
    reg = ToolRegistry()
    reg.register(_Hinter())
    loop = _loop(tmp_path, _CallThenDone("hinter", {}), reg)
    result = await loop.arun_turn("go")
    block = result.tool_results[0]
    assert "did the thing" in block.output
    assert "Hint: do X next" in block.output
    assert block.data and block.data.get("hint") == "do X next"


async def test_fix_renders_into_error_block(tmp_path: Path) -> None:
    reg = ToolRegistry()
    reg.register(_Failer())
    loop = _loop(tmp_path, _CallThenDone("failer", {}), reg)
    result = await loop.arun_turn("go")
    block = result.tool_results[0]
    assert block.is_error
    assert "it broke" in block.output and "Fix: try Y" in block.output
    assert block.data and block.data.get("fix") == "try Y"


async def test_denial_names_the_remedy(tmp_path: Path) -> None:
    reg = ToolRegistry()
    reg.register(_Writer())
    loop = _loop(
        tmp_path,
        _CallThenDone("writer", {"path": "x"}),
        reg,
        policy=PermissionPolicy(PermissionMode.DENY),
    )
    result = await loop.arun_turn("go")
    block = result.tool_results[0]
    assert block.is_error
    assert "permission denied" in block.output.lower()
    assert "Fix:" in block.output and "acceptEdits" in block.output
    assert block.data and "acceptEdits" in (block.data.get("fix") or "")


# ── builtin rails ────────────────────────────────────────────────────────────────


async def test_read_file_not_found_has_fix(tmp_path: Path) -> None:
    res = await ReadFileTool().execute({"path": "nope.txt"}, ToolContext(workspace_root=tmp_path))
    assert res.is_error and res.fix and ("list_dir" in res.fix or "glob" in res.fix)


async def test_edit_not_found_has_fix(tmp_path: Path) -> None:
    res = await EditFileTool().execute(
        {"path": "nope.txt", "old_string": "a", "new_string": "b"},
        ToolContext(workspace_root=tmp_path),
    )
    assert res.is_error and res.fix and "write_file" in res.fix


async def test_edit_old_string_not_found_has_fix(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("hello world", encoding="utf-8")
    res = await EditFileTool().execute(
        {"path": "f.txt", "old_string": "zzz", "new_string": "b"},
        ToolContext(workspace_root=tmp_path),
    )
    assert res.is_error and res.fix and "re-read" in res.fix.lower()
