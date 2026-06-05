"""Tests for the Recipe Cursor (Slice 2): a verify-before-finish gate.

Pure-state tests for :class:`RecipeCursor`, plus loop-integration tests proving that
with ``recipe_mode`` on, a create-and-run turn cannot end until the written .py has been
run successfully — and ends gracefully as ``recipe_stalled`` if it never is.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from zakcode.agent.loop import AgentLoop
from zakcode.agent.recipe import RecipeCursor
from zakcode.messages import Message, ToolResultBlock
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.store import Session
from zakcode.tools import default_registry


def _c(call_id: str, name: str, **args: object) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=dict(args))


def _r(tool_use_id: str, *, path: str | None = None, is_error: bool = False) -> ToolResultBlock:
    data = {"path": path} if path else None
    return ToolResultBlock(tool_use_id=tool_use_id, output="ok", is_error=is_error, data=data)


# ── pure cursor logic ─────────────────────────────────────────────────────────


def test_cursor_disabled_never_gates() -> None:
    c = RecipeCursor(enabled=False)
    c.observe([_c("w", "write_file", path="a.py")], [_r("w", path="a.py")])
    assert c.needs_verification() is False


def test_cursor_gates_after_python_write() -> None:
    c = RecipeCursor(enabled=True)
    c.observe([_c("w", "write_file", path="a.py")], [_r("w", path="a.py")])
    assert c.needs_verification() is True


def test_cursor_ignores_non_python_write() -> None:
    c = RecipeCursor(enabled=True)
    c.observe([_c("w", "write_file", path="notes.txt")], [_r("w", path="notes.txt")])
    assert c.needs_verification() is False


def test_cursor_verified_by_run_referencing_file() -> None:
    c = RecipeCursor(enabled=True)
    c.observe([_c("w", "write_file", path="fizz.py")], [_r("w", path="fizz.py")])
    c.observe([_c("r", "bash", command="py fizz.py")], [_r("r")])
    assert c.needs_verification() is False


def test_cursor_run_not_referencing_file_does_not_verify() -> None:
    c = RecipeCursor(enabled=True)
    c.observe([_c("w", "write_file", path="fizz.py")], [_r("w", path="fizz.py")])
    c.observe([_c("r", "bash", command="echo hello")], [_r("r")])
    assert c.needs_verification() is True


def test_cursor_failed_run_does_not_verify() -> None:
    c = RecipeCursor(enabled=True)
    c.observe([_c("w", "write_file", path="fizz.py")], [_r("w", path="fizz.py")])
    c.observe([_c("r", "bash", command="py fizz.py")], [_r("r", is_error=True)])
    assert c.needs_verification() is True


def test_cursor_nudge_cap() -> None:
    c = RecipeCursor(enabled=True, attempt_cap=2)
    assert c.can_nudge()
    c.nudge()
    assert c.can_nudge()
    c.nudge()
    assert c.can_nudge() is False


# ── loop integration ──────────────────────────────────────────────────────────


class _ScriptedProvider(Provider):
    """Returns canned LLMResults in order; the last repeats once exhausted."""

    def __init__(self, script: Sequence[LLMResult]) -> None:
        self._script = list(script)
        self.calls = 0

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:
        idx = min(self.calls, len(self._script) - 1)
        self.calls += 1
        return self._script[idx]

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=8192)


def _loop(provider: _ScriptedProvider, tmp_path: Path, **kw: Any) -> AgentLoop:
    return AgentLoop(
        provider,
        default_registry(),
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        max_iterations=10,
        **kw,
    )


def test_recipe_gate_forces_a_run_before_completing(tmp_path: Path) -> None:
    write = LLMResult(
        tool_calls=[_c("w1", "write_file", path="prog.py", content="print('hi')\n")]
    )
    done = LLMResult(text="All done!")
    run = LLMResult(tool_calls=[_c("r1", "bash", command="echo prog.py")])
    provider = _ScriptedProvider([write, done, run, done])
    result = asyncio.run(_loop(provider, tmp_path, recipe_mode=True).arun_turn("make prog.py"))
    assert result.stop_reason == "completed"
    # write -> "done"(blocked+nudged) -> run -> "done"(now verified) = 4 provider calls.
    assert provider.calls == 4


def test_recipe_gate_stalls_gracefully_after_cap(tmp_path: Path) -> None:
    write = LLMResult(tool_calls=[_c("w1", "write_file", path="p.py", content="print('x')\n")])
    done = LLMResult(text="done")  # never runs the file
    provider = _ScriptedProvider([write, done])
    loop = _loop(provider, tmp_path, recipe_mode=True, recipe_attempt_cap=2)
    result = asyncio.run(loop.arun_turn("make p.py"))
    assert result.stop_reason == "recipe_stalled"


def test_recipe_disabled_completes_without_a_run(tmp_path: Path) -> None:
    write = LLMResult(tool_calls=[_c("w1", "write_file", path="p.py", content="print('x')\n")])
    done = LLMResult(text="done")
    provider = _ScriptedProvider([write, done])
    result = asyncio.run(_loop(provider, tmp_path, recipe_mode=False).arun_turn("make p.py"))
    assert result.stop_reason == "completed"
    assert provider.calls == 2  # no nudge: write, done
