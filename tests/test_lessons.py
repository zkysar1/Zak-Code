"""Tests for the deterministic failure-lesson writer (research R1)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from zakcode.agent.lessons import LESSON_KIND, LessonWriter, _derive, _normalize
from zakcode.agent.loop import AgentLoop
from zakcode.agent.recipe import RecipeCursor
from zakcode.agent.stuck import StuckTracker
from zakcode.memory import MemoryProvider, MemoryRecord
from zakcode.messages import Message, ToolResultBlock
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.store import Session
from zakcode.tools.base import Tool, ToolContext, ToolRegistry, ToolResult, ToolSpec


def _c(call_id: str, name: str, **args: object) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=dict(args))


def _r(tool_use_id: str, *, is_error: bool = False, output: str = "ok", path: str | None = None):
    data = {"path": path} if path else None
    return ToolResultBlock(tool_use_id=tool_use_id, output=output, is_error=is_error, data=data)


class _FakeMemory(MemoryProvider):
    """In-memory MemoryProvider whose search returns all records (so dedup logic is exercised)."""

    def __init__(self) -> None:
        self.records: list[MemoryRecord] = []

    def add(self, text, *, kind="note", tags=None, source=""):
        rec = MemoryRecord(text=text, kind=kind, tags=tags or [], source=source)
        self.records.append(rec)
        return rec

    def search(self, query, *, limit=5):
        return list(self.records)[:limit]

    def recent(self, *, limit=10):
        return list(self.records)[-limit:]

    def delete(self, memory_id):
        before = len(self.records)
        self.records = [r for r in self.records if r.id != memory_id]
        return len(self.records) < before

    def count(self):
        return len(self.records)


def _recipe_recovered(command: str = "py a.py") -> RecipeCursor:
    """A cursor that ran a created file, FAILED, then SUCCEEDED (a recipe recovery)."""
    c = RecipeCursor(enabled=True)
    c.observe([_c("w", "write_file", path="a.py")], [_r("w", path="a.py")])
    c.observe([_c("r1", "bash", command=command)], [_r("r1", is_error=True)])
    c.observe([_c("r2", "bash", command=command)], [_r("r2")])
    return c


def _stuck_recovered(tool: str = "boom") -> StuckTracker:
    """A tracker that fired a recovery action with a repeatedly-failing ``tool``."""
    t = StuckTracker(nudge_at=1, narrow_at=2, stop_at=3, repeated_failure_at=2)
    t.observe([_c("c1", tool, k="a")], [_r("c1", is_error=True)], assistant_text="")
    t.next_action()  # streak 1 -> NUDGE (took_action True)
    t.observe([_c("c2", tool, k="a")], [_r("c2", is_error=True)], assistant_text="")
    t.next_action()
    return t


# ── recovered_command + error_signatures ─────────────────────────────────────────


def test_recipe_recovery_command_captured() -> None:
    c = _recipe_recovered()
    assert c.recovered_command == "py a.py"
    assert c.verified is True


def test_no_recovery_when_first_run_succeeds() -> None:
    c = RecipeCursor(enabled=True)
    c.observe([_c("w", "write_file", path="a.py")], [_r("w", path="a.py")])
    c.observe([_c("r", "bash", command="py a.py")], [_r("r")])  # succeeds first try
    assert c.recovered_command is None  # no prior failure -> not a recovery


# ── _derive (gate selection) ─────────────────────────────────────────────────────


def test_derive_prefers_recipe_recovery() -> None:
    text, tags = _derive(StuckTracker(), _recipe_recovered(), stop_reason="completed")
    assert "py a.py" in text and "recovery" in tags


def test_derive_stuck_recovery_when_no_recipe() -> None:
    out = _derive(_stuck_recovered(), RecipeCursor(enabled=True), stop_reason="completed")
    assert out is not None
    text, tags = out
    assert "boom" in text and "stuck-recovery" in tags


def test_derive_none_on_clean_turn() -> None:
    assert _derive(StuckTracker(), RecipeCursor(enabled=True), stop_reason="completed") is None


def test_derive_stuck_requires_completed_stop() -> None:
    # Gate A only fires on a clean completion (a stuck STOP/doom_loop is not a recovery).
    assert _derive(_stuck_recovered(), RecipeCursor(enabled=True), stop_reason="stuck") is None


# ── LessonWriter ─────────────────────────────────────────────────────────────────


def test_maybe_write_no_memory_is_noop() -> None:
    writer = LessonWriter(None, source="lesson:x")
    wrote = writer.maybe_write(
        _stuck_recovered(), RecipeCursor(enabled=True), stop_reason="completed"
    )
    assert wrote is False


def test_maybe_write_writes_one_lesson() -> None:
    mem = _FakeMemory()
    wrote = LessonWriter(mem, source="lesson:s1").maybe_write(
        StuckTracker(), _recipe_recovered(), stop_reason="completed"
    )
    assert wrote is True
    assert mem.count() == 1
    rec = mem.records[0]
    assert rec.kind == LESSON_KIND and rec.source == "lesson:s1"


def test_maybe_write_dedups_equivalent_lessons() -> None:
    mem = _FakeMemory()
    writer = LessonWriter(mem, source="lesson:s1")
    assert writer.maybe_write(StuckTracker(), _recipe_recovered(), stop_reason="completed") is True
    # A second identical recovery does NOT append a duplicate.
    assert writer.maybe_write(StuckTracker(), _recipe_recovered(), stop_reason="completed") is False
    assert mem.count() == 1


def test_maybe_write_redacts_secrets() -> None:
    mem = _FakeMemory()
    secret_cmd = "py a.py --key=sk-abcdefghijklmnop0123456789"
    LessonWriter(mem, source="x").maybe_write(
        StuckTracker(), _recipe_recovered(secret_cmd), stop_reason="completed"
    )
    assert mem.count() == 1
    assert "sk-abcdefghijklmnop0123456789" not in mem.records[0].text


def test_normalize_collapses_paths_and_numbers() -> None:
    a = _normalize("run C:/x/app.py line 42 hash deadbeefcafe")
    b = _normalize("run C:/y/other.py line 99 hash 0123456789ab")
    assert a == b  # volatile bits collapse to the same key


def test_gate_a_lessons_for_different_tools_not_deduped() -> None:
    # review2 #2: two stuck-recovery lessons for DIFFERENT failing tools differ only in the tool
    # name (heavy boilerplate -> Jaccard > 0.8), but distinct tags must keep them both.
    mem = _FakeMemory()
    writer = LessonWriter(mem, source="x")

    def write(tool: str) -> bool:
        return writer.maybe_write(
            _stuck_recovered(tool), RecipeCursor(enabled=True), stop_reason="completed"
        )

    assert write("boom") is True
    assert write("crash") is True  # different tag identity -> not a duplicate
    assert mem.count() == 2
    assert write("boom") is False  # same tool again -> duplicate
    assert mem.count() == 2


def test_gate_b_no_cross_file_misattribution() -> None:
    # review2 #3: file a fails then recovers; file b then succeeds on its FIRST run (never failed)
    # and must NOT overwrite the genuine recovery command.
    c = RecipeCursor(enabled=True)
    c.observe([_c("wa", "write_file", path="a.py")], [_r("wa", path="a.py")])
    c.observe([_c("ra1", "bash", command="py a.py")], [_r("ra1", is_error=True)])
    c.observe([_c("ra2", "bash", command="py a.py")], [_r("ra2")])
    assert c.recovered_command == "py a.py"
    c.observe([_c("wb", "write_file", path="b.py")], [_r("wb", path="b.py")])
    c.observe([_c("rb", "bash", command="py b.py")], [_r("rb")])  # b first-try success
    assert c.recovered_command == "py a.py"  # still a's genuine recovery, not b's


# ── loop integration ─────────────────────────────────────────────────────────────


class _Scripted(Provider):
    def __init__(self, results: list[LLMResult]) -> None:
        self._results = list(results)
        self.calls = 0

    async def acomplete(
        self, messages: list[Message], *, system=None, tools=None, response_format=None, **kw: Any
    ) -> LLMResult:
        r = self._results[min(self.calls, len(self._results) - 1)]
        self.calls += 1
        return r

    def count_tokens(self, messages, *, system=None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=8192)


class _Boom(Tool):
    spec = ToolSpec(name="boom", description="Always errors.")

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.error("boom")


def test_loop_stuck_recovery_writes_one_lesson(tmp_path: Path) -> None:
    # boom a/b/a (non-consecutive repeat -> repeated-failure + a nudge, no doom), then recover.
    reg = ToolRegistry()
    reg.register(_Boom())
    script = [
        LLMResult(tool_calls=[_c("c1", "boom", k="a")]),
        LLMResult(tool_calls=[_c("c2", "boom", k="b")]),
        LLMResult(tool_calls=[_c("c3", "boom", k="a")]),
        LLMResult(text="done"),
    ]
    mem = _FakeMemory()
    loop = AgentLoop(
        _Scripted(script),
        reg,
        Session(cwd=str(tmp_path), model="t"),
        workspace_root=tmp_path,
        max_iterations=10,
        memory_provider=mem,
    )
    result = asyncio.run(loop.arun_turn("do it"))
    assert result.stop_reason == "completed"
    lessons = [r for r in mem.records if r.kind == LESSON_KIND]
    assert len(lessons) == 1 and "boom" in lessons[0].text


def test_loop_clean_turn_writes_no_lesson(tmp_path: Path) -> None:
    mem = _FakeMemory()
    loop = AgentLoop(
        _Scripted([LLMResult(text="hi")]),
        ToolRegistry(),
        Session(cwd=str(tmp_path), model="t"),
        workspace_root=tmp_path,
        max_iterations=10,
        memory_provider=mem,
    )
    asyncio.run(loop.arun_turn("hi"))
    assert [r for r in mem.records if r.kind == LESSON_KIND] == []
