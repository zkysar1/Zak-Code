"""A scoped test run does not verify the whole turn (ADR-0141).

Field probe, 2026-09-11, a 35B model on zak-code's own tree, in BOTH arms of a two-arm run.
The task was to add three helpers to ``src/zakcode/providers/text_tools.py`` with tests. It
did that. Then it ran::

    uv run pytest tests/test_text_tools.py -v 2>&1 | tail -30

saw ``67 passed``, and closed with "All done -- 67/67 tests pass."

The FULL suite on that tree was **2 failed, 3671 passed**. The same tree with the model's
changes reverted was **3656 passed, 0 failed** -- so the two reds were caused by its own
change, in test files it never ran. ``_suite_verified``'s contract is that a green test-runner
run "verified the whole turn"; a run scoped to one file cannot, and the gate credited it anyway.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from zakcode.agent.loop import _SCOPE_NUDGE, AgentLoop
from zakcode.agent.recipe import RecipeCursor, _suite_run_is_scoped
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

#: The field command, verbatim.
FIELD = "uv run pytest tests/test_text_tools.py -v 2>&1 | tail -30"


# ── the predicate ────────────────────────────────────────────────────────────


def test_the_field_command_is_scoped() -> None:
    assert _suite_run_is_scoped(FIELD)


def test_a_file_a_node_id_or_a_filter_narrows_the_run() -> None:
    assert _suite_run_is_scoped("pytest tests/test_text_tools.py")
    assert _suite_run_is_scoped("pytest a.py::test_b")
    assert _suite_run_is_scoped("pytest -k truncate")
    assert _suite_run_is_scoped("pytest -k=truncate")
    assert _suite_run_is_scoped("pytest --lf")
    assert _suite_run_is_scoped("pytest --last-failed")
    # cd + env-manager wrapper + wrapper flag, all in front of the selector.
    assert _suite_run_is_scoped("cd sub && uv run --no-sync pytest tests/x.py")


def test_running_the_whole_suite_is_not_scoped() -> None:
    for command in (
        "pytest",
        "pytest -q",
        "pytest --tb=short -q",
        "uv run pytest -q",
        "uv run --no-sync pytest",
        "npm test",
        "cargo test",
    ):
        assert not _suite_run_is_scoped(command), command


def test_python_dash_m_is_the_module_launcher_not_a_marker_filter() -> None:
    # `-m` narrows for pytest (`pytest -m slow`) and launches for python
    # (`python -m pytest`). They are indistinguishable by token, and treating `-m` as a
    # selector would call the single most common way to run a suite "scoped".
    assert not _suite_run_is_scoped("python -m pytest")
    assert not _suite_run_is_scoped("uv run python -m pytest")


def test_a_directory_is_how_projects_spell_the_whole_suite() -> None:
    assert not _suite_run_is_scoped("pytest tests/ -q")
    assert not _suite_run_is_scoped("pytest tests -q")


def test_a_selector_shaped_token_downstream_of_the_runner_is_not_one() -> None:
    # The scan runs only on the segment the classifier calls the runner, so a later
    # pipeline stage that happens to name a .py file is not read as a selector.
    assert not _suite_run_is_scoped("pytest -q | grep foo.py")
    assert not _suite_run_is_scoped("pytest -q | tail -30")


# ── the cursor ───────────────────────────────────────────────────────────────


def _run(cursor: RecipeCursor, command: str, output: str) -> None:
    cursor.observe(
        [ToolCall(id="r", name="bash", arguments={"command": command})],
        [ToolResultBlock(tool_use_id="r", output=output, is_error=False)],
    )


def _wrote(cursor: RecipeCursor, path: str) -> None:
    cursor.observe(
        [ToolCall(id="w", name="write_file", arguments={"path": path})],
        [ToolResultBlock(tool_use_id="w", output="written", data={"path": path})],
    )


GREEN = "============ 67 passed in 0.16s ============"


def test_a_scoped_green_run_verifies_the_turn_but_is_marked_scoped() -> None:
    c = RecipeCursor(enabled=True)
    _wrote(c, "solver.py")
    _run(c, FIELD, GREEN)
    assert not c.needs_verification()  # ADR-0136's credit is untouched
    assert c.suite_scoped_only  # ...but the turn knows it never ran the rest


def test_an_unscoped_green_run_clears_it() -> None:
    c = RecipeCursor(enabled=True)
    _wrote(c, "solver.py")
    _run(c, "uv run --no-sync pytest -q", GREEN)
    assert not c.needs_verification()
    assert not c.suite_scoped_only


def test_a_scoped_run_after_an_unscoped_one_does_not_re_arm_the_rail() -> None:
    c = RecipeCursor(enabled=True)
    _wrote(c, "solver.py")
    _run(c, "uv run pytest -q", GREEN)
    _run(c, FIELD, GREEN)
    assert not c.suite_scoped_only  # the whole suite HAS been seen this turn


def test_a_fresh_write_invalidates_an_earlier_unscoped_run() -> None:
    # The reason the reset sites matter: code written AFTER the full suite ran was never
    # covered by it, so the turn is back to owing an unscoped run.
    c = RecipeCursor(enabled=True)
    _wrote(c, "solver.py")
    _run(c, "uv run pytest -q", GREEN)
    assert not c.suite_scoped_only
    _wrote(c, "other.py")
    _run(c, FIELD, GREEN)
    assert c.suite_scoped_only


def test_a_turn_verified_file_by_file_is_untouched() -> None:
    # ADR-0110's per-file fallback is a different, explicit path; no suite ran, so there is
    # no suite-shaped claim to qualify and the rail must stay silent.
    c = RecipeCursor(enabled=True)
    _wrote(c, "solver.py")
    _run(c, "python solver.py", "ok")
    assert not c.suite_scoped_only


# ── the loop ─────────────────────────────────────────────────────────────────


class _Bash(Tool):
    spec = ToolSpec(
        name="bash",
        description="fake bash",
        required_permission=PermissionTier.DANGER_FULL_ACCESS,
        concurrency=ConcurrencyClass.NEVER_PARALLEL,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        command = str(args.get("command", ""))
        if "RED" in command:
            # A failing pytest exits NON-ZERO, and ADR-0139 makes an unpiped run's verdict
            # its exit status -- so a fake that returned ok() here would be credited green
            # however red its text, and would test nothing.
            return ToolResult.error("=========== 2 failed, 65 passed in 0.2s ===========")
        return ToolResult.ok(GREEN)


class _Write(Tool):
    spec = ToolSpec(
        name="write_file",
        description="fake write",
        required_permission=PermissionTier.WORKSPACE_WRITE,
        concurrency=ConcurrencyClass.NEVER_PARALLEL,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = args.get("path", "solver.py")
        return ToolResult.ok("written", data={"path": path})


class _Sequence(Provider):
    def __init__(self, *results: LLMResult) -> None:
        self._results = list(results)
        self.calls = 0

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:
        self.calls += 1
        return self._results[min(self.calls, len(self._results)) - 1]

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=200_000)


def _text(text: str) -> LLMResult:
    return LLMResult(text=text, finish_reason="stop")


def _bash(command: str) -> LLMResult:
    return LLMResult(
        tool_calls=[ToolCall(id="c1", name="bash", arguments={"command": command})],
        finish_reason="tool_calls",
    )


def _write_call() -> LLMResult:
    return LLMResult(
        tool_calls=[ToolCall(id="c1", name="write_file", arguments={"path": "solver.py"})],
        finish_reason="tool_calls",
    )


def _loop(tmp_path: Path, provider: Provider) -> AgentLoop:
    registry = ToolRegistry()
    registry.register(_Bash())
    registry.register(_Write())
    return AgentLoop(
        provider,
        registry,
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        max_iterations=20,
    )


def _rails(loop: AgentLoop) -> list[str]:
    return [m.text for m in loop.session.messages if m.role == "user" and m.text]


#: No first-person work verb and no digits: the ADR-0033 and ADR-0044 guards sit upstream of
#: this one and would otherwise consume the completion, so a pass here would be hollow.
DONE = "The helpers are in place and the scoped tests are green."


def test_the_field_turn_is_asked_once_for_the_unscoped_run(tmp_path: Path) -> None:
    provider = _Sequence(_write_call(), _bash(FIELD), _text(DONE), _text(DONE))
    loop = _loop(tmp_path, provider)
    asyncio.run(loop.arun_turn("add three helpers to the text tools"))
    assert sum(_SCOPE_NUDGE in r for r in _rails(loop)) == 1


def test_a_turn_that_ran_the_whole_suite_is_never_asked(tmp_path: Path) -> None:
    provider = _Sequence(_write_call(), _bash("uv run --no-sync pytest -q"), _text(DONE))
    loop = _loop(tmp_path, provider)
    asyncio.run(loop.arun_turn("add three helpers to the text tools"))
    # The scripted length is what makes this non-hollow: any gate firing would re-prompt.
    assert provider.calls == 3
    assert not any(_SCOPE_NUDGE in r for r in _rails(loop))


def test_a_red_scoped_run_belongs_to_the_recipe_gate_not_this_one(tmp_path: Path) -> None:
    # A scoped run that FAILS credits nothing, so needs_verification() is still true and the
    # recipe gate owns the turn. Two rails for one problem would be the obvious way to get
    # this wrong.
    provider = _Sequence(
        _write_call(), _bash("uv run pytest tests/test_text_tools.py -k RED"), _text(DONE)
    )
    loop = _loop(tmp_path, provider)
    asyncio.run(loop.arun_turn("add three helpers to the text tools"))
    assert not any(_SCOPE_NUDGE in r for r in _rails(loop))


def test_the_unscoped_run_the_rail_asks_for_ends_the_turn(tmp_path: Path) -> None:
    # The rail's whole purpose: the model takes the advice, runs the suite unscoped, and the
    # turn finishes. If the rail re-fired here it would be a loop, not a nudge.
    provider = _Sequence(
        _write_call(),
        _bash(FIELD),
        _text(DONE),
        _bash("uv run --no-sync pytest -q"),
        _text(DONE),
    )
    loop = _loop(tmp_path, provider)
    asyncio.run(loop.arun_turn("add three helpers to the text tools"))
    assert sum(_SCOPE_NUDGE in r for r in _rails(loop)) == 1
