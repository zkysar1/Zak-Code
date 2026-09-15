"""Guide-check rail (ADR-0177): a turn that changed files with a guide in context is asked once,
before it finishes, to check the changed files against the guide's rules.

Thrust 27 (2026-09-15): with the delta-assert rule folded INTO the prompt in either form, the
27B complied about 60% of the time, and its misses narrated "delta" and then asserted the
absolute — the rule in hand, the default at the keyboard. The rail is structural (edits ran, a
guide was folded), never keyed on the completion's text, fires once per turn, and only in an
unattended session (no one at the prompt to catch a rule miss).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from zakcode.agent.loop import _GUIDE_CHECK_NUDGE, AgentLoop
from zakcode.config import PermissionTier, load_settings
from zakcode.messages import Message
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

GUIDE = (
    "# metrics\n\nA small metrics library.\n\n## Constraints\n\n"
    "- Assert on before/after deltas for shared counters, never absolutes.\n"
)


class _Writer(Tool):
    spec = ToolSpec(
        name="write_file",
        description="write",
        required_permission=PermissionTier.WORKSPACE_WRITE,
        concurrency=ConcurrencyClass.NEVER_PARALLEL,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok("wrote tests/test_metrics.py")


class _Sequence(Provider):
    """Plays back scripted completions in order; repeats the last one forever."""

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


def _write() -> LLMResult:
    return LLMResult(
        tool_calls=[ToolCall(id="c1", name="write_file", arguments={})],
        finish_reason="tool_calls",
    )


def _loop(tmp_path: Path, provider: Provider, *, attended: bool = False) -> AgentLoop:
    registry = ToolRegistry()
    registry.register(_Writer())
    return AgentLoop(
        provider,
        registry,
        Session(cwd=str(tmp_path), model="test"),
        settings=load_settings(workspace_root=tmp_path),
        workspace_root=tmp_path,
        permission_policy=None if attended else PermissionPolicy(PermissionMode.BYPASS),
        max_iterations=20,
    )


def _checks(loop: AgentLoop) -> int:
    return sum(
        _GUIDE_CHECK_NUDGE in m.text for m in loop.session.messages if m.role == "user" and m.text
    )


def test_a_turn_that_changed_files_with_a_guide_in_context_is_asked_for_the_check_once(
    tmp_path: Path,
) -> None:
    (tmp_path / "CLAUDE.md").write_text(GUIDE, encoding="utf-8")
    provider = _Sequence(
        _write(),
        _text("Added test_record_hit_increments."),
        _text("Checked: the test asserts a before/after delta; every rule is met."),
    )
    loop = _loop(tmp_path, provider)
    asyncio.run(loop.arun_turn("add a test for record_hit"))
    assert provider.calls == 3  # the write, the finish that is asked to check, the check
    assert _checks(loop) == 1
    assert loop.prompt_builder.last_guides == ("CLAUDE.md",)


def test_the_check_is_asked_once_even_when_the_model_then_edits_again(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text(GUIDE, encoding="utf-8")
    provider = _Sequence(
        _write(),
        _text("Added the test."),
        _write(),  # the check found an absolute and fixed it
        _text("Fixed: the test now asserts the delta."),
        _text("Nothing further."),
    )
    loop = _loop(tmp_path, provider)
    asyncio.run(loop.arun_turn("add a test for record_hit"))
    assert provider.calls == 4  # the second finish is not asked again
    assert _checks(loop) == 1


def test_no_guide_in_context_means_no_check(tmp_path: Path) -> None:
    provider = _Sequence(_write(), _text("Added the test."))
    loop = _loop(tmp_path, provider)
    asyncio.run(loop.arun_turn("add a test for record_hit"))
    assert provider.calls == 2
    assert _checks(loop) == 0
    assert loop.prompt_builder.last_guides == ()


def test_a_turn_without_file_changes_is_not_checked(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text(GUIDE, encoding="utf-8")
    provider = _Sequence(_text("The counters live in metrics/counters.py."))
    loop = _loop(tmp_path, provider)
    asyncio.run(loop.arun_turn("where do the counters live?"))
    assert provider.calls == 1
    assert _checks(loop) == 0


def test_a_readme_alone_is_not_a_guide(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# metrics\n\nA small metrics library.\n", encoding="utf-8")
    provider = _Sequence(_write(), _text("Added the test."))
    loop = _loop(tmp_path, provider)
    asyncio.run(loop.arun_turn("add a test for record_hit"))
    assert provider.calls == 2
    assert _checks(loop) == 0
    assert "README.md" not in loop.prompt_builder.last_guides


def test_an_attended_session_is_not_checked(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text(GUIDE, encoding="utf-8")
    provider = _Sequence(_write(), _text("Added the test."))
    loop = _loop(tmp_path, provider, attended=True)
    asyncio.run(loop.arun_turn("add a test for record_hit"))
    assert provider.calls == 2  # someone is at the prompt; no round trip
    assert _checks(loop) == 0
