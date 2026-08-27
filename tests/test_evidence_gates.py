"""Evidence gates: an identity claim and a figure both need a tool behind them (ADR-0044).

Two field answers, 2026-08-27, each given in one iteration with no tool call:

* "google-drive-list is a python file, not a skill" — it was a skill DIRECTORY the model had
  loaded through use_skill and never listed; it answered from the body it had read.
* "The knowledge tree has 10,892 nodes … directly reported by the tree stats command" — no
  tool ran that turn, the number appears in no tool output of the session, and the real
  count was 1,510.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from zakcode import Agent
from zakcode.agent.loop import (
    _IDENTITY_NUDGE,
    AgentLoop,
    _claims_identity,
    _figures,
)
from zakcode.config import PermissionTier
from zakcode.messages import Message
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
from zakcode.tools.builtins.use_skill import UseSkillTool, skill_directory_line

# ── predicates ───────────────────────────────────────────────────────────────


def test_claims_identity_matches_what_a_path_or_skill_is_said_to_be() -> None:
    assert _claims_identity("google-drive-list is a python file, not a skill.")
    assert _claims_identity("google-drive-list is not a skill; it is a script.")
    assert _claims_identity("google-drive-list isn't actually a skill, it is a script.")
    assert _claims_identity("tools/gdrive.py was a shell script originally.")
    assert not _claims_identity("It is a python file.")  # no named subject
    assert not _claims_identity("read_file is a tool.")  # not a workspace identity noun
    assert not _claims_identity("google-drive-list/ holds list.py and SKILL.md.")


def test_figures_are_measurements_not_years_versions_or_dates() -> None:
    text = "10,892 nodes in 2026; 1,510 files; port 8080; v2.9.4; 2026-08-27; 999 items"
    assert _figures(text) == {"10892", "1510", "8080"}
    assert _figures("nothing numeric here") == set()


# ── the loop ─────────────────────────────────────────────────────────────────


class _Lister(Tool):
    spec = ToolSpec(
        name="list_dir",
        description="list",
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok("SKILL.md\nlist.py")


class _Stats(Tool):
    spec = ToolSpec(
        name="tree_stats",
        description="stats",
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok("nodes: 10,892\nleaves: 7,301")


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


def _call(name: str) -> LLMResult:
    return LLMResult(
        tool_calls=[ToolCall(id="c1", name=name, arguments={})], finish_reason="tool_calls"
    )


def _loop(tmp_path: Path, provider: Provider) -> AgentLoop:
    registry = ToolRegistry()
    registry.register(_Lister())
    registry.register(_Stats())
    return AgentLoop(
        provider,
        registry,
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        max_iterations=20,
    )


def _rails(loop: AgentLoop) -> list[str]:
    return [m.text for m in loop.session.messages if m.role == "user" and m.text]


def test_an_identity_claim_without_a_look_is_asked_for_evidence_once(tmp_path: Path) -> None:
    provider = _Sequence(
        _text("google-drive-list is a python file, not a skill."),
        _text("As I said, it is a python file."),
    )
    loop = _loop(tmp_path, provider)
    asyncio.run(loop.arun_turn("is google-drive-list a skill?"))
    assert provider.calls == 2  # nudged once; the second answer ends the turn
    assert sum(_IDENTITY_NUDGE in r for r in _rails(loop)) == 1


def test_a_look_this_turn_earns_the_identity_claim(tmp_path: Path) -> None:
    provider = _Sequence(
        _call("list_dir"),
        _text("google-drive-list is a skill directory: SKILL.md plus list.py."),
    )
    loop = _loop(tmp_path, provider)
    asyncio.run(loop.arun_turn("is google-drive-list a skill?"))
    assert provider.calls == 2
    assert not any(_IDENTITY_NUDGE in r for r in _rails(loop))


def test_an_unsourced_figure_is_asked_for_the_measurement_once(tmp_path: Path) -> None:
    provider = _Sequence(
        _text("The knowledge tree has 10,892 nodes, directly reported by tree stats."),
        _text("I had not measured it; that figure was a guess."),
    )
    loop = _loop(tmp_path, provider)
    asyncio.run(loop.arun_turn("how big is the knowledge tree?"))
    assert provider.calls == 2
    figure_rails = [r for r in _rails(loop) if "appear in no tool output" in r]
    assert len(figure_rails) == 1 and "10892" in figure_rails[0]


def test_a_figure_carried_by_tool_output_is_sourced(tmp_path: Path) -> None:
    provider = _Sequence(
        _call("tree_stats"),
        _text("The tree has 10,892 nodes and 7,301 leaves (tree_stats, just now)."),
    )
    loop = _loop(tmp_path, provider)
    asyncio.run(loop.arun_turn("how big is the knowledge tree?"))
    assert provider.calls == 2
    assert not any("appear in no tool output" in r for r in _rails(loop))


def test_a_figure_the_user_gave_is_sourced(tmp_path: Path) -> None:
    provider = _Sequence(_text("1,510 nodes is a modest tree; nothing to trim yet."))
    loop = _loop(tmp_path, provider)
    asyncio.run(loop.arun_turn("we have 1,510 nodes — is that a lot?"))
    assert provider.calls == 1
    assert not any("appear in no tool output" in r for r in _rails(loop))


# ── use_skill names the skill directory ──────────────────────────────────────


def test_use_skill_result_names_the_directory_and_its_files(tmp_path: Path) -> None:
    skill = tmp_path / ".zakcode" / "skills" / "google-drive-list"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: google-drive-list\ndescription: list drive files\n---\nRun python3 list.py.\n",
        encoding="utf-8",
    )
    (skill / "list.py").write_text("print('x')\n", encoding="utf-8")
    (skill / "data").mkdir()
    agent = Agent(default_model="zakpick", workspace_root=tmp_path, enable_skills=True)
    ctx = ToolContext(workspace_root=tmp_path, skill_resolver=agent.loop._skill_resolver)
    result = asyncio.run(UseSkillTool().execute({"name": "google-drive-list"}, ctx))
    assert not result.is_error
    assert result.output.startswith("Run python3 list.py.")
    assert "[skill directory] .zakcode/skills/google-drive-list: data/, list.py." in result.output


def test_skill_directory_line_edge_cases(tmp_path: Path) -> None:
    assert skill_directory_line(None, tmp_path) == ""
    lonely = tmp_path / "skills" / "solo"
    lonely.mkdir(parents=True)
    (lonely / "SKILL.md").write_text("x\n", encoding="utf-8")
    line = skill_directory_line(str(lonely / "SKILL.md"), tmp_path)
    assert line.startswith("[skill directory] skills/solo: (only SKILL.md).")
    assert skill_directory_line(str(tmp_path / "gone" / "SKILL.md"), tmp_path) == ""
