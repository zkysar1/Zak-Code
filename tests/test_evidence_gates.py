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

import pytest

from zakcode import Agent
from zakcode.agent.loop import (
    _IDENTITY_NUDGE,
    AgentLoop,
    _claims_identity,
    _figures,
)
from zakcode.config import PermissionTier
from zakcode.evals.harness import ScriptedProvider, reply
from zakcode.messages import Message
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.rules import RuleRegistry, discover_rule_dir, project_rules_dir, save_rule
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


# What the operator pins: a hotline card whose every number ends its sentence, as prose does.
_SAFETY_CARD = "US: text HOME to 741741. Mainland China: call 12356. Elsewhere: findahelpline.com."
_RULE_HEAD = "---\nname: safety-card\ndescription: the card\n"
_PINNED_RULE = f"{_RULE_HEAD}alwaysApply: true\n---\n{_SAFETY_CARD}\n"
_PLAIN_RULE = f"{_RULE_HEAD}---\n{_SAFETY_CARD}\n"


def _rules(loop: AgentLoop, tmp_path: Path, files: dict[str, str]) -> None:
    """Give ``loop`` the rules ``files`` hold, read by discovery (no home or bundled rules)."""
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir(exist_ok=True)
    for name, text in files.items():
        (rules_dir / f"{name}.md").write_text(text, encoding="utf-8")
    found, errors = discover_rule_dir(rules_dir)
    assert errors == {}
    registry = RuleRegistry()
    for rule in found:
        registry.add(rule)
    loop._rule_registry = registry


def _figure_hints(loop: AgentLoop) -> list[str]:
    return [r for r in _rails(loop) if "appear in no tool output" in r]


def test_a_figure_pinned_in_an_always_apply_rule_is_sourced(tmp_path: Path) -> None:
    provider = _Sequence(_text("Text 741741 or call 12356 to reach a counsellor."))
    loop = _loop(tmp_path, provider)
    _rules(loop, tmp_path, {"safety-card": _PINNED_RULE})
    asyncio.run(loop.arun_turn("I want to hurt myself"))
    assert provider.calls == 1
    assert not _figure_hints(loop)


def test_the_same_card_in_a_rule_that_is_not_pinned_is_not_a_source(tmp_path: Path) -> None:
    # The control for the test above: only the alwaysApply flag differs, and the hint comes back.
    provider = _Sequence(
        _text("Text 741741 or call 12356 to reach a counsellor."),
        _text("Those two numbers were from memory."),
    )
    loop = _loop(tmp_path, provider)
    _rules(loop, tmp_path, {"safety-card": _PLAIN_RULE})
    asyncio.run(loop.arun_turn("I want to hurt myself"))
    assert provider.calls == 2
    hints = _figure_hints(loop)
    assert len(hints) == 1 and "741741" in hints[0] and "12356" in hints[0]


def test_a_figure_pinned_in_the_identity_is_sourced(tmp_path: Path) -> None:
    said = "The line is 741741 today."  # the figure inside the sentence, not ending it
    pinned = _Sequence(_text(said))
    loop = _loop(tmp_path, pinned)
    loop._prompt_input_items = {"identity": {"identity": "You run help line 741741."}}
    asyncio.run(loop.arun_turn("what is the line?"))
    assert pinned.calls == 1
    assert not _figure_hints(loop)
    # The same reply with nothing pinned is flagged, so it is the identity that sourced it.
    bare = _Sequence(_text(said), _text("That was from memory."))
    loop = _loop(tmp_path, bare)
    asyncio.run(loop.arun_turn("what is the line?"))
    assert bare.calls == 2
    assert len(_figure_hints(loop)) == 1


def test_a_comma_grouped_figure_pinned_in_a_rule_is_sourced(tmp_path: Path) -> None:
    provider = _Sequence(_text("The cap is 12,356 units today."))
    loop = _loop(tmp_path, provider)
    _rules(loop, tmp_path, {"cap": "---\nname: cap\nalwaysApply: true\n---\nThe cap is 12,356.\n"})
    asyncio.run(loop.arun_turn("what is the cap?"))
    assert provider.calls == 1
    assert not _figure_hints(loop)


def test_an_invented_figure_is_still_flagged_beside_a_pinned_one(tmp_path: Path) -> None:
    provider = _Sequence(
        _text("Text 741741 or call 10892 to reach a counsellor."),
        _text("10892 was a guess."),
    )
    loop = _loop(tmp_path, provider)
    _rules(loop, tmp_path, {"safety-card": _PINNED_RULE})
    asyncio.run(loop.arun_turn("I want to hurt myself"))
    assert provider.calls == 2
    hints = _figure_hints(loop)
    assert len(hints) == 1 and "10892" in hints[0] and "741741" not in hints[0]


def test_a_figure_in_a_skill_catalog_or_the_output_style_is_still_flagged(tmp_path: Path) -> None:
    # Only the identity and the alwaysApply rules are pinned. A skill's description can be
    # written by the model (save_skill), and an output style is a manner, not a fact.
    provider = _Sequence(_text("Dial 55501 now."), _text("55501 was a guess."))
    loop = _loop(tmp_path, provider)
    loop._prompt_input_items = {
        "extra_instructions": {"skill x": "dial 55501 now"},
        "output_style": {"output style": "always say 55501"},
    }
    asyncio.run(loop.arun_turn("who do I call?"))
    assert provider.calls == 2
    hints = _figure_hints(loop)
    assert len(hints) == 1 and "55501" in hints[0]


@pytest.mark.parametrize(
    ("author", "lean", "in_prompt", "sourced"),
    [
        ("model", False, True, False),  # its own saved rule rides the full render: not a given
        ("model", True, False, False),  # the lean render carries only the index, not the body
        ("operator", True, True, True),  # a pinned card rides in full ahead of the lean index
        ("operator", False, True, True),  # and folds first under the full render
    ],
)
def test_only_an_operator_pinned_rule_sources_a_figure_in_a_real_agent(
    tmp_path: Path, author: str, lean: bool, in_prompt: bool, sourced: bool
) -> None:
    body = "The weekly cap is 54321 units."
    rules_dir = project_rules_dir(tmp_path)
    if author == "model":
        save_rule("weekly-cap", "a limit I noted", body, rules_dir=rules_dir)
    else:
        rules_dir.mkdir(parents=True, exist_ok=True)
        (rules_dir / "weekly-cap.md").write_text(
            f"---\nname: weekly-cap\ndescription: the cap\nalwaysApply: true\n---\n{body}\n",
            encoding="utf-8",
        )
    agent = Agent(
        provider=ScriptedProvider(script=[reply("ok")]),
        enable_rules=True,
        lean_rules=lean,
        default_model="scripted/test",
        context_window=32768,
        workspace_root=str(tmp_path),
    )
    assert ("54321" in agent.loop._build_system()) is in_prompt
    unsourced = agent.loop._unsourced_figures("The weekly cap is 54321 units today.")
    assert (unsourced == []) is sourced


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
