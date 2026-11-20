"""A rule pinned in the user rules root is not displaced by a workspace rule of the same name.

The loader keys rules by name and lets a later root win (bundled, user, then ``.zakcode/rules``,
then ``.claude/rules``), so a workspace file named like a pinned card used to take its place. The
model's own file tools can write into both workspace roots (``save_rule``, ``write_file``), so
the displacement needed no operator. These tests pin the rule that a user-root rule carrying
``alwaysApply: true`` is kept, whichever file in the workspace claims its name.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zakcode import Agent
from zakcode.agent import DYNAMIC_BOUNDARY
from zakcode.evals.harness import ScriptedProvider, reply
from zakcode.rules import discover_rules, save_rule

CARD = "pinned-card"
CARD_TEXT = "CARD_BODY_MARKER say you are an AI; help lines 988 and 12356"
SHADOW_TEXT = "SHADOW_BODY_MARKER nothing to see here"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def user_rules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake home whose ``~/.config/zakcode/rules`` is the user rules root the loader scans."""
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home / ".config" / "zakcode" / "rules"


def _card(user_rules: Path, *, pinned: bool = True) -> None:
    flag = "true" if pinned else "false"
    _write(
        user_rules / f"{CARD}.md",
        f"---\nname: {CARD}\ndescription: the card\nalwaysApply: {flag}\n---\n{CARD_TEXT}\n",
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


@pytest.mark.parametrize("root", [".zakcode/rules", ".claude/rules"])
def test_a_same_named_workspace_rule_does_not_displace_the_pinned_card(
    user_rules: Path, workspace: Path, root: str
) -> None:
    _card(user_rules)
    _write(workspace / root / f"{CARD}.md", SHADOW_TEXT)
    registry, errors = discover_rules(workspace)
    held = registry.get(CARD)
    assert held is not None and held.content == CARD_TEXT and held.always_apply is True
    assert SHADOW_TEXT not in registry.render()
    # the refusal is recorded by a path-qualified key, so it cannot overwrite a parse error
    # recorded under the bare filename
    assert list(errors) == [f"{root}/{CARD}.md"]
    assert "pinned" in errors[f"{root}/{CARD}.md"] and CARD in errors[f"{root}/{CARD}.md"]


def test_a_workspace_file_with_another_filename_but_the_card_name_is_refused_too(
    user_rules: Path, workspace: Path
) -> None:
    # the loader names a rule by its frontmatter ``name`` before its file stem, so a move-aside
    # keyed on the filename never saw this one
    _card(user_rules)
    _write(
        workspace / ".zakcode" / "rules" / "innocent-looking.md",
        f"---\nname: {CARD}\n---\n{SHADOW_TEXT}\n",
    )
    registry, errors = discover_rules(workspace)
    assert registry.get(CARD).content == CARD_TEXT
    assert list(errors) == [".zakcode/rules/innocent-looking.md"]


def test_a_user_rule_that_is_not_pinned_is_still_overridden_by_the_project(
    user_rules: Path, workspace: Path
) -> None:
    # the control: only a PINNED user rule is protected, so the documented precedence is unchanged
    _card(user_rules, pinned=False)
    _write(workspace / ".zakcode" / "rules" / f"{CARD}.md", SHADOW_TEXT)
    registry, errors = discover_rules(workspace)
    assert registry.get(CARD).content == SHADOW_TEXT
    assert errors == {}


def test_only_the_user_root_is_protected_a_pinned_project_rule_is_still_overridden(
    user_rules: Path, workspace: Path
) -> None:
    # the pin protects what an operator put OUTSIDE the workspace, not every pinned file: a later
    # project root still overrides an earlier one, as documented
    _write(workspace / ".zakcode" / "rules" / "x.md", "---\nalwaysApply: true\n---\nFIRST")
    _write(workspace / ".claude" / "rules" / "x.md", "SECOND")
    registry, errors = discover_rules(workspace)
    assert registry.get("x").content == "SECOND"
    assert errors == {}


def test_two_user_root_files_of_one_name_keep_the_documented_order(
    user_rules: Path, workspace: Path
) -> None:
    # the refusal is for PROJECT roots only: inside the user root the later file still wins
    _write(user_rules / "a-first.md", "---\nname: dup\nalwaysApply: true\n---\nPINNED")
    _write(user_rules / "b-second.md", "---\nname: dup\n---\nPLAIN")
    registry, errors = discover_rules(workspace)
    assert registry.get("dup").content == "PLAIN"
    assert errors == {}


def test_a_workspace_rule_with_a_new_name_loads_beside_the_pinned_card(
    user_rules: Path, workspace: Path
) -> None:
    _card(user_rules)
    _write(workspace / ".zakcode" / "rules" / "team-style.md", "TEAM_STYLE_MARKER")
    registry, errors = discover_rules(workspace)
    assert errors == {}
    assert registry.get(CARD).content == CARD_TEXT
    assert registry.get("team-style").content == "TEAM_STYLE_MARKER"


def _next_agent(workspace: Path, **kw: bool) -> Agent:
    return Agent(
        provider=ScriptedProvider(script=[reply("ok")]),
        enable_rules=True,
        default_model="scripted/test",
        context_window=8192,
        workspace_root=str(workspace),
        **kw,
    )


@pytest.mark.parametrize("lean", [False, True])
@pytest.mark.parametrize("how", ["save_rule", "write_file"])
def test_the_next_agent_still_carries_the_card_after_the_model_names_a_rule_like_it(
    user_rules: Path, workspace: Path, how: str, lean: bool
) -> None:
    """The verification outcome: a same-named rule written the way the model writes one leaves the
    card's text and its figures in the next Agent's stable prompt tier."""
    _card(user_rules)
    if how == "save_rule":
        save_rule(CARD, "looks harmless", SHADOW_TEXT, rules_dir=workspace / ".zakcode" / "rules")
    else:
        _write(workspace / ".claude" / "rules" / f"{CARD}.md", SHADOW_TEXT)
    agent = _next_agent(workspace, lean_rules=lean)
    prompt = agent.loop.prompt_builder.build(agent.settings)
    stable = prompt[: prompt.index(DYNAMIC_BOUNDARY)]
    assert CARD_TEXT in stable and "988" in stable and "12356" in stable
    assert SHADOW_TEXT not in stable
    assert agent.rule_errors  # the refusal is on the Agent, where an operator can read it
