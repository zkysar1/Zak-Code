"""The framework agents a new chat offers to start, read from the workspace (ADR-0351).

An agent is a directory under ``agents/`` holding a ``self.md``, offered only where the workspace
holds the framework's start skill, since that is what the offer runs. Hermetic: tmp dirs only.
"""

from __future__ import annotations

from pathlib import Path

from zakcode.session.framework_agents import START_SKILL, start_line, startable_agents


def mind(root: Path, *agents: str) -> Path:
    """A workspace holding the framework's start skill and, per name, an agent with a self.md."""
    (root / START_SKILL).parent.mkdir(parents=True)
    (root / START_SKILL).write_text("---\nname: start\n---\n", encoding="utf-8")
    for name in agents:
        (root / "agents" / name).mkdir(parents=True)
        (root / "agents" / name / "self.md").write_text(f"# {name}\n", encoding="utf-8")
    return root


def test_each_agent_with_a_self_md_is_offered_by_name(tmp_path: Path) -> None:
    root = mind(tmp_path, "scout", "keeper")
    (root / "agents" / "half-made" / "session").mkdir(parents=True)  # no self.md: not an agent
    (root / "agents" / "README.md").write_text("the agents live here\n", encoding="utf-8")
    assert startable_agents(root) == ["keeper", "scout"]


def test_without_the_start_skill_nothing_is_offered(tmp_path: Path) -> None:
    root = mind(tmp_path, "scout")
    (root / START_SKILL).unlink()
    assert startable_agents(root) == []


def test_a_framework_with_no_agent_yet_offers_none(tmp_path: Path) -> None:
    assert startable_agents(mind(tmp_path)) == []
    assert startable_agents(tmp_path / "no-such-workspace") == []


def test_a_name_that_cannot_stand_in_a_start_line_is_not_offered(tmp_path: Path) -> None:
    root = mind(tmp_path, "scout", "two words", "-flag", ".hidden")
    assert startable_agents(root) == ["scout"]


def test_the_start_line_starts_the_agent_in_assistant_mode() -> None:
    assert start_line("scout") == "/start scout --mode assistant"
