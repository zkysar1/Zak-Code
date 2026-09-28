"""A session keeps the workspace inputs its system prompt was built from (ADR-0260).

A restart into a new build, a resume and a served turn each build the prompt in a new process
(or a new Agent). Reading the rules, guides, skills and survey from disk again there moved the
prompt whenever any of them had changed, and an engine that caches by exact prefix then
re-processed the whole conversation. The session pins the text instead, and the first turn of
the new process is told what changed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from zakcode import Agent
from zakcode.agent.pinned_inputs import MAX_ITEM_CHARS, change_notice
from zakcode.evals.harness import ScriptedProvider, reply
from zakcode.messages import Message
from zakcode.session.store import Session

_ASK = "Tidy the widget module"
_NOTICE = "Files that feed your system prompt changed"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _workspace(root: Path) -> None:
    _write(
        root / ".zakcode" / "rules" / "style.md", "Indent with TABS_BEFORE.\nKeep lines short.\n"
    )
    _write(root / "AGENTS.md", "# Guide\n\nRun GUIDE_BEFORE before committing.\n")
    _write(
        root / ".zakcode" / "skills" / "helper" / "SKILL.md",
        "---\nname: helper\ndescription: DESC_BEFORE\n---\nbody",
    )
    _write(root / "src" / "widget.py", "x = 1\n")


def _change(root: Path) -> None:
    """Edit what the prompt is read from, as a merge in the workspace between two processes."""
    _write(
        root / ".zakcode" / "rules" / "style.md", "Indent with SPACES_AFTER.\nKeep lines short.\n"
    )
    _write(root / "AGENTS.md", "# Guide\n\nRun GUIDE_AFTER before committing.\n")
    _write(
        root / ".zakcode" / "skills" / "helper" / "SKILL.md",
        "---\nname: helper\ndescription: DESC_AFTER\n---\nbody",
    )
    _write(root / "docs" / "notes.md", "a new file, which only the survey lists\n")


def _agent(
    root: Path, session: Session | None = None, *, replies: int = 1, compaction: bool = False
) -> Agent:
    return Agent(
        provider=ScriptedProvider(script=[reply("ok") for _ in range(replies)]),
        session=session,
        enable_rules=True,
        enable_skills=True,
        enable_compaction=compaction,
        default_model="scripted/test",
        context_window=32768,
        workspace_root=str(root),
    )


def _saved(agent: Agent) -> Session:
    """The session as the next process loads it from the store."""
    return Session.model_validate_json(agent.session.model_dump_json())


def _first_process(root: Path) -> Agent:
    _workspace(root)
    agent = _agent(root)
    asyncio.run(agent.loop.arun_turn(_ASK))
    return agent


def _notices(agent: Agent) -> list[str]:
    return [m.text for m in agent.session.messages if m.role == "user" and _NOTICE in m.text]


def test_a_restart_sends_the_same_system_prompt_after_the_workspace_changed(
    tmp_path: Path,
) -> None:
    first = _first_process(tmp_path)
    sent_first = first.provider.systems[0]
    assert sent_first is not None
    for marker in ("TABS_BEFORE", "GUIDE_BEFORE", "DESC_BEFORE", "src/widget.py"):
        assert marker in sent_first
    _change(tmp_path)

    again = _agent(tmp_path, _saved(first))
    asyncio.run(again.loop.arun_turn("[harness] restarted into a new build; continue"))

    assert again.provider.systems == [sent_first]
    assert again.loop._build_system() == again.loop._build_system()  # and it holds still


def test_a_new_session_reads_the_changed_workspace(tmp_path: Path) -> None:
    # The positive control for the test above: every change is on disk and discoverable, so a
    # new session sends all of it, the survey's new file included. Without this, the equality
    # above would also hold for changes that never reached the workspace.
    first = _first_process(tmp_path)
    _change(tmp_path)

    fresh = _agent(tmp_path)
    asyncio.run(fresh.loop.arun_turn(_ASK))

    sent = fresh.provider.systems[0]
    assert sent is not None and sent != first.provider.systems[0]
    for marker in ("SPACES_AFTER", "GUIDE_AFTER", "DESC_AFTER", "docs/notes.md"):
        assert marker in sent
    assert _notices(fresh) == []  # a new session has nothing to be told


def test_the_first_turn_after_a_restart_is_told_what_changed_once(tmp_path: Path) -> None:
    first = _first_process(tmp_path)
    _change(tmp_path)
    again = _agent(tmp_path, _saved(first), replies=2)
    kick = "[harness] restarted into a new build; continue"
    asyncio.run(again.loop.arun_turn(kick))

    [notice] = _notices(again)
    texts = [m.text for m in again.session.messages]
    assert texts.index(notice) == texts.index(kick) + 1  # read right after the turn's message
    assert "- rule style changed:" in notice
    assert "-Indent with TABS_BEFORE." in notice and "+Indent with SPACES_AFTER." in notice
    assert "- guide AGENTS.md changed:" in notice and "+Run GUIDE_AFTER" in notice
    assert "- skill helper changed:" in notice and "+DESC_AFTER" in notice
    assert "notes.md" not in notice  # the survey is pinned without a word

    # Said once: not again on this process's next turn, nor by the process after it.
    asyncio.run(again.loop.arun_turn("next"))
    third = _agent(tmp_path, _saved(again))
    asyncio.run(third.loop.arun_turn(kick))
    assert len(_notices(third)) == 1
    assert third.provider.systems == first.provider.systems  # and the prompt still holds


def test_a_compaction_brings_the_prompt_up_to_the_workspace(tmp_path: Path) -> None:
    first = _first_process(tmp_path)
    _change(tmp_path)
    again = _agent(tmp_path, _saved(first), compaction=True)
    assert "TABS_BEFORE" in again.loop._build_system()
    for i in range(4):
        again.session.add_message(Message.assistant_text(f"Looked at part {i}."))
        again.session.add_message(Message.user(f"Now part {i}"))

    assert asyncio.run(again.loop.compact_now()) is True

    caught_up = again.loop._build_system()
    for marker in ("SPACES_AFTER", "GUIDE_AFTER", "DESC_AFTER", "docs/notes.md"):
        assert marker in caught_up


def test_a_session_saved_by_an_older_build_is_pinned_without_a_notice(tmp_path: Path) -> None:
    # An older build saved no pins and no record: this process reads the workspace as it finds
    # it, pins that, and has nothing to compare against, so it says nothing.
    first = _first_process(tmp_path)
    older = _saved(first)
    older.prompt_inputs.clear()
    older.prompt_input_items.clear()
    _change(tmp_path)

    again = _agent(tmp_path, older)
    asyncio.run(again.loop.arun_turn("continue"))

    assert _notices(again) == []
    sent = again.provider.systems[0]
    assert sent is not None and "SPACES_AFTER" in sent
    assert again.session.prompt_inputs["rules"] and again.session.prompt_input_items["rules"]


def test_the_notice_names_what_it_cannot_quote() -> None:
    session = Session(cwd="/w", model="m")
    session.prompt_inputs["rules"] = "the pinned rules text"
    session.prompt_input_items["rules"] = {"rule big": "one\n", "rule gone": "two\n"}
    now = {"rule big": "grown\n" * MAX_ITEM_CHARS, "rule fresh": "Say so.\n"}

    notice = change_notice(session, {"rules": now})

    assert notice is not None
    assert "- rule big changed (not quoted here: too long)." in notice
    assert "- rule fresh is new:\n    Say so." in notice
    assert "- rule gone was removed." in notice
    assert "read_rule returns a rule's current text" in notice
    assert session.prompt_input_items["rules"] == now
    assert change_notice(session, {"rules": now}) is None
