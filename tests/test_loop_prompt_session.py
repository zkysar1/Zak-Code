"""The loop names its own session in the system prompt (ADR-0072).

A framework that runs several sessions of one agent keys per-session state by session id
and hands that id to every hook as ``session_id``; the model had no way to read it. The
loop passes ``self.session.id`` to the prompt builder, so the environment section carries
the same id the hooks receive.
"""

from __future__ import annotations

from pathlib import Path

from zakcode import Agent
from zakcode.agent import DYNAMIC_BOUNDARY
from zakcode.config import Settings
from zakcode.evals.harness import ScriptedProvider, reply


def test_build_system_names_the_loop_session_id(tmp_path: Path) -> None:
    agent = Agent(
        provider=ScriptedProvider(script=[reply("ok")]),
        settings=Settings(
            default_model="scripted/test", context_window=8192, workspace_root=tmp_path
        ),
    )
    session_id = agent.loop.session.id
    assert session_id
    prompt = agent.loop._build_system()
    context = prompt[prompt.index(DYNAMIC_BOUNDARY) :]
    assert f"- Session id: {session_id}" in context


def test_build_system_keys_the_guide_fold_by_the_sessions_first_user_message(
    tmp_path: Path,
) -> None:
    # ADR-0173: the fold of a guide past the per-file cap keeps the sections the task is about
    # first. The loop keys it by the session's first user message — the same message on every
    # later turn, so the context tier does not move within a session.
    from zakcode.agent.prompt import MAX_CONTEXT_FILE_CHARS
    from zakcode.messages import Message

    padding = [f"## Section {i}\n\n" + f"Plain guidance paragraph {i}. " * 14 for i in range(20)]
    layout = "## Widget layout\n\nThe widget lives at `widgets/<name>/layout.yaml`."
    layout += "\n\n" + "A layout names every part of the widget and where it sits. " * 30
    guide = "\n\n".join(["# Guide", *padding, layout])
    assert guide.index(layout) > MAX_CONTEXT_FILE_CHARS
    (tmp_path / "AGENTS.md").write_text(guide + "\n", encoding="utf-8")
    agent = Agent(
        provider=ScriptedProvider(script=[reply("ok")]),
        settings=Settings(
            default_model="scripted/test", context_window=8192, workspace_root=tmp_path
        ),
    )
    assert "## Widget layout" not in agent.loop._build_system()  # no task yet: the ADR-0172 fold
    agent.loop.session.add_message(Message.user("Implement widget_layout_path(name) in app/w.py"))
    keyed = agent.loop._build_system()
    assert "## Widget layout" in keyed and "layout.yaml" in keyed
    agent.loop.session.add_message(Message.user("Now the gizmo report, nothing about widgets"))
    assert agent.loop._build_system() == keyed  # the first ask keys the fold for the session


def test_a_compaction_leaves_the_fold_where_the_first_ask_put_it(tmp_path: Path) -> None:
    # ADR-0233: a compaction summarizes the first ask away. Read from the history after that,
    # the task was whatever user message the compaction kept, and the system prompt moved
    # (measured on a Mind: about 51,600 characters in, at every compaction). The task is
    # pinned on the session the first time it is read, so the prompt holds.
    import asyncio

    from zakcode.agent.compact import is_summary
    from zakcode.agent.prompt import MAX_CONTEXT_FILE_CHARS
    from zakcode.messages import Message
    from zakcode.session.store import Session

    padding = [f"## Section {i}\n\n" + f"Plain guidance paragraph {i}. " * 14 for i in range(20)]
    layout = "## Widget layout\n\nThe widget lives at `widgets/<name>/layout.yaml`."
    layout += "\n\n" + "A layout names every part of the widget and where it sits. " * 30
    guide = "\n\n".join(["# Guide", *padding, layout])
    assert guide.index(layout) > MAX_CONTEXT_FILE_CHARS
    (tmp_path / "AGENTS.md").write_text(guide + "\n", encoding="utf-8")
    agent = Agent(
        provider=ScriptedProvider(script=[reply("Asked for widget_layout_path; not started.")]),
        settings=Settings(
            default_model="scripted/test", context_window=8192, workspace_root=tmp_path
        ),
        enable_compaction=True,
    )
    loop = agent.loop
    first_ask = "Implement widget_layout_path(name) in app/w.py"
    loop.session.add_message(Message.user(first_ask))
    keyed = loop._build_system()
    assert "## Widget layout" in keyed
    for i in range(4):
        loop.session.add_message(Message.assistant_text(f"Looked at part {i}."))
        loop.session.add_message(Message.user(f"Now the gizmo report, part {i}"))

    assert asyncio.run(loop.compact_now()) is True
    assert is_summary(loop.session.messages[0])
    assert all(m.text != first_ask for m in loop.session.messages)  # the ask was summarized
    assert loop._build_system() == keyed
    restored = Session.model_validate_json(loop.session.model_dump_json())
    assert restored.task == first_ask  # a resumed session keeps it too

    # Positive control: without the pin the kept tail's first user message keys the fold.
    loop.session.task = ""
    assert loop._build_system() != keyed
