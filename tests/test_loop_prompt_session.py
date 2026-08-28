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
