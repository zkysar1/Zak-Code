"""The harness ships NO cross-session memory — that is claude-mind's job.

See ``docs/PERSISTENCE-BOUNDARY.md``: the substrate records the transcript (``SessionStore``,
powering ``/resume``) and exposes generic seams; it does not remember, recall, or learn. These
tests pin the removal: a bare Agent registers no ``remember``/``recall`` tools and holds no memory
store, while the GENERIC seams a Mind attaches its own recall to — the context-hook and
lifecycle-hook registration points — remain present and usable.
"""

from __future__ import annotations

from pathlib import Path

from zakcode import Agent
from zakcode.config import Settings
from zakcode.hooks import HookEvent


def _agent(tmp_path: Path) -> Agent:
    return Agent(
        settings=Settings(
            default_model="scripted/test", context_window=8192, workspace_root=tmp_path
        )
    )


def test_default_agent_ships_no_memory(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    assert agent.registry.get("remember") is None  # no model-facing memory tools
    assert agent.registry.get("recall") is None
    assert not hasattr(agent, "memory")  # no memory store/attribute on the harness


def test_no_enable_memory_flag(tmp_path: Path) -> None:
    # The opt-in flag is gone; the config carries no memory_* knobs.
    assert not any("memory" in name for name in Settings.model_fields)


def test_generic_recall_seams_survive(tmp_path: Path) -> None:
    # claude-mind brings its own recall via these generic, non-memory-specific seams — they must
    # still be present and accept a Mind's hooks after the memory subsystem is gone.
    agent = _agent(tmp_path)
    agent.hook_manager.register_context(lambda payload: "background a mind supplies")
    agent.hook_manager.register_lifecycle(HookEvent.SESSION_START, lambda payload: None)
    assert agent.hook_manager.has_context_hooks()  # the recall seam (PRE_LLM_CALL injection)
    assert agent.hook_manager.has_lifecycle_hooks(HookEvent.SESSION_START)  # the encode/prime seam
