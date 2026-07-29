"""Tests for per-role model routing (the "three specialized models" pattern).

A mind can route cheap roles (planner sub-agent, compaction summarizer) to a cheaper/local
model while the generator (main loop) keeps default_model. Hermetic: scripted providers, no
network. ``_provider_for`` is the seam; the provider built from settings is exercised offline
(litellm construction is lazy — no key/network needed to build the wrapper).
"""

from __future__ import annotations

from pathlib import Path

from zakcode import Agent
from zakcode.agent.budget import IterationBudget
from zakcode.agent.loop import AgentLoop
from zakcode.agent.subagent import SubAgentDefinition, SubAgentRunner
from zakcode.config import Settings
from zakcode.messages import Message
from zakcode.providers.base import Capabilities, LLMResult, Provider
from zakcode.session.store import Session
from zakcode.tools.base import ToolRegistry
from zakcode.usage import Usage


class _Scripted(Provider):
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    async def acomplete(self, messages, *, system=None, tools=None, response_format=None, **kw):
        self.calls += 1
        return LLMResult(text=self.text, usage=Usage(total_tokens=1))

    def count_tokens(self, messages, *, system=None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=8192)


# ── Agent._provider_for (the routing seam) ───────────────────────────────────────


def test_provider_for_injected_always_returns_main(tmp_path: Path) -> None:
    # An injected provider can't be rebuilt per model, so every role uses it.
    inj = _Scripted("x")
    agent = Agent(workspace_root=tmp_path, provider=inj)
    assert agent._provider_for(None) is agent.provider
    assert agent._provider_for(agent.settings.default_model) is agent.provider
    assert agent._provider_for("openai/gpt-4o") is agent.provider  # injected -> no rebuild


def test_provider_for_builds_distinct_cached_provider(tmp_path: Path) -> None:
    # No injected provider -> built from settings; a NON-default model builds a distinct,
    # cached provider; the default model / None reuse the main provider.
    agent = Agent(workspace_root=tmp_path, default_model="openai/gpt-4o")
    routed = agent._provider_for("openai/gpt-4o-mini")
    assert routed is not agent.provider
    assert agent._provider_for("openai/gpt-4o-mini") is routed  # cached (no rebuild)
    assert agent._provider_for("openai/gpt-4o") is agent.provider  # default -> main
    assert agent._provider_for(None) is agent.provider


def test_routed_cross_backend_clears_endpoint(tmp_path: Path) -> None:
    # A custom api_base set for the openai/* default must NOT leak onto an ollama_chat/* role
    # (it would send the local role to the cloud gateway). Same-backend keeps the endpoint.
    agent = Agent(workspace_root=tmp_path, default_model="openai/gpt-4o", api_base="http://gw/v1")
    assert agent.provider.inner.api_base == "http://gw/v1"  # type: ignore[attr-defined]
    cross = agent._provider_for("ollama_chat/qwen2.5:3b")
    assert cross.inner.api_base != "http://gw/v1"  # type: ignore[attr-defined]  # endpoint not carried over
    same = agent._provider_for("openai/gpt-4o-mini")
    assert same.inner.api_base == "http://gw/v1"  # type: ignore[attr-defined]  # same backend keeps it


# ── sub-agent routing ─────────────────────────────────────────────────────────────


def _runner(tmp_path: Path, default_p: Provider, provider_for) -> SubAgentRunner:
    return SubAgentRunner(
        provider=default_p,
        registry=ToolRegistry(),
        settings=Settings(default_model="default/model", workspace_root=tmp_path),
        budget=IterationBudget(10),
        provider_for=provider_for,
    )


async def test_subagent_routes_to_role_model(tmp_path: Path) -> None:
    default_p = _Scripted("default-said")
    routed_p = _Scripted("routed-said")
    runner = _runner(tmp_path, default_p, lambda m: routed_p if m == "cheap/x" else default_p)
    result = await runner.run(SubAgentDefinition(name="planner", model="cheap/x"), "go")
    assert "routed-said" in result.summary  # the routed (cheap) provider answered
    assert routed_p.calls >= 1 and default_p.calls == 0


async def test_subagent_without_model_uses_parent_provider(tmp_path: Path) -> None:
    default_p = _Scripted("default-said")
    routed_p = _Scripted("routed-said")
    runner = _runner(tmp_path, default_p, lambda m: routed_p)
    result = await runner.run(SubAgentDefinition(name="general"), "go")  # no model
    assert "default-said" in result.summary  # parent provider, provider_for not consulted
    assert default_p.calls >= 1 and routed_p.calls == 0


def test_subagent_definitions_pick_up_model_roles(tmp_path: Path) -> None:
    # The Agent applies model_roles to the built-in plan/general definitions.
    agent = Agent(
        workspace_root=tmp_path,
        provider=_Scripted("x"),
        enable_subagents=True,
        model_roles={"planner": "cheap/plan", "subagent": "cheap/general"},
    )
    spawner = agent.loop.spawner
    assert spawner is not None
    defs = {name: spawner._defs[name] for name in spawner.available_types()}  # type: ignore[attr-defined]
    assert defs["plan"].model == "cheap/plan"
    assert defs["general-purpose"].model == "cheap/general"


# ── compaction summarizer routing ──────────────────────────────────────────────────


async def test_compaction_summarizer_uses_routed_provider(tmp_path: Path) -> None:
    gen, summ = _Scripted("gen"), _Scripted("a summary")
    loop = AgentLoop(
        gen,
        ToolRegistry(),
        Session(cwd=str(tmp_path), model="t"),
        summarizer_provider=summ,
        max_iterations=5,
    )
    out = await loop._summarize_for_compaction([Message.user("old stuff")])
    assert out == "a summary"
    assert summ.calls == 1 and gen.calls == 0


async def test_compaction_summarizer_defaults_to_main_provider(tmp_path: Path) -> None:
    gen = _Scripted("gen-summary")
    loop = AgentLoop(gen, ToolRegistry(), Session(cwd=str(tmp_path), model="t"), max_iterations=5)
    out = await loop._summarize_for_compaction([Message.user("x")])
    assert out == "gen-summary" and gen.calls == 1


# ── g-016-83: routed models must not be decommissioned ───────────────────────
#
# THE DEFECT THIS PINS. Groq removed ``qwen/qwen3-32b`` from its catalog around
# 2026-07-19. The registry recorded that — in a COMMENT — and its successor's
# capabilities were registered the same day. But DEFAULT_CATEGORY_MODELS kept
# routing ``quick_code`` and ``plan`` at the dead model for ten days, because a
# comment cannot be asserted and nothing checked. Confirmed against Groq's live
# /v1/models on 2026-07-29: qwen3-32b ABSENT, qwen3.6-27b PRESENT.
#
# Hermetic: reads the static registry only, no network.


def test_routed_models_are_not_decommissioned() -> None:
    """Every default category route resolves to a model still in the catalog."""
    from zakcode.providers.registry import get_capabilities
    from zakcode.providers.routing import DEFAULT_CATEGORY_MODELS

    dead = {
        category: model.litellm_string
        for category, model in DEFAULT_CATEGORY_MODELS.items()
        if get_capabilities(model.litellm_string).decommissioned
    }
    assert not dead, (
        f"DEFAULT_CATEGORY_MODELS routes to decommissioned model(s): {dead}. "
        "Repoint the category to the catalog successor and update the registry."
    )


def test_decommissioned_flag_actually_discriminates() -> None:
    """The guard above is only meaningful if the flag is ever True.

    A test asserting 'nothing is decommissioned' passes trivially when NOTHING is
    ever marked — the same vacuity that let the original defect through. Pin that
    the flag is set on the model we know is dead, so the check has real teeth.
    """
    from zakcode.providers.registry import get_capabilities

    assert get_capabilities("groq/qwen/qwen3-32b").decommissioned is True
    assert get_capabilities("groq/qwen/qwen3.6-27b").decommissioned is False
