"""A zakpick entry names its own tool-calling mode (ADR-0304).

Reported 2026-10-14 against a LiteLLM 1.9x gateway in front of Bedrock/Vertex: one route accepts
``tools`` and drops them, answering HTTP 200 with no tool call, so there is no refusal to learn
from (ADR-0303 learns from a refusal). It needs the text protocol while the other routes behind the
same endpoint do native tools genuinely, and ``tool_calling_mode`` was global, so the two could not
share one zakpick. An entry may now say ``auto``, ``native`` or ``text``; one that says nothing
inherits ``Settings.tool_calling_mode``.

Hermetic: no network, no credentials. The routes are neutral aliases with a declared window, and
nothing is called: the providers are built and their modes read.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

import zakcode
from zakcode.providers.base import Provider
from zakcode.providers.routing import ZakpickModel
from zakcode.providers.text_tools import TextToolCallingProvider

WINDOW = 131_072


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # The cwd .env, the user config home and real env vars can all set these: isolate them, so
    # a box whose environment says ZAKCODE_TOOL_CALLING_MODE=native cannot change a verdict.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ZAKCODE_HOME", str(tmp_path / "confighome"))
    for var in (
        "ZAKCODE_FALLBACK_MODEL",
        "ZAKCODE_DEFAULT_MODEL",
        "ZAKCODE_ZAKPICK_MODELS",
        "ZAKCODE_TOOL_CALLING_MODE",
        "ZAKCODE_EXTRA_BODY",
    ):
        monkeypatch.delenv(var, raising=False)
    yield


def _route(model: str, **extra: object) -> dict[str, object]:
    return {"model": model, "source": "openai", "context_window": WINDOW, **extra}


def _agent(tmp_path: Path, models: dict[str, object], **settings: object) -> zakcode.Agent:
    return zakcode.Agent(
        default_model="zakpick", workspace_root=tmp_path, zakpick_models=models, **settings
    )


def _mode(provider: Provider) -> str:
    assert isinstance(provider, TextToolCallingProvider)
    return provider.mode


def _routed(agent: zakcode.Agent, category: str) -> Provider:
    return agent._resolve_task_provider(category)[0]


# ── the entry ────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "given, kept", [("text", "text"), ("Native", "native"), (" auto ", "auto")]
)
def test_an_entry_may_name_a_tool_calling_mode(given: str, kept: str) -> None:
    assert ZakpickModel(model="route-a", tool_calling_mode=given).tool_calling_mode == kept


def test_an_entry_names_none_unless_told() -> None:
    assert ZakpickModel(model="route-a").tool_calling_mode is None


def test_an_unknown_mode_is_refused_at_load() -> None:
    with pytest.raises(ValidationError, match="auto, native or text"):
        ZakpickModel(model="route-a", tool_calling_mode="textual")


# ── the providers ────────────────────────────────────────────────────────────────────────────


def test_two_entries_naming_different_modes_get_providers_in_those_modes(tmp_path: Path) -> None:
    agent = _agent(
        tmp_path,
        {
            "deep_code": _route("route-a", tool_calling_mode="native"),
            "summarize": _route("route-b", tool_calling_mode="text"),
        },
    )
    assert _mode(_routed(agent, "summarize")) == "text"
    assert _mode(_routed(agent, "deep_code")) == "native"
    # The startup provider is deep_code's, so it is built in deep_code's mode.
    assert _mode(agent.provider) == "native"


def test_the_same_model_in_two_modes_gets_two_providers(tmp_path: Path) -> None:
    """One model named by two categories in two modes: a cache keyed on the model alone, or the
    default-model shortcut, would hand the second category the first one's provider."""
    agent = _agent(
        tmp_path,
        {
            "deep_code": _route("route-a", tool_calling_mode="native"),
            "summarize": _route("route-a", tool_calling_mode="text"),
            "plan": _route("route-a", tool_calling_mode="native"),
        },
    )
    summarize, deep, plan = (_routed(agent, c) for c in ("summarize", "deep_code", "plan"))
    assert _mode(summarize) == "text"
    assert summarize is not deep
    # Control: the same model in the same mode is the same provider.
    assert plan is deep is agent.provider


def test_another_model_in_two_modes_gets_two_providers(tmp_path: Path) -> None:
    """The same for a model the session did not start on: there the provider cache, not the
    default-model shortcut, is what keeps the two apart."""
    agent = _agent(
        tmp_path,
        {
            "deep_code": _route("route-a"),
            "summarize": _route("route-b", tool_calling_mode="text"),
            "plan": _route("route-b", tool_calling_mode="native"),
        },
    )
    summarize, plan = _routed(agent, "summarize"), _routed(agent, "plan")
    assert (_mode(summarize), _mode(plan)) == ("text", "native")
    # Control: asked again, a category gets its own cached provider back.
    assert _routed(agent, "summarize") is summarize


def test_a_call_with_no_route_keeps_the_startup_entrys_mode(tmp_path: Path) -> None:
    """A failover or a role override names a model and nothing more. On the startup model it is
    handed the startup provider, in that entry's mode, as it keeps that entry's caching switch
    (ADR-0298): the entry's word is about this model's route."""
    agent = _agent(
        tmp_path,
        {"deep_code": _route("route-a", tool_calling_mode="text")},
        tool_calling_mode="native",
    )
    assert agent._provider_for(agent.settings.default_model) is agent.provider
    assert _mode(agent.provider) == "text"


def test_a_mode_alone_keys_the_provider_cache(tmp_path: Path) -> None:
    """At the seam: a mode with no other per-entry knob still keys the cache, so one model asked
    for in two modes gets two providers."""
    agent = _agent(
        tmp_path,
        {"deep_code": _route("route-a", tool_calling_mode="text")},
        tool_calling_mode="native",
    )
    model = agent.settings.default_model
    auto = agent._provider_for(model, tool_calling_mode="auto")
    native = agent._provider_for(model, tool_calling_mode="native")
    assert (_mode(auto), _mode(native)) == ("auto", "native")


def test_an_entry_without_the_field_inherits_the_setting(tmp_path: Path) -> None:
    agent = _agent(
        tmp_path,
        {"deep_code": _route("route-a"), "summarize": _route("route-b")},
        tool_calling_mode="text",
    )
    assert _mode(_routed(agent, "summarize")) == "text"
    assert _mode(agent.provider) == "text"


def test_the_startup_entrys_mode_is_not_inherited_by_the_others(tmp_path: Path) -> None:
    """deep_code's own word is about deep_code's route. An entry that names no mode inherits the
    setting, not the mode of whichever entry the session happened to start on."""
    agent = _agent(
        tmp_path,
        {
            "deep_code": _route("route-a", tool_calling_mode="native"),
            "summarize": _route("route-a"),
            "plan": _route("route-b"),
        },
        tool_calling_mode="text",
    )
    assert _mode(agent.provider) == "native"
    assert _mode(_routed(agent, "summarize")) == "text"
    assert _mode(_routed(agent, "plan")) == "text"


def test_an_entrys_auto_is_resolved_for_its_own_model(tmp_path: Path) -> None:
    """``auto`` on an entry goes through the same resolution as the global one: an Ollama model
    is answered through the text protocol, any other keeps ``auto``."""
    agent = _agent(
        tmp_path,
        {
            "deep_code": _route("route-a", tool_calling_mode="auto"),
            "summarize": {
                "model": "qwen3:32b",
                "source": "local",
                "context_window": 32_768,
                "tool_calling_mode": "auto",
            },
        },
        tool_calling_mode="native",
    )
    assert _mode(_routed(agent, "summarize")) == "text"
    assert _mode(agent.provider) == "auto"
