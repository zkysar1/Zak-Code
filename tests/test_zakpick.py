"""zakpick task-category model routing (the general form of per-role routing).

Hermetic and offline — zakpick assigns a concrete model per category, so there is no
availability probing to mock. Covers the (model, source)→litellm mapping, the built-in Groq
defaults + user overrides, the fail-UP main-turn classifier, the config validator, and the
Agent + loop wiring (startup model, per-category routing, model overrides, sub-agent categories,
the soft quick→deep latch that only ever switches between the user's two coder models).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import zakcode
from zakcode.config import Settings
from zakcode.providers import routing as r
from zakcode.providers.routing import ZakpickModel


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # Settings read the cwd .env, the user config home, and real env vars — isolate ALL of them
    # so a dev box's .env (e.g. a configured ZAKCODE_FALLBACK_MODEL or ZAKCODE_ZAKPICK_MODELS)
    # can never change a verdict.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ZAKCODE_HOME", str(tmp_path / "confighome"))
    for var in ("ZAKCODE_FALLBACK_MODEL", "ZAKCODE_DEFAULT_MODEL", "ZAKCODE_ZAKPICK_MODELS"):
        monkeypatch.delenv(var, raising=False)
    yield


# ── (model, source) → litellm string ────────────────────────────────────────────


@pytest.mark.parametrize(
    ("model", "source", "expected"),
    [
        ("openai/gpt-oss-120b", "groq", "groq/openai/gpt-oss-120b"),
        ("llama-3.1-8b-instant", "groq", "groq/llama-3.1-8b-instant"),
        ("qwen3:32b", "local", "ollama_chat/qwen3:32b"),
        ("qwen3", "ollama", "ollama_chat/qwen3"),
        ("gpt-4o", "openai", "openai/gpt-4o"),
        ("claude-sonnet-4-6", "anthropic", "anthropic/claude-sonnet-4-6"),
    ],
)
def test_litellm_string(model: str, source: str, expected: str) -> None:
    assert ZakpickModel(model=model, source=source).litellm_string == expected


def test_source_defaults_to_groq() -> None:
    assert ZakpickModel(model="qwen/qwen3-32b").source == "groq"
    assert ZakpickModel(model="qwen/qwen3-32b").litellm_string == "groq/qwen/qwen3-32b"


# ── built-in defaults + overrides ────────────────────────────────────────────────


def test_default_category_models_cover_every_category() -> None:
    # Every routed category (and the seam-only classify) has a built-in default.
    for category in ("quick_code", "deep_code", "summarize", "plan", "delegate", "classify"):
        assert category in r.DEFAULT_CATEGORY_MODELS


def test_defaults_are_groq_and_graduated() -> None:
    s = Settings(default_model="zakpick", workspace_root=".")
    assert r.model_for_category("classify", s) == "groq/llama-3.1-8b-instant"  # cheapest
    assert r.model_for_category("summarize", s) == "groq/openai/gpt-oss-20b"
    assert r.model_for_category("quick_code", s) == "groq/openai/gpt-oss-20b"
    assert r.model_for_category("plan", s) == "groq/qwen/qwen3-32b"
    assert r.model_for_category("deep_code", s) == "groq/openai/gpt-oss-120b"  # tools-reliable
    assert r.model_for_category("delegate", s) == "groq/openai/gpt-oss-120b"


def test_default_avoids_tools_unreliable_model() -> None:
    # llama-3.3-70b-versatile is flagged tools_unreliable — it must not be a tool-using default.
    s = Settings(default_model="zakpick", workspace_root=".")
    routed = {r.model_for_category(c, s) for c in r.ROUTED_CATEGORIES}
    assert all("llama-3.3-70b" not in m for m in routed)


def test_user_override_wins_and_flips_source() -> None:
    s = Settings(
        default_model="zakpick",
        workspace_root=".",
        zakpick_models={"deep_code": {"model": "qwen3:32b", "source": "local"}},
    )
    assert r.model_for_category("deep_code", s) == "ollama_chat/qwen3:32b"  # overridden → local
    assert r.model_for_category("quick_code", s) == "groq/openai/gpt-oss-20b"  # still default


# ── classifier (the one automatic decision) ──────────────────────────────────────


@pytest.mark.parametrize(
    ("last_user_len", "context_frac", "signal_latched", "expected"),
    [
        (50, 0.10, False, "quick_code"),  # short + small context → the user's quick model
        (900, 0.10, False, "deep_code"),  # long request → the user's deep model
        (50, 0.40, False, "deep_code"),  # large context → deep
        (50, 0.10, True, "deep_code"),  # a struggle signal ALWAYS escalates to deep
    ],
)
def test_classify_main_turn(
    last_user_len: int, context_frac: float, signal_latched: bool, expected: str
) -> None:
    assert (
        r.classify_main_turn(
            last_user_len=last_user_len, context_frac=context_frac, signal_latched=signal_latched
        )
        == expected
    )


def test_classify_has_no_iteration_input() -> None:
    # Regression fence: steady-state tool loops must NOT promote on iteration count, only on a
    # real struggle signal — so the signature carries no iteration parameter.
    import inspect

    assert set(inspect.signature(r.classify_main_turn).parameters) == {
        "last_user_len",
        "context_frac",
        "signal_latched",
    }


# ── describe + config validation ─────────────────────────────────────────────────


def test_describe_zakpick_shows_model_and_source_not_classify() -> None:
    s = Settings(default_model="zakpick", workspace_root=".")
    out = r.describe_zakpick(s)
    assert out.startswith("zakpick —")
    assert "gpt-oss-120b (groq)" in out  # model (source), not a raw litellm slug
    assert "classification" not in out  # seam-only category is never advertised


def test_config_rejects_unknown_category() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="unrecognized zakpick_models"):
        Settings(
            default_model="zakpick", workspace_root=".", zakpick_models={"planr": {"model": "x"}}
        )


# ── Agent wiring (offline — no probes; providers build lazily) ───────────────────


def test_agent_zakpick_startup_uses_deep_code_model(tmp_path: Path) -> None:
    agent = zakcode.Agent(default_model="zakpick", workspace_root=tmp_path)
    assert agent._zakpick is True
    # startup model = the deep_code category's model (the safe/capable one).
    assert agent.settings.default_model == "groq/openai/gpt-oss-120b"
    assert agent.session.model == "groq/openai/gpt-oss-120b"


def test_agent_resolves_distinct_providers_per_category(tmp_path: Path) -> None:
    agent = zakcode.Agent(default_model="zakpick", workspace_root=tmp_path)
    summ_provider, summ_model = agent._resolve_task_provider("summarize")
    deep_provider, deep_model = agent._resolve_task_provider("deep_code")
    assert summ_model == "groq/openai/gpt-oss-20b"
    assert deep_model == "groq/openai/gpt-oss-120b"
    assert summ_provider is not deep_provider  # different models → distinct cached providers


def test_agent_main_provider_updates_active_model(tmp_path: Path) -> None:
    agent = zakcode.Agent(default_model="zakpick", workspace_root=tmp_path)
    agent._main_provider_for("quick_code")
    assert agent._active_model == "groq/openai/gpt-oss-20b"  # easy turn → quick coder
    agent._main_provider_for("deep_code")
    assert agent._active_model == "groq/openai/gpt-oss-120b"  # hard turn → deep coder


def test_agent_override_changes_startup_and_routing(tmp_path: Path) -> None:
    agent = zakcode.Agent(
        default_model="zakpick",
        workspace_root=tmp_path,
        zakpick_models={"deep_code": {"model": "qwen3:32b", "source": "local"}},
    )
    assert agent.settings.default_model == "ollama_chat/qwen3:32b"
    _provider, model = agent._resolve_task_provider("deep_code")
    assert model == "ollama_chat/qwen3:32b"


def test_agent_subagent_definitions_carry_categories(tmp_path: Path) -> None:
    agent = zakcode.Agent(default_model="zakpick", workspace_root=tmp_path, enable_subagents=True)
    spawner = agent.loop.spawner
    assert spawner is not None
    defs = {name: spawner._defs[name] for name in spawner.available_types()}  # type: ignore[attr-defined]
    assert defs["plan"].category == "plan"
    assert defs["general-purpose"].category == "delegate"
    assert defs["plan"].model is None  # no model_roles override → routes by category


def test_agent_zakpick_failover_is_fallback_only(tmp_path: Path) -> None:
    # zakpick never substitutes a model the user didn't choose: with no fallback_model, failover
    # returns None (the turn ends as provider_error). model_resolution stays None.
    agent = zakcode.Agent(default_model="zakpick", workspace_root=tmp_path)
    assert agent.model_resolution is None
    from zakcode.providers.base import RequestFailed

    assert agent._model_failover(RequestFailed("boom")) is None


def test_non_zakpick_unaffected(tmp_path: Path) -> None:
    # An explicit model leaves zakpick entirely off — the legacy single-provider path.
    agent = zakcode.Agent(default_model="ollama_chat/qwen2.5", workspace_root=tmp_path)
    assert agent._zakpick is False
    assert agent._main_provider_for is not None  # method exists but the loop never calls it
    # the loop got no main_provider_for (single provider path)
    assert agent.loop.main_provider_for is None


# ── loop wiring: the main-turn router fires per iteration ────────────────────────


class _ScriptedText:
    """A buffered provider that returns fixed text and no tool calls (so the turn completes)."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    async def acomplete(self, messages, *, system=None, tools=None, response_format=None, **kw):
        from zakcode.providers.base import LLMResult
        from zakcode.usage import Usage

        self.calls += 1
        return LLMResult(text=self.text, usage=Usage(total_tokens=1))

    def count_tokens(self, messages, *, system=None) -> int:
        return 10

    def capabilities(self):
        from zakcode.providers.base import Capabilities

        return Capabilities(supports_tools=True, context_window=8192)


async def test_loop_routes_main_turn_by_category(tmp_path: Path) -> None:
    from zakcode.agent.loop import AgentLoop
    from zakcode.session.store import Session
    from zakcode.tools.base import ToolRegistry

    cheap = _ScriptedText("cheap answered")
    strong = _ScriptedText("strong answered")

    def main_for(category: str):
        return cheap if category == "quick_code" else strong

    loop = AgentLoop(
        strong,  # startup provider; the router re-selects on the first iteration
        ToolRegistry(),
        Session(cwd=str(tmp_path), model="t"),
        main_provider_for=main_for,
        max_iterations=3,
    )
    result = await loop.arun_turn("hi")  # short, no signals → quick_code → cheap
    text = "\n".join(m.text for m in result.assistant_messages)
    assert "cheap answered" in text
    assert cheap.calls >= 1 and strong.calls == 0  # the cheap model drove the easy turn
