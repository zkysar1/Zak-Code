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

    def __init__(self, text: str, model: str = "") -> None:
        self.text = text
        self.calls = 0
        self._model = model

    async def acomplete(self, messages, *, system=None, tools=None, response_format=None, **kw):
        from zakcode.providers.base import LLMResult
        from zakcode.usage import Usage

        self.calls += 1
        return LLMResult(text=self.text, usage=Usage(total_tokens=7, cost_usd=0.01))

    def count_tokens(self, messages, *, system=None) -> int:
        return 10

    def capabilities(self):
        from zakcode.providers.base import Capabilities

        return Capabilities(supports_tools=True, context_window=8192)

    def model_id(self) -> str:
        return self._model


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


# ── Phase 1: per-model cost attribution ──────────────────────────────────────────


def test_usage_carries_model_and_add_collapses_on_mismatch() -> None:
    from zakcode.usage import Usage

    a = Usage(total_tokens=3, cost_usd=0.01, model="m1")
    b = Usage(total_tokens=4, cost_usd=0.02, model="m1")
    c = Usage(total_tokens=5, cost_usd=0.03, model="m2")
    assert (a + b).model == "m1"  # same model survives the sum
    assert (a + c).model == ""  # mixed sum has no single model
    assert (a + b).total_tokens == 7


def test_session_usage_by_model_groups(tmp_path: Path) -> None:
    from zakcode.session.store import Session
    from zakcode.usage import Usage

    s = Session(cwd=str(tmp_path), model="t")
    s.add_usage(Usage(total_tokens=3, cost_usd=0.01), model="cheap/m")
    s.add_usage(Usage(total_tokens=4, cost_usd=0.02), model="deep/m")
    s.add_usage(Usage(total_tokens=5, cost_usd=0.03), model="cheap/m")
    by_model = s.usage_by_model()
    assert by_model["cheap/m"].total_tokens == 8
    assert by_model["cheap/m"].cost_usd == pytest.approx(0.04)
    assert by_model["deep/m"].total_tokens == 4
    assert s.cumulative_usage().total_tokens == 12  # total unchanged by tagging


def test_provider_model_id() -> None:
    # The default Provider returns "" (unattributed); LiteLLM exposes its model and the text
    # wrapper forwards it — so a routed turn's usage gets the right model tag.
    from zakcode.config import Settings
    from zakcode.providers.litellm_provider import LiteLLMProvider
    from zakcode.providers.text_tools import TextToolCallingProvider

    inner = LiteLLMProvider(Settings(default_model="groq/openai/gpt-oss-120b", workspace_root="."))
    assert inner.model_id() == "groq/openai/gpt-oss-120b"
    wrapped = TextToolCallingProvider(inner, mode="text")
    assert wrapped.model_id() == "groq/openai/gpt-oss-120b"
    assert _ScriptedText("x").model_id() == ""  # a stub provider is simply unattributed


async def test_loop_tags_usage_with_routed_model(tmp_path: Path) -> None:
    from zakcode.agent.loop import AgentLoop
    from zakcode.session.store import Session
    from zakcode.tools.base import ToolRegistry

    cheap = _ScriptedText("done", model="groq/openai/gpt-oss-20b")
    strong = _ScriptedText("done", model="groq/openai/gpt-oss-120b")
    session = Session(cwd=str(tmp_path), model="t")
    loop = AgentLoop(
        strong,
        ToolRegistry(),
        session,
        main_provider_for=lambda c: cheap if c == "quick_code" else strong,
        max_iterations=3,
    )
    await loop.arun_turn("hi")  # easy turn → quick_code → cheap model tags the usage
    by_model = session.usage_by_model()
    assert "groq/openai/gpt-oss-20b" in by_model  # attributed to the model that actually ran
    assert by_model["groq/openai/gpt-oss-20b"].cost_usd == pytest.approx(0.01)


# ── Phase 2: TurnResult routing fields + the "cheaper sufficed" advisory ──────────


async def test_turnresult_carries_routing_fields(tmp_path: Path) -> None:
    from zakcode.agent.loop import AgentLoop
    from zakcode.session.store import Session
    from zakcode.tools.base import ToolRegistry

    p = _ScriptedText("done", model="m")
    loop = AgentLoop(
        p,
        ToolRegistry(),
        Session(cwd=str(tmp_path), model="t"),
        main_provider_for=lambda c: p,
        max_iterations=3,
    )
    result = await loop.arun_turn("hi")  # short, clean → quick_code, no escalation
    assert result.routed_category == "quick_code"
    assert result.routed_escalated is False


async def test_turnresult_routing_none_without_zakpick(tmp_path: Path) -> None:
    from zakcode.agent.loop import AgentLoop
    from zakcode.session.store import Session
    from zakcode.tools.base import ToolRegistry

    loop = AgentLoop(
        _ScriptedText("done"),
        ToolRegistry(),
        Session(cwd=str(tmp_path), model="t"),
        max_iterations=3,  # no main_provider_for → zakpick off
    )
    result = await loop.arun_turn("hi")
    assert result.routed_category is None


def _advisory_console():
    import io

    from rich.console import Console

    from zakcode.cli._theme import ZAK_THEME

    buf = io.StringIO()
    return Console(theme=ZAK_THEME, file=buf, width=100, force_terminal=False), buf


def _done(category: str | None, *, escalated: bool = False, stop: str = "completed"):
    from zakcode.events import AgentDone

    return AgentDone(
        stop_reason=stop, iterations=1, routed_category=category, routed_escalated=escalated
    )


def test_advisory_fires_once_after_clean_deep_turns() -> None:
    from zakcode.cli import _ZAKPICK_ADVISORY_AFTER, _maybe_zakpick_advisory

    console, buf = _advisory_console()
    state = [0, 0]
    for _ in range(_ZAKPICK_ADVISORY_AFTER):
        _maybe_zakpick_advisory(console, _done("deep_code"), state)
    assert "cheaper" in buf.getvalue() and "deep_code" in buf.getvalue()
    assert state[1] == 1
    # fires at most once — a further clean deep turn prints nothing more.
    buf.truncate(0)
    buf.seek(0)
    _maybe_zakpick_advisory(console, _done("deep_code"), state)
    assert buf.getvalue() == ""


def test_advisory_ignores_easy_escalated_errored_and_non_zakpick() -> None:
    from zakcode.cli import _ZAKPICK_ADVISORY_AFTER, _maybe_zakpick_advisory

    console, buf = _advisory_console()
    state = [0, 0]
    n = _ZAKPICK_ADVISORY_AFTER + 2
    for _ in range(n):
        _maybe_zakpick_advisory(console, _done("quick_code"), state)  # easy turns
        _maybe_zakpick_advisory(console, _done("deep_code", escalated=True), state)  # needed deep
        _maybe_zakpick_advisory(console, _done("deep_code", stop="stuck"), state)  # not clean
        _maybe_zakpick_advisory(console, _done(None), state)  # zakpick off
    assert buf.getvalue() == ""  # none of these count → no advisory
    assert state[0] == 0
