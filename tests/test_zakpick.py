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
from zakcode.providers.base import Provider
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
    assert r.model_for_category("summarize", s) == "groq/openai/gpt-oss-20b"  # no-tool prose
    # Repointed 2026-07-29 (g-016-83): Groq decommissioned qwen3-32b ~2026-07-19
    # (confirmed ABSENT from the live /v1/models catalog); qwen3.6-27b is the
    # catalog successor and the same tool-capable tier.
    assert r.model_for_category("quick_code", s) == "groq/qwen/qwen3.6-27b"  # tools-reliable Groq
    assert r.model_for_category("plan", s) == "groq/qwen/qwen3.6-27b"
    # deep_code/delegate moved to a first-party model whose native tool-calling is reliable
    # (Groq's open models emit tool calls its strict parser rejects — see routing.py rationale).
    assert r.model_for_category("deep_code", s) == "openai/gpt-4o-mini"  # reliable native + cached
    assert r.model_for_category("delegate", s) == "openai/gpt-4o-mini"


def test_default_avoids_tools_unreliable_model() -> None:
    # A TOOL-USING routed category must never default to a tools_unreliable model (its tool calls
    # would flake — gpt-oss-20b/120b, llama-3.3-70b). summarize/classify do NO tool calls, so the
    # flag is moot for them and they may stay on the cheap (flagged) model.
    from zakcode.providers.registry import get_capabilities

    s = Settings(default_model="zakpick", workspace_root=".")
    assert get_capabilities("groq/openai/gpt-oss-20b").tools_unreliable is True  # the cheap tier
    for category in ("quick_code", "deep_code", "delegate", "plan"):
        model = r.model_for_category(category, s)
        assert get_capabilities(model).tools_unreliable is False, (category, model)


def test_deep_tier_uses_reliable_first_party_native_tool_calling() -> None:
    # The tool-heavy capable tier (deep_code, delegate) must NOT route to Groq's open
    # models, whose native tool calls Groq's strict parser rejects with tool_use_failed
    # (gpt-oss-120b malformed JSON; llama-3.3 pseudo-XML). Live-verified 2026-06-18:
    # gpt-4o-mini completes the deep-task benchmark natively; the Groq open models do not.
    s = Settings(default_model="zakpick", workspace_root=".")
    for category in ("deep_code", "delegate"):
        spec = r.model_spec_for_category(category, s)
        assert spec.source != "groq", f"{category} must not be a Groq open model"
        assert "gpt-oss" not in spec.model and "llama-3.3" not in spec.model


def test_user_override_wins_and_flips_source() -> None:
    s = Settings(
        default_model="zakpick",
        workspace_root=".",
        zakpick_models={"deep_code": {"model": "qwen3:32b", "source": "local"}},
    )
    assert r.model_for_category("deep_code", s) == "ollama_chat/qwen3:32b"  # overridden → local
    assert r.model_for_category("quick_code", s) == "groq/qwen/qwen3.6-27b"  # still default


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
    # real struggle signal — so the signature carries no iteration parameter (difficulty_hint is
    # the SCOPE verdict, not iteration count).
    import inspect

    assert set(inspect.signature(r.classify_main_turn).parameters) == {
        "last_user_len",
        "context_frac",
        "signal_latched",
        "difficulty_hint",
    }


# ── difficulty side-call: SCOPE judgment overrides length (the real fix) ──────────


def test_classify_main_turn_difficulty_hint_overrides_length() -> None:
    # A SCOPE 'deep' verdict routes a SHORT message to deep_code — length no longer picks quick.
    assert (
        r.classify_main_turn(
            last_user_len=15, context_frac=0.0, signal_latched=False, difficulty_hint="deep_code"
        )
        == "deep_code"
    )
    # A 'quick' verdict over a small context -> quick_code...
    assert (
        r.classify_main_turn(
            last_user_len=15, context_frac=0.0, signal_latched=False, difficulty_hint="quick_code"
        )
        == "quick_code"
    )
    # ...but a 'quick' verdict still yields to the context-size guard (accumulated state).
    assert (
        r.classify_main_turn(
            last_user_len=15, context_frac=0.9, signal_latched=False, difficulty_hint="quick_code"
        )
        == "deep_code"
    )
    # A latched struggle signal still wins over any hint.
    assert (
        r.classify_main_turn(
            last_user_len=15, context_frac=0.0, signal_latched=True, difficulty_hint="quick_code"
        )
        == "deep_code"
    )
    # No hint (classifier unavailable / bare loop) -> the legacy length heuristic (short -> quick).
    assert (
        r.classify_main_turn(last_user_len=15, context_frac=0.0, signal_latched=False)
        == "quick_code"
    )


def test_should_consult_classifier() -> None:
    # The ambiguous case (short + small context, where the length rule would say quick) is the
    # only one worth a model call.
    assert r.should_consult_classifier("add pdf support", 0.0) is True
    # A long request is decided deep_code with NO call (length only fast-paths UP to deep).
    assert r.should_consult_classifier("x" * 2000, 0.0) is False
    # An already-large context is decided deep_code with NO call.
    assert r.should_consult_classifier("tiny", 0.9) is False


def test_parse_difficulty_fails_up() -> None:
    assert r.parse_difficulty({"difficulty": "quick"}) == "quick_code"
    assert r.parse_difficulty({"difficulty": "deep"}) == "deep_code"
    # Anything unexpected resolves to deep_code — never strand a real task on the cheap model.
    assert r.parse_difficulty({"difficulty": "enormous"}) == "deep_code"
    assert r.parse_difficulty({}) == "deep_code"
    assert r.parse_difficulty(None) == "deep_code"
    assert r.parse_difficulty("quick") == "deep_code"  # not even a dict


def test_difficulty_schema_validates_a_verdict() -> None:
    from zakcode.providers.structured import coerce_structured

    # The schema accepts the two enum values and rejects anything else (so a garbled verdict
    # is caught by complete_structured's validation, then fails up in parse_difficulty).
    assert coerce_structured('{"difficulty": "deep"}', schema=r.DIFFICULTY_SCHEMA) == {
        "difficulty": "deep"
    }


# ── describe + config validation ─────────────────────────────────────────────────


def test_describe_zakpick_shows_model_and_source_not_classify() -> None:
    s = Settings(default_model="zakpick", workspace_root=".")
    out = r.describe_zakpick(s)
    assert out.startswith("zakpick —")
    assert "gpt-4o-mini (openai)" in out  # model (source), not a raw litellm slug
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
    assert agent.settings.default_model == "openai/gpt-4o-mini"
    assert agent.session.model == "openai/gpt-4o-mini"


def test_agent_resolves_distinct_providers_per_category(tmp_path: Path) -> None:
    agent = zakcode.Agent(default_model="zakpick", workspace_root=tmp_path)
    summ_provider, summ_model = agent._resolve_task_provider("summarize")
    deep_provider, deep_model = agent._resolve_task_provider("deep_code")
    assert summ_model == "groq/openai/gpt-oss-20b"
    assert deep_model == "openai/gpt-4o-mini"
    assert summ_provider is not deep_provider  # different models → distinct cached providers


def test_agent_main_provider_updates_active_model(tmp_path: Path) -> None:
    agent = zakcode.Agent(default_model="zakpick", workspace_root=tmp_path)
    agent._main_provider_for("quick_code")
    assert agent._active_model == "groq/qwen/qwen3.6-27b"  # easy turn → reliable quick coder
    agent._main_provider_for("deep_code")
    assert agent._active_model == "openai/gpt-4o-mini"  # hard turn → reliable deep coder


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


class _ScriptedText(Provider):
    """A provider returning fixed text and no tool calls (so the turn completes). Subclasses
    Provider so it inherits the default ``astream`` — usable on both the buffered and streaming
    paths from the same stub."""

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


async def test_loop_difficulty_classifier_routes_short_but_deep(tmp_path: Path) -> None:
    # The real fix: a SHORT prompt the length heuristic would send to quick_code is routed to the
    # capable coder when the injected SCOPE classifier judges it deep ("build a pdf reader").
    from zakcode.agent.loop import AgentLoop
    from zakcode.session.store import Session
    from zakcode.tools.base import ToolRegistry

    cheap = _ScriptedText("cheap answered")
    strong = _ScriptedText("strong answered")
    seen: list[str] = []

    def main_for(category: str):
        seen.append(category)
        return cheap if category == "quick_code" else strong

    async def classifier(user_text: str, context_frac: float):
        from zakcode.providers.routing import DifficultyVerdict

        return DifficultyVerdict("deep_code")

    loop = AgentLoop(
        cheap,
        ToolRegistry(),
        Session(cwd=str(tmp_path), model="t"),
        main_provider_for=main_for,
        difficulty_classifier=classifier,
        max_iterations=3,
    )
    result = await loop.arun_turn("add pdf support")  # SHORT — length alone would say quick_code
    text = "\n".join(m.text for m in result.assistant_messages)
    assert "strong answered" in text  # the SCOPE classifier routed it to the capable coder
    assert strong.calls >= 1 and cheap.calls == 0
    assert "deep_code" in seen and "quick_code" not in seen
    assert result.routed_category == "deep_code"
    assert result.routed_escalated is False  # deep from the start, not an escalation


async def test_loop_difficulty_classifier_quick_routes_cheap(tmp_path: Path) -> None:
    # A genuinely small task: the classifier says quick, so the cheap coder drives (no overpay).
    from zakcode.agent.loop import AgentLoop
    from zakcode.session.store import Session
    from zakcode.tools.base import ToolRegistry

    cheap = _ScriptedText("cheap answered")
    strong = _ScriptedText("strong answered")

    async def classifier(user_text: str, context_frac: float):
        from zakcode.providers.routing import DifficultyVerdict

        return DifficultyVerdict("quick_code")

    loop = AgentLoop(
        strong,
        ToolRegistry(),
        Session(cwd=str(tmp_path), model="t"),
        main_provider_for=lambda c: cheap if c == "quick_code" else strong,
        difficulty_classifier=classifier,
        max_iterations=3,
    )
    result = await loop.arun_turn("fix the typo on line 5")
    assert "cheap answered" in "\n".join(m.text for m in result.assistant_messages)
    assert cheap.calls >= 1 and strong.calls == 0
    assert result.routed_category == "quick_code"


def _classify_stub(json_text: str) -> _ScriptedText:
    """A provider that returns a fixed JSON body (the classify side-call's verdict)."""
    return _ScriptedText(json_text)


async def test_agent_classify_difficulty_judges_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = zakcode.Agent(default_model="zakpick", workspace_root=tmp_path)
    deep = _classify_stub('{"difficulty": "deep"}')
    monkeypatch.setattr(agent, "_resolve_task_provider", lambda c: (deep, "classify/m"))
    assert (await agent._classify_difficulty("add pdf support", 0.0)).category == "deep_code"
    assert deep.calls == 1  # the cheap classify model was consulted exactly once

    quick = _classify_stub('{"difficulty": "quick"}')
    monkeypatch.setattr(agent, "_resolve_task_provider", lambda c: (quick, "classify/m"))
    verdict = await agent._classify_difficulty("fix the typo on line 5", 0.0)
    assert verdict.category == "quick_code"
    assert verdict.skill is None  # no catalog, no skill


async def test_agent_classify_difficulty_long_skips_the_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = zakcode.Agent(default_model="zakpick", workspace_root=tmp_path)
    stub = _classify_stub('{"difficulty": "quick"}')
    monkeypatch.setattr(agent, "_resolve_task_provider", lambda c: (stub, "classify/m"))
    # A long request fast-paths to deep_code with NO model call (length only escalates UP).
    assert (await agent._classify_difficulty("x" * 2000, 0.0)).category == "deep_code"
    assert stub.calls == 0


async def test_agent_classify_difficulty_fails_up_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = zakcode.Agent(default_model="zakpick", workspace_root=tmp_path)

    class _Raises(Provider):
        async def acomplete(self, messages, *, system=None, tools=None, response_format=None, **kw):
            from zakcode.providers.base import RequestFailed

            raise RequestFailed("classify model down")

        def count_tokens(self, messages, *, system=None) -> int:
            return 0

        def capabilities(self):
            from zakcode.providers.base import Capabilities

            return Capabilities()

    monkeypatch.setattr(agent, "_resolve_task_provider", lambda c: (_Raises(), "classify/m"))
    # A provider error during classification FAILS UP to the reliable coder, never crashes.
    assert (await agent._classify_difficulty("add pdf support", 0.0)).category == "deep_code"


async def test_agent_classify_difficulty_fails_up_on_garbage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = zakcode.Agent(default_model="zakpick", workspace_root=tmp_path)
    garbage = _classify_stub("I think this is a deep task, but here is prose, not JSON.")
    monkeypatch.setattr(agent, "_resolve_task_provider", lambda c: (garbage, "classify/m"))
    # Output that never validates as the schema also fails up to deep_code.
    assert (await agent._classify_difficulty("add pdf support", 0.0)).category == "deep_code"


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


# ── Review follow-ups: loop-level escalation, streaming path, precedence ──────────


async def test_loop_doom_signal_latches_escalation(tmp_path: Path) -> None:
    # A real struggle signal (a doom-loop from a repeated identical tool batch) must latch the
    # turn to the deep coder and report it as an escalation. classify-with-latch→deep_code and
    # category→provider selection are unit-tested above; this guards that a *real* in-loop signal
    # drives the report (a wiring regression that silently never latches would fail here).
    from zakcode.agent.loop import AgentLoop
    from zakcode.config import PermissionTier
    from zakcode.providers.base import Capabilities, LLMResult, ToolCall
    from zakcode.session.store import Session
    from zakcode.tools.base import (
        ConcurrencyClass,
        Tool,
        ToolContext,
        ToolRegistry,
        ToolResult,
        ToolSpec,
    )

    class _Echo(Tool):
        spec = ToolSpec(
            name="echo",
            description="echo",
            parameters={"type": "object", "properties": {}},
            required_permission=PermissionTier.READ_ONLY,
            concurrency=ConcurrencyClass.READ_ONLY_SAFE,
        )

        async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
            return ToolResult.ok("ok")

    class _DoomProvider(Provider):
        def __init__(self) -> None:
            self.calls = 0

        async def acomplete(self, messages, *, system=None, tools=None, response_format=None, **kw):
            from zakcode.usage import Usage

            self.calls += 1
            return LLMResult(
                tool_calls=[ToolCall(id="c1", name="echo", arguments={})],
                usage=Usage(total_tokens=1),
            )

        def count_tokens(self, messages, *, system=None) -> int:
            return 5

        def capabilities(self) -> Capabilities:
            return Capabilities(supports_tools=True, context_window=8192)

        def model_id(self) -> str:
            return "cheap/m"

    cheap = _DoomProvider()
    deep = _ScriptedText("fixed", model="deep/m")
    reg = ToolRegistry()
    reg.register(_Echo())
    loop = AgentLoop(
        cheap,
        reg,
        Session(cwd=str(tmp_path), model="t"),
        main_provider_for=lambda c: cheap if c == "quick_code" else deep,
        max_iterations=10,
    )
    result = await loop.arun_turn("hi")
    # The doom signal latches escalation to the deep coder; the built-in recovery then re-enters on
    # that deep model, which completes -- escalation + recovery rescue the turn instead of dooming.
    assert result.stop_reason == "completed"
    assert result.routed_escalated is True
    assert result.routed_category == "deep_code"


async def test_streaming_path_routes_and_tags_usage(tmp_path: Path) -> None:
    # The CLI runs the STREAMING path in production; mirror the buffered routing test there so the
    # hand-duplicated streaming twin (selection + usage tag + routed_* on AgentDone) can't diverge.
    from zakcode.agent.loop import AgentLoop
    from zakcode.events import AgentDone
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
    done = None
    async for ev in loop.astream_turn("hi"):  # short → quick_code → cheap
        if isinstance(ev, AgentDone):
            done = ev
    assert done is not None and done.routed_category == "quick_code"
    assert cheap.calls >= 1 and strong.calls == 0
    assert "groq/openai/gpt-oss-20b" in session.usage_by_model()  # tagged on the streaming path


def test_model_roles_wins_over_zakpick_category(tmp_path: Path) -> None:
    # Under zakpick, an explicit model_roles override still wins over the category default (one
    # precedence chain, no third mechanism) — the combined case the per-system tests didn't cover.
    agent = zakcode.Agent(
        default_model="zakpick",
        workspace_root=tmp_path,
        enable_subagents=True,
        model_roles={"planner": "openai/gpt-4o"},
    )
    spawner = agent.loop.spawner
    assert spawner is not None
    plan_def = spawner._defs["plan"]  # type: ignore[attr-defined]
    assert plan_def.model == "openai/gpt-4o"  # explicit role override wins
    assert plan_def.category == "plan"  # category remains as the fallback when no model is set
