"""A reasoning DEPTH is one config knob, rendered per backend where the backend takes one
(ADR-0182).

Field origin 2026-09-16: a served Mind on Vertex asked how to make ``gemini-3.8-flash`` reason
harder and no configurable path existed — ``ZakpickModel.thinking`` is an on/off switch in
llama.cpp's body form (rendered per backend since ADR-0181), and litellm's real Gemini knob,
``reasoning_effort`` (a ``thinkingLevel`` on Gemini 3, a ``thinkingBudget`` on 2.5), was nowhere
in the config surface. Now ``Settings.reasoning_effort`` sets it fleet-wide and
``zakpick_models[...].reasoning_effort`` per category; the provider renders it at the one
request chokepoint only where litellm flags the model reasoning-capable — inert elsewhere,
never a 400 — and withholds it whenever the switch says off. Hermetic: request assembly only,
no network; litellm's own mapper is exercised in-process.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

import zakcode
from zakcode.config import Settings
from zakcode.providers.litellm_provider import LiteLLMProvider
from zakcode.providers.routing import (
    REASONING_EFFORT_LEVELS,
    ZakpickModel,
    normalise_reasoning_effort,
    thinking_extra_body,
)
from zakcode.providers.thinking import (
    GEMINI_THINKING_OFF_EFFORT,
    REASONING_EFFORT_KWARG,
    reasoning_effort_reaches,
    render_reasoning_effort,
    rendered_thinking_wire_names,
)

POD = "http://pod.local:8080/v1"
POD_MODEL = "openai/zds-qwen3.8-27b"
GEMINI_3 = "vertex_ai/gemini-3.8-flash"
MSGS = [{"role": "user", "content": "hi"}]


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Settings read the cwd .env, the config home and real env vars — isolate all three so a
    # dev box's configuration can never change a verdict here.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ZAKCODE_HOME", str(tmp_path / "confighome"))
    for var in (
        "ZAKCODE_REASONING_EFFORT",
        "ZAKCODE_ZAKPICK_MODELS",
        "ZAKCODE_DEFAULT_MODEL",
        "ZAKCODE_API_BASE",
        "ZAKCODE_EXTRA_BODY",
    ):
        monkeypatch.delenv(var, raising=False)


def _yes(_model: str) -> bool:
    return True


def _no(_model: str) -> bool:
    return False


# ── the levels ────────────────────────────────────────────────────────────────


def test_the_levels_are_exactly_litellms_reasoning_effort_literal() -> None:
    """The accepted values are litellm's, read from the installed version — a level Zak Code
    accepts is one every litellm backend mapping knows how to spell."""
    from litellm.types.llms.openai import REASONING_EFFORT

    assert set(REASONING_EFFORT_LEVELS) == set(get_args(REASONING_EFFORT))


@pytest.mark.parametrize(
    ("raw", "level"),
    [("high", "high"), (" High ", "high"), ("MINIMAL", "minimal"), ("", None), (None, None)],
)
def test_a_level_is_normalised_for_case_and_whitespace(raw: object, level: str | None) -> None:
    assert normalise_reasoning_effort(raw) == level


@pytest.mark.parametrize("raw", ["hgih", "very-high", 3, True])
def test_anything_else_is_refused_naming_the_levels(raw: object) -> None:
    with pytest.raises(ValueError, match="none, minimal, low, medium, high, xhigh"):
        normalise_reasoning_effort(raw)


# ── the config surface ────────────────────────────────────────────────────────


def test_the_fleet_wide_level_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZAKCODE_REASONING_EFFORT", "High")
    assert Settings().reasoning_effort == "high"


def test_unset_means_the_models_own_default() -> None:
    assert Settings().reasoning_effort is None


def test_a_misspelled_level_fails_at_load(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZAKCODE_REASONING_EFFORT", "hgih")
    with pytest.raises(ValidationError, match="reasoning_effort must be one of"):
        Settings()


def test_a_category_carries_its_own_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "ZAKCODE_ZAKPICK_MODELS",
        json.dumps(
            {
                "deep_code": {
                    "model": "gemini-3.8-flash",
                    "source": "vertex_ai",
                    "reasoning_effort": "high",
                }
            }
        ),
    )
    spec = Settings().zakpick_models["deep_code"]
    assert spec.reasoning_effort == "high"
    assert spec.litellm_string == GEMINI_3


def test_a_category_cannot_be_both_off_and_deep() -> None:
    with pytest.raises(ValidationError, match="not a depth"):
        ZakpickModel(model="m", source="vertex_ai", thinking=False, reasoning_effort="low")
    # "on" beside a depth is fine, and so is a depth alone.
    assert ZakpickModel(model="m", thinking=True, reasoning_effort="low").reasoning_effort == "low"
    assert ZakpickModel(model="m", reasoning_effort="xhigh").reasoning_effort == "xhigh"


# ── the pure renderer ─────────────────────────────────────────────────────────


def test_a_reasoning_capable_cloud_model_gets_the_level() -> None:
    kwargs = render_reasoning_effort(GEMINI_3, None, "high", {}, supports_reasoning=_yes)
    assert kwargs == {REASONING_EFFORT_KWARG: "high"}


def test_no_level_renders_nothing() -> None:
    assert render_reasoning_effort(GEMINI_3, None, None, {}, supports_reasoning=_yes) == {}


def test_a_model_litellm_does_not_flag_reasoning_capable_gets_nothing() -> None:
    assert render_reasoning_effort("openai/gpt-4.1", None, "high", {}, supports_reasoning=_no) == {}


def test_a_self_hosted_server_behind_api_base_is_never_given_the_kwarg() -> None:
    """litellm's generic-OpenAI path drops it before the request, and the servers that take a
    level take it in their own body form — ``extra_body``'s job, kept verbatim there."""
    assert reasoning_effort_reaches(POD_MODEL, POD, supports_reasoning=_yes) is False
    assert render_reasoning_effort(POD_MODEL, POD, "high", {}, supports_reasoning=_yes) == {}
    # The same model name with NO api_base is a hosted destination: litellm's predicate rules.
    assert reasoning_effort_reaches(POD_MODEL, None, supports_reasoning=_yes) is True


def test_off_wins_over_a_depth() -> None:
    off, on = thinking_extra_body(False), thinking_extra_body(True)
    assert render_reasoning_effort(GEMINI_3, None, "high", off, supports_reasoning=_yes) == {}
    kwargs = render_reasoning_effort(GEMINI_3, None, "high", on, supports_reasoning=_yes)
    assert kwargs == {REASONING_EFFORT_KWARG: "high"}


def test_the_rendered_depth_shares_the_switchs_wire_names() -> None:
    """A backend refusing the depth names a wire field (``thinkingLevel``, …); the ADR-0181
    repair maps any of them back to the one kwarg both renderings ride."""
    names = rendered_thinking_wire_names({REASONING_EFFORT_KWARG: "high"})
    assert "thinkingLevel" in names and "reasoning_effort" in names


# ── at the request chokepoint, with litellm's real predicate ──────────────────


def _gemini3(**kw: object) -> LiteLLMProvider:
    return LiteLLMProvider(model=GEMINI_3, context_window=1_000_000, **kw)


def test_the_level_reaches_litellm_as_a_top_level_kwarg_for_gemini_3() -> None:
    kwargs = _gemini3(reasoning_effort="high")._build_kwargs(MSGS, None)
    assert kwargs[REASONING_EFFORT_KWARG] == "high"
    assert "extra_body" not in kwargs


def test_a_gemini_2_5_model_gets_the_level_too() -> None:
    provider = LiteLLMProvider(
        model="vertex_ai_beta/gemini-2.5-pro", context_window=1_000_000, reasoning_effort="low"
    )
    assert provider._build_kwargs(MSGS, None)[REASONING_EFFORT_KWARG] == "low"


def test_the_fleet_wide_level_is_read_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZAKCODE_REASONING_EFFORT", "medium")
    provider = LiteLLMProvider(Settings(), model=GEMINI_3, context_window=1_000_000)
    assert provider.reasoning_effort == "medium"
    assert provider._build_kwargs(MSGS, None)[REASONING_EFFORT_KWARG] == "medium"


def test_an_explicit_level_wins_over_the_settings_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZAKCODE_REASONING_EFFORT", "medium")
    provider = LiteLLMProvider(
        Settings(), model=GEMINI_3, context_window=1_000_000, reasoning_effort="low"
    )
    assert provider._build_kwargs(MSGS, None)[REASONING_EFFORT_KWARG] == "low"


def test_the_default_request_shape_is_unchanged_without_a_level() -> None:
    kwargs = _gemini3()._build_kwargs(MSGS, None)
    assert REASONING_EFFORT_KWARG not in kwargs and "extra_body" not in kwargs


def test_a_non_reasoning_model_is_inert_and_says_so_once(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="zakcode.providers")
    provider = LiteLLMProvider(
        model="openai/gpt-4.1", context_window=1_000_000, reasoning_effort="high"
    )
    assert provider.reasoning_effort == "high"  # kept as configured — diagnostic
    assert REASONING_EFFORT_KWARG not in provider._build_kwargs(MSGS, None)
    assert [r for r in caplog.records if "the level is not sent" in r.getMessage()]


def test_the_pod_request_is_byte_identical_to_before() -> None:
    provider = LiteLLMProvider(
        model=POD_MODEL, api_base=POD, context_window=131072, reasoning_effort="high"
    )
    kwargs = provider._build_kwargs(MSGS, None, extra_body=thinking_extra_body(True))
    assert REASONING_EFFORT_KWARG not in kwargs
    assert kwargs["extra_body"] == thinking_extra_body(True)


def test_the_overflow_retry_switches_thinking_off_and_withholds_the_depth() -> None:
    """ADR-0056's per-call off switch against Gemini: the off rendering, never the configured
    depth, for that one request."""
    kwargs = _gemini3(reasoning_effort="high")._build_kwargs(
        MSGS, None, extra_body=thinking_extra_body(False)
    )
    assert kwargs[REASONING_EFFORT_KWARG] == GEMINI_THINKING_OFF_EFFORT


def test_a_refused_depth_stays_out_of_every_later_request() -> None:
    provider = _gemini3(reasoning_effort="high")
    assert provider._build_kwargs(MSGS, None)[REASONING_EFFORT_KWARG] == "high"
    provider.rejected_request_fields.append("reasoning_effort")
    assert REASONING_EFFORT_KWARG not in provider._build_kwargs(MSGS, None)


# ── litellm's own mapping, in the installed version ───────────────────────────


def test_litellm_maps_the_level_to_a_thinking_level_on_gemini_3() -> None:
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )

    optional = VertexGeminiConfig().map_openai_params(
        {REASONING_EFFORT_KWARG: "high"}, {}, "gemini-3.8-flash", drop_params=True
    )
    assert optional["thinkingConfig"]["thinkingLevel"] == "high"


def test_litellm_maps_the_level_to_a_thinking_budget_on_gemini_2_5() -> None:
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )

    optional = VertexGeminiConfig().map_openai_params(
        {REASONING_EFFORT_KWARG: "low"}, {}, "gemini-2.5-flash", drop_params=True
    )
    assert isinstance(optional["thinkingConfig"]["thinkingBudget"], int)


# ── the Agent wiring, both ways ───────────────────────────────────────────────


def _inner(provider: object) -> LiteLLMProvider:
    inner = getattr(provider, "inner", provider)
    assert isinstance(inner, LiteLLMProvider)
    return inner


def test_a_categorys_level_reaches_its_provider_and_no_other(tmp_path: Path) -> None:
    agent = zakcode.Agent(
        default_model="zakpick",
        workspace_root=tmp_path,
        zakpick_models={
            "deep_code": {
                "model": "gemini-3.8-flash",
                "source": "vertex_ai",
                "reasoning_effort": "high",
            },
            "summarize": {"model": "gemini-3.8-flash", "source": "vertex_ai"},
        },
    )
    deep, deep_model = agent._resolve_task_provider("deep_code")
    summ, summ_model = agent._resolve_task_provider("summarize")
    assert deep_model == summ_model == GEMINI_3
    assert deep is not summ  # same model, different depth → distinct cached providers
    assert _inner(deep)._build_kwargs(MSGS, None)[REASONING_EFFORT_KWARG] == "high"
    assert REASONING_EFFORT_KWARG not in _inner(summ)._build_kwargs(MSGS, None)


def test_a_category_without_a_level_inherits_the_fleet_wide_one(tmp_path: Path) -> None:
    agent = zakcode.Agent(
        default_model="zakpick",
        workspace_root=tmp_path,
        reasoning_effort="medium",
        zakpick_models={
            "deep_code": {
                "model": "gemini-3.8-flash",
                "source": "vertex_ai",
                "reasoning_effort": "high",
            },
            "plan": {"model": "gemini-3.8-flash", "source": "vertex_ai"},
        },
    )
    deep, _ = agent._resolve_task_provider("deep_code")
    plan, _ = agent._resolve_task_provider("plan")
    assert _inner(deep)._build_kwargs(MSGS, None)[REASONING_EFFORT_KWARG] == "high"
    assert _inner(plan)._build_kwargs(MSGS, None)[REASONING_EFFORT_KWARG] == "medium"
