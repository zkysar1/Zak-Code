"""The ``hipaa_mode`` model allowlist and the predicate behind it.

``hipaa_mode`` exists so a home that handles protected health information can be CERTAIN no
call reaches a model nobody confirmed. Like ``local_only`` it must never fail open, so these
tests pin the refusal, not just the happy path, at both layers (the pairing and the stub
pattern follow ``tests/test_local_only.py``):

* ``Agent._assert_hipaa_models`` — startup, names every offender at once.
* ``LiteLLMProvider._build_kwargs`` — the call itself, catches routes nobody enumerated.
"""

from __future__ import annotations

import pytest

from zakcode.config import Settings
from zakcode.providers.base import ProviderError
from zakcode.providers.hipaa import HipaaModelViolation, check_model, normalize_model
from zakcode.providers.litellm_provider import LiteLLMProvider
from zakcode.providers.routing import ZakpickModel

MESSAGES = [{"role": "user", "content": "hi"}]


# ── the predicate ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("model", "allowed", "expected"),
    [
        # OpenAI's own prefix and the bare name are one model, in either direction.
        ("openai/gpt-4o", ["gpt-4o"], True),
        ("gpt-4o", ["openai/gpt-4o"], True),
        ("OpenAI/GPT-4o", ["gpt-4o"], True),
        ("openai/gpt-4o-mini", ["gpt-4o", "gpt-4o-mini"], True),
        # Exact match only: a sibling, a dated snapshot, and a shorter name are other models.
        ("openai/gpt-4o-mini", ["gpt-4o"], False),
        ("openai/gpt-4o-2024-08-06", ["gpt-4o"], False),
        ("openai/gpt-4", ["gpt-4o"], False),
        # Only OpenAI's own prefix is dropped. Another provider, or a generic-OpenAI base,
        # is a different destination and has to be listed by its full name.
        ("anthropic/gpt-4o", ["gpt-4o"], False),
        ("openai_like/gpt-4o", ["gpt-4o"], False),
        ("anthropic/claude-sonnet-4-6", ["claude-sonnet-4-6"], False),
        ("anthropic/claude-sonnet-4-6", ["anthropic/claude-sonnet-4-6"], True),
        # Fail closed: nothing to compare, or nothing listed, permits nothing.
        ("openai/gpt-4o", [], False),
        ("openai/gpt-4o", None, False),
        ("openai/gpt-4o", ["", "   "], False),
        ("", ["gpt-4o"], False),
        (None, ["gpt-4o"], False),
    ],
)
def test_check_model(model: str | None, allowed: list[str] | None, expected: bool) -> None:
    ok, reason = check_model(model, allowed)
    assert ok is expected
    assert reason, "a refusal must always carry an operator-actionable reason"


def test_an_empty_allowlist_says_it_is_empty() -> None:
    ok, reason = check_model("openai/gpt-4o", [])
    assert not ok
    assert "empty" in reason


def test_a_refusal_lists_what_is_allowed() -> None:
    ok, reason = check_model("openai/gpt-4o-mini", ["gpt-4o", "openai/gpt-4"])
    assert not ok
    assert "gpt-4, gpt-4o" in reason


def test_normalize_model() -> None:
    assert normalize_model(" OpenAI/GPT-4o ") == "gpt-4o"
    assert normalize_model("anthropic/x") == "anthropic/x"
    assert normalize_model(None) == ""


def test_violation_is_not_a_provider_error() -> None:
    """A HipaaModelViolation must NOT ride the loop's failover machinery.

    ProviderError is recoverable: the loop answers it by trying a DIFFERENT model. For this
    refusal that is the one reaction that would send patient data to a model nobody confirmed,
    so the violation is deliberately outside that hierarchy and nothing retries it.
    """
    assert not issubclass(HipaaModelViolation, ProviderError)
    assert issubclass(HipaaModelViolation, RuntimeError)


# ── the settings ──────────────────────────────────────────────────────────────


def test_hipaa_mode_defaults_off_with_an_empty_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fail-closed check must newly refuse only what the operator opted into."""
    monkeypatch.delenv("ZAKCODE_HIPAA_MODE", raising=False)
    monkeypatch.delenv("ZAKCODE_HIPAA_ALLOWED_MODELS", raising=False)
    settings = Settings(_env_file=None)
    assert settings.hipaa_mode is False
    assert settings.hipaa_allowed_models == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("gpt-4o", ["gpt-4o"]),
        ("gpt-4o,openai/gpt-4o-mini", ["gpt-4o", "openai/gpt-4o-mini"]),
        ("gpt-4o  gpt-4o-mini", ["gpt-4o", "gpt-4o-mini"]),
        ('["gpt-4o", "openai/gpt-4o-mini"]', ["gpt-4o", "openai/gpt-4o-mini"]),
    ],
)
def test_the_allowlist_parses_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: list[str]
) -> None:
    monkeypatch.setenv("ZAKCODE_HIPAA_MODE", "true")
    monkeypatch.setenv("ZAKCODE_HIPAA_ALLOWED_MODELS", raw)
    settings = Settings(_env_file=None)
    assert settings.hipaa_mode is True
    assert settings.hipaa_allowed_models == expected


# ── layer 2: the application gate (the actual guarantee) ──────────────────────


def _provider(model: str, allowed: list[str], *, mode: bool = True) -> LiteLLMProvider:
    return LiteLLMProvider(
        Settings(default_model=model, hipaa_mode=mode, hipaa_allowed_models=allowed, _env_file=None)
    )


def test_a_listed_model_builds_the_request() -> None:
    kwargs = _provider("openai/gpt-4o", ["gpt-4o"])._build_kwargs(MESSAGES, None)
    assert kwargs["model"] == "openai/gpt-4o"


def test_an_unlisted_model_is_refused_at_the_request_builder() -> None:
    with pytest.raises(HipaaModelViolation, match="not on the allowlist") as exc:
        _provider("openai/gpt-4o-mini", ["gpt-4o"])._build_kwargs(MESSAGES, None)
    # The operator learns the model and the remedy from the refusal alone.
    assert "openai/gpt-4o-mini" in str(exc.value)
    assert "ZAKCODE_HIPAA_ALLOWED_MODELS" in str(exc.value)


def test_an_empty_allowlist_refuses_every_call() -> None:
    with pytest.raises(HipaaModelViolation, match="empty"):
        _provider("openai/gpt-4o", [])._build_kwargs(MESSAGES, None)


def test_hipaa_mode_off_changes_nothing() -> None:
    provider = _provider("openai/gpt-4o-mini", [], mode=False)
    assert provider.hipaa_mode is False
    assert provider._build_kwargs(MESSAGES, None)["model"] == "openai/gpt-4o-mini"


def test_a_routed_model_keeps_the_allowlist() -> None:
    """Roles, zakpick categories and failover all build their provider through
    ``Agent._build_provider`` from a COPY of the settings. The allowlist has to survive that
    copy, or the one chokepoint would guard only the default model."""
    from zakcode import Agent

    settings = Settings(
        default_model="openai/gpt-4o",
        hipaa_mode=True,
        hipaa_allowed_models=["gpt-4o", "gpt-4o-mini"],
        _env_file=None,
    )
    stub = _Stub(settings)
    allowed = Agent._build_provider(stub, "openai/gpt-4o-mini")
    assert allowed.inner._build_kwargs(MESSAGES, None)["model"] == "openai/gpt-4o-mini"  # type: ignore[attr-defined]
    refused = Agent._build_provider(stub, "anthropic/claude-sonnet-4-6")
    with pytest.raises(HipaaModelViolation):
        refused.inner._build_kwargs(MESSAGES, None)  # type: ignore[attr-defined]


# ── layer 1: the startup gate (anticipation) ──────────────────────────────────
#
# _assert_hipaa_models reads only ``self.settings`` and ``self._zakpick``, so it is exercised
# against a stub rather than a fully-constructed Agent (which would build providers and probe
# the network). Same code path, no side effects.


class _Stub:
    def __init__(self, settings: Settings, *, zakpick: bool = False) -> None:
        self.settings = settings
        self._zakpick = zakpick


def _assert(settings: Settings, *, zakpick: bool = False) -> None:
    from zakcode import Agent

    Agent._assert_hipaa_models(_Stub(settings, zakpick=zakpick))  # type: ignore[arg-type]


def _hipaa(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "default_model": "openai/gpt-4o",
        "hipaa_mode": True,
        "hipaa_allowed_models": ["gpt-4o"],
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


def test_startup_gate_passes_for_a_fully_listed_config() -> None:
    _assert(
        _hipaa(
            fallback_model="openai/gpt-4o-mini",
            model_roles={"judge": "openai/gpt-4o-mini"},
            hipaa_allowed_models=["gpt-4o", "gpt-4o-mini"],
        )
    )


def test_startup_gate_catches_an_unlisted_fallback_model() -> None:
    """The headline case: the primary is listed, but a failover would reach an unlisted model."""
    with pytest.raises(HipaaModelViolation) as exc:
        _assert(_hipaa(fallback_model="anthropic/claude-sonnet-4-6"))
    assert "fallback_model" in str(exc.value)
    assert "anthropic/claude-sonnet-4-6" in str(exc.value)


def test_startup_gate_names_every_offender_at_once() -> None:
    with pytest.raises(HipaaModelViolation) as exc:
        _assert(
            _hipaa(
                default_model="openai/gpt-4o-mini",
                fallback_model="anthropic/claude-sonnet-4-6",
                model_roles={"judge": "openai/gpt-5-nano"},
            )
        )
    message = str(exc.value)
    for label in ("default_model", "fallback_model", "model_roles['judge']"):
        assert label in message


def test_startup_gate_catches_unset_zakpick_categories() -> None:
    """The case that makes this gate worth writing.

    An operator turns on hipaa_mode, switches to zakpick and assigns only deep_code, a config
    that LOOKS listed. The categories they did not assign still route, and they fall through to
    the built-in defaults. Checking only the assignments would report this clean, which is the
    most dangerous false negative: a quiet turn on a default model nobody confirmed.
    """
    settings = _hipaa(zakpick_models={"deep_code": ZakpickModel(model="gpt-4o")})
    with pytest.raises(HipaaModelViolation) as exc:
        _assert(settings, zakpick=True)
    message = str(exc.value)
    for category in ("quick_code", "summarize", "plan", "classify", "delegate"):
        assert f"zakpick '{category}'" in message
    assert "zakpick 'deep_code'" not in message


def test_startup_gate_passes_zakpick_when_every_default_is_listed() -> None:
    _assert(
        _hipaa(
            zakpick_models={"deep_code": ZakpickModel(model="gpt-4o")},
            hipaa_allowed_models=["gpt-4o", "gpt-4o-mini"],
        ),
        zakpick=True,
    )


def test_startup_gate_refuses_when_the_allowlist_is_empty() -> None:
    with pytest.raises(HipaaModelViolation) as exc:
        _assert(_hipaa(hipaa_allowed_models=[]))
    assert "default_model" in str(exc.value)
    assert "empty" in str(exc.value)


def test_startup_gate_does_not_accuse_a_sentinel() -> None:
    """``auto``/``zakpick`` name no model; by the time the gate runs they are already resolved
    to concrete ones, so a sentinel still standing is not a call target."""
    _assert(_hipaa(default_model="zakpick"))


def test_startup_gate_is_inert_when_hipaa_mode_is_off() -> None:
    _assert(
        _hipaa(
            hipaa_mode=False,
            hipaa_allowed_models=[],
            fallback_model="anthropic/claude-sonnet-4-6",
        )
    )


def test_the_agent_constructor_runs_the_startup_gate(tmp_path) -> None:
    """The gate only guarantees anything if construction calls it. A full Agent with an
    unlisted default model must refuse before a provider is built or a call is made."""
    from zakcode import Agent

    settings = _hipaa(default_model="openai/gpt-4o-mini", workspace_root=tmp_path)
    with pytest.raises(HipaaModelViolation, match="default_model"):
        Agent(settings=settings)
