"""BYOK — a member's own provider key, taken from their environment's vault (g-369-11).

The feature is one function, and almost everything worth pinning about it is a NEGATIVE:
what it must decline to touch. The positive path is a dict lookup.

WHY THE OVERLAY GOES THROUGH ``os.environ`` AT ALL, since a reader will ask. For a bare
cloud model ``LiteLLMProvider`` deliberately OMITS ``api_key`` from the call so litellm
reads the standard variable itself; an explicit key there once shadowed the real one and
produced a spurious AuthError (audit3 #6). The environment is therefore the only seam
that reaches real inference, and ``test_cloud_call_still_omits_api_key`` pins that the
decision this feature depends on has not been quietly reversed underneath it.

MUTATION TABLE — RUN, and the interesting row is the one that SURVIVED (guard-1475).
Baseline 14/14 green, restored 14/14 green:

  M1  drop the baseline restore (assign only — the naive version)   -> RED  (2)
  M2  replace ``values_for(NAMES)`` with a raw ``_load()``          -> GREEN (survived)
  M2a the same mutation, after adding the two accessor tests below  -> RED  (2)
  M3  make the overlay a no-op                                      -> RED  (7)

M2 survived because the recognised-name filter exists TWICE — once in ``values_for``
and again in the overlay loop, which iterates the recognised names rather than the
file's. Defense in depth is the right design; a test suite that cannot SEE one of the
two layers is not, and a corpus-level assertion reported green through a mutation that
removed a security filter. The accessor is now pinned directly, which is what moved M2a
to RED. Recorded rather than quietly fixed: an aggregate that passes through the defect
it was written to catch is the failure mode worth naming (guard-1793).

WHY A BASELINE RATHER THAN A PLAIN ASSIGNMENT. A long-lived process freezes its
environment at exec time, so the naive version — overlay once, assign — pins the member's
key for the life of the runtime and makes DELETING the secret appear to do nothing until
someone restarts the box. The baseline is what makes removal an event.
"""

from __future__ import annotations

import json
import os

import pytest

from zakcode.providers import resolve as R


class _Settings:
    """Minimal stand-in — the overlay reads exactly one attribute."""

    def __init__(self, secrets_file=None):
        self.secrets_file = secrets_file


@pytest.fixture(autouse=True)
def _clean_baseline():
    """The baseline is process state by design; a test must not inherit another's."""
    R._PLATFORM_KEY_BASELINE.clear()
    yield
    R._PLATFORM_KEY_BASELINE.clear()


def _vault(tmp_path, mapping):
    p = tmp_path / "secrets.json"
    p.write_text(json.dumps(mapping), encoding="utf-8")
    return p


# ── feature-off is total ─────────────────────────────────────────────────────


def test_no_secrets_file_leaves_the_environment_exactly_as_found(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "platform-key")
    assert R.apply_vault_provider_keys(_Settings(None)) == []
    assert os.environ["ANTHROPIC_API_KEY"] == "platform-key"


def test_unreadable_vault_is_survivable_and_changes_nothing(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "platform-key")
    # A directory where a file is expected: the loader's OSError path.
    assert R.apply_vault_provider_keys(_Settings(tmp_path)) == []
    assert os.environ["ANTHROPIC_API_KEY"] == "platform-key"


def test_malformed_vault_json_changes_nothing(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "platform-key")
    bad = tmp_path / "secrets.json"
    bad.write_text("{not json", encoding="utf-8")
    assert R.apply_vault_provider_keys(_Settings(bad)) == []
    assert os.environ["GROQ_API_KEY"] == "platform-key"


# ── the positive path ────────────────────────────────────────────────────────


def test_member_key_overrides_the_deployment_key(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "platform-key")
    v = _vault(tmp_path, {"ANTHROPIC_API_KEY": "member-key"})
    assert R.apply_vault_provider_keys(_Settings(v)) == ["ANTHROPIC_API_KEY"]
    assert os.environ["ANTHROPIC_API_KEY"] == "member-key"


def test_the_return_is_names_only(monkeypatch, tmp_path):
    # It is logged. A value in the return would become a value in a log line.
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    v = _vault(tmp_path, {"GROQ_API_KEY": "gsk_member_value_here"})
    got = R.apply_vault_provider_keys(_Settings(v))
    assert got == ["GROQ_API_KEY"]
    assert not any("gsk_member_value_here" in n for n in got)


# ── the properties a naive implementation gets wrong ─────────────────────────


def test_a_key_saved_after_startup_is_picked_up_without_a_restart(monkeypatch, tmp_path):
    """The vault's whole point: save a secret, use it on the next call, no restart."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "platform-key")
    v = _vault(tmp_path, {})
    assert R.apply_vault_provider_keys(_Settings(v)) == []
    assert os.environ["ANTHROPIC_API_KEY"] == "platform-key"

    v.write_text(json.dumps({"ANTHROPIC_API_KEY": "member-key"}), encoding="utf-8")
    assert R.apply_vault_provider_keys(_Settings(v)) == ["ANTHROPIC_API_KEY"]
    assert os.environ["ANTHROPIC_API_KEY"] == "member-key"


def test_removing_the_vault_entry_restores_the_deployment_key(monkeypatch, tmp_path):
    """Without the baseline this is the silent bug: deletion appears to do nothing."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "platform-key")
    v = _vault(tmp_path, {"ANTHROPIC_API_KEY": "member-key"})
    R.apply_vault_provider_keys(_Settings(v))
    assert os.environ["ANTHROPIC_API_KEY"] == "member-key"

    v.write_text(json.dumps({}), encoding="utf-8")
    assert R.apply_vault_provider_keys(_Settings(v)) == []
    assert os.environ["ANTHROPIC_API_KEY"] == "platform-key"


def test_removal_with_no_deployment_key_unsets_rather_than_stranding(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    v = _vault(tmp_path, {"OPENAI_API_KEY": "member-key"})
    R.apply_vault_provider_keys(_Settings(v))
    assert os.environ["OPENAI_API_KEY"] == "member-key"

    v.write_text(json.dumps({}), encoding="utf-8")
    R.apply_vault_provider_keys(_Settings(v))
    assert "OPENAI_API_KEY" not in os.environ


# ── the security boundary ────────────────────────────────────────────────────


def test_an_unrelated_member_secret_is_never_placed_in_the_environment(monkeypatch, tmp_path):
    """The load-bearing negative.

    A member's vault holds arbitrary service keys — that is the feature. Only the
    recognised PROVIDER names may reach the process environment; anything else stays
    behind the ``{{secret:NAME}}`` substitution boundary, where the value is resolved into
    one outbound request and scrubbed out of everything model-facing. Putting a member's
    WEATHER_API_KEY in ``os.environ`` would hand it to every subprocess the agent spawns.
    """
    v = _vault(tmp_path, {"WEATHER_API_KEY": "w-value", "ANTHROPIC_API_KEY": "member-key"})
    applied = R.apply_vault_provider_keys(_Settings(v))
    assert applied == ["ANTHROPIC_API_KEY"]
    assert "WEATHER_API_KEY" not in os.environ


def test_values_for_narrows_independently_of_the_overlay_loop(tmp_path):
    """The name filter exists TWICE and each layer needs its own proof.

    ``apply_vault_provider_keys`` asks the vault for the recognised names AND then
    iterates only those names — so removing either filter alone leaves behaviour
    identical, and a corpus-level test cannot tell you whether either works. Measured:
    a mutation replacing ``values_for(...)`` with a raw ``_load()`` passed the whole
    file. Defense in depth is the right design here; an assertion that cannot see one
    of the two layers is not (guard-1793). This pins the accessor directly.
    """
    from zakcode.tools.builtins._secrets import SecretsProvider

    f = tmp_path / "s.json"
    f.write_text(
        json.dumps({"ANTHROPIC_API_KEY": "member-key", "WEATHER_API_KEY": "w"}),
        encoding="utf-8",
    )
    got = SecretsProvider(f).values_for(["ANTHROPIC_API_KEY"])
    assert got == {"ANTHROPIC_API_KEY": "member-key"}
    assert "WEATHER_API_KEY" not in got


def test_values_for_omits_absent_names_rather_than_raising(tmp_path):
    """Asking "did the member happen to save this?" — a no is an answer, not an error."""
    from zakcode.tools.builtins._secrets import SecretsProvider

    f = tmp_path / "s.json"
    f.write_text(json.dumps({"ANTHROPIC_API_KEY": "k"}), encoding="utf-8")
    assert SecretsProvider(f).values_for(["GROQ_API_KEY"]) == {}


def test_the_recognised_set_is_derived_from_the_sources_not_a_second_list():
    """Anti-drift: a provider added to _EXTERNAL_SOURCES is covered with no second edit."""
    assert frozenset(s.key_env for s in R._EXTERNAL_SOURCES.values()) == R.VAULT_PROVIDER_KEY_NAMES
    assert "ANTHROPIC_API_KEY" in R.VAULT_PROVIDER_KEY_NAMES


def test_an_overlaid_key_is_still_scrubbed_from_subprocess_environments(monkeypatch, tmp_path):
    """A member key must inherit the platform key's subprocess hygiene, not bypass it."""
    from zakcode.secrets import provider_key_env_names

    v = _vault(tmp_path, {"ANTHROPIC_API_KEY": "member-key"})
    R.apply_vault_provider_keys(_Settings(v))
    assert "ANTHROPIC_API_KEY" in provider_key_env_names()


# ── non-regression on the decision this feature rests upon ───────────────────


def test_cloud_call_still_omits_api_key_so_litellm_reads_the_environment():
    """audit3 #6. If this ever reverses, the overlay stops reaching real inference.

    The failure would be SILENT in the worst way: the probe in this module reads
    ``os.environ`` and would still go green on the member's key, so resolution would look
    correct while the actual call authenticated as the deployment.
    """
    from zakcode.providers.litellm_provider import LiteLLMProvider

    msgs = [{"role": "user", "content": "hi"}]

    cloud = LiteLLMProvider(model="anthropic/claude-haiku-4-5", api_key="explicit-key")
    assert "api_key" not in cloud._build_kwargs(msgs, None)

    # The other direction, so the assertion above discriminates instead of passing
    # because nothing ever forwards a key: the generic-endpoint case the explicit
    # api_key exists for STILL forwards it.
    generic = LiteLLMProvider(
        model="openai/local-model", api_base="http://127.0.0.1:8080/v1", api_key="explicit-key"
    )
    assert generic._build_kwargs(msgs, None).get("api_key") == "explicit-key"
