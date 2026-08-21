"""``local_only`` must not trust an arbitrary ``api_base``.

The guard classifies by MODEL PREFIX, so before the allowlist any ``openai/*`` model with
an ``api_base`` counted as local — including a base aimed at an LLM gateway that forwards
to metered providers. Measured 2026-08-21 against a litellm gateway fronting
deepinfra/groq/openai: the call was NOT refused.

The allowlist is opt-in: empty keeps the historical behavior so no working config starts
refusing (guard-1562); non-empty makes an unlisted base count as metered.
"""

from __future__ import annotations

import pytest

from zakcode.config import Settings
from zakcode.providers.endpoints import (
    api_base_is_trusted,
    classify_destination,
    is_local_model,
)

POD = "http://10.0.0.250:9090/v1"
GATEWAY = "http://10.63.163.1:4000"


class TestEmptyAllowlistIsBackwardCompatible:
    def test_any_base_is_local_when_allowlist_empty(self) -> None:
        assert is_local_model("openai/zds-qwen3.8-27b", POD, []) is True
        assert is_local_model("openai/zds-qwen3.8-27b", GATEWAY, []) is True

    def test_none_allowlist_matches_empty(self) -> None:
        assert is_local_model("openai/some-model", GATEWAY, None) is True

    def test_default_argument_preserves_old_two_arg_callers(self) -> None:
        assert is_local_model("openai/some-model", POD) is True


class TestAllowlistRefusesUnlistedBase:
    def test_listed_base_is_local(self) -> None:
        ok, reason = classify_destination("openai/zds-qwen3.8-27b", POD, [POD])
        assert ok is True
        assert POD in reason

    def test_unlisted_gateway_is_refused(self) -> None:
        ok, reason = classify_destination("openai/zds-qwen3.8-27b", GATEWAY, [POD])
        assert ok is False
        assert "ZAKCODE_LOCAL_API_BASES" in reason
        assert GATEWAY in reason

    def test_missing_base_is_not_trusted_when_allowlist_set(self) -> None:
        assert api_base_is_trusted(None, [POD]) is False


class TestNormalization:
    @pytest.mark.parametrize(
        "configured,listed",
        [
            (POD, POD + "/"),
            (POD + "/", POD),
            (POD.upper(), POD),
            ("  " + POD + "  ", POD),
        ],
    )
    def test_trailing_slash_case_and_whitespace_do_not_matter(
        self, configured: str, listed: str
    ) -> None:
        assert api_base_is_trusted(configured, [listed]) is True

    def test_blank_entries_do_not_widen_the_allowlist(self) -> None:
        # An all-blank list is a SET allowlist that happens to match nothing, so it
        # trusts nothing. Deliberately fail-CLOSED: the alternative (treat it as unset
        # and trust every base) is the exact fail-open shape this allowlist exists to
        # close. Note ZAKCODE_LOCAL_API_BASES="" parses to [] via the env validator,
        # which IS the unset case — this only arises from programmatic construction.
        assert api_base_is_trusted(GATEWAY, ["", "   "]) is False
        # A blank alongside a real entry is skipped without disturbing the match.
        assert api_base_is_trusted(GATEWAY, ["", POD]) is False
        assert api_base_is_trusted(POD, ["", POD]) is True


class TestUnaffectedClasses:
    def test_named_cloud_prefix_still_metered_regardless_of_allowlist(self) -> None:
        assert is_local_model("groq/llama-3.3-70b-versatile", GATEWAY, [GATEWAY]) is False

    def test_ollama_still_local_regardless_of_allowlist(self) -> None:
        assert is_local_model("ollama_chat/qwen3", None, [POD]) is True

    def test_generic_without_base_still_metered(self) -> None:
        assert is_local_model("openai/gpt-4o", None, [POD]) is False


class TestSettingsPlumbing:
    def test_env_var_parses_as_a_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ZAKCODE_LOCAL_API_BASES", f"{POD}, {GATEWAY}")
        assert Settings().local_api_bases == [POD, GATEWAY]

    def test_defaults_to_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ZAKCODE_LOCAL_API_BASES", raising=False)
        assert Settings().local_api_bases == []

    def test_provider_reads_it_from_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ZAKCODE_LOCAL_API_BASES", POD)
        from zakcode.providers.litellm_provider import LiteLLMProvider

        provider = LiteLLMProvider(settings=Settings(default_model="openai/x", api_base=POD))
        assert provider.local_api_bases == [POD]
