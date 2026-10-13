"""Session-affinity routing key (`prompt_cache_key`) + llama.cpp context-error mapping.

WHY THIS FILE EXISTS
The zds inference pod routes each conversation to the engine already holding
its KV-cache prefix. Without an explicit key it infers one by fingerprinting
the message head — an inference that has produced two measured production
incidents (same-template collision; volatile-tail churn). The OpenAI-standard
``prompt_cache_key`` body param removes the inference entirely, and OpenAI
itself uses the same field for the same cache-routing purpose, so one request
body works against every OpenAI-compatible vendor with only the URL changing.

Two wire-level constraints pin the implementation:

* It must ride ``extra_body`` — ``drop_params=True`` silently discards unknown
  TOP-LEVEL kwargs, so a top-level ``prompt_cache_key`` would vanish without
  an error (the exact failure mode drop_params exists to create).
* It must be scoped to OpenAI-compatible destinations — other clouds reject
  unknown body params outright.

Separately: llama.cpp reports context overflow as a plain 400 whose phrasing
("request (N tokens) exceeds the available context size (M tokens)") is NOT in
litellm's context-window sniff list (probed 2026-08-28: recognized False), so
it mapped to a generic BadRequestError and the agent loop's compact-and-retry
recovery never fired — measured the same day as 8 failed turns that
should each have been a silent compaction.
"""

import pytest

from zakcode.config import Settings
from zakcode.providers.base import ContextWindowExceeded
from zakcode.providers.litellm_provider import LiteLLMProvider

MSGS = [{"role": "user", "content": "hi"}]


def test_prompt_cache_key_rides_extra_body_for_generic_endpoint() -> None:
    p = LiteLLMProvider(
        model="openai/zds-qwen3.6-35b", api_base="http://10.0.0.205:9090/v1", context_window=131072
    )
    kwargs = p._build_kwargs(MSGS, None, prompt_cache_key="zakcode/sess-1")
    assert kwargs["extra_body"]["prompt_cache_key"] == "zakcode/sess-1"
    # NEVER top-level: drop_params would silently discard it there.
    assert "prompt_cache_key" not in kwargs


def test_prompt_cache_key_preserves_configured_extra_body() -> None:
    p = LiteLLMProvider(
        model="openai/zds-qwen3.6-35b",
        api_base="http://10.0.0.205:9090/v1",
        extra_body={"reasoning_budget": 0},
        context_window=131072,
    )
    kwargs = p._build_kwargs(MSGS, None, prompt_cache_key="zakcode/sess-2")
    assert kwargs["extra_body"]["reasoning_budget"] == 0
    assert kwargs["extra_body"]["prompt_cache_key"] == "zakcode/sess-2"
    # The configured mapping is copied, not mutated.
    assert p.extra_body == {"reasoning_budget": 0}


def test_prompt_cache_key_omitted_for_named_cloud_providers() -> None:
    """anthropic/... rejects unknown body params — the key must not reach it."""
    p = LiteLLMProvider(model="anthropic/claude-sonnet-4-5")
    kwargs = p._build_kwargs(MSGS, None, prompt_cache_key="zakcode/sess-3")
    assert "prompt_cache_key" not in (kwargs.get("extra_body") or {})
    assert "prompt_cache_key" not in kwargs


def test_prompt_cache_key_absent_when_not_passed() -> None:
    """The default request shape stays byte-identical to before the feature."""
    p = LiteLLMProvider(
        model="openai/zds-qwen3.6-35b", api_base="http://10.0.0.205:9090/v1", context_window=131072
    )
    kwargs = p._build_kwargs(MSGS, None)
    assert "extra_body" not in kwargs


def test_prompt_cache_key_stays_out_once_the_provider_was_refused_it() -> None:
    """ADR-0285: the key is added AFTER the refused-field filter, so it must read that filter's
    list itself — otherwise the call re-issued after a refusal carried the key straight back."""
    p = LiteLLMProvider(
        model="openai/zds-qwen3.6-35b", api_base="http://10.0.0.205:9090/v1", context_window=131072
    )
    p.rejected_request_fields.append("prompt_cache_key")
    kwargs = p._build_kwargs(MSGS, None, prompt_cache_key="zakcode/sess-4")
    assert "extra_body" not in kwargs


def _switched_off(**extra: object) -> Settings:
    return Settings(
        default_model="openai/zds-qwen3.6-35b",
        api_base="http://10.0.0.205:9090/v1",
        context_window=131072,
        send_prompt_cache_key=False,
        _env_file=None,
        **extra,  # type: ignore[arg-type]
    )


def test_prompt_cache_key_switch_is_on_by_default_and_reads_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert Settings(_env_file=None).send_prompt_cache_key is True
    monkeypatch.setenv("ZAKCODE_SEND_PROMPT_CACHE_KEY", "false")
    assert Settings(_env_file=None).send_prompt_cache_key is False


def test_prompt_cache_key_is_not_sent_when_the_switch_is_off() -> None:
    p = LiteLLMProvider(_switched_off())
    assert p.send_prompt_cache_key is False
    kwargs = p._build_kwargs(MSGS, None, prompt_cache_key="zakcode/sess-5")
    assert "extra_body" not in kwargs


def test_the_switch_off_leaves_the_rest_of_the_configured_body_alone() -> None:
    p = LiteLLMProvider(_switched_off(extra_body={"reasoning_budget": 0}))
    kwargs = p._build_kwargs(MSGS, None, prompt_cache_key="zakcode/sess-6")
    assert kwargs["extra_body"] == {"reasoning_budget": 0}


def test_a_provider_built_without_settings_still_sends_the_key() -> None:
    """The switch is read from settings only, so every provider built the old way is unchanged."""
    p = LiteLLMProvider(
        model="openai/zds-qwen3.6-35b", api_base="http://10.0.0.205:9090/v1", context_window=131072
    )
    assert p.send_prompt_cache_key is True


def test_llama_cpp_context_overflow_maps_to_context_window_exceeded() -> None:
    """The exact phrasing a llama.cpp pod's engines emit, wrapped the way litellm
    surfaces it. Must map to ContextWindowExceeded so the loop's
    compact-and-retry recovery fires instead of failing the turn."""
    exc = Exception(
        "litellm.BadRequestError: OpenAIException - request (131103 tokens) "
        "exceeds the available context size (131072 tokens), try increasing it"
    )
    mapped = LiteLLMProvider._map_error(exc)
    assert isinstance(mapped, ContextWindowExceeded)


def test_loop_key_is_per_session_by_default_and_per_workspace_when_identity_is_stable() -> None:
    """The key is an input the endpoint acts on (ADR-0157 fourth addendum): under
    stable_prompt_identity it must not vary with the session, and must still separate workspaces."""
    from pathlib import Path
    from types import SimpleNamespace

    from zakcode.agent.loop import AgentLoop

    stub = SimpleNamespace(
        settings=SimpleNamespace(stable_prompt_identity=False),
        session=SimpleNamespace(id="abc"),
        workspace_root=Path("/w"),
    )
    assert AgentLoop._prompt_cache_key(stub) == "zakcode/abc"
    stub.settings.stable_prompt_identity = True
    stable = AgentLoop._prompt_cache_key(stub)
    assert stable.startswith("zakcode/ws-") and stable != "zakcode/abc"
    stub.session.id = "def"
    assert AgentLoop._prompt_cache_key(stub) == stable  # a new session: same key
    stub.workspace_root = Path("/elsewhere")
    assert AgentLoop._prompt_cache_key(stub) != stable  # a different workspace: a different key


def test_every_provider_call_site_in_the_loop_uses_the_helper() -> None:
    """Three call sites carried the literal f-string; a fourth added the same way would silently
    reintroduce the per-run variation under the flag."""
    from pathlib import Path

    loop_py = Path(__file__).resolve().parent.parent / "src/zakcode/agent/loop.py"
    src = loop_py.read_text(encoding="utf-8")
    without_helper = src.replace('        return f"zakcode/{self.session.id}"', "")
    assert 'f"zakcode/{self.session.id}"' not in without_helper
    # Three conversation calls (buffered, streamed, summarizer) and, since ADR-0257, three side
    # calls on the loop's own model (critic, plan critique, quality gate).
    assert src.count("prompt_cache_key=self._prompt_cache_key()") == 6
