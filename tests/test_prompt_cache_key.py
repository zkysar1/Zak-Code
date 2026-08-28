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
recovery never fired — measured on zakpod1 the same day as 8 failed turns that
should each have been a silent compaction.
"""
from zakcode.providers.base import ContextWindowExceeded
from zakcode.providers.litellm_provider import LiteLLMProvider

MSGS = [{"role": "user", "content": "hi"}]


def test_prompt_cache_key_rides_extra_body_for_generic_endpoint() -> None:
    p = LiteLLMProvider(model="openai/zds-qwen3.6-35b", api_base="http://10.0.0.205:9090/v1", context_window=131072)
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
    p = LiteLLMProvider(model="openai/zds-qwen3.6-35b", api_base="http://10.0.0.205:9090/v1", context_window=131072)
    kwargs = p._build_kwargs(MSGS, None)
    assert "extra_body" not in kwargs


def test_llama_cpp_context_overflow_maps_to_context_window_exceeded() -> None:
    """The exact phrasing zakpod1's engines emit, wrapped the way litellm
    surfaces it. Must map to ContextWindowExceeded so the loop's
    compact-and-retry recovery fires instead of failing the turn."""
    exc = Exception(
        "litellm.BadRequestError: OpenAIException - request (131103 tokens) "
        "exceeds the available context size (131072 tokens), try increasing it"
    )
    mapped = LiteLLMProvider._map_error(exc)
    assert isinstance(mapped, ContextWindowExceeded)
