"""Prompt-cache emission + cache-token accounting (parity #2).

cache_control content blocks are stamped on the stable system prefix for caching-capable
Anthropic models (and only those — OpenAI caches automatically and would reject the
annotation); cache_read/cache_creation tokens are read from the response usage for any
backend. No network: litellm-shaped dicts through the provider's pure builders.
"""

from __future__ import annotations

from zakcode.agent.prompt import DYNAMIC_BOUNDARY
from zakcode.providers.litellm_provider import (
    _CACHE_BOUNDARY,
    LiteLLMProvider,
    _supports_cache_control,
)
from zakcode.usage import Usage

_SYSTEM = f"STABLE identity + tool guidance\n\n{DYNAMIC_BOUNDARY}\n\nDYNAMIC git status here"


# ── the boundary constant must not drift from the agent's marker ──────────────


def test_cache_boundary_matches_prompt_module() -> None:
    assert _CACHE_BOUNDARY == DYNAMIC_BOUNDARY


# ── Usage cache fields ────────────────────────────────────────────────────────


def test_usage_cache_fields_add() -> None:
    a = Usage(prompt_tokens=100, cache_read_tokens=80, cache_creation_tokens=20)
    b = Usage(prompt_tokens=50, cache_read_tokens=40, cache_creation_tokens=0)
    total = a + b
    assert total.cache_read_tokens == 120
    assert total.cache_creation_tokens == 20
    assert total.prompt_tokens == 150


def test_usage_cache_fields_default_zero() -> None:
    u = Usage(prompt_tokens=10)
    assert u.cache_read_tokens == 0 and u.cache_creation_tokens == 0


# ── _extract_usage reads both vendor shapes ───────────────────────────────────


def test_extract_usage_anthropic_cache_shape() -> None:
    resp = {
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 20,
            "total_tokens": 1020,
            "cache_read_input_tokens": 900,
            "cache_creation_input_tokens": 100,
        },
    }
    u = LiteLLMProvider._extract_usage(resp)
    assert u.cache_read_tokens == 900
    assert u.cache_creation_tokens == 100


def test_extract_usage_openai_nested_cached_tokens() -> None:
    resp = {
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 500,
            "completion_tokens": 10,
            "total_tokens": 510,
            "prompt_tokens_details": {"cached_tokens": 384},
        },
    }
    u = LiteLLMProvider._extract_usage(resp)
    assert u.cache_read_tokens == 384
    assert u.cache_creation_tokens == 0


def test_extract_usage_no_cache_fields_is_zero() -> None:
    resp = {
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
    u = LiteLLMProvider._extract_usage(resp)
    assert u.cache_read_tokens == 0 and u.cache_creation_tokens == 0


# ── cache_control emission gating ─────────────────────────────────────────────


def _system_blocks(provider: LiteLLMProvider, system: str) -> object:
    wire = provider._translate_messages([], system)
    kwargs = provider._build_kwargs(wire, tools=None)
    return kwargs["messages"][0]["content"]


def test_anthropic_emits_cache_control_on_stable_block() -> None:
    provider = LiteLLMProvider(model="anthropic/claude-sonnet-4-6", temperature=0.0)
    content = _system_blocks(provider, _SYSTEM)
    assert isinstance(content, list) and len(content) == 2
    stable, dynamic = content
    assert stable["cache_control"] == {"type": "ephemeral"}
    assert "STABLE identity" in stable["text"]
    assert stable["text"].rstrip().endswith(DYNAMIC_BOUNDARY)  # boundary rides the stable block
    assert "cache_control" not in dynamic  # dynamic tier is never cached
    assert "DYNAMIC git status" in dynamic["text"]


def test_openai_leaves_system_a_plain_string() -> None:
    provider = LiteLLMProvider(model="openai/gpt-4o", temperature=0.0)
    content = _system_blocks(provider, _SYSTEM)
    assert isinstance(content, str)  # no cache_control annotation for OpenAI
    assert content == _SYSTEM


def test_groq_leaves_system_a_plain_string() -> None:
    provider = LiteLLMProvider(model="groq/llama-3.3-70b-versatile", temperature=0.0)
    assert isinstance(_system_blocks(provider, _SYSTEM), str)


def test_anthropic_without_boundary_left_plain() -> None:
    provider = LiteLLMProvider(model="anthropic/claude-sonnet-4-6", temperature=0.0)
    content = _system_blocks(provider, "a bare system prompt with no boundary marker")
    assert isinstance(content, str)  # nothing to split -> no caching annotation


def test_supports_cache_control_gate() -> None:
    assert _supports_cache_control("anthropic/claude-sonnet-4-6") is True
    assert _supports_cache_control("claude-opus-4-8") is True
    assert _supports_cache_control("openai/gpt-4o") is False
    assert _supports_cache_control("groq/llama-3.3-70b-versatile") is False
    assert _supports_cache_control("ollama_chat/llama3.1") is False
