"""Tri-provider metadata: registry entries, key detection, response shapes, cost.

Audit P0-1 (a-e). Test names match the audit's acceptance list verbatim
(`test_anthropic_registry`, `test_groq_registry`, `test_provider_key_status`,
`test_anthropic_cost`). Response-shape tests feed litellm-SHAPED plain dicts through
the normalizers — `_get` reads dicts and attribute objects alike — so no network or
key is ever needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zakcode.cli import _PROVIDER_KEY_ENV, _provider_key_status
from zakcode.providers.litellm_provider import LiteLLMProvider
from zakcode.providers.registry import _lookup_static, get_capabilities

# ── registry statics (audit P0-1a; acceptance 1 + 2) ─────────────────────────


def test_anthropic_registry() -> None:
    """Static lookup pins the standard 200k window for the Claude 4.x family.

    Deliberately 200k, NOT litellm's advertised 1M long-context beta: the runtime
    sends no beta header, so budgeting compaction against 1M would overflow the real
    request limit. The static entry must win over the litellm fallback.
    """
    for model in (
        "claude-sonnet-4-20250514",
        "anthropic/claude-sonnet-4-20250514",
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    ):
        static = _lookup_static(model)
        assert static is not None, f"no static registry entry resolves {model!r}"
        assert static.context_window == 200_000
        assert static.supports_tools and static.supports_vision and static.supports_caching
        # Resolution order is static-first, so the public lookup agrees.
        assert get_capabilities(model).context_window == 200_000


def test_groq_registry() -> None:
    """Static lookup returns the real (128k+) windows for current Groq models."""
    caps = _lookup_static("groq/llama-3.3-70b-versatile")
    assert caps is not None
    assert caps.context_window >= 128_000
    assert caps.max_output == 32_768
    assert caps.supports_tools
    for model in ("groq/llama-3.1-8b-instant", "groq/openai/gpt-oss-120b", "groq/qwen/qwen3-32b"):
        static = _lookup_static(model)
        assert static is not None, f"no static registry entry resolves {model!r}"
        assert static.context_window >= 128_000
        assert static.supports_tools
    assert get_capabilities("groq/llama-3.3-70b-versatile").context_window >= 128_000


# ── CLI key detection (audit P0-1c; acceptance 3) ────────────────────────────


def test_provider_key_status(monkeypatch) -> None:
    """The info panel detects ANTHROPIC_API_KEY and GROQ_API_KEY (presence only)."""
    assert "ANTHROPIC_API_KEY" in _PROVIDER_KEY_ENV
    assert "GROQ_API_KEY" in _PROVIDER_KEY_ENV
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    status = _provider_key_status()
    assert status["ANTHROPIC_API_KEY"] is True
    assert status["GROQ_API_KEY"] is True
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.delenv("GROQ_API_KEY")
    status = _provider_key_status()
    assert status["ANTHROPIC_API_KEY"] is False
    assert status["GROQ_API_KEY"] is False


# ── docs (audit P0-1b; acceptance 4) ─────────────────────────────────────────


def test_env_example_documents_all_three_providers() -> None:
    text = (Path(__file__).resolve().parents[1] / ".env.example").read_text(encoding="utf-8")
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY"):
        assert name in text, f".env.example is missing a commented {name} example"
    assert "anthropic/" in text and "groq/" in text and "openai/" in text


# ── provider-shaped response normalization (audit P0-1d) ─────────────────────


def _response(message: dict, *, usage: dict | None = None, cost: float | None = None) -> dict:
    doc: dict = {"choices": [{"message": message, "finish_reason": "stop"}]}
    if usage is not None:
        doc["usage"] = usage
    if cost is not None:
        doc["_hidden_params"] = {"response_cost": cost}
    return doc


def test_anthropic_thinking_kept_out_of_text() -> None:
    """Extended thinking (litellm: ``reasoning_content``) lands on ``LLMResult.thinking``
    and never pollutes the assistant ``text`` or the tool calls."""
    result = LiteLLMProvider._normalize(
        _response(
            {
                "content": "The answer is 4.",
                "reasoning_content": "2+2: carry nothing, it's 4.",
                "tool_calls": [
                    {
                        "id": "t1",
                        "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
                    }
                ],
            }
        )
    )
    assert result.text == "The answer is 4."
    assert result.thinking == "2+2: carry nothing, it's 4."
    assert "carry nothing" not in result.text
    assert [c.name for c in result.tool_calls] == ["read_file"]
    assert result.tool_calls[0].arguments == {"path": "a.txt"}


def test_groq_usage_metadata_parses() -> None:
    """Groq rides extra usage fields (queue/prompt/completion times, ``x_groq``) on the
    OpenAI shape; the normalizer must take the token counts and ignore the rest."""
    result = LiteLLMProvider._normalize(
        {
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 25,
                "completion_tokens": 7,
                "total_tokens": 32,
                "queue_time": 0.018,
                "prompt_time": 0.002,
                "completion_time": 0.011,
                "total_time": 0.013,
            },
            "x_groq": {"id": "req_01jq"},
        }
    )
    assert result.text == "hi"
    assert result.usage.prompt_tokens == 25
    assert result.usage.completion_tokens == 7
    assert result.usage.total_tokens == 32


def test_openai_shape_still_normalizes_without_thinking() -> None:
    """Control case: a plain OpenAI-shaped response leaves ``thinking`` empty."""
    result = LiteLLMProvider._normalize(
        _response({"content": "plain"}, usage={"prompt_tokens": 3, "completion_tokens": 1})
    )
    assert result.text == "plain"
    assert result.thinking == ""
    assert result.usage.total_tokens == 4  # derived from the parts


def test_streaming_reasoning_delta_yields_no_text_event() -> None:
    """A streamed reasoning fragment (``delta.reasoning_content``) must not surface as
    assistant text; a later real content delta still streams normally."""
    events, _fr = LiteLLMProvider._parse_chunk(
        {"choices": [{"delta": {"reasoning_content": "mulling it over"}}]}
    )
    assert events == []
    events, _fr = LiteLLMProvider._parse_chunk({"choices": [{"delta": {"content": "4"}}]})
    assert len(events) == 1 and events[0].text == "4"


# ── cost accounting (audit P0-1e; acceptance 10) ─────────────────────────────


def test_anthropic_cost() -> None:
    """Cost extraction returns the nonzero litellm-computed cost for an Anthropic response."""
    result = LiteLLMProvider._normalize(
        _response(
            {"content": "ok", "reasoning_content": "brief"},
            usage={"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200},
            cost=0.006,
        )
    )
    assert result.usage.cost_usd == pytest.approx(0.006)
    assert result.usage.cost_usd > 0


def test_groq_cost() -> None:
    """Same for a Groq response (litellm's pricing DB covers Groq — probed 2026-06-10)."""
    result = LiteLLMProvider._normalize(
        _response(
            {"content": "ok"},
            usage={"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200},
            cost=0.000748,
        )
    )
    assert result.usage.cost_usd == pytest.approx(0.000748)
    assert result.usage.cost_usd > 0
