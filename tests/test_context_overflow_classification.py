"""A context overflow that reaches us in a gateway's words is classified as one (ADR-0293).

A LiteLLM proxy in front of Bedrock speaks the OpenAI wire, so litellm sniffs the error body
with its ``openai`` list, which does not know Bedrock's phrasing, and the overflow arrives as a
plain ``BadRequestError``. That maps to ``RequestFailed``: the loop's compact-and-retry never
runs and the turn ends ``provider_error`` where one compaction would have saved it.

These tests build the exception the way the live path does (the openai SDK's error for the
status, run through litellm's own ``exception_type`` for provider ``openai``) and check what
zakcode makes of it. Hermetic: no network, no credentials, and the host name is the reserved
``.invalid`` one.
"""

from __future__ import annotations

from typing import Any

import httpx
import litellm
import openai
import pytest

import zakcode.providers.litellm_provider as lp
from zakcode.messages import Message
from zakcode.providers.base import (
    CONTEXT_OVERFLOW_MARKERS,
    ContextWindowExceeded,
    ProviderUnavailable,
    RateLimited,
    RequestFailed,
    context_overflow_marker,
)
from zakcode.providers.litellm_provider import LiteLLMProvider

_SDK_ERRORS = {
    400: openai.BadRequestError,
    429: openai.RateLimitError,
    500: openai.InternalServerError,
}


def gateway_error(status: int, message: str) -> Exception:
    """The exception ``litellm.acompletion`` raises for a gateway's error body.

    The openai SDK builds its error from the response; litellm's wrapper then runs
    ``exception_type`` for the route's provider (``openai`` for an ``openai/<alias>`` model
    behind an ``api_base``) and raises the result. Both steps are the real ones.
    """
    body = {"error": {"message": message, "type": None, "param": None, "code": str(status)}}
    request = httpx.Request("POST", "https://gateway.invalid/v1/chat/completions")
    response = httpx.Response(status, request=request, json=body)
    sdk_error = _SDK_ERRORS[status](
        message=f"Error code: {status} - {body}", response=response, body=body["error"]
    )
    try:
        mapped = litellm.exception_type(
            model="Claude4.5",
            original_exception=sdk_error,
            custom_llm_provider="openai",
            completion_kwargs={},
            extra_kwargs={},
        )
    except Exception as raised:  # noqa: BLE001 — litellm raises the mapped exception
        return raised
    return mapped


def bedrock_body(inner: str) -> str:
    """A LiteLLM proxy's body for a Bedrock refusal: its own prefix, the vendor JSON, and the
    router's trailer."""
    return (
        "litellm.BadRequestError: BedrockException - "
        f'{{"message":"{inner}"}}. Received Model Group=Claude4.5\n'
        "Available Model Group Fallbacks=None"
    )


INPUT_TOO_LONG = bedrock_body(
    "The model returned the following errors: Input is too long for requested model."
)
PROMPT_TOO_LONG = bedrock_body(
    "The model returned the following errors: prompt is too long: 213456 tokens > 200000 maximum"
)
TOO_MANY_INPUT_TOKENS = bedrock_body(
    "Too many input tokens. Max input tokens: 8192, request input token count: 9100"
)
# The proxy ran litellm's own mapping on its side and wrote the class name into the body.
PROXY_CLASS_NAME = INPUT_TOO_LONG.replace(
    "litellm.BadRequestError:", "litellm.ContextWindowExceededError: litellm.BadRequestError:", 1
)

#: The wordings a gateway in front of Bedrock sends for an overflow, and that litellm's
#: ``openai`` route does not name (``test_litellm_leaves_these_wordings_a_plain_bad_request``).
UNNAMED_BY_LITELLM = [
    pytest.param(PROXY_CLASS_NAME, id="proxy-wrote-its-class-name"),
    pytest.param(INPUT_TOO_LONG, id="bedrock-input-is-too-long"),
    pytest.param(PROMPT_TOO_LONG, id="anthropic-prompt-is-too-long"),
    pytest.param(TOO_MANY_INPUT_TOKENS, id="bedrock-too-many-input-tokens"),
]


@pytest.mark.parametrize("message", UNNAMED_BY_LITELLM)
def test_litellm_leaves_these_wordings_a_plain_bad_request(message: str) -> None:
    """The premise of the fix, measured: litellm 1.86.2 does not name these as overflows for
    an ``openai`` route. If this fails, litellm has learned the wording and the matching
    marker in ``CONTEXT_OVERFLOW_MARKERS`` may be retired, but only after this file's other
    tests are rerun without it."""
    mapped = gateway_error(400, message)
    assert type(mapped).__name__ == "BadRequestError", type(mapped).__name__


@pytest.mark.parametrize("message", UNNAMED_BY_LITELLM)
def test_a_gateway_overflow_litellm_did_not_name_is_a_context_overflow(message: str) -> None:
    mapped = LiteLLMProvider._map_error(gateway_error(400, message))
    assert isinstance(mapped, ContextWindowExceeded), type(mapped).__name__


@pytest.mark.parametrize(
    "message",
    [
        pytest.param(
            "This model's maximum context length is 8192 tokens. However, you requested 9000 "
            "tokens (8000 in the messages, 1000 in the completion).",
            id="openai-chat-completions-sentence",
        ),
        pytest.param(
            "Your input exceeds the context window of this model. Please adjust your input "
            "and try again.",
            id="openai-responses-sentence",
        ),
        pytest.param(
            "input length and `max_tokens` exceed context limit: 195000 + 8192 > 200000, "
            "decrease input length or max_tokens and try again",
            id="anthropic-max-tokens-does-not-fit",
        ),
    ],
)
def test_the_overflows_litellm_names_still_classify(message: str) -> None:
    mapped = LiteLLMProvider._map_error(gateway_error(400, message))
    assert isinstance(mapped, ContextWindowExceeded), type(mapped).__name__


def test_the_llama_cpp_wording_is_still_an_overflow() -> None:
    """The phrasing the inline check handled before the markers moved into one list."""
    exc = Exception(
        "litellm.BadRequestError: OpenAIException - request (131103 tokens) "
        "exceeds the available context size (131072 tokens), try increasing it"
    )
    assert isinstance(LiteLLMProvider._map_error(exc), ContextWindowExceeded)


def test_an_overflow_reported_as_a_server_error_is_an_overflow_not_an_outage() -> None:
    """A gateway can report a provider's 400 as a 500. Retried as an outage it would fail the
    same way every time, so the text wins over the status class."""
    mapped = LiteLLMProvider._map_error(gateway_error(500, INPUT_TOO_LONG))
    assert isinstance(mapped, ContextWindowExceeded), type(mapped).__name__


# ── what must NOT become an overflow ──────────────────────────────────────────────────────────


def test_a_token_rate_throttle_is_a_throttle() -> None:
    """Bedrock's token-rate refusal says "Too many tokens"; compacting cures nothing here."""
    error = gateway_error(
        429,
        "litellm.RateLimitError: BedrockException - "
        '{"message":"Too many tokens, please wait before trying again."}',
    )
    mapped = LiteLLMProvider._map_error(error)
    assert isinstance(mapped, RateLimited)
    assert not isinstance(mapped, ContextWindowExceeded)


def test_a_throttle_is_never_an_overflow_whatever_its_text_says() -> None:
    """The check sits after the rate-limit type: a 429 that happens to quote an overflow
    phrase is still waited out, not compacted."""
    error = gateway_error(
        429, "Rate limit reached. Your last request said: prompt is too long. Retry in 20s."
    )
    mapped = LiteLLMProvider._map_error(error)
    assert isinstance(mapped, RateLimited)
    assert not isinstance(mapped, ContextWindowExceeded)


def test_an_unrelated_bad_request_stays_a_request_failure() -> None:
    error = gateway_error(
        400,
        bedrock_body(
            "The model returned the following errors: messages.0.content: "
            "Input should be a valid list"
        ),
    )
    assert type(LiteLLMProvider._map_error(error)) is RequestFailed


def test_a_plain_server_error_is_still_waited_out_as_unavailable() -> None:
    mapped = LiteLLMProvider._map_error(gateway_error(500, "upstream connect error"))
    assert isinstance(mapped, ProviderUnavailable)


def test_no_marker_matches_a_known_throttle_sentence() -> None:
    """Every phrase here is a refusal that compaction cannot cure. The list is built from
    wordings that name the INPUT's size; this keeps a later addition from reading as one."""
    throttles = [
        "Too many tokens, please wait before trying again.",
        "Too many requests, please wait before trying again.",
        "Rate limit reached for gpt-4 in organization org-x on tokens per min (TPM): "
        "Limit 30000, Used 25000, Requested 9000. Please try again in 10s.",
        "This request would exceed your organization's rate limit of 30,000 input tokens "
        "per minute.",
        "You exceeded your current quota, please check your plan and billing details.",
    ]
    for sentence in throttles:
        assert context_overflow_marker(sentence) is None, sentence


# ── the marker list itself ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("marker", CONTEXT_OVERFLOW_MARKERS)
def test_every_marker_is_found_whatever_its_case(marker: str) -> None:
    assert marker == marker.lower()
    assert (
        context_overflow_marker(f"upstream said: {marker.upper()} (see the model card)") == marker
    )


def test_the_marker_reads_an_exception_as_well_as_text() -> None:
    assert (
        context_overflow_marker(ValueError("Prompt is too long: 9 tokens")) == "prompt is too long"
    )
    assert context_overflow_marker("fine") is None


# ── the whole provider path, buffered and streamed ────────────────────────────────────────────


def _provider() -> LiteLLMProvider:
    # A gateway alias litellm has no metadata for: the window is declared, as an operator's
    # config entry declares it (ADR-0066), because the provider refuses to guess one.
    return LiteLLMProvider(
        model="openai/Claude4.5",
        api_key="sk-test",
        api_base="https://gateway.invalid/v1",
        context_window=200000,
        temperature=0.0,
    )


def _raising(error: BaseException) -> Any:
    async def _acompletion(**_kwargs: Any) -> Any:
        raise error

    return _acompletion


async def test_acomplete_raises_the_overflow_the_loop_recovers_from(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lp.litellm, "acompletion", _raising(gateway_error(400, INPUT_TOO_LONG)))
    with pytest.raises(ContextWindowExceeded) as raised:
        await _provider().acomplete([Message.user("hi")])
    assert "Input is too long for requested model" in str(raised.value)


async def test_astream_raises_the_overflow_the_loop_recovers_from(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lp.litellm, "acompletion", _raising(gateway_error(400, PROMPT_TOO_LONG)))
    with pytest.raises(ContextWindowExceeded) as raised:
        async for _event in _provider().astream([Message.user("hi")]):
            pass
    assert "prompt is too long" in str(raised.value)


def test_the_message_that_propagates_is_redacted() -> None:
    """The verdict reads the redacted text, and so does everything downstream of it."""
    leaked = "sk-test-0123456789abcdefghijklmnop"
    message = bedrock_body(f"Input is too long for requested model; api_key={leaked} was echoed")
    mapped = LiteLLMProvider._map_error(gateway_error(400, message))
    assert isinstance(mapped, ContextWindowExceeded)
    assert leaked not in str(mapped)
    assert "[REDACTED]" in str(mapped)
