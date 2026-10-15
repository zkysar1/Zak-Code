"""A refusal warning quotes the provider's own message, cut at a word boundary (ADR-0316).

Found on 2026-10-15 by a hostile review of ADR-0303. The warning a learned refusal prints, for
tool use, the temperature, the stop field and the stop limit, a field refused by name (ADR-0181)
and the reasoning-effort remedy (ADR-0188), quoted litellm's whole exception text and cut it at
200 characters wherever the 200th fell. Behind a LiteLLM proxy that text is litellm's wrapper, the
proxy's own wrapper and Bedrock's JSON: ``litellm.BadRequestError: OpenAIException -
litellm.BadRequestError: BedrockException - {"message":"..."}``. Each warning now quotes the
provider's own message, as ADR-0311 reads it, with its whitespace collapsed, and cuts a longer one
at a word boundary with an ellipsis.

Hermetic: no network, no credentials. Each exception is the one litellm's own client raises for
the 400 body, as ``tests/test_refusal_from_message.py`` builds it.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

import pytest

import zakcode.providers.litellm_provider as lp
from tests.test_refusal_from_message import LIVE, Scripted, bare, raised_for
from tests.test_stop_refusal import LIMIT_SENTENCE, NO_STOP_SENTENCE, SENTINELS, TOOL_REPLY, TOOLS
from tests.test_temperature_tools_refusal import (
    GATEWAY,
    HOWS,
    ROUTE,
    TEMPERATURE_SENTENCE,
    TOOL_USE_SENTENCE,
    _run,
    _wrapped,
)
from zakcode.providers import base
from zakcode.providers.litellm_provider import LiteLLMProvider

#: Bedrock's validator refusing a field by name, as ADR-0181's tests measured it.
FIELD_SENTENCE = (
    "Malformed input request: #: extraneous key [seed] is not permitted, please reformat "
    "your input and try again."
)
#: The remedy OpenAI names for function tools with a reasoning effort, as ADR-0188 measured it.
EFFORT_SENTENCE = (
    "Function tools with reasoning_effort are not supported for gpt-5.7-nova in "
    "/v1/chat/completions. To use function tools, use /v1/responses or set reasoning_effort "
    "to 'none'."
)
#: What a backend may add after its sentence; no word in it is longer than 11 characters.
MORE = "The route takes top_p and max_tokens, and refuses every other sampling field. " * 4


@pytest.fixture(autouse=True)
def _no_request_pacing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The provider spaces request starts a second apart; that sleep is not under test."""
    monkeypatch.setattr(lp, "_MIN_REQUEST_INTERVAL_S", 0.0)


def _gateway_provider(**kw: Any) -> LiteLLMProvider:
    return LiteLLMProvider(
        model=ROUTE, api_base=GATEWAY, api_key="sk-test", context_window=131_072, **kw
    )


#: Each refusal that is learned with a warning: its sentence, the provider that learns it, what
#: the call carries, and the gateway's reply to the re-issued call.
REFUSALS = [
    pytest.param(
        TOOL_USE_SENTENCE, lambda: _wrapped("native"), {"tools": TOOLS}, TOOL_REPLY, id="tool-use"
    ),
    pytest.param(
        TEMPERATURE_SENTENCE,
        lambda: _gateway_provider(temperature=0.0),
        {},
        "ok",
        id="temperature",
    ),
    pytest.param(NO_STOP_SENTENCE, _gateway_provider, {"stop": SENTINELS}, "ok", id="no-stop"),
    pytest.param(LIMIT_SENTENCE, _gateway_provider, {"stop": SENTINELS}, "ok", id="stop-limit"),
    pytest.param(
        FIELD_SENTENCE, lambda: _gateway_provider(extra_body={"seed": 42}), {}, "ok", id="field"
    ),
]


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


async def _quote_in_the_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    body: dict[str, Any],
    provider: Any,
    how: str,
    reply: str,
    **carried: Any,
) -> str:
    """One call the gateway answers with ``body`` and then ``reply``: what the one warning it
    printed quotes, the text in parentheses before its dash."""
    exc = await raised_for(monkeypatch, body)
    monkeypatch.setattr(lp.litellm, "acompletion", gateway := Scripted(exc, reply))
    caplog.set_level(logging.WARNING, logger="zakcode.providers")

    await _run(provider, how, **carried)

    # The premise: litellm's text has its wrapper in front of the message. And it was learned.
    assert str(exc).startswith("litellm.BadRequestError: ")
    assert len(gateway.requests) == 2
    [warning] = _warnings(caplog)
    found = re.search(r" \((.*)\) — ", warning)
    assert found is not None, warning
    return found.group(1)


# ── the warnings ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("sentence, make, carried, reply", REFUSALS)
@pytest.mark.parametrize("shape", LIVE)
@pytest.mark.parametrize("how", HOWS)
async def test_a_refusal_warning_quotes_the_providers_own_message(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    how: str,
    shape: Callable[[str], dict[str, Any]],
    sentence: str,
    make: Callable[[], Any],
    carried: dict[str, Any],
    reply: str,
) -> None:
    quoted = await _quote_in_the_warning(
        monkeypatch, caplog, shape(sentence), make(), how, reply, **carried
    )

    assert quoted == sentence


@pytest.mark.parametrize("shape", LIVE)
async def test_the_effort_remedy_warning_quotes_the_providers_own_message(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    shape: Callable[[str], dict[str, Any]],
) -> None:
    """The buffered path is the one that learns this remedy from the provider's words."""
    quoted = await _quote_in_the_warning(
        monkeypatch,
        caplog,
        shape(EFFORT_SENTENCE),
        _gateway_provider(),
        "acomplete",
        "ok",
        tools=TOOLS,
    )

    assert quoted == EFFORT_SENTENCE


@pytest.mark.parametrize("how", HOWS)
async def test_a_long_message_is_quoted_up_to_a_word_boundary_with_an_ellipsis(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, how: str
) -> None:
    message = f"{TEMPERATURE_SENTENCE} {MORE}"

    quoted = await _quote_in_the_warning(
        monkeypatch, caplog, bare(message), _gateway_provider(temperature=0.0), how, "ok"
    )

    flat = " ".join(message.split())
    assert quoted.endswith("…")
    assert len(quoted) <= base.PROVIDER_QUOTE_MAX_CHARS
    # Cut after a whole word, and after the last word that fits.
    assert flat.startswith(quoted[:-1])
    assert flat[len(quoted) - 1] == " "
    assert len(quoted) > base.PROVIDER_QUOTE_MAX_CHARS - 12


# ── the quote ────────────────────────────────────────────────────────────────────────────────


def test_a_short_message_is_quoted_whole_with_its_whitespace_collapsed() -> None:
    text = "litellm.BadRequestError: OpenAIException - This model\n  doesn't support tool use. "

    assert base.quoted_provider_message(text) == TOOL_USE_SENTENCE


@pytest.mark.parametrize(
    "message, quoted",
    [
        pytest.param("x" * 150 + " " + "y" * 100, "x" * 150 + "…", id="inside-a-word"),
        pytest.param(
            "x" * 99 + " " + "x" * 99 + " " + "y" * 10,
            "x" * 99 + " " + "x" * 99 + "…",
            id="at-a-word-boundary",
        ),
        pytest.param("x" * 300, "x" * 199 + "…", id="one-long-word"),
        pytest.param("x" * 200, "x" * 200, id="exactly-the-limit"),
    ],
)
def test_a_message_past_the_limit_is_cut_at_its_last_word_boundary(
    message: str, quoted: str
) -> None:
    assert base.PROVIDER_QUOTE_MAX_CHARS == 200
    assert base.quoted_provider_message(message) == quoted


def test_the_quote_is_taken_from_the_whole_message_not_the_bounded_one() -> None:
    """The quote is taken from the whole message, not from the bounded one the readers read: a
    message whose first 400 characters are mostly whitespace is quoted whole, and is marked as
    cut only where it was cut."""
    message = "word" + " " * 400 + "more"

    assert base.quoted_provider_message(message) == "word more"
    assert base.quoted_provider_message(message + " " + "z" * 300) == "word more…"
