"""The per-call output cap is clamped to the room the window leaves (ADR-0308).

Reported 2026-10-14, through a LiteLLM 1.9x gateway in front of Bedrock/Vertex: a route
whose window is 32,768 tokens adds the requested output to the prompt and refuses the sum,
"This model's maximum context length is 32768 tokens. However, you requested 8192 output
tokens and your prompt contains at least 24577 input tokens ...". The compaction threshold
on that window is 27,034 (the window, less the 5% margin, less the 4,096 answer room of an
unmapped alias), so a prompt between 24,577 and 27,034 tokens went out uncompacted with the
8,192 default cap and was refused; the loop then compacted and retried (#767), one wasted
request each time. Now the provider sends no more output than the window has room for: the
window, less the prompt's token count, less that same 5% margin, never under a floor.

The count is litellm's tokenizer, as ``count_tokens`` makes it, not four characters a token:
on id-dense tool output that estimate runs well under the truth, which would leave the refused
cap as it was. Most tests here stand a deterministic tokenizer in for litellm's, one token a
character, so the numbers are exact; one uses the real tokenizer on dense content.

Hermetic: ``litellm.acompletion`` is faked by a stand-in route that counts the prompt with
the same tokenizer and refuses with the measured wording when prompt plus requested output
passes its window. litellm's tokenizer reads encodings bundled with it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace
from typing import Any

import litellm
import pytest

import zakcode.providers.litellm_provider as lp
from zakcode.messages import Message
from zakcode.providers.base import ContextWindowExceeded, StreamTextDelta
from zakcode.providers.litellm_provider import LiteLLMProvider

GATEWAY = "https://gateway.invalid/v1"
WINDOW = 32_768
MARGIN = int(WINDOW * 0.05)  # 1,638: the compaction check's estimate margin (ADR-0240)
REAL_COUNTER = litellm.token_counter


def _one_a_character(
    model: str = "", messages: list[dict[str, Any]] | None = None, tools: Any = None
) -> int:
    """The stand-in tokenizer: one token for each character of message text and tool schema."""
    text = sum(len(m["content"]) for m in messages or [] if isinstance(m.get("content"), str))
    return text + (len(json.dumps(tools)) if tools else 0)


@pytest.fixture(autouse=True)
def tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lp.litellm, "token_counter", _one_a_character)


def _prompt(tokens: int) -> list[Message]:
    """One user message the stand-in tokenizer counts as ``tokens``."""
    return [Message.user("x" * tokens)]


def _dense_jsonl(rows: int) -> str:
    """Tool output of the id-dense kind: goal ids, short shas, timestamps and paths."""
    lines = []
    for i in range(rows):
        digest = hashlib.sha1(str(i).encode()).hexdigest()
        lines.append(
            json.dumps(
                {
                    "id": f"g-{100 + i % 300}-{i * 37 % 12000}",
                    "sha": digest[:10],
                    "status": ("pending", "completed", "blocked")[i % 3],
                    "at": f"2026-10-15T10:{i % 60:02d}:{i * 7 % 60:02d}",
                    "path": f"core/scripts/{digest[10:14]}-{i}.py:{i * 13 % 999}",
                }
            )
        )
    return "\n".join(lines)


class BadRequestError(Exception):
    """Named exactly: the provider matches litellm's class by MRO NAME (plus status)."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def _refusal(requested: int, prompt: int) -> str:
    """The measured wording, with the route's own numbers."""
    return (
        "litellm.BadRequestError: OpenAIException - This model's maximum context length is "
        f"{WINDOW} tokens. However, you requested {requested} output tokens and your prompt "
        f"contains at least {prompt} input tokens, for a total of at least "
        f"{requested + prompt} tokens. Please reduce the length of the input prompt or the "
        "number of requested output tokens."
    )


class Route:
    """A stand-in for a 32k route: records every request, refuses one whose prompt plus
    requested output passes the window, and answers otherwise. It counts the prompt with
    ``count``, by default whatever tokenizer litellm has at the time of the call."""

    def __init__(self, window: int = WINDOW, count: Callable[..., int] | None = None) -> None:
        self.window = window
        self.count = count
        self.calls: list[dict[str, Any]] = []

    @staticmethod
    def cap(kwargs: dict[str, Any]) -> int:
        return int(kwargs.get("max_completion_tokens") or kwargs["max_tokens"])

    def prompt(self, kwargs: dict[str, Any]) -> int:
        count = self.count or lp.litellm.token_counter
        return int(
            count(model=kwargs["model"], messages=kwargs["messages"], tools=kwargs.get("tools"))
        )

    async def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        prompt = self.prompt(kwargs)
        if prompt + self.cap(kwargs) > self.window:
            raise BadRequestError(_refusal(self.cap(kwargs), prompt))
        if kwargs.get("stream"):
            return self._stream()
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=None), finish_reason="stop"
                )
            ],
            usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=1, total_tokens=2),
            _hidden_params={"response_cost": 0.0},
        )

    @staticmethod
    async def _stream() -> AsyncIterator[Any]:
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="ok", tool_calls=None), finish_reason="stop"
                )
            ],
            usage=None,
        )


@pytest.fixture
def route(monkeypatch: pytest.MonkeyPatch) -> Route:
    fake = Route()
    monkeypatch.setattr(lp.litellm, "acompletion", fake)
    return fake


def _provider(model: str = "openai/route-a", **kw: Any) -> LiteLLMProvider:
    if model.startswith("openai/route"):
        kw.setdefault("api_base", GATEWAY)
    return LiteLLMProvider(model=model, api_key="sk-test", context_window=WINDOW, **kw)


async def _call(provider: LiteLLMProvider, path: str, messages: list[Message], **kw: Any) -> str:
    if path == "buffered":
        return (await provider.acomplete(messages, **kw)).text
    text = ""
    async for ev in provider.astream(messages, **kw):
        if isinstance(ev, StreamTextDelta):
            text += ev.text
    return text


def test_the_fixture_prompt_counts_as_named() -> None:
    wire = LiteLLMProvider._translate_messages(_prompt(26_000), None)
    assert lp.litellm.token_counter(model="openai/route-a", messages=wire) == 26_000


# ── the report's case ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("path", ["buffered", "streamed"])
async def test_a_32k_window_with_a_26k_prompt_sends_a_clamped_cap(route: Route, path: str) -> None:
    assert await _call(_provider(), path, _prompt(26_000)) == "ok"
    assert len(route.calls) == 1  # no refused request first
    assert route.calls[0]["max_tokens"] == WINDOW - MARGIN - 26_000  # 5,130


async def test_on_main_that_prompt_was_refused_as_an_overflow() -> None:
    """The stand-in's refusal is the overflow the loop compacts on: what an unclamped cap
    met. Here it is provoked directly, to show the route refuses 8,192 on 26,000."""
    fake = Route()
    with pytest.raises(BadRequestError) as caught:
        await fake(
            model="openai/route-a",
            messages=LiteLLMProvider._translate_messages(_prompt(26_000), None),
            max_tokens=8_192,
        )
    assert "you requested 8192 output tokens" in str(caught.value)
    assert isinstance(LiteLLMProvider._map_error(caught.value), ContextWindowExceeded)


# ── what the count is ─────────────────────────────────────────────────────────


async def test_the_count_is_the_tokenizers_not_four_characters_a_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On id-dense tool output four characters a token counts well under the tokenizer, and
    a clamp from that estimate would still send the cap the route refuses. With litellm's own
    tokenizer, the route here counting the same way, the clamped call is taken."""
    monkeypatch.setattr(lp.litellm, "token_counter", REAL_COUNTER)
    fake = Route()
    monkeypatch.setattr(lp.litellm, "acompletion", fake)
    messages = [Message.user(_dense_jsonl(440))]
    wire = LiteLLMProvider._translate_messages(messages, None)
    counted = REAL_COUNTER(model="openai/route-a", messages=wire)
    rough = LiteLLMProvider._rough_token_estimate(wire)
    # The premise: the tokenizer's count leaves less room than the default cap, the rough
    # estimate leaves more, so a clamp from the estimate would send 8,192 and be refused.
    assert counted > WINDOW - MARGIN - 8_192 > rough
    assert counted + 8_192 > WINDOW

    assert (await _provider().acomplete(messages)).text == "ok"
    assert len(fake.calls) == 1
    assert fake.calls[0]["max_tokens"] == WINDOW - MARGIN - counted


async def test_a_counter_that_fails_falls_back_to_the_rough_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(**_: Any) -> int:
        raise ValueError("no tokenizer for this model")

    monkeypatch.setattr(lp.litellm, "token_counter", broken)
    fake = Route(count=lambda **kw: LiteLLMProvider._rough_token_estimate(kw["messages"]))
    monkeypatch.setattr(lp.litellm, "acompletion", fake)
    messages = [Message.user("a" * ((26_000 - 4) * 4))]  # a rough estimate of 26,000
    assert (await _provider().acomplete(messages)).text == "ok"
    assert fake.calls[0]["max_tokens"] == WINDOW - MARGIN - 26_000


async def test_the_tool_schemas_count_against_the_room(route: Route) -> None:
    """The route counts the tool definitions as prompt; so does the clamp."""
    tools = [
        {
            "type": "function",
            "function": {"name": "read", "description": "d" * 4_000, "parameters": {}},
        }
    ]
    await _provider().acomplete(_prompt(26_000), tools=tools)
    assert route.calls[0]["max_tokens"] == WINDOW - MARGIN - 26_000 - len(json.dumps(tools))


# ── the clamp's bounds ────────────────────────────────────────────────────────


@pytest.mark.parametrize("path", ["buffered", "streamed"])
async def test_a_prompt_with_room_keeps_the_default_cap(route: Route, path: str) -> None:
    assert await _call(_provider(), path, _prompt(1_000)) == "ok"
    assert route.calls[0]["max_tokens"] == 8_192


async def test_room_exactly_the_default_keeps_it(route: Route) -> None:
    await _provider().acomplete(_prompt(WINDOW - MARGIN - 8_192))
    assert route.calls[0]["max_tokens"] == 8_192


async def test_one_token_less_room_clamps_by_one(route: Route) -> None:
    await _provider().acomplete(_prompt(WINDOW - MARGIN - 8_191))
    assert route.calls[0]["max_tokens"] == 8_191


async def test_a_prompt_that_fills_the_window_is_sent_the_floor(route: Route) -> None:
    """Below the floor the clamp stops: an answer of a few tokens is no answer, and a
    request the route then refuses is the overflow the loop compacts on, as before."""
    with pytest.raises(ContextWindowExceeded):
        await _provider().acomplete(_prompt(WINDOW - 300))
    assert route.calls[0]["max_tokens"] == lp._MIN_CLAMPED_OUTPUT_TOKENS
    assert lp._MIN_CLAMPED_OUTPUT_TOKENS == 1_024


async def test_the_floor_is_not_a_raise(route: Route) -> None:
    """A caller's own cap under the floor is left as it is."""
    await _provider().acomplete(_prompt(WINDOW - 300), max_tokens=200)
    assert route.calls[0]["max_tokens"] == 200


# ── a caller's own cap ────────────────────────────────────────────────────────


async def test_a_callers_larger_cap_is_clamped_too(route: Route) -> None:
    """The compaction summarizer asks for 32,768 (ADR-0275); on a 32k window with a
    10,000-token prompt that request could never be taken."""
    assert (await _provider().acomplete(_prompt(10_000), max_tokens=32_768)).text == "ok"
    assert route.calls[0]["max_tokens"] == WINDOW - MARGIN - 10_000


async def test_a_callers_smaller_cap_is_kept(route: Route) -> None:
    await _provider().acomplete(_prompt(26_000), max_tokens=512)
    assert route.calls[0]["max_tokens"] == 512


# ── where it applies ──────────────────────────────────────────────────────────


async def test_openais_own_api_clamps_the_renamed_cap(route: Route) -> None:
    """On OpenAI's own API the cap is sent as ``max_completion_tokens`` (ADR-0279); the
    clamp is on that name too."""
    await _provider("openai/gpt-6-luna").acomplete(_prompt(26_000))
    call = route.calls[0]
    assert "max_tokens" not in call
    assert call["max_completion_tokens"] == WINDOW - MARGIN - 26_000


async def test_a_large_window_is_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = Route(window=131_072)
    monkeypatch.setattr(lp.litellm, "acompletion", fake)
    provider = LiteLLMProvider(
        model="openai/route-b", api_base=GATEWAY, api_key="sk-test", context_window=131_072
    )
    await provider.acomplete(_prompt(26_000))
    assert fake.calls[0]["max_tokens"] == 8_192


def test_the_margin_is_the_compaction_checks() -> None:
    from zakcode.agent.compact import ESTIMATE_MARGIN_FRACTION

    assert lp._CLAMP_MARGIN_FRACTION == ESTIMATE_MARGIN_FRACTION
