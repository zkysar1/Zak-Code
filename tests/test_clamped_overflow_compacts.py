"""A reasoning overflow under a cap the window clamped compacts before it retries (ADR-0310).

Reported 2026-10-15 from a re-test of main against a LiteLLM 1.9x gateway in front of
Bedrock/Vertex. A route whose window is 32,768 tokens refuses a request whose prompt plus
requested output passes the window, and a reasoning model there needed about 7,000 output tokens.
With a prompt of 25,000 tokens the clamp of ADR-0308 held the 8,192 default to the 6,130 the
window leaves, the reasoning filled it, and the call ended ``length`` with nothing visible. The
overflow retry of ADR-0307 found no room past the default, so it asked for the default, the clamp
held that to the same room less the rail, and after the third try the turn gave up with "output
cap exhausted by reasoning". Nothing compacted, because no request was refused for the window:
the clamp made every one fit. Before the clamp, the refusal compacted the transcript and the retry
answered.

The band is a prompt past the room the default cap needs, the window less 5% less 8,192, and
under the compaction threshold, which reserves the 4,096-token answer room of an unmapped alias.
Now a call the clamp held under the cap it asked for, and that ends on that cap with nothing
visible, compacts first, and the overflow retry is sized to the prompt that is left.

Hermetic: ``litellm.acompletion`` is faked by a stand-in route that counts the prompt with a
stand-in tokenizer, one token a character, refuses a request whose prompt plus requested output
passes its window with the measured wording, summarizes when asked to compact, and answers only
when the cap it was sent leaves room for the reasoning it needs. The system prompt is pinned, so
the numbers do not move with the real one.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import zakcode.providers.litellm_provider as lp
from zakcode.agent import SystemPromptBuilder
from zakcode.agent.compact import ESTIMATE_MARGIN_FRACTION, Compactor
from zakcode.agent.loop import _SUMMARY_OUTPUT_FLOOR, AgentLoop, _answer_room
from zakcode.config import load_settings
from zakcode.messages import Message
from zakcode.providers.base import Capabilities, LLMResult, Provider, StreamDone
from zakcode.providers.litellm_provider import LiteLLMProvider
from zakcode.providers.text_tools import TextToolCallingProvider
from zakcode.session.store import Session
from zakcode.tools.base import Tool, ToolContext, ToolRegistry, ToolResult, ToolSpec

GATEWAY = "https://gateway.invalid/v1"
MODEL = "openai/route-a"
SYSTEM = "You are a coding agent."
QUESTION = "question"
#: The output a reasoning model needs before it answers, as the report measured.
NEEDS = 7_000
#: Messages in the session before the question: with it, the 15 the report's compaction folded.
HISTORY = 14
#: Reasoning that no cap ends.
FOREVER = 10**9
SUMMARY = (
    "<summary>"
    + "The user and the agent worked through the build, one step at a time, and agreed on "
    "what each step changes. " * 8 + "</summary>"
)


def _one_a_character(
    model: str = "", messages: list[dict[str, Any]] | None = None, tools: Any = None
) -> int:
    """The stand-in tokenizer: one token for each character of message text and tool schema."""
    text = sum(len(m["content"]) for m in messages or [] if isinstance(m.get("content"), str))
    return text + (len(json.dumps(tools)) if tools else 0)


@pytest.fixture(autouse=True)
def _stand_ins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lp.litellm, "token_counter", _one_a_character)
    # The provider spaces request starts a second apart; that sleep is not under test.
    monkeypatch.setattr(lp, "_MIN_REQUEST_INTERVAL_S", 0.0)


class FixedPrompt(SystemPromptBuilder):
    """A system prompt of a known size."""

    def build(self, settings: Any, **kwargs: Any) -> str:
        return SYSTEM


class BadRequestError(Exception):
    """Named like litellm's: the provider matches the class by MRO name, plus the status."""

    status_code = 400


def _refusal(window: int, requested: int, prompt: int) -> str:
    """The measured wording, with the route's own numbers."""
    return (
        "litellm.BadRequestError: OpenAIException - This model's maximum context length is "
        f"{window} tokens. However, you requested {requested} output tokens and your prompt "
        f"contains at least {prompt} input tokens, for a total of at least "
        f"{requested + prompt} tokens. Please reduce the length of the input prompt or the "
        "number of requested output tokens."
    )


def _completion(
    text: str, reasoning: str, finish: str, prompt: int, spent: int, call: str | None
) -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=text,
                    reasoning_content=reasoning or None,
                    tool_calls=[_tool_call(call)] if call is not None else None,
                ),
                finish_reason=finish,
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt, completion_tokens=spent, total_tokens=prompt + spent
        ),
        _hidden_params={"response_cost": 0.0},
    )


def _tool_call(arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        index=0,
        id="c1",
        type="function",
        function=SimpleNamespace(name="echo", arguments=arguments),
    )


async def _chunks(text: str, reasoning: str, finish: str, call: str | None) -> AsyncIterator[Any]:
    if reasoning:
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None, reasoning_content=reasoning),
                    finish_reason=None,
                )
            ],
            usage=None,
        )
    delta = SimpleNamespace(
        content=text or None, tool_calls=[_tool_call(call)] if call is not None else None
    )
    yield SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish)], usage=None)


class EchoTool(Tool):
    spec = ToolSpec(name="echo", description="Echo back the provided text.")

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(output=str(args.get("text", "")))


class Route:
    """A stand-in for one route through the gateway, every request recorded.

    It refuses a request whose prompt plus requested output passes its window, as the 32k route
    does. A request that asks it to compact gets a summary. Any other call needs ``needs`` output
    tokens of reasoning: under that cap it ends ``length`` with nothing visible (or stops on its
    own, ``stops``), and at or over it it gives the next of ``replies``: the answer, or a call of
    the echo tool whose output is ``echo`` characters long.
    """

    def __init__(
        self,
        *,
        window: int,
        needs: int,
        replies: tuple[str, ...] = ("answer",),
        echo: int = 0,
        stops: bool = False,
    ) -> None:
        self.window = window
        self.needs = needs
        self.replies = list(replies)
        self.echo = echo
        self.stops = stops
        self.calls: list[dict[str, Any]] = []
        self.refused = 0

    async def __call__(self, **kwargs: Any) -> Any:
        cap = int(kwargs.get("max_completion_tokens") or kwargs["max_tokens"])
        messages = kwargs["messages"]
        prompt = _one_a_character(messages=messages, tools=kwargs.get("tools"))
        system = messages[0].get("content") if messages[0]["role"] == "system" else ""
        summarizer = isinstance(system, str) and system.startswith("You are compacting")
        self.calls.append({"cap": cap, "prompt": prompt, "summarizer": summarizer})
        if prompt + cap > self.window:
            self.refused += 1
            raise BadRequestError(_refusal(self.window, cap, prompt))
        call: str | None = None
        if summarizer:
            text, reasoning, finish, spent = SUMMARY, "", "stop", len(SUMMARY)
        elif cap < self.needs:
            finish = "stop" if self.stops else "length"
            text, reasoning, spent = "", "still working it out", cap
        elif (self.replies.pop(0) if self.replies else "answer") == "tool":
            text, reasoning, finish, spent = "", "worked it out", "tool_calls", self.needs
            call = json.dumps({"text": "y" * self.echo})
        else:
            text, reasoning, finish, spent = "the answer", "worked it out", "stop", self.needs
        if kwargs.get("stream"):
            return _chunks(text, reasoning, finish, call)
        return _completion(text, reasoning, finish, prompt, spent, call)

    def main(self) -> list[dict[str, Any]]:
        """The conversation's calls, the summarizer's left out."""
        return [call for call in self.calls if not call["summarizer"]]

    def summaries(self) -> int:
        return sum(1 for call in self.calls if call["summarizer"])


def _loop(tmp_path: Path, *, window: int, prompt: int, tools: bool = False) -> AgentLoop:
    """A loop whose first call is about a ``prompt``-token request: the pinned system prompt,
    the history, and the question, and the echo tool's schema when ``tools``."""
    provider = LiteLLMProvider(
        model=MODEL, api_base=GATEWAY, api_key="sk-test", context_window=window
    )
    session = Session(cwd=str(tmp_path), model=MODEL)
    each = (prompt - len(SYSTEM) - len(QUESTION)) // HISTORY
    for i in range(HISTORY):
        body = (f"step {i}: " + "x" * each)[:each]
        session.add_message(Message.user(body) if i % 2 == 0 else Message.assistant_text(body))
    registry = ToolRegistry()
    if tools:
        registry.register(EchoTool())
    return AgentLoop(
        provider,
        registry,
        session,
        settings=load_settings(workspace_root=tmp_path),
        compactor=Compactor(),
        prompt_builder=FixedPrompt(),
    )


async def _run(loop: AgentLoop, path: str) -> tuple[str | None, list[str]]:
    """The turn's stop reason and, on the streamed path, its status lines."""
    if path == "buffered":
        result = await loop.arun_turn(QUESTION)
        return result.stop_reason, []
    events = [ev async for ev in loop.astream_turn(QUESTION)]
    done = [ev for ev in events if getattr(ev, "stop_reason", None) is not None]
    return done[-1].stop_reason if done else None, [getattr(ev, "message", "") for ev in events]


def _notes(loop: AgentLoop, kind: str) -> list[str]:
    return [e.detail for e in loop._trace.of_kind("intervention") if e.data.get("kind") == kind]


def _band(window: int) -> tuple[int, int]:
    """The prompts the clamp holds under the default cap that do not compact: past the room the
    default needs, up to the compaction threshold of an unmapped alias."""
    low = window - int(window * lp._CLAMP_MARGIN_FRACTION) - lp._MAX_COMPLETION_TOKENS
    high = Compactor().threshold(window, answer_room=_answer_room(Capabilities()))
    return low, high


# ── the band ─────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("window", "band"), [(32_768, (22_938, 27_034)), (128_000, (113_408, 115_200))]
)
def test_the_band_is_where_three_numbers_disagree(window: int, band: tuple[int, int]) -> None:
    # The clamp's margin is the compaction check's, and the default the overflow retry measures
    # against is the provider's own. The threshold reserves the 4,096 answer room of an unmapped
    # alias where the provider sends 8,192, so between the two the clamp holds every call down.
    assert lp._CLAMP_MARGIN_FRACTION == ESTIMATE_MARGIN_FRACTION
    assert _SUMMARY_OUTPUT_FLOOR == lp._MAX_COMPLETION_TOKENS
    assert _answer_room(Capabilities()) == 4_096
    assert _band(window) == band


# ── the scenario ─────────────────────────────────────────────────────────────────────────────


# On the 128,000-token window the band is narrower, and each retry's rail moves the prompt
# toward the threshold, so a need of 7,000 there compacts on the threshold by the first retry;
# 8,000 at 114,000 stays in the band for every try, as 7,000 at 25,000 does on 32,768.
@pytest.mark.parametrize(
    ("window", "prompt", "needs"), [(32_768, 25_000, NEEDS), (128_000, 114_000, 8_000)]
)
@pytest.mark.parametrize("path", ["buffered", "streamed"])
async def test_a_clamped_overflow_compacts_and_the_retry_answers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    path: str,
    window: int,
    prompt: int,
    needs: int,
) -> None:
    route = Route(window=window, needs=needs)
    monkeypatch.setattr(lp.litellm, "acompletion", route)
    loop = _loop(tmp_path, window=window, prompt=prompt)
    stop, statuses = await _run(loop, path)
    first = route.main()[0]
    # The premise: the first call sits in the band, and the clamp held it under the need.
    low, high = _band(window)
    assert low < first["prompt"] <= high
    assert first["cap"] < needs
    assert stop == "completed"
    # One summary between the overflow and the retry, and a retry with the room to answer.
    assert route.summaries() == 1
    assert len(route.main()) == 2
    assert route.main()[1]["cap"] >= needs
    assert route.main()[1]["prompt"] < first["prompt"]
    # No request was refused for the window: the clamp still makes every one fit.
    assert route.refused == 0
    assert loop.session.messages[-1].text == "the answer"
    assert len(_notes(loop, "overflow_compacted")) == 1
    if path == "streamed":
        assert any("under a cap the window held down" in s for s in statuses)


async def test_a_route_that_always_overflows_compacts_once_and_still_gives_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    route = Route(window=32_768, needs=FOREVER)
    monkeypatch.setattr(lp.litellm, "acompletion", route)
    loop = _loop(tmp_path, window=32_768, prompt=25_000)
    stop, _ = await _run(loop, "buffered")
    assert stop == "gave_up"
    # Compacted once: after it the retries ask for room the window has, and are not held down.
    assert route.summaries() == 1
    assert [call["cap"] for call in route.main()][1:] == [16_384, 16_384]
    [note] = _notes(loop, "gave_up")
    assert "output cap" in note and "reasoning" in note


@pytest.mark.parametrize("path", ["buffered", "streamed"])
async def test_a_turn_that_fills_the_band_again_summarizes_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
) -> None:
    # A long turn: the retry answers with a tool call whose output brings the prompt back into
    # the band, and the next call overflows under the clamp again. That is a new run of
    # overflows, so it starts the ladder again with a summary, as the first one did.
    route = Route(window=32_768, needs=NEEDS, replies=("tool", "answer"), echo=16_500)
    monkeypatch.setattr(lp.litellm, "acompletion", route)
    loop = _loop(tmp_path, window=32_768, prompt=25_000, tools=True)
    stop, _ = await _run(loop, path)
    assert stop == "completed"
    main = route.main()
    assert len(main) == 4
    # Both overflowed calls were held down by the window, and each was followed by a summary.
    assert main[0]["cap"] < NEEDS and main[2]["cap"] < NEEDS
    assert _band(32_768)[0] < main[2]["prompt"] <= _band(32_768)[1]
    assert route.summaries() == 2
    assert len(_notes(loop, "overflow_compacted")) == 2


# ── left alone ───────────────────────────────────────────────────────────────────────────────


async def test_reasoning_that_stopped_on_its_own_under_a_clamp_is_not_compacted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The cap did not end it, so a smaller prompt would not either.
    route = Route(window=32_768, needs=FOREVER, stops=True)
    monkeypatch.setattr(lp.litellm, "acompletion", route)
    loop = _loop(tmp_path, window=32_768, prompt=25_000)
    stop, _ = await _run(loop, "buffered")
    assert stop == "gave_up"
    assert route.main()[0]["cap"] < lp._MAX_COMPLETION_TOKENS
    assert route.summaries() == 0
    assert _notes(loop, "overflow_compacted") == []


@pytest.mark.parametrize("path", ["buffered", "streamed"])
async def test_a_prompt_below_the_band_is_left_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
) -> None:
    route = Route(window=32_768, needs=NEEDS)
    monkeypatch.setattr(lp.litellm, "acompletion", route)
    loop = _loop(tmp_path, window=32_768, prompt=20_000)
    stop, _ = await _run(loop, path)
    assert route.main()[0]["prompt"] <= _band(32_768)[0]
    assert stop == "completed"
    assert [call["cap"] for call in route.main()] == [lp._MAX_COMPLETION_TOKENS]
    assert route.summaries() == 0
    assert _notes(loop, "overflow_compacted") == []


@pytest.mark.parametrize("path", ["buffered", "streamed"])
async def test_a_clamped_call_with_room_enough_is_left_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
) -> None:
    route = Route(window=32_768, needs=3_000)
    monkeypatch.setattr(lp.litellm, "acompletion", route)
    loop = _loop(tmp_path, window=32_768, prompt=25_000)
    stop, _ = await _run(loop, path)
    assert stop == "completed"
    [call] = route.main()
    assert 3_000 <= call["cap"] < lp._MAX_COMPLETION_TOKENS
    assert route.summaries() == 0


@pytest.mark.parametrize("path", ["buffered", "streamed"])
async def test_an_overflow_the_clamp_did_not_cause_is_not_compacted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
) -> None:
    route = Route(window=131_072, needs=12_000)
    monkeypatch.setattr(lp.litellm, "acompletion", route)
    loop = _loop(tmp_path, window=131_072, prompt=2_000)
    stop, _ = await _run(loop, path)
    assert stop == "completed"
    assert [call["cap"] for call in route.main()] == [8_192, 16_384]
    assert route.summaries() == 0
    assert _notes(loop, "overflow_compacted") == []


# ── what the provider says about the cap it sent ─────────────────────────────────────────────


@pytest.mark.parametrize(("prompt", "clamped"), [(25_000, True), (20_000, False)])
async def test_the_result_says_when_the_clamp_held_the_cap(
    monkeypatch: pytest.MonkeyPatch, prompt: int, clamped: bool
) -> None:
    route = Route(window=32_768, needs=1_000)
    monkeypatch.setattr(lp.litellm, "acompletion", route)
    provider = LiteLLMProvider(
        model=MODEL, api_base=GATEWAY, api_key="sk-test", context_window=32_768
    )
    messages = [Message.user("x" * prompt)]
    result = await provider.acomplete(messages)
    done = [ev async for ev in provider.astream(messages) if isinstance(ev, StreamDone)]
    sent = [call["cap"] for call in route.calls]
    assert sent[0] == sent[1] and (sent[0] < lp._MAX_COMPLETION_TOKENS) is clamped
    expected = sent[0] if clamped else None
    assert result.clamped_output_cap == expected
    assert [ev.clamped_output_cap for ev in done] == [expected]


async def test_a_clamp_is_not_carried_to_the_next_call(monkeypatch: pytest.MonkeyPatch) -> None:
    # The provider records the clamp of each request it builds, so a request it did not clamp,
    # after one it did, says it went out as asked.
    route = Route(window=32_768, needs=1_000)
    monkeypatch.setattr(lp.litellm, "acompletion", route)
    provider = LiteLLMProvider(
        model=MODEL, api_base=GATEWAY, api_key="sk-test", context_window=32_768
    )
    held = await provider.acomplete([Message.user("x" * 25_000)])
    free = await provider.acomplete([Message.user("x" * 20_000)])
    done = [
        ev
        async for ev in provider.astream([Message.user("x" * 20_000)])
        if isinstance(ev, StreamDone)
    ]
    assert held.clamped_output_cap is not None
    assert free.clamped_output_cap is None
    assert [ev.clamped_output_cap for ev in done] == [None]


async def test_a_cap_the_floor_keeps_as_asked_is_not_clamped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With less room than the floor, a caller's cap under the floor goes out as asked
    # (ADR-0308). The window did not hold it down, so the result does not say it did.
    route = Route(window=32_768, needs=0)
    monkeypatch.setattr(lp.litellm, "acompletion", route)
    provider = LiteLLMProvider(
        model=MODEL, api_base=GATEWAY, api_key="sk-test", context_window=32_768
    )
    result = await provider.acomplete([Message.user("x" * 31_000)], max_tokens=1_000)
    assert [call["cap"] for call in route.calls] == [1_000]
    assert result.clamped_output_cap is None


class HeldDown(Provider):
    """A backend without its own streaming, whose every call the window held to 6,130 tokens."""

    async def acomplete(self, messages: list[Message], **kwargs: Any) -> LLMResult:
        return LLMResult(finish_reason="length", clamped_output_cap=6_130)

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities()


@pytest.mark.parametrize("text_tools", [False, True], ids=["default stream", "text tools"])
async def test_a_buffered_result_streamed_keeps_its_clamp(text_tools: bool) -> None:
    # The base provider's stream and the text tool protocol's both replay a buffered result,
    # and the cap the window held it to goes with it.
    provider: Provider = HeldDown()
    if text_tools:
        provider = TextToolCallingProvider(provider, mode="text")
    echo = {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Echo back the provided text.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    events = [ev async for ev in provider.astream([Message.user(QUESTION)], tools=[echo])]
    assert [ev.clamped_output_cap for ev in events if isinstance(ev, StreamDone)] == [6_130]
