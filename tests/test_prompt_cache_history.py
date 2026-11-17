"""The conversation's cache breakpoint on a route that is Anthropic by name (ADR-0385).

Until ADR-0385 the one ``cache_control`` breakpoint a request carried sat on the system prompt's
stable block. An Anthropic model caches only what a request marks, so it read that block back on
every call and was sent the whole conversation again at the full input price. One more
breakpoint now rides the last assistant or tool message that has text, so each call reads the
conversation the call before it wrote.

Where it sits is the whole design. The loop appends a per-call tail of user messages after the
persisted history (ADR-0193) and drops it before the next request, so a breakpoint behind the
tail would write a prefix no later request can read. The breakpoint therefore never sits on a
user message, and a side call, which sends one user message, gets none.

Hermetic. The request shape is read off the provider's pure builders, and what reaches the wire
is read with ``httpx.AsyncClient.send`` replaced by a canned answer, so it is what litellm's own
client puts in an Anthropic request. No network, no credentials.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

import zakcode.providers.litellm_provider as lp
from zakcode.agent.prompt import DYNAMIC_BOUNDARY
from zakcode.config import Settings
from zakcode.messages import Message, TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock
from zakcode.providers.litellm_provider import LiteLLMProvider, _conversation_breakpoint

ANTHROPIC = "anthropic/claude-sonnet-4-6"
#: A bare name the locked litellm routes as Anthropic's by itself, prefix or no prefix.
BARE = "claude-sonnet-4-6"
GATEWAY = "https://gateway.invalid/v1"
SYSTEM = f"stable identity and tool guidance\n\n{DYNAMIC_BOUNDARY}\n\nsession context"
PLAIN_SYSTEM = "a side call's own instruction, with no boundary marker"
MARKER = {"type": "ephemeral"}
TAIL = "the live plan, said again for this one call"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Settings read the cwd .env, the config home and real env vars: isolate all three so a dev
    # box's configuration can never change a verdict.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ZAKCODE_HOME", str(tmp_path / "confighome"))
    for var in (
        "ZAKCODE_CACHE_CONVERSATION",
        "ZAKCODE_PROMPT_CACHING",
        "ZAKCODE_DEFAULT_MODEL",
        "ZAKCODE_API_BASE",
        "ZAKCODE_EXTRA_BODY",
        "ZAKCODE_ANTHROPIC_POOL",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(lp, "_MIN_REQUEST_INTERVAL_S", 0.0)


# ── a small conversation, in the engine's own vocabulary ────────────────────────────────────


def _calls(text: str, *ids: str) -> Message:
    """An assistant message that says ``text`` and calls one tool per id."""
    blocks: list[Any] = [TextBlock(text=text)] if text else []
    blocks += [ToolUseBlock(id=i, name="read_file", input={"path": i}) for i in ids]
    return Message(role="assistant", blocks=blocks)


def _results(**outputs: str) -> Message:
    return Message.tool_results(
        [ToolResultBlock(tool_use_id=i, output=text) for i, text in outputs.items()]
    )


def _first_call() -> list[Message]:
    return [Message.user("read the two files and compare them")]


def _second_call() -> list[Message]:
    return [
        *_first_call(),
        _calls("I will read both.", "t1", "t2"),
        _results(t1="contents of a", t2="contents of b"),
    ]


def _third_call() -> list[Message]:
    return [*_second_call(), _calls("One more.", "t3"), _results(t3="contents of c")]


def _provider(model: str = ANTHROPIC, **settings: Any) -> LiteLLMProvider:
    return LiteLLMProvider(
        Settings(
            default_model=model,
            api_key="sk-test",
            context_window=200_000,
            _env_file=None,
            **settings,
        )
    )


def _request(
    provider: LiteLLMProvider, messages: list[Message], system: str = SYSTEM
) -> list[dict[str, Any]]:
    wire = provider._translate_messages(messages, system)
    built: list[dict[str, Any]] = provider._build_kwargs(wire, None)["messages"]
    return built


def _marked(request: list[dict[str, Any]]) -> list[int]:
    """The conversation messages that carry a breakpoint, counted without the code under test."""
    return [i for i, message in enumerate(request) if i and "cache_control" in message]


def _system_marks(request: list[dict[str, Any]]) -> int:
    content = request[0].get("content")
    if not isinstance(content, list):
        return 0
    return sum(1 for block in content if isinstance(block, dict) and "cache_control" in block)


# ── where the breakpoint sits ───────────────────────────────────────────────────────────────


def test_the_breakpoint_sits_on_the_last_tool_result() -> None:
    request = _request(_provider(), _second_call())

    assert [m["role"] for m in request] == ["system", "user", "assistant", "tool", "tool"]
    assert _marked(request) == [4]
    assert request[4]["cache_control"] == MARKER
    assert request[4]["content"] == "contents of b"
    assert _system_marks(request) == 1, "the system block keeps its own"


@pytest.mark.parametrize("tail", [1, 2, 3])
def test_the_per_call_tail_is_never_inside_the_cached_prefix(tail: int) -> None:
    """The loop appends the tail after the persisted history and drops it before the next
    request (ADR-0193). A breakpoint on it, or behind it, would be written and never read."""
    with_tail = _request(_provider(), [*_second_call(), *[Message.user(TAIL)] * tail])
    without = _request(_provider(), _second_call())

    assert _marked(with_tail) == _marked(without) == [4]
    assert all("cache_control" not in message for message in with_tail[5:])


def test_a_turns_new_user_message_waits_one_call() -> None:
    """This layer cannot tell the turn's own user message from a tail, so the breakpoint stays
    on the answer before it. The next request caches the message, as the test below shows."""
    history = [*_second_call(), Message.assistant_text("They differ on line 3.")]
    request = _request(_provider(), [*history, Message.user("and the fourth file?")])

    assert request[5]["role"] == "assistant" and request[6]["role"] == "user"
    assert _marked(request) == [5]


def test_the_breakpoint_moves_forward_with_the_conversation() -> None:
    second = _request(_provider(), [*_second_call(), Message.user(TAIL)])
    third = _request(_provider(), [*_third_call(), Message.user(TAIL)])

    assert _marked(second) == [4]
    assert _marked(third) == [6]
    assert third[6]["content"] == "contents of c"


def test_a_first_call_carries_only_the_system_blocks_breakpoint() -> None:
    request = _request(_provider(), [*_first_call(), Message.user(TAIL)])

    assert _marked(request) == []
    assert _system_marks(request) == 1


def test_a_side_call_carries_no_breakpoint_at_all() -> None:
    """A side call (the summarizer, a judge, a classifier) sends its own instruction and one
    user message. It is asked once, so a write there is a cost with nothing to read it."""
    request = _request(_provider(), [Message.user("summarize this transcript")], PLAIN_SYSTEM)

    assert request[0]["content"] == PLAIN_SYSTEM
    assert _marked(request) == []


def test_a_conversation_with_no_boundary_in_its_system_prompt_is_left_alone() -> None:
    request = _request(_provider(), _second_call(), PLAIN_SYSTEM)

    assert _marked(request) == []
    assert _system_marks(request) == 0


def test_a_message_with_no_text_is_passed_over() -> None:
    """The marker needs a block to sit on: an empty tool result has none, and neither has an
    assistant message that only calls tools."""
    empty_last = [
        *_first_call(),
        _calls("", "t1", "t2"),
        _results(t1="contents of a", t2=""),
    ]
    assert _marked(_request(_provider(), empty_last)) == [3]

    all_empty = [
        *_first_call(),
        _calls("I will read both.", "t1", "t2"),
        _results(t1="", t2="  "),
    ]
    request = _request(_provider(), all_empty)
    assert _marked(request) == [2], "the assistant message's own text carries it"

    nothing_to_sit_on = [*_first_call(), _calls("", "t1"), _results(t1="")]
    assert _marked(_request(_provider(), nothing_to_sit_on)) == []


def test_a_thinking_block_does_not_take_the_marker() -> None:
    answer = Message(
        role="assistant",
        blocks=[ThinkingBlock(text="reasoning", signature="sig"), TextBlock(text="Done.")],
    )
    request = _request(_provider(), [*_second_call(), answer, Message.user("next")])

    assert _marked(request) == [5]
    assert request[5]["content"] == "Done."
    assert "cache_control" not in json.dumps(request[5]["thinking_blocks"])


@pytest.mark.parametrize(
    "messages",
    [_first_call(), _second_call(), _third_call(), [*_third_call(), Message.user(TAIL)]],
)
def test_a_request_uses_at_most_two_of_the_four_breakpoints(messages: list[Message]) -> None:
    request = _request(_provider(), messages)

    assert _system_marks(request) + len(_marked(request)) <= 2


def test_building_the_same_messages_again_marks_the_same_message() -> None:
    """A call re-issued after a refusal, and a streamed call's rebuild, build their kwargs from
    the same wire messages a second time."""
    provider = _provider()
    wire = provider._translate_messages([*_third_call(), Message.user(TAIL)], SYSTEM)

    first = provider._build_kwargs(wire, None)["messages"]
    assert _marked(first) == [6]
    again = provider._build_kwargs(wire, None)["messages"]

    assert _marked(again) == [6]
    assert _system_marks(again) == 1

    # A library that strips markers from the list it is handed, as litellm does on other
    # routes (ADR-0298), leaves the split system block and nothing else. The next build puts
    # the conversation's marker back.
    del wire[6]["cache_control"]
    assert _marked(provider._build_kwargs(wire, None)["messages"]) == [6]


def test_the_helper_names_no_message_when_there_is_none_to_name() -> None:
    assert _conversation_breakpoint([]) is None
    assert _conversation_breakpoint([{"role": "system", "content": "s"}]) is None
    only_users = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    assert _conversation_breakpoint(only_users) is None
    leading = [{"role": "assistant", "content": "a"}, {"role": "user", "content": "u"}]
    assert _conversation_breakpoint(leading) is None, "index 0 is the system slot"


# ── which routes take it ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("model", ["anthropic/claude-sonnet-4-6", "claude-opus-4-8"])
def test_a_route_that_is_anthropic_by_name_takes_it(model: str) -> None:
    assert _marked(_request(_provider(model), _second_call())) == [4]


def test_a_claude_name_behind_an_api_base_takes_it_too() -> None:
    """The name is the whole rule, wherever the request goes. A bare ``claude…`` model with an
    API base is sent to that base, and it carried the system block's marker there on the name
    alone before ADR-0385, so the conversation's marker follows the first."""
    provider = _provider(BARE, api_base=GATEWAY)
    kwargs = provider._build_kwargs(provider._translate_messages(_second_call(), SYSTEM), None)

    assert kwargs["api_base"] == GATEWAY
    assert _system_marks(kwargs["messages"]) == 1
    assert _marked(kwargs["messages"]) == [4]


@pytest.mark.parametrize("model", ["anthropic/claude-sonnet-4-6", "claude-opus-4-8"])
def test_opting_in_changes_nothing_on_a_route_the_name_vouches_for(model: str) -> None:
    """``prompt_caching`` is the operator's word for a route that is not Anthropic by name. Set
    on a route that is Anthropic by name it is redundant: the request is the by-name request,
    both breakpoints on the messages and no copy of them in ``extra_body``."""
    messages = [*_third_call(), Message.user(TAIL)]
    opted = _provider(model, prompt_caching=True)
    request = _request(opted, messages)

    assert request == _request(_provider(model), messages)
    assert _system_marks(request) == 1
    assert _marked(request) == [6]
    kwargs = opted._build_kwargs(opted._translate_messages(messages, SYSTEM), None)
    assert "messages" not in kwargs.get("extra_body", {})


@pytest.mark.parametrize(
    "model", ["openai/gpt-4o", "groq/llama-3.3-70b-versatile", "ollama_chat/llama3.1"]
)
def test_a_route_that_takes_no_breakpoint_is_sent_what_it_was(model: str) -> None:
    request = _request(_provider(model), _second_call())

    assert request[0]["content"] == SYSTEM
    assert _marked(request) == []


@pytest.mark.parametrize(
    "model",
    [
        "openai/Claude4.5",
        "hosted_vllm/Claude4.5",
        "bedrock/anthropic.claude-sonnet-4-6",
        "vertex_ai/claude-sonnet-4-6",
    ],
)
def test_an_opted_in_route_keeps_the_one_breakpoint_it_had(model: str) -> None:
    """A route that is not Anthropic by name carries its breakpoint only because it opted in:
    ADR-0298's switch is the operator's word that it takes the marker on the system block. No
    gateway was measured with a second one, and a gateway that refused it by name would be
    served the plain request, losing the first."""
    provider = _provider(model, api_base=GATEWAY, prompt_caching=True)
    request = _request(provider, [*_third_call(), Message.user(TAIL)])

    assert _system_marks(request) == 1
    assert _marked(request) == []
    carried = provider._build_kwargs(provider._translate_messages(_third_call(), SYSTEM), None).get(
        "extra_body", {}
    )
    assert "cache_control" not in json.dumps(carried.get("messages", [])[1:])


# ── the switch ──────────────────────────────────────────────────────────────────────────────


def test_the_switch_is_on_by_default() -> None:
    assert Settings(_env_file=None).cache_conversation is True
    assert LiteLLMProvider(model=ANTHROPIC, temperature=0.0).cache_conversation is True


def test_the_switch_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZAKCODE_CACHE_CONVERSATION", "false")

    assert Settings(_env_file=None).cache_conversation is False


def test_with_the_switch_off_the_request_is_the_one_sent_before() -> None:
    request = _request(_provider(cache_conversation=False), [*_third_call(), Message.user(TAIL)])

    assert _marked(request) == []
    assert _system_marks(request) == 1
    stable, context = request[0]["content"]
    assert stable["cache_control"] == MARKER and "cache_control" not in context


# ── the wire: what litellm's own client sends to an Anthropic route ─────────────────────────


class Wire:
    """Every request litellm's own client sends, answered with a refusal nothing retries."""

    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []
        self.urls: list[str] = []


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> Wire:
    seen = Wire()

    async def send(self: httpx.AsyncClient, request: httpx.Request, **kw: Any) -> httpx.Response:
        seen.bodies.append(json.loads(request.content.decode("utf-8")))
        seen.urls.append(str(request.url))
        refusal = {"type": "error", "error": {"type": "invalid_request_error", "message": "x"}}
        return httpx.Response(400, json=refusal, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    # litellm builds no Anthropic request without a key to put on it. This one goes nowhere:
    # every request is answered above.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    return seen


async def _sent(wire: Wire, provider: LiteLLMProvider, messages: list[Message]) -> dict[str, Any]:
    """The Anthropic-shaped body of the one request ``messages`` produce."""
    wire.bodies.clear()
    wire.urls.clear()
    with contextlib.suppress(Exception):  # the canned answer is a refusal; the request is real
        await provider.acomplete(messages, system=SYSTEM)
    return wire.bodies[0]


def _blocks(body: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Every content block of a request in the order the API renders it, each with its role."""
    flat: list[tuple[str, dict[str, Any]]] = [("system", block) for block in body["system"]]
    for message in body["messages"]:
        flat += [(message["role"], block) for block in message["content"]]
    return flat


def _breakpoints(body: dict[str, Any]) -> list[int]:
    return [i for i, (_, block) in enumerate(_blocks(body)) if "cache_control" in block]


def _unmarked(blocks: list[tuple[str, dict[str, Any]]]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (role, {k: v for k, v in block.items() if k != "cache_control"}) for role, block in blocks
    ]


async def test_on_the_wire_the_marker_is_on_the_tool_result_and_the_tail_follows_it(
    wire: Wire,
) -> None:
    body = await _sent(wire, _provider(), [*_second_call(), Message.user(TAIL)])

    last = body["messages"][-1]
    assert [block["type"] for block in last["content"]] == ["tool_result", "tool_result", "text"]
    first_result, second_result, tail = last["content"]
    assert "cache_control" not in first_result
    assert second_result["cache_control"] == MARKER
    assert tail == {"type": "text", "text": TAIL}
    assert body["system"][0]["cache_control"] == MARKER
    assert len(_breakpoints(body)) == 2


async def test_on_the_wire_an_answers_marker_is_on_its_text_block(wire: Wire) -> None:
    answer = Message(
        role="assistant",
        blocks=[ThinkingBlock(text="reasoning", signature="sig"), TextBlock(text="Done.")],
    )
    body = await _sent(wire, _provider(), [*_second_call(), answer, Message.user("next")])

    assistant = body["messages"][-2]
    assert assistant["role"] == "assistant"
    assert [block["type"] for block in assistant["content"]] == ["thinking", "text"]
    assert "cache_control" not in assistant["content"][0]
    assert assistant["content"][1]["cache_control"] == MARKER


async def test_what_one_call_marks_is_a_prefix_of_the_next_call(wire: Wire) -> None:
    """The property the cache read rests on. Everything up to the block one call marks must
    reappear unchanged at the head of the next call, whose tail is different text and whose own
    breakpoint has moved forward. The markers themselves are left out of the comparison: a
    block marked earlier is still read."""
    provider = _provider()
    earlier = await _sent(wire, provider, [*_second_call(), Message.user("tail of call two")])
    later = await _sent(wire, provider, [*_third_call(), Message.user("tail of call three")])

    written = _breakpoints(earlier)[-1]
    read_back = _breakpoints(later)[-1]
    assert read_back > written
    assert _unmarked(_blocks(later))[: written + 1] == _unmarked(_blocks(earlier))[: written + 1]
    assert earlier["tools"] == later["tools"] if "tools" in earlier else "tools" not in later


async def test_a_streamed_call_carries_the_conversations_breakpoint_too(wire: Wire) -> None:
    wire.bodies.clear()
    with contextlib.suppress(Exception):
        async for _ in _provider().astream([*_second_call(), Message.user(TAIL)], system=SYSTEM):
            pass

    body = wire.bodies[0]
    assert body["stream"] is True
    assert body["messages"][-1]["content"][1]["cache_control"] == MARKER
    assert len(_breakpoints(body)) == 2


async def test_with_the_switch_off_the_wire_carries_the_system_blocks_marker_alone(
    wire: Wire,
) -> None:
    body = await _sent(
        wire, _provider(cache_conversation=False), [*_second_call(), Message.user(TAIL)]
    )

    assert _breakpoints(body) == [0]


async def test_on_the_wire_a_claude_name_behind_an_api_base_is_an_anthropic_request_to_it(
    wire: Wire,
) -> None:
    """Measured on litellm 1.86.2: a bare ``claude…`` model with an API base is posted to that
    base in Anthropic's own shape, so both markers reach whatever answers there. A server that
    took the first and refused the second would fail every call until the switch is off."""
    provider = _provider(BARE, api_base=GATEWAY)
    body = await _sent(wire, provider, [*_second_call(), Message.user(TAIL)])

    assert httpx.URL(wire.urls[0]).host == "gateway.invalid"
    assert body["system"][0]["cache_control"] == MARKER
    assert body["messages"][-1]["content"][1]["cache_control"] == MARKER
    assert len(_breakpoints(body)) == 2


async def test_on_the_wire_a_request_that_asks_for_thinking_carries_both_breakpoints(
    wire: Wire,
) -> None:
    """A Claude 5 route sends ``thinking`` and replays its earlier thinking blocks. What the
    API does with a replayed block is the API's and cannot be pinned here. That the request
    carries both breakpoints beside ``thinking`` can."""
    provider = _provider("anthropic/claude-sonnet-5-5")
    body = await _sent(wire, provider, [*_second_call(), Message.user(TAIL)])

    assert "thinking" in body
    assert body["messages"][-1]["content"][1]["cache_control"] == MARKER
    assert len(_breakpoints(body)) == 2


# ── the premise: where litellm puts a message's marker ──────────────────────────────────────


async def test_litellm_moves_a_messages_marker_onto_its_block(wire: Wire) -> None:
    """Why a marker on the message is enough. litellm 1.86.2 puts a tool message's marker on
    its ``tool_result`` block and an assistant message's on its text block, and leaves the
    caller's own messages as they were. If an upgrade changes any of that, this says so."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "ask"},
        {
            "role": "assistant",
            "content": "reading",
            "tool_calls": [
                {
                    "id": "t1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "t1",
            "content": "contents",
            "cache_control": dict(MARKER),
        },
        {"role": "assistant", "content": "Done.", "cache_control": dict(MARKER)},
        {"role": "user", "content": "next"},
    ]

    with contextlib.suppress(Exception):
        await lp.litellm.acompletion(
            model=ANTHROPIC,
            messages=messages,
            api_key="sk-test",
            api_base="https://gateway.invalid",
        )

    sent = wire.bodies[0]["messages"]
    tool_result = sent[2]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["cache_control"] == MARKER
    answer = sent[3]["content"][0]
    assert answer["type"] == "text" and answer["cache_control"] == MARKER
    assert messages[2]["cache_control"] == MARKER, "litellm left the caller's list as it was"
    assert messages[3]["cache_control"] == MARKER
