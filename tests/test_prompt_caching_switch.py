"""An explicit, per-route prompt-caching switch for a Claude behind an OpenAI-compatible gateway
(ADR-0298).

``_supports_cache_control`` is true only for ``anthropic/`` and ``claude*`` model strings, so a
Claude reached as ``openai/Claude4.5`` through a gateway never got a breakpoint. The switch
(``Settings.prompt_caching`` for the default model, ``ZakpickModel.prompt_caching`` per category)
is the operator's word that THIS route takes ``cache_control``: off by default, never inferred
from the model's name, and reported as ``Capabilities.supports_caching``.

Measured on litellm 1.86.2 (2026-10-14): emitting the breakpoint is not enough. litellm's OpenAI
chat config deletes every ``cache_control`` from the messages before the request leaves, in
place, on an ``openai/`` and a ``hosted_vllm/`` route, and keeps it on ``anthropic/``. So with
the switch on the marked messages ride ``extra_body``, which the OpenAI client merges over
``messages`` when it builds the body, and litellm is handed a copy to strip.

Hermetic: ``httpx.AsyncClient.send`` is replaced by a canned answer, so what is asserted about
the wire is what litellm's own client puts in the request. No network, no credentials.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

import zakcode
import zakcode.providers.litellm_provider as lp
from zakcode.agent.prompt import DYNAMIC_BOUNDARY
from zakcode.config import Settings
from zakcode.messages import Message
from zakcode.providers.base import StreamUsage
from zakcode.providers.litellm_provider import LiteLLMProvider
from zakcode.providers.routing import ZakpickModel

GATEWAY = "https://gateway.invalid/v1"
CLAUDE_ALIAS = "openai/Claude4.5"
SYSTEM = f"stable identity and tool guidance\n\n{DYNAMIC_BOUNDARY}\n\ndynamic git status"
PLAIN_SYSTEM = "a bare system prompt with no boundary marker"
MARKER = {"type": "ephemeral"}

#: The usage a gateway sent on a later turn of one conversation (ADR-0294): the cached part is in
#: the total and nowhere else.
UNCACHED, CACHED, COMPLETION = 3, 49_445, 120
GATEWAY_USAGE = {
    "prompt_tokens": UNCACHED,
    "completion_tokens": COMPLETION,
    "total_tokens": UNCACHED + CACHED + COMPLETION,
}

#: Routes whose litellm config removes the marker (measured), so the carry applies.
STRIPPED_ROUTES = ["openai/Claude4.5", "hosted_vllm/Claude4.5"]


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Settings read the cwd .env, the config home and real env vars: isolate all three so a dev
    # box's configuration can never change a verdict. The provider's one-second request pacing
    # is not under test either.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ZAKCODE_HOME", str(tmp_path / "confighome"))
    for var in (
        "ZAKCODE_PROMPT_CACHING",
        "ZAKCODE_ZAKPICK_MODELS",
        "ZAKCODE_DEFAULT_MODEL",
        "ZAKCODE_API_BASE",
        "ZAKCODE_EXTRA_BODY",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(lp, "_MIN_REQUEST_INTERVAL_S", 0.0)


# ── the wire ────────────────────────────────────────────────────────────────────────────────


def _sse(usage: dict[str, int]) -> bytes:
    chunk = {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "alias"}
    frames = [
        {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ready"}}]},
        {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {**chunk, "choices": [], "usage": usage},
    ]
    body = b"".join(b"data: " + json.dumps(frame).encode() + b"\n\n" for frame in frames)
    return body + b"data: [DONE]\n\n"


class Wire:
    """Every request litellm's own client sends, answered from a canned body."""

    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []
        #: A 400 sentence for the FIRST request, so the call is re-issued once.
        self.refuse_first: str | None = None
        self.usage: dict[str, int] = {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> Wire:
    seen = Wire()

    async def send(self: httpx.AsyncClient, request: httpx.Request, **kw: Any) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen.bodies.append(body)
        if seen.refuse_first and len(seen.bodies) == 1:
            return httpx.Response(400, json={"message": seen.refuse_first}, request=request)
        if body.get("stream"):
            return httpx.Response(
                200,
                content=_sse(seen.usage),
                headers={"content-type": "text/event-stream"},
                request=request,
            )
        answer = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "alias",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ready"},
                    "finish_reason": "stop",
                }
            ],
            "usage": seen.usage,
        }
        return httpx.Response(200, json=answer, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    return seen


def _breakpoints_in(messages: Any) -> int:
    """How many cache breakpoints these messages carry, counted without the code under test."""
    return sum(
        1
        for message in messages or ()
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict) and "cache_control" in block
    )


def _marked(text: str = "stable") -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": text, "cache_control": dict(MARKER)},
                {"type": "text", "text": "tail"},
            ],
        },
        {"role": "user", "content": "hi"},
    ]


def _provider(model: str = CLAUDE_ALIAS, *, switch: bool = False) -> LiteLLMProvider:
    return LiteLLMProvider(
        Settings(
            default_model=model,
            api_base=GATEWAY,
            api_key="sk-test",
            context_window=131_072,
            prompt_caching=switch,
            _env_file=None,
        )
    )


def _build(provider: LiteLLMProvider, system: str) -> dict[str, Any]:
    wire_messages = provider._translate_messages([Message.user("hi")], system)
    return provider._build_kwargs(wire_messages, None)


# ── the switch ──────────────────────────────────────────────────────────────────────────────


def test_the_switch_is_off_by_default() -> None:
    assert Settings(_env_file=None).prompt_caching is False
    assert ZakpickModel(model="m").prompt_caching is False


def test_the_switch_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZAKCODE_PROMPT_CACHING", "true")

    assert Settings(_env_file=None).prompt_caching is True


def test_a_category_carries_its_own_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = {"model": "Claude4.5", "source": "openai"}
    monkeypatch.setenv(
        "ZAKCODE_ZAKPICK_MODELS",
        json.dumps({"deep_code": {**entry, "prompt_caching": True}, "summarize": entry}),
    )

    models = Settings(_env_file=None).zakpick_models

    assert models["deep_code"].prompt_caching is True
    assert models["summarize"].prompt_caching is False


@pytest.mark.parametrize(
    "model",
    [
        "openai/Claude4.5",
        "openai/claude-sonnet-4-5",
        "hosted_vllm/Claude4.5",
        # The registry knows these two take caching. The name still gives no breakpoint: only an
        # ``anthropic/`` or a ``claude`` prefix does, and nothing here widens that.
        "bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0",
        "vertex_ai/claude-sonnet-4-5",
    ],
)
def test_a_claude_alias_gets_no_breakpoint_unless_the_route_says_so(model: str) -> None:
    kwargs = _build(_provider(model), SYSTEM)

    assert kwargs["messages"][0]["content"] == SYSTEM
    assert "messages" not in (kwargs.get("extra_body") or {})


@pytest.mark.parametrize("model", [*STRIPPED_ROUTES, "openai/gpt-oss-120b"])
def test_the_switch_puts_the_breakpoint_on_the_stable_block(model: str) -> None:
    """Whatever the model is called: the switch is the operator's word, not a guess."""
    kwargs = _build(_provider(model, switch=True), SYSTEM)

    stable, dynamic = kwargs["extra_body"]["messages"][0]["content"]
    assert stable["cache_control"] == MARKER
    assert stable["text"].rstrip().endswith(DYNAMIC_BOUNDARY)
    assert "cache_control" not in dynamic
    assert "dynamic git status" in dynamic["text"]


def test_a_system_prompt_without_the_boundary_carries_nothing_even_with_the_switch() -> None:
    kwargs = _build(_provider(switch=True), PLAIN_SYSTEM)

    assert kwargs["messages"][0]["content"] == PLAIN_SYSTEM
    assert "messages" not in (kwargs.get("extra_body") or {})


@pytest.mark.parametrize("switch", [False, True])
def test_an_anthropic_route_is_the_same_with_the_switch_or_without(switch: bool) -> None:
    """litellm keeps the marker on this route, so nothing rides ``extra_body``: OpenAI-shaped
    messages in an Anthropic request body would replace its own."""
    kwargs = _build(_provider("anthropic/claude-sonnet-4-6", switch=switch), SYSTEM)

    assert kwargs["messages"][0]["content"][0]["cache_control"] == MARKER
    assert "messages" not in (kwargs.get("extra_body") or {})


def test_the_capabilities_report_the_switch() -> None:
    assert _provider(switch=False).capabilities().supports_caching is False
    assert _provider(switch=True).capabilities().supports_caching is True


# ── the premise: what litellm does with the marker ──────────────────────────────────────────


@pytest.mark.parametrize("model", STRIPPED_ROUTES)
async def test_litellm_removes_the_marker_from_the_messages_it_is_given(
    wire: Wire, model: str
) -> None:
    """Why the carry exists. If a litellm upgrade stops doing this, the carry is dead weight and
    this test says so; if it starts doing it on more routes, the carry's route list is short."""
    messages = _marked()

    await lp.litellm.acompletion(
        model=model, messages=messages, api_key="sk-test", api_base=GATEWAY
    )

    assert _breakpoints_in(wire.bodies[0]["messages"]) == 0
    assert _breakpoints_in(messages) == 0, "litellm strips the caller's own list in place"


async def test_litellm_keeps_the_marker_on_an_anthropic_route(wire: Wire) -> None:
    with contextlib.suppress(Exception):  # the canned answer is OpenAI-shaped; the request is not
        await lp.litellm.acompletion(
            model="anthropic/claude-sonnet-4-6",
            messages=_marked(),
            api_key="sk-test",
            api_base="https://gateway.invalid",
        )

    assert "cache_control" in json.dumps(wire.bodies[0])


# ── what reaches the gateway ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("model", STRIPPED_ROUTES)
async def test_without_the_switch_no_breakpoint_reaches_the_wire(wire: Wire, model: str) -> None:
    await _provider(model).acomplete([Message.user("hi")], system=SYSTEM)

    [body] = wire.bodies
    assert _breakpoints_in(body["messages"]) == 0
    assert body["messages"][0]["content"] == SYSTEM


@pytest.mark.parametrize("model", STRIPPED_ROUTES)
async def test_with_the_switch_the_breakpoint_reaches_the_wire(wire: Wire, model: str) -> None:
    await _provider(model, switch=True).acomplete([Message.user("hi")], system=SYSTEM)

    [body] = wire.bodies
    stable, dynamic = body["messages"][0]["content"]
    assert stable["cache_control"] == MARKER
    assert "cache_control" not in dynamic
    assert [message["role"] for message in body["messages"]] == ["system", "user"]
    assert "messages" not in body.get("extra_body", {}), "the client merged it into the body"


async def test_a_streamed_call_carries_the_breakpoint_too(wire: Wire) -> None:
    events = [
        event async for event in _provider(switch=True).astream([Message.user("hi")], system=SYSTEM)
    ]

    [body] = wire.bodies
    assert body["stream"] is True
    assert body["messages"][0]["content"][0]["cache_control"] == MARKER
    assert any(isinstance(event, StreamUsage) for event in events)


async def test_a_reissued_call_carries_the_breakpoint_again(wire: Wire) -> None:
    """litellm strips the list it is handed in place. Had it been handed the provider's own, the
    call re-issued after a refusal would be rebuilt from messages with no marker left."""
    wire.refuse_first = (
        "This model doesn't support the stopSequences field. Remove stopSequences and try again."
    )

    result = await _provider(switch=True).acomplete(
        [Message.user("hi")], system=SYSTEM, stop=["</tool_call>"]
    )

    assert result.text == "ready"
    assert ["stop" in body for body in wire.bodies] == [True, False]
    assert [_breakpoints_in(body["messages"]) for body in wire.bodies] == [1, 1]


async def test_a_gateway_that_refuses_the_marker_is_served_the_plain_request(wire: Wire) -> None:
    """The carried messages are a field WE added, so a refusal that names the path to the marker
    is learned like any other (ADR-0181): the call is re-issued once without it, and the session
    stays plain. Without the guard the re-issued call carried it straight back (ADR-0285)."""
    wire.refuse_first = "messages.0.content.0.cache_control: Extra inputs are not permitted"
    provider = _provider(switch=True)

    first = await provider.acomplete([Message.user("hi")], system=SYSTEM)
    later = await provider.acomplete([Message.user("again")], system=SYSTEM)

    assert first.text == later.text == "ready"
    assert [_breakpoints_in(body["messages"]) for body in wire.bodies] == [1, 0, 0]
    assert "messages" in provider.rejected_request_fields


# ── the cached prompt a gateway reports only in its total (ADR-0294) ─────────────────────────


async def test_a_buffered_call_on_the_opted_in_route_reports_the_cached_prompt(
    wire: Wire,
) -> None:
    """The breakpoint check runs after the call, on what litellm stripped from the request
    kwargs. It reads the messages that went on the wire, or the correction never fires."""
    wire.usage = GATEWAY_USAGE

    result = await _provider(switch=True).acomplete([Message.user("hi")], system=SYSTEM)

    assert result.usage.prompt_tokens == UNCACHED + CACHED
    assert result.usage.cache_unsplit_tokens == CACHED


async def test_a_streamed_call_on_the_opted_in_route_reports_the_cached_prompt(
    wire: Wire,
) -> None:
    wire.usage = GATEWAY_USAGE

    events = [e async for e in _provider(switch=True).astream([Message.user("hi")], system=SYSTEM)]

    [streamed] = [e.usage for e in events if isinstance(e, StreamUsage)]
    assert streamed.prompt_tokens == UNCACHED + CACHED
    assert streamed.cache_unsplit_tokens == CACHED
    assert streamed.total_tokens == UNCACHED + CACHED + COMPLETION


async def test_without_the_switch_the_same_report_is_left_as_it_is(wire: Wire) -> None:
    wire.usage = GATEWAY_USAGE

    result = await _provider().acomplete([Message.user("hi")], system=SYSTEM)

    assert result.usage.prompt_tokens == UNCACHED
    assert result.usage.cache_unsplit_tokens == 0


async def test_a_streamed_call_without_the_switch_leaves_the_report_as_it_is(wire: Wire) -> None:
    """The backend's own total is read back only for a request that carried a breakpoint. With
    none, the excess of the total over prompt plus completion may be something else (ADR-0294)."""
    wire.usage = GATEWAY_USAGE

    events = [e async for e in _provider().astream([Message.user("hi")], system=SYSTEM)]

    [streamed] = [e.usage for e in events if isinstance(e, StreamUsage)]
    assert streamed.prompt_tokens == UNCACHED
    assert streamed.cache_unsplit_tokens == 0
    assert streamed.total_tokens == UNCACHED + COMPLETION, "litellm's own total, untouched"


async def test_litellm_rebuilds_a_streams_usage_without_the_backends_total(wire: Wire) -> None:
    """The premise of reading the raw chunks back (ADR-0298), measured on litellm 1.86.2: the
    usage chunk a stream yields is prompt plus completion, the backend's total survives only in
    the wrapper's stored chunks. If a litellm bump changes either half, this names which."""
    import litellm

    wire.usage = GATEWAY_USAGE
    stream = await litellm.acompletion(
        model=CLAUDE_ALIAS,
        messages=[{"role": "user", "content": "hi"}],
        api_base=GATEWAY,
        api_key="sk-test",
        stream=True,
        stream_options={"include_usage": True},
    )
    yielded = [chunk async for chunk in stream if getattr(chunk, "usage", None) is not None]

    assert [c.usage.total_tokens for c in yielded] == [UNCACHED + COMPLETION]
    assert LiteLLMProvider._stream_reported_total(stream) == UNCACHED + CACHED + COMPLETION


class _KeptChunks:
    """The one thing of litellm's stream wrapper the helper reads: the raw chunks it kept."""

    def __init__(self, chunks: Any) -> None:
        self.chunks = chunks


def test_the_reported_total_is_the_one_on_the_last_usage_chunk() -> None:
    kept = _KeptChunks(
        [
            {"choices": [{"delta": {"content": "a"}}]},
            {"usage": {"prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 41_000}},
            {"usage": {"prompt_tokens": 3, "completion_tokens": 120, "total_tokens": 49_568}},
            {"choices": []},
        ]
    )

    assert LiteLLMProvider._stream_reported_total(kept) == 49_568


@pytest.mark.parametrize(
    "stream",
    [object(), _KeptChunks(None), _KeptChunks(42), _KeptChunks([]), _KeptChunks([{"choices": []}])],
    ids=["no-chunks-attribute", "none", "not-iterable", "empty", "no-usage-chunk"],
)
def test_a_stream_that_kept_no_usage_reports_no_total(stream: Any) -> None:
    """Nothing kept, or kept in a shape this does not know: 0, which is the old behaviour."""
    assert LiteLLMProvider._stream_reported_total(stream) == 0


# ── the switch follows the route ────────────────────────────────────────────────────────────


def _alias(**extra: object) -> dict[str, object]:
    return {"model": "Claude4.5", "source": "openai", "context_window": 131_072, **extra}


def _agent(tmp_path: Path, **kwargs: Any) -> zakcode.Agent:
    return zakcode.Agent(workspace_root=tmp_path, api_base=GATEWAY, api_key="sk-test", **kwargs)


def test_the_startup_category_takes_its_entrys_switch(tmp_path: Path) -> None:
    agent = _agent(
        tmp_path,
        default_model="zakpick",
        zakpick_models={"deep_code": _alias(prompt_caching=True), "summarize": _alias()},
    )

    assert agent.settings.prompt_caching is True
    assert agent.provider.capabilities().supports_caching is True


def test_two_categories_naming_one_model_keep_their_own_switch(tmp_path: Path) -> None:
    agent = _agent(
        tmp_path,
        default_model="zakpick",
        zakpick_models={"deep_code": _alias(prompt_caching=True), "summarize": _alias()},
    )

    summarize, model = agent._resolve_task_provider("summarize")
    deep, _ = agent._resolve_task_provider("deep_code")

    assert model == CLAUDE_ALIAS
    assert summarize.capabilities().supports_caching is False
    assert deep.capabilities().supports_caching is True


def test_two_categories_naming_another_model_keep_their_own_switch(tmp_path: Path) -> None:
    """Neither is the default model's route, so both are built, and a cache key that left the
    switch out would hand the second one the first one's provider."""
    agent = _agent(
        tmp_path,
        default_model="zakpick",
        zakpick_models={
            "deep_code": {"model": "gpt-oss-120b", "source": "openai", "context_window": 131_072},
            "plan": _alias(prompt_caching=True),
            "delegate": _alias(),
        },
    )

    plan, _ = agent._resolve_task_provider("plan")
    delegate, _ = agent._resolve_task_provider("delegate")

    assert plan.capabilities().supports_caching is True
    assert delegate.capabilities().supports_caching is False


def test_the_default_models_switch_is_read_from_settings(tmp_path: Path) -> None:
    agent = _agent(
        tmp_path, default_model=CLAUDE_ALIAS, context_window=131_072, prompt_caching=True
    )

    assert agent.provider.capabilities().supports_caching is True


def test_another_model_does_not_inherit_the_default_models_switch(tmp_path: Path) -> None:
    """The switch is a fact about a route. A fallback or a role model is another route."""
    agent = _agent(
        tmp_path, default_model=CLAUDE_ALIAS, context_window=131_072, prompt_caching=True
    )

    other = agent._build_provider("ollama_chat/llama3.1", context_window=8_192)

    assert other.capabilities().supports_caching is False
