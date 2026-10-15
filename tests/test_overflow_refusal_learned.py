"""An override of the overflow retry that a provider refused naming neither is learned for the
session (ADR-0315).

Read on 2026-10-15. A refusal of the overflow retry that names nothing drops the thinking switch
first and the raised cap second (ADR-0307), on that call's own overrides, so nothing was kept.
Where the cap was the offender, every overflow retry of the session paid two refused requests,
and the request that was taken carried neither override, so a model that needed thinking off to
answer never got it. Now the override whose drop let a call through is remembered for that
provider for the session, and later overflow retries leave it off. CONFIG.md said a call whose
field is refused by name is "re-issued once"; with several refusals it is re-issued once per
refusal learned, at most ``_MAX_REQUEST_REPAIRS`` times, and the rows now say so.

Hermetic: the loop meets scripted routes, and the wire tests fake ``litellm.acompletion``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import zakcode.providers.litellm_provider as lp
from zakcode.agent.loop import AgentLoop
from zakcode.config import load_settings
from zakcode.messages import Message
from zakcode.providers.base import AuthError, Capabilities, LLMResult, Provider, RequestFailed
from zakcode.providers.litellm_provider import LiteLLMProvider
from zakcode.providers.thinking import THINKING_SWITCH_KEY
from zakcode.session.store import Session
from zakcode.tools.base import ToolRegistry
from zakcode.usage import Usage

#: The cap every call is sent unless it asks for another (ADR-0018).
DEFAULT_CAP = lp._MAX_COMPLETION_TOKENS
#: The overflow retry's cap where the window has room (ADR-0307).
RAISED = 2 * DEFAULT_CAP
#: A refusal that names no field, Bedrock's own words.
UNNAMED = "The model returned the following errors: Validation Error"
#: Reasoning that no cap ends.
FOREVER = 10**9
CONFIG = Path(__file__).resolve().parents[1] / "docs" / "CONFIG.md"


@pytest.fixture(autouse=True)
def _no_request_pacing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The provider spaces request starts a second apart; that sleep is not under test."""
    monkeypatch.setattr(lp, "_MIN_REQUEST_INTERVAL_S", 0.0)


class Route(Provider):
    """A route to a reasoning model as the loop meets it, each call recorded as (switch, cap).

    A call carrying an override the route ``refuses`` (the thinking switch, or a cap of its own)
    is refused naming nothing, as Bedrock's 400 reaches the loop, and takes no step. Otherwise
    each scripted step is the output a completion needs before it answers: under the cap the call
    asked for (the default when it asked for none) it ends ``length`` with nothing visible. A
    step that is an exception is raised instead.
    """

    def __init__(
        self, script: list[int | Exception], *, refuses: frozenset[str] = frozenset()
    ) -> None:
        self.script = list(script)
        self.refuses = refuses
        self.calls: list[tuple[bool, int | None]] = []
        self.refused = 0

    async def acomplete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        switch, cap = bool(kwargs.get("extra_body")), kwargs.get("max_tokens")
        self.calls.append((switch, cap))
        if ("switch" in self.refuses and switch) or ("cap" in self.refuses and cap is not None):
            self.refused += 1
            raise RequestFailed(UNNAMED)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        room = cap or DEFAULT_CAP
        spent = min(room, step)
        usage = Usage(prompt_tokens=100, completion_tokens=spent, total_tokens=100 + spent)
        if room < step:
            return LLMResult(finish_reason="length", usage=usage)
        return LLMResult(text="the answer", finish_reason="stop", usage=usage)

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(context_window=131_072)


def _make_loop(provider: Provider, tmp_path: Path, **kwargs: Any) -> AgentLoop:
    return AgentLoop(
        provider,
        ToolRegistry(),
        Session(cwd=str(tmp_path), model="openai/route-a"),
        settings=load_settings(workspace_root=tmp_path),
        **kwargs,
    )


async def _turn(loop: AgentLoop, path: str) -> str | None:
    """One turn's stop reason, on the buffered or the streamed path."""
    if path == "buffered":
        return (await loop.arun_turn("question")).stop_reason
    events = [ev async for ev in loop.astream_turn("question")]
    done = [ev for ev in events if getattr(ev, "stop_reason", None) is not None]
    return done[-1].stop_reason if done else None


def _notes(loop: AgentLoop, kind: str) -> list[str]:
    return [e.detail for e in loop._trace.of_kind("intervention") if e.data.get("kind") == kind]


# ── learned for the provider, for the session ────────────────────────────────────────────────


@pytest.mark.parametrize("path", ["buffered", "streamed"])
async def test_a_refused_raised_cap_is_left_off_the_next_retry(tmp_path: Path, path: str) -> None:
    # The cap is the offender. The first retry pays two refusals, the switch dropped first, and
    # the call is taken without either; the next retry of the session sends the switch alone.
    route = Route([FOREVER, 4_000, FOREVER, 4_000], refuses=frozenset({"cap"}))
    loop = _make_loop(route, tmp_path)
    assert await _turn(loop, path) == "completed"
    first = list(route.calls)
    assert len(_notes(loop, "overflow_retry_refused")) == 2
    [learned] = _notes(loop, "overflow_override_learned")
    assert "the raised output cap" in learned
    assert await _turn(loop, path) == "completed"
    assert first == [(False, None), (True, RAISED), (False, RAISED), (False, None)]
    assert route.calls[len(first) :] == [(False, None), (True, None)]
    assert route.refused == 2
    assert _notes(loop, "overflow_retry_refused") == []


@pytest.mark.parametrize("path", ["buffered", "streamed"])
async def test_a_refused_switch_is_left_off_the_next_retry(tmp_path: Path, path: str) -> None:
    # The switch is the offender, as the report of ADR-0307 measured on one route: the next
    # retry sends the raised cap alone.
    route = Route([FOREVER, 12_000, FOREVER, 12_000], refuses=frozenset({"switch"}))
    loop = _make_loop(route, tmp_path)
    assert await _turn(loop, path) == "completed"
    first = list(route.calls)
    [learned] = _notes(loop, "overflow_override_learned")
    assert "the thinking switch" in learned
    assert await _turn(loop, path) == "completed"
    assert first == [(False, None), (True, RAISED), (False, RAISED)]
    assert route.calls[len(first) :] == [(False, None), (False, RAISED)]
    assert route.refused == 1


async def test_a_route_that_refuses_both_is_taught_each_once(tmp_path: Path) -> None:
    # The cap is learned on the first retry, its drop being the one that let the call through,
    # and the switch on the second; the third pays nothing.
    route = Route([FOREVER, 4_000] * 3, refuses=frozenset({"switch", "cap"}))
    loop = _make_loop(route, tmp_path)
    refused = []
    for _ in range(3):
        before = route.refused
        assert await _turn(loop, "buffered") == "completed"
        refused.append(route.refused - before)
    assert refused == [2, 1, 0]
    assert route.calls[-2:] == [(False, None), (False, None)]


# ── what is not learned ──────────────────────────────────────────────────────────────────────


async def test_another_provider_is_sent_both_overrides(tmp_path: Path) -> None:
    taught = Route([FOREVER, 4_000], refuses=frozenset({"cap"}))
    loop = _make_loop(taught, tmp_path)
    assert await _turn(loop, "buffered") == "completed"
    other = Route([FOREVER, 12_000])
    loop.provider = other  # as a change of route swaps it
    assert await _turn(loop, "buffered") == "completed"
    assert other.calls == [(False, None), (True, RAISED)]


@pytest.mark.parametrize("path", ["buffered", "streamed"])
async def test_a_call_another_provider_took_teaches_nothing(tmp_path: Path, path: str) -> None:
    # The switch is refused and the call without it fails some other way, so the turn fails over
    # and another provider takes it. Nothing says the first would have, so it learns nothing.
    first = Route([FOREVER, AuthError("expired"), FOREVER, 12_000], refuses=frozenset({"switch"}))
    other = Route([12_000])
    loop = _make_loop(first, tmp_path, model_failover=lambda exc: (other, "route-b"))
    assert await _turn(loop, path) == "completed"
    assert other.calls == [(False, RAISED)]
    assert len(_notes(loop, "overflow_retry_refused")) == 1
    assert _notes(loop, "overflow_override_learned") == []
    loop.provider = first
    assert await _turn(loop, path) == "completed"
    assert first.calls[-3:] == [(False, None), (True, RAISED), (False, RAISED)]


async def test_a_new_session_is_sent_both_overrides(tmp_path: Path) -> None:
    route = Route([FOREVER, 4_000, FOREVER, 4_000], refuses=frozenset({"cap"}))
    assert await _turn(_make_loop(route, tmp_path), "buffered") == "completed"
    assert await _turn(_make_loop(route, tmp_path), "buffered") == "completed"
    assert route.refused == 4


# ── on the wire, through the provider ────────────────────────────────────────────────────────


class BadRequestError(Exception):
    """Named like litellm's: the provider matches the class by MRO name, plus the status."""

    status_code = 400


def _switch(kwargs: dict[str, Any]) -> bool:
    return THINKING_SWITCH_KEY in (kwargs.get("extra_body") or {})


def _completion(answered: bool, spent: int) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="the answer" if answered else "", tool_calls=None),
                finish_reason="stop" if answered else "length",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1_000, completion_tokens=spent, total_tokens=1_000),
        _hidden_params={"response_cost": 0.0},
    )


async def _chunks(answered: bool) -> AsyncIterator[Any]:
    yield SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content="the answer" if answered else None, tool_calls=None),
                finish_reason="stop" if answered else "length",
            )
        ],
        usage=None,
    )


class Gateway:
    """A scripted ``litellm.acompletion`` for one route, every request recorded. The route holds
    an output limit of its own, 12,000, and refuses a cap past it with a 400 that names nothing.
    Its model answers within 2,000 tokens with thinking off, and needs 20,000 with it on."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if kwargs["max_tokens"] > 12_000:
            raise BadRequestError(f"litellm.BadRequestError: OpenAIException - {UNNAMED}")
        needs = 2_000 if _switch(kwargs) else 20_000
        answered = kwargs["max_tokens"] >= needs
        if kwargs.get("stream"):
            return _chunks(answered)
        return _completion(answered, min(needs, kwargs["max_tokens"]))


@pytest.mark.parametrize("path", ["buffered", "streamed"])
async def test_on_the_wire_a_refused_cap_is_learned_and_the_switch_answers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
) -> None:
    # The first retry is refused with the switch and the raised cap, then with the cap, and is
    # taken plain; thinking on, it overflows again. The second retry leaves the cap off, keeps
    # the switch, and answers, where it used to pay the same two refusals and give up.
    gateway = Gateway()
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)
    provider = LiteLLMProvider(
        model="openai/route-a",
        api_base="https://gateway.invalid/v1",
        api_key="sk-test",
        context_window=131_072,
    )
    loop = _make_loop(provider, tmp_path)
    assert await _turn(loop, path) == "completed"
    assert [call["max_tokens"] for call in gateway.calls] == [8_192, 16_384, 16_384, 8_192, 8_192]
    assert [_switch(call) for call in gateway.calls] == [False, True, False, False, True]
    # Nothing in the refusal names a field, so the provider learns nothing by name.
    assert provider.rejected_request_fields == []


# ── the bound the docs state ─────────────────────────────────────────────────────────────────


def test_config_states_the_bound_on_re_issuing_a_refused_field() -> None:
    # A call whose field a provider refuses by name is re-issued once per refusal learned, at
    # most _MAX_REQUEST_REPAIRS times (ADR-0303): the rows that said "once" state the bound.
    text = " ".join(CONFIG.read_text(encoding="utf-8").split())
    assert lp._MAX_REQUEST_REPAIRS == 3
    assert "re-issued once" not in text
    assert text.count("once per refusal learned, at most three times") == 4
