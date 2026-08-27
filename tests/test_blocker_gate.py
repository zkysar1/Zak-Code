"""Blocker-without-evidence guard + route-label unwrapping (ADR-0036).

Field incident 2026-08-27 (serene): the model read a hook's source, decided the session id
it injects "is not available in this execution environment", and ended three turns on that
sentence. No tool call had failed — the skill's own one-line check was never run, and would
have passed. A blocker nobody measured is a conclusion, not a finding: one nudge asks for
the failing probe or the next step.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from zakcode.agent.loop import (
    _BLOCKER_NUDGE,
    AgentLoop,
    _claims_blocker,
    _provider_label,
)
from zakcode.events import AgentStatus
from zakcode.messages import Message
from zakcode.providers.base import (
    Capabilities,
    LLMResult,
    Provider,
    ProviderStreamEvent,
    StreamDone,
    StreamTextDelta,
    ToolCall,
)
from zakcode.session.store import Session
from zakcode.tools.base import ToolRegistry

SERENE_BLOCKERS = [
    "I am blocked because the MIND_SID environment variable, necessary for the /start "
    "skill, is not available in this execution environment.",
    "I cannot force the recovery without the MIND_SID.",
    "I cannot proceed with the --recover or --force options because the agent's session "
    "ID (MIND_SID) is not available.",
    "Please provide the MIND_SID or ensure it's set correctly in the environment.",
]


def test_blocker_claims_are_first_person_only() -> None:
    for text in SERENE_BLOCKERS:
        assert _claims_blocker(text), text
    assert _claims_blocker("I'm stuck: the API key is missing.")
    # Answers that mention absence are not blocker claims.
    assert not _claims_blocker("Two fields are missing from the config: name and port.")
    assert not _claims_blocker("The service is not available on port 80; nginx listens on 8080.")
    assert not _claims_blocker("I'm not blocked — continuing with the next step.")
    assert not _claims_blocker("Done. The file now has three sections.")


class _Script(Provider):
    def __init__(self, results: list[LLMResult]) -> None:
        self._results = results
        self.calls = 0

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:
        self.calls += 1
        return self._results[self.calls - 1]

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=200_000)


class _Stream(Provider):
    def __init__(self, texts: list[str]) -> None:
        self._texts = texts
        self.calls = 0

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:  # pragma: no cover — streaming path only
        raise AssertionError("buffered path must not run")

    async def astream(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> AsyncIterator[ProviderStreamEvent]:
        self.calls += 1
        yield StreamTextDelta(text=self._texts[self.calls - 1])
        yield StreamDone(finish_reason="stop")

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=200_000)


def _loop(provider: Provider, tmp_path: Path) -> AgentLoop:
    return AgentLoop(
        provider,
        ToolRegistry(),
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        max_iterations=6,
    )


def _rails(loop: AgentLoop) -> list[str]:
    return [m.text for m in loop.session.messages if m.role == "user"][1:]


def test_unmeasured_blocker_is_nudged_once_then_the_turn_continues(tmp_path: Path) -> None:
    provider = _Script(
        [
            LLMResult(text=SERENE_BLOCKERS[0]),
            LLMResult(text="Probed: `echo $MIND_SID` prints a value. Continuing — bound sera."),
        ]
    )
    loop = _loop(provider, tmp_path)
    result = asyncio.run(loop.arun_turn("/start sera --mode assistant"))
    assert result.stop_reason == "completed"
    assert provider.calls == 2
    assert any(_BLOCKER_NUDGE in r for r in _rails(loop))
    assert loop._turn_struggle  # a measured-nothing blocker is a struggle signal


def test_a_blocker_a_failed_tool_call_demonstrated_is_not_nudged(tmp_path: Path) -> None:
    provider = _Script(
        [
            LLMResult(
                tool_calls=[ToolCall(id="c1", name="no_such_tool", arguments={})]
            ),  # fails: unknown tool → an error result block
            LLMResult(text="I am blocked: the tool this needs is not registered here."),
        ]
    )
    loop = _loop(provider, tmp_path)
    result = asyncio.run(loop.arun_turn("use the special tool"))
    assert result.stop_reason == "completed"
    assert provider.calls == 2
    assert loop._turn_tool_errors == 1
    assert not any(_BLOCKER_NUDGE in r for r in _rails(loop))


def test_streaming_unmeasured_blocker_is_nudged_and_announced(tmp_path: Path) -> None:
    provider = _Stream(
        [
            SERENE_BLOCKERS[1],
            "Probed it: the id is present. Proceeding with the recovery step.",
        ]
    )
    loop = _loop(provider, tmp_path)

    async def _collect() -> list[Any]:
        return [ev async for ev in loop.astream_turn("can you force the recovery?")]

    events = asyncio.run(_collect())
    statuses = [ev.message for ev in events if isinstance(ev, AgentStatus)]
    assert any(s.startswith("completion declares a blocker") for s in statuses)
    assert provider.calls == 2
    assert any(_BLOCKER_NUDGE in r for r in _rails(loop))


class _Inner:
    model = "vertex_ai/gemini-2.5-flash-lite"


class _Wrapper:
    def __init__(self) -> None:
        self.inner = _Inner()


class _Nameless:
    pass


def test_route_label_unwraps_adapters_to_the_model() -> None:
    assert _provider_label(_Wrapper()) == "vertex_ai/gemini-2.5-flash-lite"
    assert _provider_label(_Inner()) == "vertex_ai/gemini-2.5-flash-lite"
    assert _provider_label(_Nameless()) == "_Nameless"
