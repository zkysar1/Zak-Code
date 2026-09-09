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
    _REFUSAL_BLOCKER_NUDGE,
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
from zakcode.tools.builtins.write_file import WriteFileTool

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


# ── refusal-is-not-a-blocker (ADR-0118) ───────────────────────────────────────
# Field incident 2026-09-09 (serene): every write of a mangled .py was REFUSED by the write
# firewall (nothing landed), the model read the refusals as the environment "reporting
# syntax errors on valid code", declared an environmental blocker, and asked the user to
# apply a one-line fix by hand. A refusal of the model's own content is not a measured
# blocker: the gate treats a turn whose only failures are such refusals like a turn with
# none, and its nudge says whose problem it is.

HANDOFFS = [
    "I was unable to apply it programmatically due to persistent errors; manual "
    "intervention is required.",
    "You will need to apply the fix manually: change line 47 to use double quotes.",
    "Please apply this change manually and re-run the script.",
    "The user must edit google-drive-list.py and replace the string on line 47.",
]


def test_handing_the_user_an_edit_is_a_blocker_claim() -> None:
    for text in HANDOFFS:
        assert _claims_blocker(text), text
    # advice about a third party, or an answer that merely describes work, is not
    assert not _claims_blocker("The user should apply for a token in the console.")
    assert not _claims_blocker("You must apply for access before the API works.")
    assert not _claims_blocker("The change applies the fix on line 47.")


def _write_loop(provider: Provider, tmp_path: Path, *, max_iterations: int = 6) -> AgentLoop:
    registry = ToolRegistry()
    registry.register(WriteFileTool())
    return AgentLoop(
        provider,
        registry,
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        max_iterations=max_iterations,
    )


def test_a_blocker_over_refusals_of_own_content_gets_the_refusal_nudge(tmp_path: Path) -> None:
    broken = {"path": "tool.py", "content": "def f():\n    x = 'abc\n"}
    provider = _Script(
        [
            LLMResult(tool_calls=[ToolCall(id="c1", name="write_file", arguments=broken)]),
            LLMResult(text=HANDOFFS[0]),
            LLMResult(
                tool_calls=[
                    ToolCall(
                        id="c2",
                        name="write_file",
                        arguments={"path": "tool.py", "content": "def f():\n    x = 'abc'\n"},
                    )
                ]
            ),
            LLMResult(text="Fixed: the string on line 2 was unterminated; tool.py now parses."),
        ]
    )
    # Four iterations: the run-the-script cursor (a different rail) would hold a .py write
    # open past this script; the refusal nudge and the corrected write are what is under test.
    loop = _write_loop(provider, tmp_path, max_iterations=4)
    asyncio.run(loop.arun_turn("write tool.py"))
    assert provider.calls == 4
    assert loop._turn_tool_errors == 1 and loop._turn_content_refusals == 1
    rails = _rails(loop)
    assert any(_REFUSAL_BLOCKER_NUDGE in r for r in rails)
    assert not any(_BLOCKER_NUDGE in r for r in rails)
    assert (tmp_path / "tool.py").read_text() == "def f():\n    x = 'abc'\n"
    assert loop._turn_struggle


def test_a_refusal_beside_a_real_failure_still_counts_as_evidence(tmp_path: Path) -> None:
    broken = {"path": "tool.py", "content": "def f(:\n"}
    provider = _Script(
        [
            LLMResult(
                tool_calls=[
                    ToolCall(id="c1", name="write_file", arguments=broken),
                    ToolCall(id="c2", name="no_such_tool", arguments={}),
                ]
            ),
            LLMResult(text="I am blocked: the tool this needs is not registered here."),
        ]
    )
    loop = _write_loop(provider, tmp_path)
    result = asyncio.run(loop.arun_turn("use the special tool"))
    assert result.stop_reason == "completed"
    assert provider.calls == 2
    assert loop._turn_tool_errors == 2 and loop._turn_content_refusals == 1
    assert not any(_BLOCKER_NUDGE in r or _REFUSAL_BLOCKER_NUDGE in r for r in _rails(loop))
