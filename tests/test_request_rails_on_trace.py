"""ADR-0204: the transcript is not the prompt, so the trace says what rode each request; and
a gate that speaks to the model says so on the trace.

Zak Code ends a request with up to three messages the session never stores (hook context, the
turn's prompt context, the plan reminder in one of its two forms). Four served samples could
not see that the finished plan's "answer now" line was the LAST thing the model read after a
refused stop, because no session file and no trace held it. Every main request's ``usage``
event now lists the kinds that rode it (an empty list when none did, so a missing key can only
mean an older build), and says when the tail was withheld for the prompt cache (ADR-0193).

The plan gate, the recipe gate and the project-verifier gate each nudged the model without a
trace event, so a turn they had kept open read, from the trace, like a model that carried on
by itself. Each now notes the nudge, in both twins.

Hermetic: scripted providers, no network. The same scripts drive the buffered and the
streaming twin (the base ``astream`` wraps ``acomplete``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from zakcode.agent.loop import AgentLoop
from zakcode.config import Settings
from zakcode.messages import Message
from zakcode.permissions import PermissionMode, PermissionPolicy
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.store import Session
from zakcode.tools import default_registry
from zakcode.tools.base import Tool, ToolContext, ToolRegistry, ToolResult, ToolSpec
from zakcode.usage import Usage

MODEL = "fake/scripted"


class _Scripted(Provider):
    """Canned completions in order (the last repeats), under a fixed model id. Keeps, per call,
    which plan reminder the request it was HANDED ended with: the wire's side of the claim."""

    def __init__(self, results: list[LLMResult]) -> None:
        self._results = results
        self.calls = 0
        self.ended_with: list[str | None] = []

    async def acomplete(self, messages: Any, *, system: Any = None, tools: Any = None, **kw: Any):  # type: ignore[override]
        self.ended_with.append(_reminder_kind(messages[-1]) if messages else None)
        index = min(self.calls, len(self._results) - 1)
        self.calls += 1
        return self._results[index]

    def model_id(self) -> str:
        return MODEL

    def count_tokens(self, messages: Any, *, system: Any = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=8192)


def _reminder_kind(message: Message) -> str | None:
    """Which plan reminder a message IS, read from its own text and not from any label."""
    text = "".join(str(getattr(block, "text", "") or "") for block in message.blocks)
    if text.startswith("[plan] Plan complete"):
        return "plan_complete"
    return "plan" if text.startswith("[plan] ") else None


def _plan(*statuses: str) -> LLMResult:
    tasks = [{"title": f"S{i}", "status": s, "note": "ok"} for i, s in enumerate(statuses)]
    call = ToolCall(id=f"p{len(statuses)}", name="update_plan", arguments={"tasks": tasks})
    return LLMResult(text="", tool_calls=[call], usage=Usage(total_tokens=1))


def _say(text: str) -> LLMResult:
    return LLMResult(text=text, tool_calls=[], usage=Usage(total_tokens=1))


def _judge_ok() -> LLMResult:
    """The decomposition judge's verdict (ADR-0050): a side call, strong, so it stays silent."""
    scores = {"coverage": 0.9, "granularity": 0.9, "ordering": 0.9, "soundness": 0.9}
    return LLMResult(text=json.dumps({"scores": scores}), usage=Usage(total_tokens=2))


def _review_ok() -> LLMResult:
    """The fresh-eyes review of a finished plan (ADR-0117): a side call."""
    return LLMResult(text='{"approved": true, "issues": ""}', usage=Usage(total_tokens=2))


def _loop(provider: Provider, tmp_path: Path, **kw: Any) -> AgentLoop:
    session = Session(cwd=str(tmp_path), model=MODEL)
    registry = kw.pop("registry", None) or default_registry()
    return AgentLoop(provider, registry, session, workspace_root=tmp_path, max_iterations=20, **kw)


async def _run(loop: AgentLoop, text: str, *, streamed: bool) -> None:
    if streamed:
        async for _ in loop.astream_turn(text):
            pass
    else:
        await loop.arun_turn(text)


def _requests(loop: AgentLoop) -> list[tuple[list[str], bool]]:
    """``(rails, rested)`` of every main request of the turn, in order. A ``usage`` event with
    no ``rails`` key is a failure by itself: absence must never mean "nothing rode"."""
    return [(e.data["rails"], e.data["rails_rested"]) for e in loop._trace.of_kind("usage")]


def _gates(loop: AgentLoop) -> list[str]:
    return [
        str(e.data.get("kind"))
        for e in loop._trace.of_kind("intervention")
        if str(e.data.get("kind")).endswith(("_gate", "_stalled", "_failed", "_unresolved"))
    ]


# ── what rode each request ───────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_each_request_says_which_rails_rode_it(tmp_path: Path, streamed: bool) -> None:
    """No plan yet: nothing rides. A live plan: the checklist. A finished plan: its one
    "answer now" line, which is a DIFFERENT message and is named as one."""
    script = [
        _plan("in_progress", "pending"),  # request 1, assembled before any plan exists
        _judge_ok(),
        _plan("done", "done"),  # request 2 carried the live checklist
        _say("both steps are done: here is the answer"),  # request 3 carried the closing line
        _review_ok(),
    ]
    provider = _Scripted(script)
    loop = _loop(provider, tmp_path)
    await _run(loop, "two steps", streamed=streamed)
    assert _requests(loop) == [([], False), (["plan"], False), (["plan_complete"], False)]
    assert all(e.data.get("streamed", False) is streamed for e in loop._trace.of_kind("usage"))
    # The label against the wire, not against itself: calls 0, 2 and 3 are the main requests
    # (1 is the judge, 4 the review), and each ended with exactly the reminder its label names.
    assert [provider.ended_with[i] for i in (0, 2, 3)] == [None, "plan", "plan_complete"]


@pytest.mark.asyncio
async def test_a_withheld_tail_is_said_to_be_withheld(tmp_path: Path) -> None:
    """On a model whose cache reuses only whole prompts the reminder rides every OTHER call
    (ADR-0193). A request that went out without it must not read as "there was no plan": it
    says the tail rested. The finished plan's line never rests."""
    script = [
        _plan("in_progress", "pending"),
        _judge_ok(),
        _say("on it"),  # narrates: the plan gate keeps the turn open
        _plan("done", "in_progress"),
        _say("on it"),
        _plan("done", "done"),
        _say("both done: the answer"),
        _review_ok(),
    ]
    provider = _Scripted(script)
    loop = _loop(provider, tmp_path)
    loop.session.tail_sparse_models.append(MODEL)
    await loop.arun_turn("two steps")
    assert _requests(loop) == [
        ([], False),  # no plan yet, and nothing to withhold
        (["plan"], False),
        ([], True),  # withheld: the plan is still there
        (["plan"], False),
        ([], True),
        (["plan_complete"], False),  # the closing line always rides
    ]
    # And the wire agrees, request by request (call 1 is the judge, call 7 the review): a
    # rested request really did end without a reminder.
    main = [provider.ended_with[i] for i in (0, 2, 3, 4, 5, 6)]
    assert main == [None, "plan", None, "plan", None, "plan_complete"]


# ── a gate that speaks says so ───────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_the_plan_gate_notes_each_nudge(tmp_path: Path, streamed: bool) -> None:
    script = [_plan("done", "pending"), _judge_ok()] + [_say("all done")] * 6
    loop = _loop(_Scripted(script), tmp_path)
    await _run(loop, "two steps", streamed=streamed)
    assert _gates(loop) == ["plan_gate", "plan_gate", "plan_unresolved"]
    nudges = [e.data for e in loop._trace.of_kind("intervention") if e.data["kind"] == "plan_gate"]
    assert [(n["nudge"], n["open_steps"]) for n in nudges] == [(1, 1), (2, 1)]


def _write(path: str) -> LLMResult:
    call = ToolCall(id="w1", name="write_file", arguments={"path": path, "content": "print(1)\n"})
    return LLMResult(tool_calls=[call])


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_the_recipe_gate_notes_its_nudge(tmp_path: Path, streamed: bool) -> None:
    """acceptEdits lets the write through and makes a shell run prompt, so the harness cannot
    run the file itself and asks the model to: the path that wrote no trace event."""
    loop = _loop(
        _Scripted([_write("p.py"), _say("done")]),
        tmp_path,
        attempt_cap=1,
        permission_policy=PermissionPolicy(PermissionMode.ACCEPT_EDITS),
    )
    await _run(loop, "make p.py", streamed=streamed)
    assert _gates(loop) == ["recipe_gate", "recipe_stalled"]


class _FakeWrite(Tool):
    spec = ToolSpec(name="write_file", description="pretend to write a file")

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok("written")


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_the_project_verifier_gate_notes_each_nudge(tmp_path: Path, streamed: bool) -> None:
    """No shell is registered, so the harness cannot run the project checks itself and asks
    the model to, once per attempt, before it gives up."""
    registry = ToolRegistry()
    registry.register(_FakeWrite())
    loop = _loop(
        _Scripted([_write("notes.txt")] + [_say("done")] * 8),
        tmp_path,
        registry=registry,
        settings=Settings(verify_command="check"),
    )
    await _run(loop, "change code", streamed=streamed)
    assert _gates(loop) == ["verify_gate", "verify_gate", "verify_gate", "verification_failed"]
