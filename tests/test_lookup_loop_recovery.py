"""ADR-0391: the exact-repeat guard sees a loop of rule lookups, and its recovery names the rules.

Measured 2026-11-17 on ten worker sessions of a host framework, from transcript metadata only: a
pair of rules asked again in the other order, or one rule of the pair alone, read to the guard as a
new batch, so a lookup loop ran on past it (one session asked the same pair, in both orders, in
nine batches over 18 minutes). And the general recovery nudge tells the model to READ again to check
the real state: after 23 of 36 such nudges to a lookup loop, the next batch read rules again.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from zakcode.agent.loop import (
    _DOOM_RECOVERY_NUDGE,
    _DOOM_UNEXECUTED,
    AgentLoop,
    _doom_recovery_texts,
)
from zakcode.agent.stuck import batch_signature, is_lookup_repeat, lookup_rules
from zakcode.messages import ToolResultBlock
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.rules import Rule, RuleRegistry
from zakcode.session.store import Session
from zakcode.tasks import TaskNetwork
from zakcode.tools.base import ToolContext, ToolRegistry
from zakcode.tools.builtins.read_rule import ReadRuleTool
from zakcode.tools.builtins.update_plan import UpdatePlanTool
from zakcode.usage import Usage


def _read(*names: str, refresh: bool = False) -> list[ToolCall]:
    return [
        ToolCall(
            id=f"c{i}",
            name="read_rule",
            arguments={"name": name, **({"refresh": True} if refresh else {})},
        )
        for i, name in enumerate(names)
    ]


def _sig(calls: list[ToolCall]):
    return batch_signature(calls)


# ── the repeat test ───────────────────────────────────────────────────────────


def test_a_pair_asked_again_in_the_other_order_is_a_repeat() -> None:
    assert _sig(_read("a", "b")) != _sig(_read("b", "a"))  # the guard's own test misses it
    assert is_lookup_repeat(_sig(_read("b", "a")), _sig(_read("a", "b")))


def test_one_rule_of_the_pair_alone_is_a_repeat() -> None:
    assert is_lookup_repeat(_sig(_read("a")), _sig(_read("a", "b")))


def test_a_lookup_that_adds_a_rule_is_not_a_repeat() -> None:
    assert not is_lookup_repeat(_sig(_read("a", "b")), _sig(_read("a")))


def test_a_refresh_is_an_ask_on_purpose_not_a_lookup() -> None:
    assert lookup_rules(_sig(_read("a", refresh=True))) is None
    assert not is_lookup_repeat(_sig(_read("a", refresh=True)), _sig(_read("a")))
    assert not is_lookup_repeat(_sig(_read("a")), _sig(_read("a", refresh=True)))


def test_a_batch_with_any_other_tool_is_not_a_lookup() -> None:
    # Another tool that also takes a ``name`` (a skill, say) is not a rule lookup.
    skill = ToolCall(id="s", name="use_skill", arguments={"name": "a"})
    mixed = [*_read("a"), skill]
    assert lookup_rules(_sig([skill])) is None
    assert lookup_rules(_sig(mixed)) is None
    assert not is_lookup_repeat(_sig(_read("a")), _sig(mixed))
    assert not is_lookup_repeat(_sig([skill]), _sig(_read("a")))


def test_no_batch_before_is_no_repeat() -> None:
    assert not is_lookup_repeat(_sig(_read("a")), None)
    assert lookup_rules(()) is None


# ── the recovery texts ────────────────────────────────────────────────────────


def _network_on(title: str) -> TaskNetwork:
    net = TaskNetwork()
    ctx = ToolContext(workspace_root=Path("/tmp"), task_network=net)
    tasks = [
        {"title": "read the record", "status": "done"},
        {"title": title, "status": "in_progress"},
    ]
    asyncio.run(UpdatePlanTool().execute({"tasks": tasks}, ctx))
    return net


def test_a_lookup_loop_gets_texts_that_name_its_rules_and_the_current_step() -> None:
    unexecuted, nudge = _doom_recovery_texts(
        _read("b-rule", "a-rule"), _network_on("Verify the goal")
    )
    assert unexecuted.startswith("Not answered: you asked read_rule for a-rule, b-rule again")
    assert "read_rule for a-rule, b-rule several times in a row" in nudge
    assert "to your current step, 'Verify the goal', and take that step's next action now" in nudge
    assert "READ it now" not in nudge


def test_without_a_plan_the_nudge_names_no_step() -> None:
    _, nudge = _doom_recovery_texts(_read("a-rule"), TaskNetwork())
    assert "to your current step and take that step's next action now" in nudge


def test_positive_control_any_other_batch_keeps_the_general_texts() -> None:
    echo = [ToolCall(id="e", name="echo", arguments={"text": "same"})]
    assert _doom_recovery_texts(echo, TaskNetwork()) == (_DOOM_UNEXECUTED, _DOOM_RECOVERY_NUDGE)
    assert _doom_recovery_texts(_read("a", refresh=True), TaskNetwork()) == (
        _DOOM_UNEXECUTED,
        _DOOM_RECOVERY_NUDGE,
    )


# ── through the loop, both turn paths ─────────────────────────────────────────


class _Scripted(Provider):
    def __init__(self, results: list[LLMResult]) -> None:
        self._results = results
        self.calls = 0

    async def acomplete(self, messages, *, system=None, tools=None, **kw) -> LLMResult:
        i = min(self.calls, len(self._results) - 1)
        self.calls += 1
        return self._results[i]

    def count_tokens(self, messages, *, system=None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(context_window=32768)


def _batch(*names: str) -> LLMResult:
    return LLMResult(text="", tool_calls=_read(*names), usage=Usage(total_tokens=1))


def _loop(tmp_path: Path, batches: list[LLMResult]) -> AgentLoop:
    registry = RuleRegistry()
    registry.add(Rule("a-rule", "# A\nDo the a thing.", Path("/x/a-rule.md")))
    registry.add(Rule("b-rule", "# B\nDo the b thing.", Path("/x/b-rule.md")))
    tools = ToolRegistry()
    tools.register(ReadRuleTool())
    return AgentLoop(
        _Scripted([*batches, LLMResult(text="done", usage=Usage(total_tokens=1))]),
        tools,
        Session(cwd=str(tmp_path), model="t"),
        rule_registry=registry,
        max_iterations=20,
    )


async def _run(loop: AgentLoop, streaming: bool) -> str:
    if streaming:
        events = [ev async for ev in loop.astream_turn("go")]
        return str(events[-1].stop_reason)
    return str((await loop.arun_turn("go")).stop_reason)


def _texts(loop: AgentLoop) -> tuple[list[str], list[str]]:
    outputs = [
        b.output for m in loop.session.messages for b in m.blocks if isinstance(b, ToolResultBlock)
    ]
    users = [m.text for m in loop.session.messages if m.role == "user"]
    return outputs, users


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "batches",
    [
        [("a-rule", "b-rule"), ("b-rule", "a-rule"), ("a-rule", "b-rule")],  # the order swaps
        [("a-rule", "b-rule"), ("a-rule",), ("b-rule",)],  # one rule of the pair at a time
    ],
    ids=["reordered", "partial"],
)
async def test_the_guard_sees_a_lookup_loop_and_its_recovery_names_the_rules(
    tmp_path: Path, streaming: bool, batches: list[tuple[str, ...]]
) -> None:
    loop = _loop(tmp_path, [_batch(*names) for names in batches])
    assert await _run(loop, streaming) == "completed"  # the recovery, then the model's answer
    outputs, users = _texts(loop)
    third = batches[2]
    refused = [o for o in outputs if o.startswith("Not answered: you asked read_rule for")]
    assert len(refused) == len(third)  # the third batch was not run
    assert any("several times in a row" in u and "a-rule, b-rule" in u for u in users)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_positive_control_lookups_that_move_on_are_not_refused(
    tmp_path: Path, streaming: bool
) -> None:
    """A new rule in each batch is reading, not a loop: every ask is answered."""
    loop = _loop(tmp_path, [_batch("a-rule"), _batch("a-rule", "b-rule")])
    assert await _run(loop, streaming) == "completed"
    outputs, users = _texts(loop)
    assert not any(o.startswith("Not answered") for o in outputs)
    assert not any("several times in a row" in u for u in users)
