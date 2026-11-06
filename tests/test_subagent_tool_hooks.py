"""A sub-agent's tool calls run the session's tool hooks, as Claude Code's do (ADR-0367).

Why this file exists. Zak Code ran every sub-agent with an empty hook set, on the reading that
Claude Code gives a sub-agent only its own frontmatter hooks. Claude Code's sub-agents guide says
the opposite: tool events "fire for the subagent's tool calls the same way they do in the main
conversation", and its hooks reference adds ``agent_id`` and ``agent_type`` to the input of a
hook that fires inside a sub-agent. So a workspace's PreToolUse gates never judged a sub-agent's
writes or commands, and a hook that records the session's activity went silent for as long as a
delegation ran: measured on a self-hosted worker, a parent waited inside one task call for
2 h 43 m with no hook firing at all.

These tests drive the real seam end to end. ``Agent(enable_subagents=True)`` runs the parent
loop, and its ``task`` call goes through ``SubAgentManager`` and ``SubAgentRunner`` into a real
child loop that runs the real shell tool. Only the model is scripted. One case hands the hooks to
a shell SCRIPT that reads its own stdin, the way a Claude Code hook does.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from zakcode import Agent
from zakcode.agent.subagent import _HANDOFF
from zakcode.config import Settings
from zakcode.hooks import (
    HookEvent,
    HookManager,
    HookPayload,
    HookSpec,
    LLMContextPayload,
    TurnEndPayload,
)
from zakcode.messages import Message
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.usage import Usage

CHILD_PROMPT = "Run the command."


class _Delegator(Provider):
    """The parent runs one command, delegates once, then answers; the child runs one command.

    A request is the child's when its system prompt carries the handoff instruction every
    sub-agent gets.
    """

    async def acomplete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kw: Any,
    ) -> LLMResult:
        usage = Usage(total_tokens=1)
        results = sum(1 for m in messages if m.role == "tool")
        if _HANDOFF in (system or ""):
            if results == 0:
                call = ToolCall(id="c1", name="Bash", arguments={"command": "echo child"})
                return LLMResult(tool_calls=[call], usage=usage)
            return LLMResult(text="CHILD-RESULT", usage=usage)
        if results == 0:
            call = ToolCall(id="p1", name="Bash", arguments={"command": "echo parent"})
            return LLMResult(tool_calls=[call], usage=usage)
        if results == 1:
            call = ToolCall(id="p2", name="task", arguments={"tasks": [{"prompt": CHILD_PROMPT}]})
            return LLMResult(tool_calls=[call], usage=usage)
        return LLMResult(text="PARENT-SYNTHESIS", usage=usage)

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=32768)


async def _run(tmp_path: Path, hooks: HookManager) -> Agent:
    agent = Agent(
        settings=Settings(
            default_model="scripted/test",
            context_window=32768,
            workspace_root=tmp_path,
            permission_mode="allow",
        ),
        provider=_Delegator(),
        enable_subagents=True,
        hook_manager=hooks,
    )
    result = await agent.arun_turn("DELEGATE")
    assert result.stop_reason == "completed"
    assert any(m.text == "PARENT-SYNTHESIS" for m in result.assistant_messages)
    return agent


async def test_a_sub_agents_tool_calls_run_the_sessions_tool_hooks_as_the_session(
    tmp_path: Path,
) -> None:
    seen: list[HookPayload] = []
    hooks = HookManager()
    hooks.register(HookEvent.PRE_TOOL_USE, seen.append)
    hooks.register(HookEvent.POST_TOOL_USE, seen.append)
    agent = await _run(tmp_path, hooks)
    parent = agent.session.id

    def calls(command: str) -> list[HookPayload]:
        return [p for p in seen if p.arguments.get("command") == command]

    mine = calls("echo parent")
    child = calls("echo child")
    # The child's command passed the session's PreToolUse and PostToolUse hooks ...
    assert [p.event for p in child] == [HookEvent.PRE_TOOL_USE, HookEvent.POST_TOOL_USE]
    # ... as the session, with the sub-agent named the way Claude Code names it.
    assert {p.session_id for p in child} == {parent}
    assert {p.agent_type for p in child} == {"general-purpose"}
    (agent_id,) = {p.agent_id for p in child}
    assert agent_id and agent_id != parent
    # The session's own calls are unchanged: no sub-agent fields.
    assert [p.event for p in mine] == [HookEvent.PRE_TOOL_USE, HookEvent.POST_TOOL_USE]
    assert {(p.session_id, p.agent_id, p.agent_type) for p in mine} == {(parent, "", "")}


async def test_the_sessions_other_hooks_stay_out_of_the_sub_agent(tmp_path: Path) -> None:
    # Claude Code fires SubagentStop, not Stop, for a sub-agent: a host's turn-end hook must
    # not run at a child's turn end. An observer, because it runs at every turn end; a
    # veto-capable hook runs only on a vetoable loop, which a child's is not. Which events the
    # child's hook set holds at all is pinned in test_subagent.py.
    turn_ends: list[TurnEndPayload] = []
    contexts: list[LLMContextPayload] = []
    hooks = HookManager()
    hooks.register_turn_end_observer(turn_ends.append)
    hooks.register_context(contexts.append)
    agent = await _run(tmp_path, hooks)
    parent = agent.session.id
    assert [p.session_id for p in turn_ends] == [parent]  # the parent's turn end, once
    assert contexts  # the parent's model calls ran the context hook ...
    assert all(CHILD_PROMPT not in p.user_text for p in contexts)  # ... the child's did not


async def test_a_shell_hook_reads_the_sub_agent_fields_from_its_own_stdin(tmp_path: Path) -> None:
    log = tmp_path / "hook-stdin.jsonl"
    script = (
        "import sys\n"
        f"open({str(log)!r}, 'a', encoding='utf-8').write(sys.stdin.read().strip() + '\\n')\n"
    )
    spec = HookSpec(
        event=HookEvent.PRE_TOOL_USE, matcher="Bash", command=[sys.executable, "-c", script]
    )
    hooks = HookManager([spec])
    agent = await _run(tmp_path, hooks)
    parent = agent.session.id
    docs = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    by_command = {d["tool_input"]["command"]: d for d in docs}
    assert sorted(by_command) == ["echo child", "echo parent"]
    child, mine = by_command["echo child"], by_command["echo parent"]
    assert child["hook_event_name"] == "PreToolUse" and child["tool_name"] == "Bash"
    assert child["session_id"] == parent
    assert child["agent_type"] == "general-purpose"
    assert child["agent_id"] and child["agent_id"] != parent
    # On the session's own call the two keys are absent, as in Claude Code.
    assert mine["session_id"] == parent
    assert "agent_id" not in mine and "agent_type" not in mine
