"""A delegated sub-agent compacts its own context instead of dying at the window.

Compaction used to stop at the delegation seam. The loop a sub-agent runs on was built with no
compactor, so a child whose transcript outgrew the model's window had no way to recover: the
overflow ended its turn as ``provider_error`` with ``(recovery: no compactor; 0/2 attempts)``.
Measured on a self-hosted worker, a sub-agent's prompt grew from 22.5k to 130.6k tokens over
three hours of reading and died at the 131,072-token window, and the parent got a stop back
instead of the child's answer.

These tests drive the real seam end to end. ``Agent(enable_subagents=True,
enable_compaction=...)`` runs the parent loop, and its ``task`` call goes through
``SubAgentManager`` and ``SubAgentRunner`` into a real child loop. Only the model is scripted. It
stands in for a server with a small window and refuses any request larger than that window, the
way a llama.cpp server does. The child reads one large file per step until its reads add up to
about twice the window, then answers. With compaction off the child dies at the window, which is
the control showing the scenario really overflows. With compaction on the child compacts and
finishes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from zakcode import Agent
from zakcode.agent.compact import SUMMARY_MARKER
from zakcode.agent.subagent import _HANDOFF
from zakcode.config import Settings
from zakcode.messages import Message, TextBlock, ToolResultBlock, ToolUseBlock
from zakcode.providers.base import (
    Capabilities,
    ContextWindowExceeded,
    LLMResult,
    Provider,
    ToolCall,
)
from zakcode.usage import Usage

#: The scripted server's context window, in tokens. Small, so a few reads overflow it.
WINDOW = 8192
#: Files the child reads before it answers. The loop clamps each result to a quarter of the
#: window, so the eight together come to about twice what the window holds.
READS = 8
#: Long enough that the summarizer's minimum-length check never rejects it.
_SUMMARY = "The sub-agent is reading the numbered files one at a time and has not finished. " * 8


def _tokens(messages: list[Message]) -> int:
    """The scripted server's size for a request: every character it carries, 3 to a token."""
    chars = 0
    for message in messages:
        for block in message.blocks:
            if isinstance(block, TextBlock):
                chars += len(block.text)
            elif isinstance(block, ToolResultBlock):
                chars += len(block.output)
            elif isinstance(block, ToolUseBlock):
                chars += len(json.dumps(block.input))
    return chars // 3


class _WindowedModel(Provider):
    """One small-window model for the parent, its child and the compaction summarizer.

    The parent delegates once and wraps up when the result comes back. A request is the child's
    when its system prompt carries the handoff instruction every sub-agent gets. The summarizer
    is recognised by its own instruction.

    With ``reports_usage=False`` the model reports no prompt size and its estimator counts
    nothing, so the loop never sees the transcript grow. Then only the server's refusal can
    start a compaction, which tests the in-turn overflow recovery on its own.
    """

    def __init__(self, *, reports_usage: bool = True) -> None:
        self.reports_usage = reports_usage
        self.reads = 0
        self.refusals = 0
        self.summaries = 0
        self.child_saw_summary = False

    async def acomplete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kw: Any,
    ) -> LLMResult:
        system = system or ""
        if system.startswith("You are compacting"):
            self.summaries += 1
            return LLMResult(text=f"<summary>{_SUMMARY}</summary>", usage=Usage(total_tokens=1))
        size = _tokens(messages)
        if size > WINDOW:
            self.refusals += 1
            raise ContextWindowExceeded(
                f"request ({size} tokens) exceeds the available context size ({WINDOW} tokens)"
            )
        usage = Usage(prompt_tokens=size if self.reports_usage else 0, total_tokens=size + 1)
        if _HANDOFF not in system:
            if any(m.role == "tool" for m in messages):
                return LLMResult(text="PARENT-SYNTHESIS", usage=usage)
            call = ToolCall(
                id="task1", name="task", arguments={"tasks": [{"prompt": "Read the files."}]}
            )
            return LLMResult(tool_calls=[call], usage=usage)
        if any(SUMMARY_MARKER in m.text for m in messages):
            self.child_saw_summary = True
        if self.reads < READS:
            self.reads += 1
            call = ToolCall(
                id=f"read{self.reads}", name="Read", arguments={"path": f"part{self.reads}.txt"}
            )
            return LLMResult(tool_calls=[call], usage=usage)
        return LLMResult(text="CHILD-RESULT", usage=usage)

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        # Characters over four, under the server's count the way a real estimate runs under it.
        return _tokens(messages) * 3 // 4 if self.reports_usage else 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=WINDOW)


async def _delegate(tmp_path: Path, model: _WindowedModel, *, compaction: bool) -> str:
    """Run one parent turn that delegates the reads, and return the task tool's output."""
    for i in range(1, READS + 1):
        # 24,000 characters a file, so every read comes back at the loop's clamp.
        (tmp_path / f"part{i}.txt").write_text(f"part {i}: {'x' * 50}\n" * 400)
    agent = Agent(
        settings=Settings(
            default_model="scripted/test",
            context_window=WINDOW,
            workspace_root=tmp_path,
            permission_mode="allow",
        ),
        provider=model,
        enable_subagents=True,
        enable_compaction=compaction,
    )
    result = await agent.arun_turn("DELEGATE")
    assert result.stop_reason == "completed"
    assert any(m.text == "PARENT-SYNTHESIS" for m in result.assistant_messages)
    (output,) = [tr.output for tr in result.tool_results]
    return output


async def test_without_compaction_the_sub_agent_dies_at_the_window(tmp_path: Path) -> None:
    # The control: the same delegation with compaction off overflows and has no recovery.
    model = _WindowedModel()
    output = await _delegate(tmp_path, model, compaction=False)
    assert "(stopped: provider_error)" in output
    assert "CHILD-RESULT" not in output
    assert model.refusals == 1
    assert model.summaries == 0
    assert model.reads < READS


@pytest.mark.parametrize("reports_usage", [True, False], ids=["threshold", "overflow-recovery"])
async def test_a_sub_agent_compacts_its_own_context_and_finishes(
    tmp_path: Path, reports_usage: bool
) -> None:
    model = _WindowedModel(reports_usage=reports_usage)
    output = await _delegate(tmp_path, model, compaction=True)
    # The child finished: its answer came back, not a stop.
    assert "CHILD-RESULT" in output
    assert "(stopped:" not in output
    # It did every read, about twice what one window holds, so its context was compacted:
    # the summarizer ran, and the child's later requests carried the summary.
    assert model.reads == READS
    assert model.summaries >= 1
    assert model.child_saw_summary
    if reports_usage:
        # The loop saw the transcript grow and compacted before the server had to refuse.
        assert model.refusals == 0
    else:
        # Blind to the size, the child ran into the window and recovered from the refusal.
        assert model.refusals >= 1
