"""The behavioral probe suite (M9).

Each probe drives the *real* agent loop with a deterministic
:class:`~zakcode.evals.harness.ScriptedProvider` and asserts an end-to-end behavior
the project must never regress:

* **completion-detection** — a turn with no tool calls ends cleanly as ``completed``.
* **safety-rejection** — a privileged tool call in ``ask`` mode with no prompter is
  denied at the core gate (fail-closed); the workspace is untouched and the loop
  keeps going so the model sees the denial.
* **plan-mode-readonly** — the read-only planner sub-agent's registry has no write
  tools at all (schema-enforced), so a plan can never mutate the workspace.
* **doom-loop-halt** — a model that repeats the same tool call forever is stopped
  with ``stop_reason="doom_loop"`` instead of burning the whole iteration budget.
* **partial-failure-recovery** — a tool that fails once then succeeds is retried by
  the model and the turn completes (no doom-loop, no crash).
* **stuck-recovery** — a model stuck on varying-arg failures (which the exact-repeat
  doom guard misses) is recovered via the nudge → narrow ladder and then halted as
  ``stuck`` rather than flailing to the iteration cap.
* **long-horizon-compaction** — once history crosses the context-window threshold the
  loop compacts (a model-written summary replaces old turns) and the turn still
  completes.

``build_default_suite(workspace)`` returns all seven as :class:`~zakcode.evals.harness.EvalCase`
objects bound to a caller-provided workspace directory (the CLI passes a temp dir).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from zakcode.config import PermissionTier
from zakcode.evals.harness import (
    EvalCase,
    ScriptedProvider,
    call_tool,
    make_agent,
    reply,
)
from zakcode.messages import Message
from zakcode.providers.base import LLMResult
from zakcode.tools.base import ConcurrencyClass, Tool, ToolContext, ToolResult, ToolSpec

# ── a custom in-eval tool: fails once, then succeeds ──────────────────────────────


class FlakyTool(Tool):
    """A READ_ONLY tool that errors on its first call and succeeds afterward.

    Used by the partial-failure-recovery probe to prove the loop surfaces a tool
    error to the model as recoverable feedback (not a crash) and can make progress on
    a retry.
    """

    spec = ToolSpec(
        name="flaky",
        description="Fails once, then succeeds (for recovery testing).",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": [],
        },
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.calls += 1
        if self.calls == 1:
            return ToolResult.error("transient failure (first call)")
        return ToolResult.ok(f"succeeded on call {self.calls}")


class AlwaysFailTool(Tool):
    """A READ_ONLY tool that ALWAYS errors (for stuck-recovery testing).

    Used by the stuck-recovery probe: the model keeps calling it with fresh args, so the
    exact-repeat doom guard never fires, but the multi-signal stuck detector still recovers
    (nudge → narrow) and then halts the turn as ``stuck`` instead of flailing forever.
    """

    spec = ToolSpec(
        name="always_fail",
        description="Always fails (for stuck-recovery testing).",
        parameters={
            "type": "object",
            "properties": {"n": {"type": "integer"}},
            "required": [],
        },
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.calls += 1
        return ToolResult.error(f"always fails (call {self.calls})")


# ── the probes ──────────────────────────────────────────────────────────────────────


async def _probe_completion(workspace: str) -> str:
    """A plain reply ends the turn as ``completed`` with the assistant's text."""
    provider = ScriptedProvider([reply("the answer is 42")])
    agent = make_agent(provider, workspace_root=workspace)
    result = await agent.arun_turn("what is the answer?")
    assert result.stop_reason == "completed", result.stop_reason
    assert result.assistant_messages[-1].text == "the answer is 42"
    assert result.iterations == 1, result.iterations
    return "no-tool reply terminated cleanly as 'completed'"


async def _probe_safety_rejection(workspace: str) -> str:
    """A bash call in ``ask`` mode with no prompter is denied; nothing executes."""
    sentinel = Path(workspace) / "PWNED.txt"
    provider = ScriptedProvider(
        [
            call_tool("bash", {"command": f"echo pwned > {sentinel}"}),
            reply("understood, I won't run that"),
        ]
    )
    # ask + no prompter ⇒ DANGER_FULL_ACCESS fails closed.
    agent = make_agent(provider, workspace_root=workspace, permission_mode="ask", prompter=None)
    result = await agent.arun_turn("run a shell command")
    assert result.stop_reason == "completed", result.stop_reason
    denied = [r for r in result.tool_results if r.data and r.data.get("permission_denied")]
    assert denied, "expected a permission-denied tool result"
    assert not sentinel.exists(), "the denied command must not have run"
    return "privileged tool call denied at the core gate; workspace untouched"


async def _probe_plan_mode_readonly(workspace: str) -> str:
    """The plan sub-agent's registry exposes no write/exec tools (schema-enforced).

    Plan Mode's read-only guarantee is structural: the planner runs against
    ``registry.subset(PLAN.allowed_tools)``, so write/exec tools are absent from the
    schema the model is ever offered. We assert that at its source — the canonical
    ``PLAN`` definition + the subset of the live default registry — which is exactly
    what :meth:`SubAgentRunner.child_registry` builds for a plan child.
    """
    from zakcode.agent.subagent import PLAN, READ_ONLY_TOOLS
    from zakcode.tools.builtins.default_registry import default_registry

    # READ_ONLY_TOOLS plus update_plan (READ_ONLY — touches only the in-memory plan), and NO
    # write/exec tools: Plan Mode's read-only-on-the-workspace guarantee stays schema-enforced.
    assert PLAN.allowed_tools == [*READ_ONLY_TOOLS, "update_plan"], PLAN.allowed_tools
    plan_registry = default_registry().subset(PLAN.allowed_tools or [])
    names = set(plan_registry.active_names())
    for write_tool in ("write_file", "edit_file", "bash"):
        assert write_tool not in names, f"plan mode must not expose {write_tool}: {names}"
    assert "read_file" in names, f"plan mode should still allow reads: {names}"
    # And the planner type is actually wired into a sub-agent-enabled Agent.
    provider = ScriptedProvider([reply("noop")])
    agent = make_agent(provider, workspace_root=workspace, enable_subagents=True)
    assert agent.loop.spawner is not None, "sub-agents must be enabled"
    assert "plan" in agent.loop.spawner.available_types(), agent.loop.spawner.available_types()
    return f"plan registry is read-only ({sorted(names)}); no write/exec tools present"


async def _probe_doom_loop_halt(workspace: str) -> str:
    """A model that repeats one tool call forever is halted with ``doom_loop``."""
    target = Path(workspace) / "scratch.txt"
    # Always request the SAME write_file call with identical args → identical signature.
    provider = ScriptedProvider([call_tool("write_file", {"path": str(target), "content": "x"})])
    agent = make_agent(provider, workspace_root=workspace)
    result = await agent.arun_turn("keep writing")
    assert result.stop_reason == "doom_loop", result.stop_reason
    # Halted early — far below a runaway count (DOOM_LOOP_THRESHOLD = 3).
    assert result.iterations <= 5, result.iterations
    return f"identical repeated tool call halted as 'doom_loop' after {result.iterations} iters"


async def _probe_partial_failure_recovery(workspace: str) -> str:
    """A tool that fails once then succeeds is retried; the turn completes."""
    flaky = FlakyTool()

    # Call flaky twice (it fails then succeeds), then finish. Distinct args each call so
    # the doom-loop guard never fires (we are testing recovery, not looping).
    def responder(messages: list[Message], system: str | None, i: int) -> LLMResult:
        if i == 0:
            return call_tool("flaky", {"value": "first"}, id="f1")
        if i == 1:
            return call_tool("flaky", {"value": "second"}, id="f2")
        return reply("recovered and finished")

    provider = ScriptedProvider([reply("unused")], responder=responder)
    agent = make_agent(provider, workspace_root=workspace, extra_tools=[flaky])
    result = await agent.arun_turn("use the flaky tool")
    assert result.stop_reason == "completed", result.stop_reason
    assert flaky.calls == 2, flaky.calls
    errored = [r for r in result.tool_results if r.is_error]
    assert errored, "the first (failing) call should surface as an error result"
    assert result.assistant_messages[-1].text == "recovered and finished"
    return "tool failure surfaced as recoverable feedback; model retried and completed"


async def _probe_stuck_recovery(workspace: str) -> str:
    """A model stuck on VARYING-arg failures is recovered, then halted as ``stuck``.

    Distinct args each iteration mean the exact-repeat doom guard never fires — only the
    multi-signal stuck ladder (nudge → narrow-to-read-only → stop) can end the turn. Proves
    the loop never flails forever on the many stall shapes the doom guard misses.
    """
    failing = AlwaysFailTool()

    def responder(messages: list[Message], system: str | None, i: int) -> LLMResult:
        # A fresh arg every iteration → identical-signature doom detection cannot trip.
        return call_tool("always_fail", {"n": i}, id=f"af{i}")

    provider = ScriptedProvider([reply("unused")], responder=responder)
    agent = make_agent(provider, workspace_root=workspace, extra_tools=[failing])
    result = await agent.arun_turn("keep trying the broken tool")
    assert result.stop_reason == "stuck", result.stop_reason
    assert result.degraded is True, result.degraded
    # Halted by the recovery ladder (nudge→narrow→stop), far below the iteration cap.
    assert result.iterations <= 6, result.iterations
    return (
        f"varying-arg failure loop recovered then halted as 'stuck' after {result.iterations} iters"
    )


async def _probe_long_horizon_compaction(workspace: str) -> str:
    """Once history crosses the window threshold, the loop compacts and still completes.

    A tiny ``context_window`` plus a per-message token weight means the running
    session crosses the 0.8 threshold within a few turns; ``_maybe_compact`` then runs
    a summarization pass (system prompt contains "compacting") before the turn.
    """

    # context_window 200, 50 tokens/msg ⇒ threshold 160 ⇒ compaction once ≳3-4 msgs.
    def responder(messages: list[Message], system: str | None, i: int) -> LLMResult:
        if system and "compacting" in system.lower():
            return reply("SUMMARY: earlier turns condensed.")
        return reply(f"turn reply {i}")

    provider = ScriptedProvider(
        [reply("unused")],
        responder=responder,
        context_window=200,
        tokens_per_message=50,
    )
    agent = make_agent(provider, workspace_root=workspace, enable_compaction=True)
    # Several turns to grow history past the threshold.
    for n in range(5):
        result = await agent.arun_turn(f"message number {n} with some content")
        assert result.stop_reason == "completed", result.stop_reason
    assert provider.summarize_calls >= 1, "expected at least one compaction summarization"
    # After compaction the session begins with the model-written summary.
    assert any(
        m.role == "system" and "summary" in m.text.lower() for m in agent.session.messages
    ), "a leading summary message should be present after compaction"
    return f"history compacted ({provider.summarize_calls} summarization pass(es)); turns completed"


def build_default_suite(workspace: str) -> list[EvalCase]:
    """The seven behavioral probes, bound to ``workspace`` (a writable temp dir)."""
    return [
        EvalCase(
            "completion-detection",
            "A no-tool reply ends the turn cleanly as 'completed'.",
            lambda: _probe_completion(workspace),
        ),
        EvalCase(
            "safety-rejection",
            "A privileged tool call fails closed at the core gate; nothing executes.",
            lambda: _probe_safety_rejection(workspace),
        ),
        EvalCase(
            "plan-mode-readonly",
            "The plan sub-agent registry has no write/exec tools (schema-enforced).",
            lambda: _probe_plan_mode_readonly(workspace),
        ),
        EvalCase(
            "doom-loop-halt",
            "An identical-call loop is stopped with stop_reason='doom_loop'.",
            lambda: _probe_doom_loop_halt(workspace),
        ),
        EvalCase(
            "partial-failure-recovery",
            "A tool that fails once then succeeds is retried; the turn completes.",
            lambda: _probe_partial_failure_recovery(workspace),
        ),
        EvalCase(
            "stuck-recovery",
            "A varying-arg failure loop (doom guard misses it) is recovered then halts 'stuck'.",
            lambda: _probe_stuck_recovery(workspace),
        ),
        EvalCase(
            "long-horizon-compaction",
            "Long histories are compacted (LLM summary) and the turn still completes.",
            lambda: _probe_long_horizon_compaction(workspace),
        ),
    ]


__all__ = ["FlakyTool", "build_default_suite"]
