"""The ``task`` tool — delegate self-contained subtasks to isolated sub-agents (M4).

When the model calls ``task``, each requested subtask is run by an isolated
sub-agent (fresh history, filtered tools, shared budget) and the **summaries** are
returned — not the children's raw transcripts — so the parent's context stays
compact. Independent subtasks in one call run **concurrently** (``asyncio.gather``).

The tool is a thin adapter over the :class:`~zakcode.tools.base.SubAgentSpawner`
placed on the :class:`~zakcode.tools.base.ToolContext` by the loop. With no spawner
(an ordinary turn, or inside a sub-agent — one-level nesting), it returns a
recoverable error rather than raising, so the model can adapt.

Delegation itself requires only :data:`READ_ONLY` permission: the *child's* tool
calls are gated by the same permission policy as the parent, so spawning cannot be
used to escape the gate.

A batch identical to one that already ran to its end in this turn is not run again while
that result is still in the conversation (ADR-0382): the answer is a pointer to it, the
same shape as ``read_rule``'s answer to a repeat (ADR-0372). The loop's exact-repeat guard
sees only CONSECUTIVE identical calls, and one other call between two dispatches of the same
batch resets it; a sub-agent batch runs for minutes to an hour, so each missed repeat costs
the whole run again.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Any

from zakcode.agent.subagent import SubAgentResult
from zakcode.config import PermissionTier
from zakcode.tools.base import ConcurrencyClass, Tool, ToolContext, ToolResult, ToolSpec

#: The answer to a batch identical to one that already ran to its end in this turn, while that
#: result is still in the conversation (ADR-0382). It names when the earlier run returned and
#: how long its result is, so the model can find it, and it names the two ways to run on purpose.
ALREADY_RUN = (
    "[already run this turn] This exact task batch ran to its end at {at} and its result "
    "({chars:,} characters) is in your context above, unchanged. Use that result: running the "
    "same batch again repeats the same work. If something it depends on has changed since, "
    "call task again with rerun=true, or change the batch to say what is different."
)

#: What the parent is told about a sub-agent whose turn ended when its context filled (ADR-0412).
#: The text below it is every reply the sub-agent wrote, so its report is the last part.
CONTEXT_FULL_NOTE = (
    "its context filled and it was asked for its report, which ends the text below; what it did "
    "not get to is a task for a new, narrower sub-agent, since this same batch would stop at the "
    "same place again"
)

#: What the parent is told about a sub-agent whose context filled and that gave no report when it
#: was asked twice (ADR-0412). Its batch did not run to its end: the parent holds no report from
#: it, so a pointer would send it back to text that is not one, and the same batch runs again.
CONTEXT_FULL_NO_REPORT_NOTE = (
    "its context filled and it was asked twice for its report and gave none, so the text below "
    "is not a report: it is what it wrote before, the output of the last tool it ran, or a line "
    "saying it wrote nothing; what it did not get to is a task for a new, narrower sub-agent"
)

#: The stop reasons of a sub-agent that ran to its end, whose batch a later identical ask may lean
#: on (ADR-0382): it completed, or it reported when its context filled (ADR-0412).
_RAN_TO_ITS_END = frozenset({"completed", "context_full"})

#: How much of an earlier result's opening is kept to ask whether that result is still in the
#: conversation. A clamp keeps a result's head and tail and a compaction or an elision takes the
#: whole result, so the head answers for the result; the whole text would not fit a clamped one.
_PROBE_CHARS = 400


def _batch_key(prepared: list[tuple[str, str]]) -> str:
    """A digest of what a batch runs: each subtask's resolved type and prompt, in order. The
    resolved type, so a batch that names the default type and one that omits it are one batch."""
    return hashlib.sha256(json.dumps(prepared, ensure_ascii=False).encode()).hexdigest()


class TaskTool(Tool):
    """Run one or more subtasks as isolated, concurrent sub-agents."""

    spec = ToolSpec(
        name="task",
        description=(
            "Delegate one or more self-contained subtasks to isolated sub-agents that run "
            "concurrently and return short summaries (not full transcripts). Use it to "
            "parallelize independent research or work. Each subtask gets a fresh context and "
            "a filtered toolset; they share one iteration budget."
        ),
        parameters={
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "description": "The subtasks to run concurrently (one sub-agent each).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subagent_type": {
                                "type": "string",
                                "description": (
                                    "Which sub-agent type to use; omit for the default."
                                ),
                            },
                            "prompt": {
                                "type": "string",
                                "description": "The self-contained instruction for this subtask.",
                            },
                        },
                        "required": ["prompt"],
                    },
                },
                "rerun": {
                    "type": "boolean",
                    "description": (
                        "true runs the batch again although an identical batch already ran to "
                        "its end in this turn. Only when something it depends on has changed."
                    ),
                },
            },
            "required": ["tasks"],
        },
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.NEVER_PARALLEL,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        spawner = ctx.spawner
        if spawner is None:
            return ToolResult.error(
                "delegation is not available here (no sub-agent spawner on the context)"
            )

        tasks = args.get("tasks")
        if tasks is None and isinstance(args.get("prompt"), str):
            # Claude Code's single-delegation shape (``Task`` / ``Agent``: description, prompt,
            # subagent_type) — the aliases resolve here (ADR-0190). One subtask; ``description``
            # is a label the batch form has no slot for.
            one: dict[str, Any] = {"prompt": args["prompt"]}
            if args.get("subagent_type"):
                one["subagent_type"] = args["subagent_type"]
            tasks = [one]
        if not isinstance(tasks, list) or not tasks:
            return ToolResult.error(
                "'tasks' must be a non-empty array of {subagent_type?, prompt} objects"
            )

        available = spawner.available_types()
        # The spawner names its own default explicitly (no reliance on list order).
        default_type = spawner.default_type()

        prepared: list[tuple[str, str]] = []
        for i, task in enumerate(tasks):
            if not isinstance(task, dict) or "prompt" not in task:
                return ToolResult.error(f"task[{i}] must be an object containing a 'prompt'")
            prompt = task["prompt"]
            # A blank prompt is a degenerate delegation (the child has no task) AND would leave the
            # child's caller_query empty, mis-attributing any skill it uses to the PARENT's turn —
            # so reject it here rather than letting an empty subtask through.
            if not isinstance(prompt, str) or not prompt.strip():
                return ToolResult.error(f"task[{i}]: 'prompt' must be a non-empty string")
            type_name = task.get("subagent_type") or default_type
            if available and type_name not in available:
                # Models routinely guess "default" (the docstrings call general-purpose "the
                # default delegate"). On slow local inference a recoverable error costs a full
                # model round-trip, so absorb that one predictable guess; anything else still
                # errors — a misspelled real type must not silently get the full toolset.
                if type_name == "default":
                    type_name = default_type
                else:
                    return ToolResult.error(
                        f"task[{i}]: unknown subagent_type {type_name!r}; available: "
                        f"{available} (omit subagent_type for the default: {default_type!r})"
                    )
            prepared.append((type_name, prompt))

        # ADR-0397: a sub-agent has its own task table, so it can neither wait on, read the
        # status of, nor stop a background command of THIS session (its TaskOutput answers that
        # no such task exists). A delegation that names one still running is refused before any
        # child starts; the parent waits with TaskOutput itself, which is one call.
        named = _running_background_task_named(ctx, [p for _t, p in prepared])
        if named is not None:
            i, task_id, label, output_file = named
            return ToolResult.error(
                f"task[{i}] names background task {task_id} of THIS session, still running "
                f"({label}). A sub-agent has its own task table: it cannot wait on, read or stop "
                f"that task (its TaskOutput answers that no such task exists). Wait for it here: "
                f'TaskOutput(task_id="{task_id}", timeout=600000) returns when it exits; its '
                f"output is in {output_file}. Delegate work on that output afterwards if you "
                "still need to.",
                data={"refused": "names_own_running_background_task", "task_id": task_id},
            )

        # A batch identical to one that ran to its end in this turn gets a pointer to that result
        # while the result is still in the conversation (ADR-0382); rerun=true runs it anyway.
        key = _batch_key(prepared)
        runs = getattr(ctx, "task_runs", None)
        in_context = getattr(ctx, "in_context", None)
        earlier = runs.get(key) if runs is not None else None
        if (
            earlier is not None
            and args.get("rerun") is not True
            and in_context is not None
            and in_context(earlier[1])
        ):
            return ToolResult.ok(ALREADY_RUN.format(at=earlier[0], chars=earlier[2]))

        # Run every subtask concurrently. return_exceptions=True so one child's
        # failure (e.g. the budget's child cap) is reported, not propagated as a
        # crash that would lose the other children's results.
        results = await asyncio.gather(
            *(spawner.spawn(type_name=tn, prompt=p) for tn, p in prepared),
            return_exceptions=True,
        )

        lines: list[str] = []
        errors = 0
        total_tokens = 0
        total_cost = 0.0
        for i, (type_name, _prompt) in enumerate(prepared):
            res = results[i]
            if isinstance(res, BaseException):
                errors += 1
                lines.append(f"[task {i + 1}] {type_name}: ERROR: {type(res).__name__}: {res}")
            elif isinstance(res, SubAgentResult):
                if res.stop_reason == "completed":
                    note = ""
                elif res.stop_reason == "context_full":
                    note = f" (stopped: context_full; {CONTEXT_FULL_NOTE})"
                elif res.stop_reason == "context_full_no_report":
                    note = f" (stopped: context_full_no_report; {CONTEXT_FULL_NO_REPORT_NOTE})"
                else:
                    note = f" (stopped: {res.stop_reason})"
                lines.append(f"[task {i + 1}] {res.name}{note}:\n{res.summary}")
                total_tokens += res.usage.total_tokens
                total_cost += res.usage.cost_usd

        output = "\n\n".join(lines)
        # Only a batch whose every subtask ran to its end is one a later identical ask may lean on;
        # an error or a stopped subtask leaves the batch to run again (ADR-0382). A subtask that
        # reported when its context filled ran to its end (ADR-0412): the same batch again would
        # fill the same window at the same place.
        if runs is not None:
            if all(
                isinstance(r, SubAgentResult) and r.stop_reason in _RAN_TO_ITS_END for r in results
            ):
                at = datetime.now(UTC).strftime("%H:%M:%SZ")
                runs[key] = (at, output[:_PROBE_CHARS], len(output))
            else:
                runs.pop(key, None)

        # Surface child token/cost spend in the structured result so it is visible to
        # the parent (the children's usage is otherwise invisible to the parent turn).
        return ToolResult(
            output=output,
            is_error=errors == len(results),
            data={
                "count": len(results),
                "errors": errors,
                "child_total_tokens": total_tokens,
                "child_cost_usd": total_cost,
            },
        )


def _running_background_task_named(
    ctx: ToolContext, prompts: list[str]
) -> tuple[int, str, str, str] | None:
    """The first prompt that names a STILL-RUNNING background task of this session, by its id
    (a whole token) or its output file, as ``(index, id, label, output_file)``; else ``None``.
    A finished task is not refused: its output file is readable from a child (ADR-0397)."""
    tasks = getattr(ctx, "background_tasks", None)
    if tasks is None:
        return None
    running = [t for t in tasks.records() if tasks.status(t)[0] == "running"]
    for i, prompt in enumerate(prompts):
        for t in running:
            whole = rf"(?<![0-9A-Za-z_-]){re.escape(t.id)}(?![0-9A-Za-z_-])"
            if re.search(whole, prompt) or (t.output_file and t.output_file in prompt):
                return i, t.id, t.label, t.output_file
    return None


__all__ = ["TaskTool"]
