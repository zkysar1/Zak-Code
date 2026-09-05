"""The ``plan_recall`` tool — the model's READ handle on the plan's record (ADR-0111).

``update_plan`` writes the plan; the harness writes the record behind it — each step's
evidence (the tool calls that ran while it was current), its outcome, and the plan's history
(ADR-0110). The re-injected plan shows the newest slice of that record on every iteration; this
tool lets the model read the REST of it on demand: what the request was, what an earlier step
did and found, whether something was already tried, what changed in the plan and when. Those
are the questions a model that has lost its context — after a compaction, a resume, or twenty
iterations of a long goal — otherwise answers by guessing or by re-doing the work.

Read-only: it touches nothing, so it is always available and never gated.
"""

from __future__ import annotations

from typing import Any

from zakcode.config import PermissionTier
from zakcode.tasks import PlanEvent, Task, TaskNetwork, clip
from zakcode.tools.base import ConcurrencyClass, Tool, ToolContext, ToolResult, ToolSpec

#: History events shown by the overview when the model does not say how many.
_DEFAULT_LAST = 8
#: Upper bound on history events in one answer — the record is bounded already; this keeps
#: one tool result from re-injecting all of it.
_MAX_LAST = 60
#: Upper bound on search hits in one answer.
_MAX_HITS = 40

_GLYPH = {"pending": " ", "in_progress": "~", "done": "x", "blocked": "!", "cancelled": "-"}


def _event_line(event: PlanEvent) -> str:
    when = event.at[11:19] if len(event.at) >= 19 else event.at
    head = f"#{event.seq} {when} {event.kind}"
    if event.step:
        head += f" step {event.step}"
    return f"{head}: {event.detail}" if event.detail else head


def _overview(network: TaskNetwork, last: int) -> str:
    lines = [_request_line(network), ""]
    leaves = network.leaves()
    if leaves:
        lines.append(f"Steps ({len(leaves)}):")
        for task in leaves:
            shown = (
                task.outcome if task.status in ("done", "cancelled") and task.outcome else task.note
            )
            detail = f" — {shown}" if shown else ""
            calls = f" · {len(task.evidence)} tool call(s)" if task.evidence else ""
            lines.append(
                f"  [{_GLYPH.get(task.status, ' ')}] {task.id} {task.title}{detail}{calls}"
            )
    else:
        lines.append("Steps: none on the board right now.")
    events = network.log[-last:] if last > 0 else []
    if events:
        lines.append("")
        folded = f" ({network.log_folded} older folded)" if network.log_folded else ""
        lines.append(f"History (last {len(events)} of {len(network.log)}{folded}):")
        lines.extend(f"  {_event_line(e)}" for e in events)
    lines.append("")
    lines.append(
        "Ask plan_recall with a step id for one step's full evidence and history, or with a "
        "query to search titles, outcomes, evidence and history."
    )
    return "\n".join(lines)


def _request_line(network: TaskNetwork) -> str:
    request = network.context.request
    return f"Request: {request}" if request else "Request: (not recorded for this plan)"


def _step_report(network: TaskNetwork, task: Task) -> str:
    lines = [
        _request_line(network),
        "",
        f"Step {task.id} [{task.status}] {task.title}",
    ]
    if task.note:
        lines.append(f"  done-condition: {task.note}")
    if task.outcome:
        lines.append(f"  outcome: {task.outcome}")
    if task.evidence:
        lines.append(f"  evidence ({len(task.evidence)} tool call(s), oldest first):")
        lines.extend(f"    - {line}" for line in task.evidence)
    else:
        lines.append("  evidence: no tool call has run while this step was current")
    history = [e for e in network.log if e.step == task.id][-_MAX_LAST:]
    if history:
        lines.append("  history:")
        lines.extend(f"    {_event_line(e)}" for e in history)
    return "\n".join(lines)


def _search(network: TaskNetwork, query: str) -> str:
    needle = query.lower()
    hits: list[str] = []
    for task in network.leaves():
        fields = (("title", task.title), ("done-condition", task.note), ("outcome", task.outcome))
        for label, value in fields:
            if value and needle in value.lower():
                hits.append(f"step {task.id} {label}: {clip(value, 120)}")
        for line in task.evidence:
            if needle in line.lower():
                hits.append(f"step {task.id} evidence: {line}")
    for event in network.log:
        if needle in event.detail.lower() or needle in event.title.lower():
            hits.append(f"history {_event_line(event)}")
    if not hits:
        return f"{_request_line(network)}\n\nNo match for {query!r} in the plan's record."
    shown = hits[:_MAX_HITS]
    more = f"\n  … {len(hits) - len(shown)} more" if len(hits) > len(shown) else ""
    return (
        f"{_request_line(network)}\n\n{len(hits)} match(es) for {query!r}:\n  "
        + "\n  ".join(shown)
        + more
    )


class PlanRecallTool(Tool):
    """Read the plan's record: the request, what each step did and produced, the history."""

    spec = ToolSpec(
        name="plan_recall",
        description=(
            "Read the plan's RECORD: what the request was, what each step did (the tool calls "
            "that ran while it was current) and produced (its outcome), and the history of the "
            "plan. Use it whenever you have lost track — 'what did I do a few steps ago', 'what "
            "did step 2 find', 'have I already tried this' — instead of re-doing work or "
            "guessing. With no arguments: the request, every step with its outcome, and the "
            "recent history. 'step' returns one step's full evidence and history; 'query' "
            "searches titles, outcomes, evidence and history. Read-only."
        ),
        parameters={
            "type": "object",
            "properties": {
                "step": {
                    "type": "string",
                    "description": "A step id as shown in the plan ('2', '3.1'): its record.",
                },
                "query": {
                    "type": "string",
                    "description": "Text to search for across the record (case-insensitive).",
                },
                "last": {
                    "type": "integer",
                    "description": (
                        f"How many recent history events the overview shows (default "
                        f"{_DEFAULT_LAST}, at most {_MAX_LAST})."
                    ),
                },
            },
        },
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        network = ctx.task_network
        if network is None:
            return ToolResult.error(
                "planning is not available here (no task network on the context)"
            )
        data = {
            "steps": len(network.leaves()),
            "events": len(network.log),
            "folded": network.log_folded,
        }
        raw_step = args.get("step")
        if raw_step not in (None, ""):
            step_id = str(raw_step).strip()
            task = network.get(step_id)
            if task is None:
                known = ", ".join(t.id for t in network.leaves()) or "none"
                return ToolResult.error(
                    f"no step {step_id!r} in the plan (steps: {known})",
                    fix="Call plan_recall with no arguments to see the current step ids.",
                )
            return ToolResult.ok(_step_report(network, task), data=data)
        raw_query = args.get("query")
        if isinstance(raw_query, str) and raw_query.strip():
            return ToolResult.ok(_search(network, raw_query.strip()), data=data)
        raw_last = args.get("last")
        last = _DEFAULT_LAST
        if isinstance(raw_last, int) and not isinstance(raw_last, bool):
            last = max(0, min(raw_last, _MAX_LAST))
        return ToolResult.ok(_overview(network, last), data=data)


__all__ = ["PlanRecallTool"]
