"""The ``await_user`` tool — the model's handle on "I need the operator" (ADR-0121).

A plan step whose completion needs a human answer — a decision only the operator can
make, a credential, an approval gate — is not work the model can advance by thinking
harder. Before this tool the model's only way to say so was prose, and prose does not
stop a loop: measured 2026-08-22 on slow local inference, the model reached a
non-skippable gate, said so correctly ("I can't advance without your explicit answer"),
and the plan-continuation re-invoked it anyway — 27 of the session's 50 iterations spent
restating "still waiting" at ~15-17 minutes a call, each restatement growing the context
(Zak-Code #181).

Calling this tool makes waiting a turn-ENDING state: the turn stops, the plan is kept
exactly as it stands (the open step stays ``in_progress`` — it is not failed, not
cancelled, not falsely marked done), and the operator's next message resumes it. That is
the difference between a paused turn and a spinning one.

Read-only and never gated: it touches nothing. It is a declaration, not an action.
"""

from __future__ import annotations

from typing import Any

from zakcode.config import PermissionTier
from zakcode.tools.base import ConcurrencyClass, Tool, ToolContext, ToolResult, ToolSpec

#: The question is echoed back capped, so one runaway argument cannot flood the transcript
#: of a turn that is about to end anyway.
_MAX_QUESTION_CHARS = 2000


class AwaitUserTool(Tool):
    """Pause the turn until the operator answers, keeping the plan intact."""

    spec = ToolSpec(
        name="await_user",
        description=(
            "Stop and wait for the operator. Call this — instead of saying you are waiting "
            "— the moment you need something ONLY a person can give: a decision between "
            "options, an approval, a credential, an answer to a question you cannot probe. "
            "The turn ends immediately and your plan is kept as it stands (the open step "
            "stays in progress), and the operator's reply resumes it. Do NOT call it for "
            "anything you can find out yourself: read the file, run the command, search. "
            "Restating that you are blocked without calling this just spends another model "
            "call on the same sentence."
        ),
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "What you need from the operator, as one direct question. Include "
                        "the options if you are asking them to choose."
                    ),
                },
            },
            "required": ["question"],
        },
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.NEVER_PARALLEL,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raw = args.get("question")
        if not isinstance(raw, str) or not raw.strip():
            return ToolResult.error(
                "'question' is required: say what you need from the operator.",
                fix="Call await_user again with the question you want answered.",
            )
        question = raw.strip()[:_MAX_QUESTION_CHARS]
        network = ctx.task_network
        open_steps = len(network.actionable_remaining()) if network is not None else 0
        kept = (
            f" Your plan is kept as it stands ({open_steps} open step(s))."
            if open_steps
            else " Your plan is kept as it stands."
        )
        return ToolResult.ok(
            f"Waiting for the operator: {question}{kept} The turn ends here — do not "
            "continue and do not restate the question.",
            data={"question": question, "open_steps": open_steps},
        )


__all__ = ["AwaitUserTool"]
