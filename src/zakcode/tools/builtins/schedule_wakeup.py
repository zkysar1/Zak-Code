"""The ``schedule_wakeup`` tool — Claude Code's ``ScheduleWakeup`` for a Zak Code session
(ADR-0094).

Arms ONE wake-up on the session: at the next idle prompt on or after the delay, the harness
hands the session the prompt as a ``(harness)`` line — the deadman net a Mind's autonomous
loop arms before every re-entry, and the re-poll a parked worker Body arms. A new call
replaces the held wake-up; ``stop`` cancels it. Nothing fires mid-turn. The mechanics live in
:mod:`zakcode.wakeup`; this is the model-facing door.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from zakcode.config import PermissionTier
from zakcode.tools.base import ConcurrencyClass, Tool, ToolContext, ToolResult, ToolSpec
from zakcode.wakeup import (
    DEFAULT_DELAY_SECONDS,
    MAX_DELAY_SECONDS,
    MIN_DELAY_SECONDS,
)


class ScheduleWakeupTool(Tool):
    """Arm (or cancel) the session's one scheduled wake-up."""

    spec = ToolSpec(
        name="schedule_wakeup",
        description=(
            "Schedule a wake-up: after delaySeconds, if the session is at its prompt, it "
            "receives 'prompt' as a harness line and continues. One wake-up is held per "
            "session — a new call replaces it; stop=true cancels it. It fires only at an idle "
            "prompt, never mid-turn. Use it as a safety net for a loop that must re-enter on "
            "its own, or to re-check external state later; never to poll work the harness "
            "already reports on."
        ),
        parameters={
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "What the session is told when the wake-up fires. The sentinel "
                        "'<<autonomous-loop-dynamic>>' means: re-enter the autonomous loop."
                    ),
                },
                "delaySeconds": {
                    "type": "integer",
                    "description": (
                        f"Seconds until it fires, clamped to [{MIN_DELAY_SECONDS}, "
                        f"{MAX_DELAY_SECONDS}] (default {DEFAULT_DELAY_SECONDS})."
                    ),
                },
                "stop": {
                    "type": "boolean",
                    "description": "true cancels the held wake-up instead of arming one.",
                },
                "reason": {
                    "type": "string",
                    "description": "Optional one-line note on why this delay (recorded only).",
                },
            },
        },
        # Touches only the session's own record — always available, like planning.
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.NEVER_PARALLEL,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        slot = ctx.wakeup_slot
        if slot is None:
            return ToolResult.error(
                "wake-ups are not available here (no session to hold one)",
                data={"armed": False},
            )
        if args.get("stop") is True:
            cancelled = slot.cancel()
            text = "Wake-up cancelled." if cancelled else "No wake-up was held; nothing to cancel."
            return ToolResult.ok(text, data={"armed": False, "cancelled": cancelled})
        prompt = args.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return ToolResult.error(
                "'prompt' is required: what the session is told when the wake-up fires "
                "(or pass stop=true to cancel the held one)",
                fix='e.g. {"prompt": "<<autonomous-loop-dynamic>>", "delaySeconds": 600}',
            )
        delay_arg = args.get("delaySeconds", args.get("delay_seconds"))
        replaced = slot.pending() is not None
        wakeup = slot.arm(prompt, delay_arg)
        due = datetime.fromtimestamp(wakeup.due_at, tz=UTC).strftime("%H:%M:%S UTC")
        return ToolResult.ok(
            f"Wake-up armed: in {wakeup.delay_seconds}s (at {due}) this session receives "
            "the prompt if it is at its prompt then, or at the first idle prompt after. "
            + ("It replaced the wake-up held before; " if replaced else "")
            + "one is held at a time.",
            data={
                "armed": True,
                "delay_seconds": wakeup.delay_seconds,
                "due_at": wakeup.due_at,
                "replaced": replaced,
            },
        )
