"""The lifecycle hook runtime.

Hooks let the operator (and, later, plugins) observe and gate the agent at defined
points — most importantly **before** and **after** each tool call. Two flavours:

* **Shell hooks** — a configured command run as an **argv array** (never through a
  shell, so model-influenced arguments can't be injected). The hook receives a JSON
  payload on **stdin** and signals via its **exit code**: ``0`` = allow, ``2`` =
  block/deny, anything else = warn-and-continue. Its stdout, if valid JSON, may
  carry a ``message`` and (for ``PreToolUse``) rewritten ``arguments``.
* **In-process callbacks** — Python callables (the surface plugins will register).

The defining invariant is **error isolation**: a hook that raises, times out, or
exits weirdly is downgraded to a warning and the turn continues. One bad hook never
breaks the loop (``docs/ARCHITECTURE.md`` / ``docs/GUARDRAILS.md``). Only ``PreToolUse``
can *block* or *mutate*; ``PostToolUse`` is observe-and-warn (the tool already ran).
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("zakcode.hooks")

#: Default per-hook timeout (seconds). A hook exceeding it is killed and treated
#: as a warning, never a block — a hung hook can't wedge the turn.
DEFAULT_HOOK_TIMEOUT = 10.0


class HookEvent(StrEnum):
    """Lifecycle points a hook can fire on. M2 implements the tool-use pair."""

    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"


class HookDecision(StrEnum):
    """A single hook's verdict."""

    ALLOW = "allow"
    BLOCK = "block"
    WARN = "warn"


class HookResult(BaseModel):
    """The aggregate outcome of running all hooks for one event.

    ``blocked`` is ``True`` iff some ``PreToolUse`` hook vetoed the call. ``messages``
    collects any human-facing notes (from blocks and warns). ``mutated_arguments``,
    when set, is the tool input after ``PreToolUse`` rewrites — the caller passes it
    to the tool in place of the original.
    """

    decision: HookDecision = HookDecision.ALLOW
    messages: list[str] = Field(default_factory=list)
    mutated_arguments: dict[str, Any] | None = None

    @property
    def blocked(self) -> bool:
        return self.decision is HookDecision.BLOCK

    @property
    def message(self) -> str:
        """All collected messages joined into one human-readable string."""
        return "; ".join(m for m in self.messages if m)


class HookPayload(BaseModel):
    """The JSON document handed to a hook (on stdin for shell hooks)."""

    event: HookEvent
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    cwd: str = ""
    # PostToolUse only: the result the tool produced.
    output: str | None = None
    is_error: bool | None = None


class HookSpec(BaseModel):
    """A configured shell hook: when to fire, what to run, and how long to wait."""

    event: HookEvent
    command: list[str] = Field(..., description="argv array; run WITHOUT a shell.")
    matcher: str = "*"  # glob over the tool name; '*' = every tool
    timeout: float = DEFAULT_HOOK_TIMEOUT

    def matches(self, tool_name: str) -> bool:
        return fnmatch.fnmatch(tool_name, self.matcher)


#: An in-process hook: ``(payload) -> HookResult | None`` (None == allow/no-op).
#: May be sync or async; both are awaited safely by the manager.
InProcessHook = Callable[[HookPayload], "HookResult | None | Awaitable[HookResult | None]"]


class HookManager:
    """Runs the configured hooks for a lifecycle event, isolating every failure."""

    def __init__(
        self,
        shell_hooks: list[HookSpec] | None = None,
        *,
        in_process: dict[HookEvent, list[InProcessHook]] | None = None,
    ) -> None:
        self.shell_hooks = list(shell_hooks or [])
        self.in_process: dict[HookEvent, list[InProcessHook]] = dict(in_process or {})

    def register(self, event: HookEvent, hook: InProcessHook) -> None:
        """Add an in-process hook (used by the plugin surface later)."""
        self.in_process.setdefault(event, []).append(hook)

    def has_hooks(self, event: HookEvent, tool_name: str) -> bool:
        """Whether any hook would fire for ``event`` on ``tool_name`` (cheap pre-check)."""
        if self.in_process.get(event):
            return True
        return any(h.event is event and h.matches(tool_name) for h in self.shell_hooks)

    async def run(self, payload: HookPayload) -> HookResult:
        """Run every matching hook for ``payload.event`` and aggregate the verdict.

        First ``BLOCK`` wins and stops the chain (only meaningful for ``PreToolUse``).
        ``PreToolUse`` mutations thread forward — each hook sees the prior's rewrite.
        Any hook failure becomes a warning; the turn is never broken.
        """
        decision = HookDecision.ALLOW
        messages: list[str] = []
        arguments = dict(payload.arguments)
        mutated = False

        # In-process hooks first (cheap, trusted), then configured shell hooks.
        for hook in self.in_process.get(payload.event, []):
            current = payload.model_copy(update={"arguments": arguments})
            one = await self._run_in_process(hook, current)
            decision, messages, arguments, mutated = self._fold(
                one, payload.event, decision, messages, arguments, mutated
            )
            if decision is HookDecision.BLOCK:
                return self._result(decision, messages, arguments, mutated)

        for spec in self.shell_hooks:
            if spec.event is not payload.event or not spec.matches(payload.tool_name):
                continue
            current = payload.model_copy(update={"arguments": arguments})
            one = await self._run_shell(spec, current)
            decision, messages, arguments, mutated = self._fold(
                one, payload.event, decision, messages, arguments, mutated
            )
            if decision is HookDecision.BLOCK:
                return self._result(decision, messages, arguments, mutated)

        return self._result(decision, messages, arguments, mutated)

    # ── aggregation helpers ───────────────────────────────────────────────────

    @staticmethod
    def _fold(
        one: HookResult | None,
        event: HookEvent,
        decision: HookDecision,
        messages: list[str],
        arguments: dict[str, Any],
        mutated: bool,
    ) -> tuple[HookDecision, list[str], dict[str, Any], bool]:
        if one is None:
            return (decision, messages, arguments, mutated)
        if one.message:
            messages.append(one.message)
        # Only PreToolUse can actually block (the tool hasn't run yet). A BLOCK from
        # any other event (e.g. PostToolUse) can't undo the call, so it is surfaced
        # as a warning instead of being silently dropped. A WARN is also a warning.
        if one.decision is HookDecision.BLOCK:
            if event is HookEvent.PRE_TOOL_USE:
                decision = HookDecision.BLOCK
            elif decision is HookDecision.ALLOW:
                decision = HookDecision.WARN
        elif one.decision is HookDecision.WARN and decision is HookDecision.ALLOW:
            decision = HookDecision.WARN
        # Only PreToolUse mutations are honored.
        if (
            event is HookEvent.PRE_TOOL_USE
            and one.mutated_arguments is not None
            and isinstance(one.mutated_arguments, dict)
        ):
            arguments = dict(one.mutated_arguments)
            mutated = True
        return (decision, messages, arguments, mutated)

    @staticmethod
    def _result(
        decision: HookDecision,
        messages: list[str],
        arguments: dict[str, Any],
        mutated: bool,
    ) -> HookResult:
        return HookResult(
            decision=decision,
            messages=messages,
            mutated_arguments=arguments if mutated else None,
        )

    # ── runners (each fully error-isolated) ───────────────────────────────────

    @staticmethod
    async def _run_in_process(hook: InProcessHook, payload: HookPayload) -> HookResult | None:
        try:
            outcome = hook(payload)
            if asyncio.iscoroutine(outcome):
                outcome = await outcome
            if outcome is None or isinstance(outcome, HookResult):
                return outcome
            logger.warning("in-process hook returned %r; ignoring", type(outcome))
            return None
        except Exception as exc:  # noqa: BLE001 — a bad hook never breaks the loop
            logger.warning("in-process hook raised %s: %s", type(exc).__name__, exc)
            return HookResult(decision=HookDecision.WARN, messages=[f"hook error: {exc}"])

    async def _run_shell(self, spec: HookSpec, payload: HookPayload) -> HookResult | None:
        """Run one shell hook as an argv array; map its exit code to a decision.

        Protocol: exit 0 = allow, exit 2 = block, anything else = warn. stdout, if
        valid JSON, may provide ``message`` and (PreToolUse) ``arguments``. Spawn
        failure / timeout / weirdness all degrade to a warning.
        """
        if not spec.command:
            return None
        stdin_bytes = payload.model_dump_json().encode("utf-8")
        try:
            proc = await asyncio.create_subprocess_exec(
                *spec.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (OSError, ValueError) as exc:
            logger.warning("hook %r failed to start: %s", spec.command, exc)
            return HookResult(decision=HookDecision.WARN, messages=[f"hook failed to start: {exc}"])

        try:
            stdout, _stderr = await asyncio.wait_for(
                proc.communicate(stdin_bytes), timeout=spec.timeout
            )
        except TimeoutError:
            with _suppress():
                proc.kill()
                await proc.wait()
            logger.warning("hook %r timed out after %ss", spec.command, spec.timeout)
            return HookResult(
                decision=HookDecision.WARN,
                messages=[f"hook timed out after {spec.timeout}s"],
            )
        except Exception as exc:  # noqa: BLE001 — never propagate a hook failure
            logger.warning("hook %r errored: %s", spec.command, exc)
            return HookResult(decision=HookDecision.WARN, messages=[f"hook error: {exc}"])

        code = proc.returncode
        message, mutated_args = self._parse_stdout(stdout)

        if code == 0:
            return HookResult(
                decision=HookDecision.ALLOW, messages=_msgs(message), mutated_arguments=mutated_args
            )
        if code == 2:
            return HookResult(
                decision=HookDecision.BLOCK,
                messages=_msgs(message) or ["blocked by hook"],
                mutated_arguments=mutated_args,
            )
        return HookResult(
            decision=HookDecision.WARN,
            messages=_msgs(message) or [f"hook exited with code {code}"],
            mutated_arguments=mutated_args,
        )

    @staticmethod
    def _parse_stdout(stdout: bytes) -> tuple[str, dict[str, Any] | None]:
        """Best-effort parse of a hook's stdout for a message + mutated arguments."""
        text = (stdout or b"").decode("utf-8", errors="replace").strip()
        if not text:
            return ("", None)
        try:
            doc = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            # Plain text on stdout is just a message.
            return (text, None)
        if not isinstance(doc, dict):
            return (text, None)
        message = doc.get("message", "")
        message = message if isinstance(message, str) else ""
        args = doc.get("arguments")
        mutated = args if isinstance(args, dict) else None
        return (message, mutated)


def _msgs(message: str) -> list[str]:
    return [message] if message else []


class _suppress:
    """Tiny context manager that swallows any exception (best-effort cleanup)."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> bool:
        return True


__all__ = [
    "HookEvent",
    "HookDecision",
    "HookResult",
    "HookPayload",
    "HookSpec",
    "HookManager",
    "InProcessHook",
    "DEFAULT_HOOK_TIMEOUT",
]
