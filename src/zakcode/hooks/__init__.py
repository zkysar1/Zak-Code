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
import os
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from zakcode._subprocess import new_group_kwargs, terminate_process_tree

logger = logging.getLogger("zakcode.hooks")

#: Default per-hook timeout (seconds). A hook exceeding it is killed and treated
#: as a warning, never a block — a hung hook can't wedge the turn.
DEFAULT_HOOK_TIMEOUT = 10.0


def _scrubbed_env(drop: list[str]) -> dict[str, str]:
    """Return a copy of ``os.environ`` with *drop* names removed."""
    child_env = dict(os.environ)
    for name in drop:
        child_env.pop(name, None)
    return child_env


def _hook_env(drop: list[str], cwd: str) -> dict[str, str]:
    """Child env for a shell hook: ``os.environ`` minus *drop*, plus ``CLAUDE_PROJECT_DIR`` set to
    the workspace root (forward-slash form for Git Bash). That is Claude Code's hook env var —
    Claude-Code hooks read it and reference scripts through it; Claude Code sets it, so we do too.
    """
    env = _scrubbed_env(drop)
    if cwd:
        env["CLAUDE_PROJECT_DIR"] = cwd.replace("\\", "/")
    return env


class HookEvent(StrEnum):
    """Lifecycle points a hook can fire on.

    The tool-use pair (M2) gates each tool call. ``PRE_LLM_CALL`` fires before every
    model completion and is the *context-injection* seam: its hooks return text that
    is folded into the turn as an ephemeral tail message (never persisted, after all
    cached content — so prompt caching is preserved). It is the plug a memory-recall
    layer, a RAG step, or a self-learning framework's retrieval script uses.
    """

    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    PRE_LLM_CALL = "PreLLMCall"
    # Session-lifecycle events. Observe-only (fire-and-forget side effects): a host
    # automation (e.g. a self-learning framework's prime / encode / serialize step)
    # registers here. They cannot block or rewrite anything — see HookManager.fire.
    SESSION_START = "SessionStart"
    SESSION_END = "SessionEnd"
    PRE_COMPACT = "PreCompact"
    # Fired (observe-only) when a skill is invoked, carrying the chosen skill name and the
    # triggering query in the LifecyclePayload ``data`` map. The seam a learning layer records
    # (query -> skill) from to learn habitual skill preferences (it does NOT pick skills).
    ON_SKILL_SELECTED = "OnSkillSelected"
    # Fired when the loop is about to end a turn. Unlike lifecycle events (observe-only)
    # and unlike PRE_TOOL_USE (which gates individual tool calls), TURN_END gates the
    # turn's exit. A BLOCK decision vetoes the stop, injects a continuation prompt, and
    # re-enters the iteration loop.
    TURN_END = "TurnEnd"


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
    """The JSON document handed to a tool-gate hook (on stdin for shell hooks).

    Shell hooks are serialized with ``by_alias=True`` so the wire shape matches the **Claude
    Code hook contract** — ``tool_input`` (not ``arguments``) and a top-level ``session_id`` —
    which frameworks like claude-mind read (e.g. ``tool_input.command``, ``session_id`` to
    resolve the agent and inject env). The in-process attribute stays ``arguments``.
    """

    event: HookEvent
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict, serialization_alias="tool_input")
    session_id: str = ""
    cwd: str = ""
    # PostToolUse only: the result the tool produced.
    output: str | None = None
    is_error: bool | None = None


class LLMContextPayload(BaseModel):
    """The JSON document handed to a ``PRE_LLM_CALL`` context hook (stdin for shells).

    Unlike :class:`HookPayload` (a tool-gate event), this carries the turn's
    triggering ``user_text`` and the loop position, so a hook can decide what
    background context to retrieve. A context hook returns *text to inject*, never
    an allow/block decision.
    """

    event: HookEvent = HookEvent.PRE_LLM_CALL
    user_text: str = ""
    cwd: str = ""
    iteration: int = 0
    message_count: int = 0


class LifecyclePayload(BaseModel):
    """The JSON document handed to a session-lifecycle hook (stdin for shells).

    Carries enough to locate the session's state — the ``session_id`` and ``cwd`` —
    plus a free-form ``data`` map for event-specific extras. A lifecycle hook runs
    for its side effects only.
    """

    event: HookEvent
    session_id: str = ""
    cwd: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class TurnEndPayload(BaseModel):
    """Payload for TURN_END hooks.

    Field names match Claude Code's Stop-hook stdin protocol so Mind's
    stop-hook.sh works unmodified: it reads ``session_id`` and
    ``last_assistant_message`` via ``json.load(sys.stdin).get()``.
    """

    event: HookEvent = HookEvent.TURN_END
    session_id: str = ""
    cwd: str = ""
    stop_reason: str = ""
    iterations: int = 0
    max_iterations: int = 0
    degraded: bool = False
    last_assistant_message: str = ""


class TurnEndResult(BaseModel):
    """Aggregate result of running TURN_END hooks."""

    vetoed: bool = False
    continuation_prompt: str = ""


#: Claude-Code tool names → the Zak Code tools they map to. A hook ``matcher`` written for Claude
#: Code (``"Skill"``, ``"Read"``, ``"Bash"``) thus fires on the equivalent Zak Code tool
#: (``use_skill``, ``read_file``, ``bash``) — so a Claude-Code framework's ``PreToolUse`` gates
#: (e.g. claude-mind's skill-dedup gate) apply to the model's calls unchanged. Also corrects case
#: (``Bash`` vs ``bash``; ``fnmatch`` is case-sensitive on POSIX). Only clear counterparts mapped.
_CLAUDE_CODE_TOOL_NAMES: dict[str, tuple[str, ...]] = {
    "use_skill": ("Skill",),
    "read_file": ("Read",),
    "write_file": ("Write",),
    "edit_file": ("Edit", "MultiEdit"),
    "bash": ("Bash",),
    "glob": ("Glob",),
    "grep": ("Grep",),
    "list_dir": ("LS",),
    "web_search": ("WebSearch",),
    "web_fetch": ("WebFetch",),
    "task": ("Task",),
    "update_plan": ("TodoWrite",),
}


class HookSpec(BaseModel):
    """A configured shell hook: when to fire, what to run, and how long to wait."""

    event: HookEvent
    command: list[str] = Field(..., description="argv array; run WITHOUT a shell.")
    matcher: str = "*"  # glob over the tool name; '*' = every tool
    timeout: float = DEFAULT_HOOK_TIMEOUT
    # Env-var names to scrub from the child environment before spawning.  Populated
    # by the settings_loader for workspace-sourced hooks (TE-R1: provider-key hygiene
    # applies to ALL workspace hooks, not just TURN_END).
    drop_env: list[str] = Field(default_factory=list)

    def matches(self, tool_name: str) -> bool:
        # Match the tool's own name OR its Claude-Code equivalent(s), so a matcher written for
        # Claude Code (e.g. "Skill") fires on the corresponding Zak Code tool ("use_skill").
        names = (tool_name, *_CLAUDE_CODE_TOOL_NAMES.get(tool_name, ()))
        return any(fnmatch.fnmatch(name, self.matcher) for name in names)


#: An in-process hook: ``(payload) -> HookResult | None`` (None == allow/no-op).
#: May be sync or async; both are awaited safely by the manager.
InProcessHook = Callable[[HookPayload], "HookResult | None | Awaitable[HookResult | None]"]

#: An in-process context hook: ``(payload) -> str | None`` returning text to inject
#: before a model call (None == nothing to add). May be sync or async.
ContextHook = Callable[[LLMContextPayload], "str | None | Awaitable[str | None]"]

#: An in-process lifecycle hook: ``(payload) -> None`` run for side effects only.
#: May be sync or async; its return value is ignored.
LifecycleHook = Callable[[LifecyclePayload], "None | Awaitable[None]"]

#: An in-process TURN_END hook: ``(payload) -> TurnEndResult | None`` (None == allow/no-op).
#: May be sync or async; both are awaited safely by the manager.
InProcessTurnEndHook = Callable[
    [TurnEndPayload], "TurnEndResult | None | Awaitable[TurnEndResult | None]"
]


class HookManager:
    """Runs the configured hooks for a lifecycle event, isolating every failure."""

    def __init__(
        self,
        shell_hooks: list[HookSpec] | None = None,
        *,
        in_process: dict[HookEvent, list[InProcessHook]] | None = None,
        context: list[ContextHook] | None = None,
    ) -> None:
        self.shell_hooks = list(shell_hooks or [])
        self.in_process: dict[HookEvent, list[InProcessHook]] = dict(in_process or {})
        # PRE_LLM_CALL in-process context hooks (return text, not a decision).
        self.context_hooks: list[ContextHook] = list(context or [])
        # Session-lifecycle in-process hooks, keyed by event (observe-only).
        self.lifecycle_hooks: dict[HookEvent, list[LifecycleHook]] = {}
        # TURN_END in-process hooks: veto-capable, first veto wins and stops the chain.
        self.turn_end_hooks: list[InProcessTurnEndHook] = []

    def register(self, event: HookEvent, hook: InProcessHook) -> None:
        """Add an in-process hook (used by the plugin surface later)."""
        self.in_process.setdefault(event, []).append(hook)

    def register_context(self, hook: ContextHook) -> None:
        """Add an in-process ``PRE_LLM_CALL`` context hook (returns text to inject)."""
        self.context_hooks.append(hook)

    def has_context_hooks(self) -> bool:
        """Whether any context hook would run (cheap pre-check before a model call)."""
        if self.context_hooks:
            return True
        return any(h.event is HookEvent.PRE_LLM_CALL for h in self.shell_hooks)

    def register_lifecycle(self, event: HookEvent, hook: LifecycleHook) -> None:
        """Add an in-process session-lifecycle hook (observe-only)."""
        self.lifecycle_hooks.setdefault(event, []).append(hook)

    def register_turn_end(self, hook: InProcessTurnEndHook) -> None:
        """Add an in-process ``TURN_END`` hook (veto-capable)."""
        self.turn_end_hooks.append(hook)

    def has_lifecycle_hooks(self, event: HookEvent) -> bool:
        """Whether any hook would fire for ``event`` (cheap pre-check)."""
        if self.lifecycle_hooks.get(event):
            return True
        return any(h.event is event for h in self.shell_hooks)

    def has_hooks(self, event: HookEvent, tool_name: str) -> bool:
        """Whether any hook would fire for ``event`` on ``tool_name`` (cheap pre-check)."""
        if event is HookEvent.TURN_END and self.turn_end_hooks:
            return True
        if self.in_process.get(event):
            return True
        return any(h.event is event and h.matches(tool_name) for h in self.shell_hooks)

    def has_turn_end_hooks(self) -> bool:
        """Whether any ``TURN_END`` hook is registered (cheap pre-check).

        Matcher-agnostic on purpose: TURN_END is not a per-tool event, so a shell
        spec's ``matcher`` (which defaults to ``"*"``) is ignored here.
        """
        return bool(self.turn_end_hooks) or any(
            spec.event is HookEvent.TURN_END for spec in self.shell_hooks
        )

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

    async def gather_context(self, payload: LLMContextPayload) -> list[str]:
        """Run every ``PRE_LLM_CALL`` context hook and collect the text to inject.

        In-process context hooks run first (cheap, trusted), then shell hooks whose
        event is ``PRE_LLM_CALL`` (matcher is ignored — there is no tool name).
        Returns the non-empty, stripped strings in run order. Every failure is
        isolated to a skipped contribution; a context hook can never block or break
        the turn (the worst case is no extra context).
        """
        collected: list[str] = []
        for hook in self.context_hooks:
            text = await self._run_context_in_process(hook, payload)
            if text and text.strip():
                collected.append(text.strip())
        for spec in self.shell_hooks:
            if spec.event is not HookEvent.PRE_LLM_CALL:
                continue
            text = await self._run_context_shell(spec, payload)
            if text and text.strip():
                collected.append(text.strip())
        return collected

    async def fire(self, payload: LifecyclePayload) -> None:
        """Fire a session-lifecycle event (observe-only; never blocks or rewrites).

        Runs in-process lifecycle hooks for ``payload.event`` first, then shell hooks
        whose event matches (the payload JSON on stdin; stdout/exit code are advisory
        and ignored). Every failure is isolated — a lifecycle hook can never break a
        turn or session. This is the seam a host's prime / encode / serialize
        automation plugs into.
        """
        for hook in self.lifecycle_hooks.get(payload.event, []):
            await self._run_lifecycle_in_process(hook, payload)
        for spec in self.shell_hooks:
            if spec.event is payload.event:
                await self._run_lifecycle_shell(spec, payload)

    # ── TURN_END dispatch (veto-capable) ─────────────────────────────────────

    async def run_turn_end(
        self, payload: TurnEndPayload, *, drop_env: list[str] | None = None
    ) -> TurnEndResult:
        """Run TURN_END hooks; first veto wins and stops the chain.

        In-process hooks run first, then shell hooks. Both can veto (block the turn
        end). ``drop_env`` lists env-var names to scrub from shell-hook children
        (provider-key hygiene).
        """
        for hook in self.turn_end_hooks:
            result = await self._run_turn_end_in_process(hook, payload)
            if result is not None and result.vetoed:
                return result
        for spec in self.shell_hooks:
            if spec.event is not HookEvent.TURN_END:
                continue
            result = await self._run_turn_end_shell(spec, payload, drop_env=drop_env)
            if result.vetoed:
                return result
        return TurnEndResult()

    @staticmethod
    async def _run_turn_end_in_process(
        hook: InProcessTurnEndHook, payload: TurnEndPayload
    ) -> TurnEndResult | None:
        """Run one in-process TURN_END hook; errors fail-open."""
        try:
            outcome = hook(payload)
            if asyncio.iscoroutine(outcome):
                outcome = await outcome
            if outcome is None or isinstance(outcome, TurnEndResult):
                return outcome
            logger.warning("turn_end in-process hook returned %r; ignoring", type(outcome))
            return None
        except Exception as exc:  # noqa: BLE001 — a bad hook never breaks the loop
            logger.warning("turn_end in-process hook raised %s: %s", type(exc).__name__, exc)
            return None

    async def _run_turn_end_shell(
        self,
        spec: HookSpec,
        payload: TurnEndPayload,
        *,
        drop_env: list[str] | None = None,
    ) -> TurnEndResult:
        """Run one TURN_END shell hook.

        Protocol (Claude Code Stop-hook compatible):
        - Exit 0 + stdout JSON ``{"decision":"block","reason":"..."}`` → vetoed
        - Exit 0 + empty stdout (or ``{"decision":"allow"}``) → allow
        - Exit 2 + stdout message → vetoed (native Zak-Code exit-2 protocol)
        - Non-zero (except 2) / timeout / crash / spawn failure → fail-open allow
        """
        if not spec.command:
            return TurnEndResult()
        stdin_bytes = payload.model_dump_json(by_alias=True).encode("utf-8")

        # Build a scrubbed child env (provider-key hygiene). The per-spec list is the
        # PRIMARY scrub (populated at settings ingestion — TE-R1: hygiene applies to
        # all workspace hooks, like every sibling shell runner); the caller-supplied
        # ``drop_env`` is layered on top, so neither path depends on the other.
        child_env = _hook_env([*spec.drop_env, *(drop_env or [])], payload.cwd)

        try:
            proc = await asyncio.create_subprocess_exec(
                *spec.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=_valid_cwd(payload.cwd),  # run at the workspace root (see _valid_cwd)
                env=child_env,
                **new_group_kwargs(),
            )
        except (OSError, ValueError) as exc:
            logger.warning("turn_end hook %r failed to start: %s", spec.command, exc)
            return TurnEndResult()

        try:
            stdout, _stderr = await asyncio.wait_for(
                proc.communicate(stdin_bytes), timeout=spec.timeout
            )
        except (TimeoutError, asyncio.CancelledError) as exc:
            await terminate_process_tree(proc)
            if isinstance(exc, asyncio.CancelledError):
                raise
            logger.warning("turn_end hook %r timed out after %ss", spec.command, spec.timeout)
            return TurnEndResult()
        except Exception as exc:  # noqa: BLE001 — never propagate a hook failure
            logger.warning("turn_end hook %r errored: %s", spec.command, exc)
            return TurnEndResult()

        code = proc.returncode
        text = (stdout or b"").decode("utf-8", errors="replace").strip()

        # Native exit-2 protocol: vetoed with the stdout text as the prompt.
        if code == 2:
            return TurnEndResult(vetoed=True, continuation_prompt=text or "Continue.")

        if code != 0:
            # Non-zero (non-2) exit: fail-open.
            return TurnEndResult()

        # Exit 0: check stdout for the Claude Code decision JSON.
        if not text:
            return TurnEndResult()
        try:
            doc = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return TurnEndResult()
        if isinstance(doc, dict) and doc.get("decision") == "block":
            reason = doc.get("reason", "Continue.")
            return TurnEndResult(
                vetoed=True,
                continuation_prompt=reason if isinstance(reason, str) else "Continue.",
            )
        return TurnEndResult()

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

    # ── context runners (PRE_LLM_CALL; each fully error-isolated) ──────────────

    @staticmethod
    async def _run_context_in_process(hook: ContextHook, payload: LLMContextPayload) -> str | None:
        try:
            outcome = hook(payload)
            if asyncio.iscoroutine(outcome):
                outcome = await outcome
            if outcome is None or isinstance(outcome, str):
                return outcome
            logger.warning("context hook returned %r; ignoring", type(outcome))
            return None
        except Exception as exc:  # noqa: BLE001 — a bad context hook never breaks the loop
            logger.warning("context hook raised %s: %s", type(exc).__name__, exc)
            return None

    async def _run_context_shell(self, spec: HookSpec, payload: LLMContextPayload) -> str | None:
        """Run one ``PRE_LLM_CALL`` shell hook; its stdout is the text to inject.

        Protocol: exit 0 = use stdout (plain text, or JSON ``{"context": "..."}``);
        any non-zero exit, spawn failure, timeout, or weirdness contributes nothing.
        A context hook can never block the turn.
        """
        if not spec.command:
            return None
        stdin_bytes = payload.model_dump_json(by_alias=True).encode("utf-8")
        child_env = _hook_env(spec.drop_env, payload.cwd)
        try:
            proc = await asyncio.create_subprocess_exec(
                *spec.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=_valid_cwd(payload.cwd),  # run at the workspace root (see _valid_cwd)
                env=child_env,
                **new_group_kwargs(),  # own process group so the whole tree is killable
            )
        except (OSError, ValueError) as exc:
            logger.warning("context hook %r failed to start: %s", spec.command, exc)
            return None

        try:
            stdout, _stderr = await asyncio.wait_for(
                proc.communicate(stdin_bytes), timeout=spec.timeout
            )
        except (TimeoutError, asyncio.CancelledError) as exc:
            # Kill the whole child tree (not just the parent) on timeout, AND on a turn
            # cancellation — CancelledError is a BaseException that the bare `except Exception`
            # below would NOT catch, so without this a cancelled turn orphans the hook. (audit4 #2)
            await terminate_process_tree(proc)
            if isinstance(exc, asyncio.CancelledError):
                raise
            logger.warning("context hook %r timed out after %ss", spec.command, spec.timeout)
            return None
        except Exception as exc:  # noqa: BLE001 — never propagate a context-hook failure
            logger.warning("context hook %r errored: %s", spec.command, exc)
            return None

        if proc.returncode != 0:
            return None
        text = (stdout or b"").decode("utf-8", errors="replace").strip()
        if not text:
            return None
        # A JSON object may wrap the text under a "context" key; plain text is used as-is.
        try:
            doc = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return text
        if isinstance(doc, dict):
            ctx = doc.get("context")
            return ctx if isinstance(ctx, str) and ctx.strip() else None
        return text

    # ── lifecycle runners (observe-only; each fully error-isolated) ────────────

    @staticmethod
    async def _run_lifecycle_in_process(hook: LifecycleHook, payload: LifecyclePayload) -> None:
        try:
            outcome = hook(payload)
            if asyncio.iscoroutine(outcome):
                await outcome
        except Exception as exc:  # noqa: BLE001 — a lifecycle hook never breaks a session
            logger.warning("lifecycle hook raised %s: %s", type(exc).__name__, exc)

    async def _run_lifecycle_shell(self, spec: HookSpec, payload: LifecyclePayload) -> None:
        """Run one lifecycle shell hook for its side effects; output is advisory."""
        if not spec.command:
            return
        stdin_bytes = payload.model_dump_json(by_alias=True).encode("utf-8")
        child_env = _hook_env(spec.drop_env, payload.cwd)
        try:
            proc = await asyncio.create_subprocess_exec(
                *spec.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=_valid_cwd(payload.cwd),  # run at the workspace root (see _valid_cwd)
                env=child_env,
                **new_group_kwargs(),  # own process group so the whole tree is killable
            )
        except (OSError, ValueError) as exc:
            logger.warning("lifecycle hook %r failed to start: %s", spec.command, exc)
            return
        try:
            await asyncio.wait_for(proc.communicate(stdin_bytes), timeout=spec.timeout)
        except (TimeoutError, asyncio.CancelledError) as exc:
            await terminate_process_tree(proc)  # tree, not parent-only (audit4 #2)
            if isinstance(exc, asyncio.CancelledError):
                raise
            logger.warning("lifecycle hook %r timed out after %ss", spec.command, spec.timeout)
        except Exception as exc:  # noqa: BLE001 — never propagate a lifecycle-hook failure
            logger.warning("lifecycle hook %r errored: %s", spec.command, exc)

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
        stdin_bytes = payload.model_dump_json(by_alias=True).encode("utf-8")
        child_env = _hook_env(spec.drop_env, payload.cwd)
        try:
            proc = await asyncio.create_subprocess_exec(
                *spec.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=_valid_cwd(payload.cwd),  # run at the workspace root (see _valid_cwd)
                env=child_env,
                **new_group_kwargs(),  # own process group so the whole tree is killable
            )
        except (OSError, ValueError) as exc:
            logger.warning("hook %r failed to start: %s", spec.command, exc)
            return HookResult(decision=HookDecision.WARN, messages=[f"hook failed to start: {exc}"])

        try:
            stdout, _stderr = await asyncio.wait_for(
                proc.communicate(stdin_bytes), timeout=spec.timeout
            )
        except (TimeoutError, asyncio.CancelledError) as exc:
            await terminate_process_tree(proc)  # tree, not parent-only (audit4 #2)
            if isinstance(exc, asyncio.CancelledError):
                raise
            logger.warning("hook %r timed out after %ss", spec.command, spec.timeout)
            return HookResult(
                decision=HookDecision.WARN,
                messages=[f"hook timed out after {spec.timeout}s"],
            )
        except Exception as exc:  # noqa: BLE001 — never propagate a hook failure
            logger.warning("hook %r errored: %s", spec.command, exc)
            return HookResult(decision=HookDecision.WARN, messages=[f"hook error: {exc}"])

        code = proc.returncode
        message, mutated_args, deny = self._parse_stdout(stdout)

        # Claude Code blocks via {"hookSpecificOutput": {"permissionDecision": "deny"}} on
        # exit 0 (not exit 2). Honor it as a BLOCK regardless of exit code.
        if deny:
            return HookResult(
                decision=HookDecision.BLOCK,
                messages=_msgs(message) or ["blocked by hook"],
                mutated_arguments=mutated_args,
            )
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
    def _parse_stdout(stdout: bytes) -> tuple[str, dict[str, Any] | None, bool]:
        """Best-effort parse of a hook's stdout: ``(message, mutated_arguments, deny)``.

        Understands two stdout shapes:

        - **Zak-native:** ``{"message": str, "arguments": {...}}``.
        - **Claude Code** (what claude-mind and other Claude-Code hooks emit):
          ``{"hookSpecificOutput": {"updatedInput": {...}, "permissionDecision":
          "allow"|"deny"|"ask", "permissionDecisionReason": str}}``. ``updatedInput`` rewrites
          the tool arguments (e.g. claude-mind's ``bash-agent-inject`` prepending
          ``export PATH=...; export MIND_SID=...`` to a Bash command); ``permissionDecision ==
          "deny"`` blocks the call — Claude Code blocks via this JSON on **exit 0**, not exit 2 —
          so it is surfaced for the caller to honor.
        """
        text = (stdout or b"").decode("utf-8", errors="replace").strip()
        if not text:
            return ("", None, False)
        try:
            doc = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            # Plain text on stdout is just a message.
            return (text, None, False)
        if not isinstance(doc, dict):
            return (text, None, False)
        message = doc.get("message", "")
        message = message if isinstance(message, str) else ""
        args = doc.get("arguments")
        mutated = args if isinstance(args, dict) else None
        deny = False
        hso = doc.get("hookSpecificOutput")
        if isinstance(hso, dict):
            updated = hso.get("updatedInput")
            if isinstance(updated, dict):
                mutated = updated
            deny = hso.get("permissionDecision") == "deny"
            reason = hso.get("permissionDecisionReason")
            if not message and isinstance(reason, str):
                message = reason
        return (message, mutated, deny)


def _valid_cwd(cwd: str) -> str | None:
    """The directory a hook should run in: the workspace root when it exists, else inherit.

    Hooks run AT the workspace root so a Claude-Code hook's relative command (e.g.
    claude-mind's ``bash core/scripts/...``) resolves. Guarded with ``isdir`` so a payload
    carrying a bogus/relative cwd never breaks hook spawn — it just inherits the parent cwd.
    """
    return cwd if cwd and os.path.isdir(cwd) else None


def _msgs(message: str) -> list[str]:
    return [message] if message else []


__all__ = [
    "HookEvent",
    "HookDecision",
    "HookResult",
    "HookPayload",
    "LLMContextPayload",
    "LifecyclePayload",
    "TurnEndPayload",
    "TurnEndResult",
    "HookSpec",
    "HookManager",
    "InProcessHook",
    "InProcessTurnEndHook",
    "ContextHook",
    "LifecycleHook",
    "DEFAULT_HOOK_TIMEOUT",
]
