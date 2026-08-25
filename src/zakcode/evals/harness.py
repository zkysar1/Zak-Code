"""The evaluation harness core: a scripted provider, a case/result/report model, and
a runner.

Everything here is deterministic and offline. :class:`ScriptedProvider` is a real
:class:`~zakcode.providers.base.Provider` whose completions come from a fixed script
or a caller-supplied responder — never a network call — so a probe's outcome depends
only on the agent's own logic. Token counting is a deterministic, history-growing
estimate and the context window is configurable, so compaction thresholds can be
crossed cheaply and predictably.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from zakcode.messages import Message
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.usage import Usage

if TYPE_CHECKING:
    from zakcode import Agent
    from zakcode.permissions import PermissionPrompter
    from zakcode.tools.base import Tool

#: A dynamic completion source: ``(messages, system, call_index) -> LLMResult``.
Responder = Callable[[list[Message], str | None, int], LLMResult]


# ── LLMResult builders (keep probe scripts terse) ────────────────────────────────


def reply(text: str, *, usage: Usage | None = None) -> LLMResult:
    """An assistant turn that ends the turn (no tool calls)."""
    return LLMResult(text=text, finish_reason="stop", usage=usage or Usage())


def call_tool(
    name: str,
    arguments: dict[str, Any] | None = None,
    *,
    id: str = "call-1",
    usage: Usage | None = None,
) -> LLMResult:
    """An assistant turn that requests a single tool call."""
    return LLMResult(
        tool_calls=[ToolCall(id=id, name=name, arguments=arguments or {})],
        finish_reason="tool_calls",
        usage=usage or Usage(),
    )


# ── the scripted, no-network provider ────────────────────────────────────────────


class ScriptedProvider(Provider):
    """A deterministic :class:`Provider` for evals and tests (never touches a network).

    Provide a static ``script`` (a queue of :class:`LLMResult`; the last entry repeats
    once the queue is exhausted) and/or a ``responder`` callable for dynamic behavior
    (e.g. "return a summary when asked to compact, otherwise reply"). The responder, if
    given, takes precedence.

    ``count_tokens`` returns ``tokens_per_message * len(messages)`` (+ one unit for a
    system prompt) — a deterministic estimate that grows with history, so a probe can
    force compaction by pairing a modest ``context_window`` with enough turns. The
    provider records ``calls``, the ``systems`` it saw, and ``summarize_calls`` (calls
    whose system prompt marks them as a compaction summarization) for assertions.
    """

    def __init__(
        self,
        script: Sequence[LLMResult] | None = None,
        *,
        responder: Responder | None = None,
        context_window: int = 8192,
        tokens_per_message: int = 50,
        supports_tools: bool = True,
    ) -> None:
        if script is None and responder is None:
            raise ValueError("ScriptedProvider needs a script or a responder")
        self._script = list(script) if script is not None else []
        self._responder = responder
        self._context_window = context_window
        self._tokens_per_message = tokens_per_message
        self._supports_tools = supports_tools
        # Observability for assertions.
        self.calls = 0
        self.systems: list[str | None] = []
        self.summarize_calls = 0

    async def acomplete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        index = self.calls
        self.systems.append(system)
        if system is not None and "compacting" in system.lower():
            self.summarize_calls += 1
        self.calls += 1
        if self._responder is not None:
            return self._responder(list(messages), system, index)
        clamped = min(index, len(self._script) - 1)
        return self._script[clamped]

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        total = self._tokens_per_message * len(messages)
        if system:
            total += self._tokens_per_message
        return total

    def capabilities(self) -> Capabilities:
        return Capabilities(
            supports_tools=self._supports_tools, context_window=self._context_window
        )


# ── case / result / report model ─────────────────────────────────────────────────


class EvalResult(BaseModel):
    """The outcome of one probe."""

    name: str
    passed: bool
    detail: str = ""
    error: str | None = None


@dataclass
class EvalCase:
    """A named behavioral probe.

    ``run`` is an async callable that asserts the expected behavior. It returns an
    optional human-readable detail string on success and raises on failure (any
    exception — ``AssertionError`` included — is caught by the runner and recorded as a
    failed :class:`EvalResult`, so one bad probe never aborts the suite).
    """

    name: str
    description: str
    run: Callable[[], Awaitable[str | None]]


class EvalReport(BaseModel):
    """The aggregate outcome of an eval run."""

    results: list[EvalResult] = Field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def ok(self) -> bool:
        """True only if every probe passed (the CI gate condition)."""
        return all(r.passed for r in self.results)


async def run_case(case: EvalCase) -> EvalResult:
    """Run one probe, turning any raised exception into a failed result."""
    try:
        detail = await case.run()
    except Exception as exc:  # noqa: BLE001 — a probe failure is data, not a crash
        return EvalResult(name=case.name, passed=False, error=f"{type(exc).__name__}: {exc}")
    return EvalResult(name=case.name, passed=True, detail=detail or "")


async def run_evals(cases: Sequence[EvalCase]) -> EvalReport:
    """Run probes sequentially (deterministic, no cross-talk) into a report."""
    results = [await run_case(case) for case in cases]
    return EvalReport(results=results)


def run_evals_sync(cases: Sequence[EvalCase]) -> EvalReport:
    """Synchronous wrapper around :func:`run_evals` (for the CLI)."""
    return asyncio.run(run_evals(cases))


# ── helpers for building agents under test ────────────────────────────────────────


def make_agent(
    provider: Provider,
    *,
    workspace_root: str,
    permission_mode: str = "allow",
    prompter: PermissionPrompter | None = None,
    enable_subagents: bool = False,
    enable_compaction: bool = False,
    extra_tools: Sequence[Tool] | None = None,
    max_iterations: int = 50,
) -> Agent:
    """Build a real :class:`~zakcode.Agent` wired to ``provider`` (no network).

    The agent uses an explicit :class:`~zakcode.permissions.PermissionPolicy` in
    ``permission_mode`` (default ``"allow"`` so tools run without prompts; a safety
    probe overrides to ``"ask"`` with no prompter to assert fail-closed). ``extra_tools``
    are registered into the live registry so a scripted provider can call them.

    ``max_iterations`` is an EXPLICIT bound, never inherited from the product default:
    the product default is 0 (unlimited — minds run for days), and a
    :class:`ScriptedProvider` repeats its last entry forever once the script is
    exhausted — an eval probe whose script ends on a tool call would spin unbounded.
    """
    from zakcode import Agent
    from zakcode.permissions import PermissionPolicy

    agent = Agent(
        provider=provider,
        permission_policy=PermissionPolicy(permission_mode, prompter=prompter),
        enable_subagents=enable_subagents,
        enable_compaction=enable_compaction,
        default_model="scripted/eval",
        workspace_root=workspace_root,
        max_iterations=max_iterations,
    )
    for tool in extra_tools or []:
        agent.registry.register(tool)
    return agent


@contextmanager
def hermetic_env() -> Iterator[None]:
    """Best-effort isolation for an eval run.

    Sets ``TZ=UTC`` and removes provider API keys from the environment for the duration,
    so a misconfigured probe can never silently reach a real provider (probes use only
    :class:`ScriptedProvider`). The previous environment is restored on exit.
    """
    keys = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TZ")
    saved = {k: os.environ.get(k) for k in keys}
    try:
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ["TZ"] = "UTC"
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
