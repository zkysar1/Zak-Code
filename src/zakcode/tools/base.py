"""Tool contract and registry.

A tool is **data plus a handler**: a declarative :class:`ToolSpec` (name, description,
JSON-schema parameters, required permission tier, concurrency class) kept separate from an
async :meth:`Tool.execute`. The agent loop discovers tools via :meth:`ToolRegistry.definitions`
(the schemas sent to the model) and invokes them via :meth:`ToolRegistry.execute` — it never
knows a tool's internals.

Design rules (see ``docs/ARCHITECTURE.md`` / ``docs/GUARDRAILS.md``):

* **Handlers never raise.** Failures are wrapped into an error :class:`ToolResult` so a bad
  tool call can never crash the loop — the model sees the error and can recover.
* **Idempotent where sensible.** A repeated call that finds the desired state already holds
  SHOULD return a benign success (a no-op), not an error — small models retry on timeouts or
  confusion, so "already done" must not read as failure (e.g. ``edit_file`` whose edit is
  already applied, or ``remember`` of an existing note). Genuine errors (wrong path, ambiguous
  match, differing content) still error. Inherently stateful tools (``bash`` / ``powershell``)
  are exempt — re-running an arbitrary command yields a real result, not a contract violation.
* **Structured I/O.** Input is a validated ``dict``; results carry optional structured
  ``data``. No lossy round-trips through strings.
* **Least privilege.** Every spec declares the narrowest :class:`~zakcode.config.PermissionTier`
  it needs and a :class:`ConcurrencyClass` describing how it may be parallelized.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import StrEnum
from fnmatch import fnmatchcase
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field, model_validator

from zakcode.artifacts import ArtifactRef
from zakcode.config import PermissionTier
from zakcode.tasks import TaskNetwork

logger = logging.getLogger("zakcode.tools")


def _match_skill_name(raw: str, skill_names: list[str]) -> str | None:
    """Canonical skill name matching ``raw`` (case-insensitive, leading slash
    stripped), or ``None``. Same normalization as the skill trigger-matcher
    (``skills/__init__.py`` ``_by_trigger``), so ``Start``/``/start``/``start`` all match.
    """
    lowered = raw.strip().lstrip("/").lower()
    for n in skill_names:
        if n.lower() == lowered:
            return n
    return None


if TYPE_CHECKING:
    # Only for the spawner's return annotation. Importing at runtime would be a
    # cycle (subagent.py imports this module); under `from __future__ import
    # annotations` the annotation is a string, so this type-only import suffices.
    from zakcode.agent.subagent import SubAgentResult


class ConcurrencyClass(StrEnum):
    """How a tool may be scheduled relative to others in the same turn.

    Used by the loop (from M1 on) to parallelize independent work safely:

    * ``READ_ONLY_SAFE`` — no side effects; may run concurrently with anything.
    * ``PATH_SCOPED`` — mutates paths; may run in parallel only with calls whose paths
      do not overlap (conflicting subtrees are serialized).
    * ``NEVER_PARALLEL`` — interactive/stateful; always runs sequentially.
    """

    READ_ONLY_SAFE = "read_only_safe"
    PATH_SCOPED = "path_scoped"
    NEVER_PARALLEL = "never_parallel"


class ToolSpec(BaseModel):
    """Declarative description of a tool (the part the model and the harness reason about)."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}},
        description="JSON Schema for the tool's arguments (an 'object' schema).",
    )
    required_permission: PermissionTier = PermissionTier.READ_ONLY
    concurrency: ConcurrencyClass = ConcurrencyClass.READ_ONLY_SAFE

    @model_validator(mode="after")
    def _check_concurrency_tier(self) -> ToolSpec:
        """Reject an *explicitly* ``READ_ONLY_SAFE`` tool that is not ``READ_ONLY`` tier.

        The scheduler parallelizes a batch only when every call is ``READ_ONLY_SAFE``,
        on the assumption that such tools have no side effects and never prompt (true
        only for ``READ_ONLY`` tier). Flagging the inconsistent combination at
        construction gives a tool author immediate feedback instead of a silent
        sequential fallback. Only an *explicit* ``concurrency=READ_ONLY_SAFE`` is
        checked: the field's default is ``READ_ONLY_SAFE``, so a writing tool that
        merely sets its tier and leaves concurrency defaulted is not rejected here
        (the loop's own tier guard keeps it off the parallel path at runtime).
        """
        if (
            "concurrency" in self.model_fields_set
            and self.concurrency is ConcurrencyClass.READ_ONLY_SAFE
            and self.required_permission is not PermissionTier.READ_ONLY
        ):
            raise ValueError(
                "a READ_ONLY_SAFE tool must be READ_ONLY tier "
                f"(tool {self.name!r} is {self.required_permission.name}); use PATH_SCOPED "
                "or NEVER_PARALLEL for a writing/dangerous tool"
            )
        return self

    def to_openai(self) -> dict[str, Any]:
        """Render this spec as an OpenAI-shaped function-tool definition.

        litellm translates this shape to whatever the target backend expects, so this is
        the single wire format the provider layer is handed.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@runtime_checkable
class SubAgentSpawner(Protocol):
    """Runs a named sub-agent on a prompt and returns its condensed result (M4).

    The loop injects a concrete spawner into :class:`ToolContext` when delegation
    is enabled, and the ``task`` tool calls it. Child loops are built **without** a
    spawner — that absence is what enforces one-level nesting (a sub-agent cannot
    itself delegate). ``runtime_checkable`` so pydantic can validate the field by
    structural ``isinstance``.
    """

    async def spawn(self, *, type_name: str, prompt: str) -> SubAgentResult: ...

    def available_types(self) -> list[str]:
        """Names of the sub-agent types this spawner can launch."""
        ...

    def default_type(self) -> str:
        """The sub-agent type used when a caller does not name one explicitly."""
        ...


@runtime_checkable
class Sampler(Protocol):
    """Produces a single raw model completion for a tool that needs to *deliberate* — i.e.
    make its own model calls rather than just touch the filesystem (the ``deep_think``
    best-of-N synthesis tool).

    The wirer (the ``Agent``) points it at the agent's strongest configured model — under
    zakpick the ``deep_code`` category, otherwise ``default_model`` — and records each call's
    usage so a deliberation's spend shows in ``/cost`` and counts against the turn budget.
    ``None`` on :class:`ToolContext` when no provider is wired (a bare/test loop), so a
    model-using tool degrades to a clean error instead of crashing. ``runtime_checkable`` so
    pydantic can validate the field structurally.
    """

    async def __call__(
        self, prompt: str, *, system: str | None = None, temperature: float = 0.0
    ) -> str: ...


class SkillLoad(BaseModel):
    """Outcome of resolving a skill for the model-facing ``use_skill`` tool.

    ``found`` is True iff ``name`` matched a discovered skill (so the tool reports "unknown
    skill" rather than crashing); ``body`` is the loaded, defanged L1 instructions on success;
    ``error`` is set iff a matched skill's file could not be read (vanished/unreadable since
    discovery). The body is carried back as the tool RESULT — never injected as a session
    message — so model-driven invocation can't corrupt mid-turn message ordering.
    """

    found: bool = False
    name: str = ""
    body: str | None = None
    error: str | None = None
    #: Set when a found skill is REFUSED by policy rather than failing to load — today the per-turn
    #: skill-invocation budget (a runaway-chain guard). Distinct from ``error`` (a file that could
    #: not be read) so the tool can report "budget exhausted" rather than "unreadable".
    denied_reason: str | None = None
    #: The loaded SKILL.md's path on success (ADR-0044). A skill is a DIRECTORY — its scripts
    #: and data sit beside the SKILL.md — and the model that never sees that directory later
    #: describes the skill from memory of its own writing ("it is a python file, not a skill").
    #: ``use_skill`` lists the siblings from this path so the answer is in the tool result.
    path: str | None = None


@runtime_checkable
class SkillResolver(Protocol):
    """Resolves a skill name to its body for the ``use_skill`` tool and fires the
    skill-selected signal (M7).

    The wirer (the ``Agent``) binds it to the session's skill registry; :meth:`load` reads the
    L1 body lazily, defangs it, and emits ``ON_SKILL_SELECTED`` (``source="tool"``) so a learning
    mind records model-driven ``(query -> skill)`` choices just like the CLI ``/<name>`` path.
    A sub-agent shares the PARENT's resolver (one registry + one per-turn budget), passing its own
    ``query`` so attribution stays per-caller. ``None`` on :class:`ToolContext` only when skills are
    disabled, so the tool degrades to a clean "skills not enabled" error rather than crashing.
    ``runtime_checkable`` so pydantic can validate the field structurally.
    """

    def names(self) -> list[str]:
        """Discovered skill names (for the tool's 'unknown skill — available: …' message)."""
        ...

    async def load(self, name: str, *, query: str = "", args: str = "") -> SkillLoad:
        """Load ``name``; ``query`` is the invoking turn's prompt, recorded as the
        ``ON_SKILL_SELECTED`` trigger (so a sub-agent attributes the skill to ITS task, not the
        parent's). A falsy ``query`` falls back to the resolver's bound (parent) agent session's
        recent user text — correct for the main agent; callers that need per-caller attribution
        (sub-agents) must pass a non-empty ``query`` (the loop stamps ``caller_query`` for this)."""
        ...

    # Optional (not part of the structural check, so a minimal test double still satisfies
    # the protocol): ``forget_loads() -> None`` — forget which bodies are "already loaded"
    # this turn (ADR-0080). The loop looks it up with getattr right after a compaction: the
    # dedup's premise — the body is still in context — no longer holds, so the next
    # ``use_skill`` must deliver the body, not a pointer. The Agent's resolver implements it.

    def body(self, name: str) -> str | None:
        """The skill's whole (defanged) body with none of the load ceremony — no budget, no
        selection signal, no dedup (ADR-0067). The loop reads it to seed a skeleton and to
        page sections from a load that delivered only page 1. ``None`` when ``name`` is
        unknown or unreadable."""
        ...


class ToolContext(BaseModel):
    """Ambient state handed to a tool at execution time.

    Carries the workspace root so file tools can scope and validate paths against it
    (see ``docs/GUARDRAILS.md`` §4) and, when delegation is enabled, a
    :class:`SubAgentSpawner` the ``task`` tool uses to launch sub-agents. ``spawner``
    is ``None`` for ordinary turns and for every sub-agent (one-level nesting).

    ``extra_workspace_roots`` (M-3 multi-root sandbox) extends the sandbox to
    additional trusted filesystem roots. When non-empty, file tools accept paths
    under any of ``[workspace_root] + extra_workspace_roots``. This enables
    cross-repo skill execution where a skill needs to read/write across the primary
    workspace and one or more external directories (e.g. a claude-mind skill
    accessing the mind repo, its world dir, and its meta dir).
    """

    model_config = {"arbitrary_types_allowed": True}

    workspace_root: Path
    extra_workspace_roots: list[Path] = Field(default_factory=list)
    spawner: SubAgentSpawner | None = None
    #: Extra environment for subprocess tools (bash/powershell) — e.g. ``HTTP(S)_PROXY`` pointing
    #: at the egress proxy when the network-egress sandbox is on. Empty for an ordinary turn.
    egress_env: dict[str, str] = Field(default_factory=dict)
    #: Environment-variable NAMES removed from the inherited environment before a
    #: subprocess tool spawns its child (the provider-key scrub — see RISKS / GUARDRAILS
    #: §6). Built by the loop from ``zakcode.secrets.provider_key_env_names``; empty when
    #: the operator opted out via ``subprocess_inherit_provider_keys=true``. Removal runs
    #: LAST, after ``egress_env`` is overlaid, so nothing can resurrect a scrubbed key.
    scrub_env: list[str] = Field(default_factory=list)
    #: The live hierarchical plan (:class:`~zakcode.tasks.TaskNetwork`) for this loop's
    #: session — the seam the ``update_plan`` tool rewrites and the loop then persists and
    #: re-injects. ``None`` for a bare/ungated loop that does not wire planning, so the
    #: tool degrades to a recoverable error rather than raising.
    task_network: TaskNetwork | None = None
    #: A :class:`Sampler` for tools that make their own model calls (``deep_think``). The
    #: ``Agent`` wires it to its strongest model and accounts the spend; ``None`` for a
    #: bare/test loop, so a model-using tool returns a clean error rather than crashing.
    sampler: Sampler | None = None
    #: A :class:`SkillResolver` the ``use_skill`` tool calls to load a skill's instructions by
    #: name (M7 model-facing invocation). The ``Agent`` wires it to the session's skill registry
    #: when ``enable_skills``; ``None`` otherwise, so the tool returns a clean "skills not enabled"
    #: error rather than crashing. (A sub-agent gets the PARENT's resolver — shared registry +
    #: budget — but its own ``caller_query`` below, so attribution stays correct.)
    skill_resolver: SkillResolver | None = None
    #: The session's :class:`~zakcode.rules.RuleRegistry`, which the ``read_rule`` tool reads to
    #: return ONE rule body by name (Vinheim Lever A chunk 2). It is the retrieval half of
    #: ``lean_rules``: ``render_index()`` puts every rule's name + summary in the prompt and the
    #: model fetches a body on demand instead of paying for all of them every turn. The ``Agent``
    #: wires it when ``enable_rules``; ``None`` otherwise, so the tool returns a clean
    #: "rules not enabled" error rather than crashing.
    rule_registry: Any | None = None
    #: The user text that triggered THIS loop's turn — passed to ``use_skill`` so the
    #: ``ON_SKILL_SELECTED`` signal records the *invoking* turn's prompt (the sub-agent's task, not
    #: the parent's originating turn). Each loop stamps its own; empty for a loop that builds a
    #: bare context, in which case the resolver falls back to its session's recent user text.
    caller_query: str = ""

    @property
    def all_workspace_roots(self) -> list[Path]:
        """All trusted roots: the primary workspace root followed by any extras."""
        return [self.workspace_root, *self.extra_workspace_roots]


class ToolResult(BaseModel):
    """The outcome of a tool invocation.

    ``output`` is the text the model sees; ``data`` optionally carries structured results
    losslessly alongside it. ``artifacts`` names files a client can download/preview without
    pushing binary content back through the model prompt.

    ``hint`` and ``fix`` are the optional *rails* a tool can hand the model: ``hint`` is a
    suggested next step on success (e.g. "saved -- reply and end"), ``fix`` is the concrete
    remedy on an error (e.g. "re-read the file; old_string must match exactly"). The agent
    loop renders whichever is set as a trailing ``Hint:`` / ``Fix:`` line in the model-facing
    text (and mirrors it into the result's structured data). Naming the next action is the
    single biggest help for a small model, which is otherwise weak at planning the next step
    and at recovering from errors.
    """

    output: str = ""
    is_error: bool = False
    data: dict[str, Any] | None = None
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    hint: str | None = None
    fix: str | None = None
    #: The output is INSTRUCTIONS, not data (a skill body, a rule): the loop's seam clamp
    #: (ADR-0023) never cuts it, because there is no "re-run narrower" for a procedure — a
    #: head-and-tail of a skill is not a shorter skill, it is a broken one (ADR-0065). An
    #: oversized verbatim result is the compactor's problem, not the clamp's.
    verbatim: bool = False

    @classmethod
    def ok(
        cls,
        output: str,
        *,
        data: dict[str, Any] | None = None,
        artifacts: list[ArtifactRef] | None = None,
        hint: str | None = None,
        verbatim: bool = False,
    ) -> ToolResult:
        """A successful result, optionally with a next-step ``hint``."""
        return cls(
            output=output,
            is_error=False,
            data=data,
            artifacts=artifacts or [],
            hint=hint,
            verbatim=verbatim,
        )

    @classmethod
    def error(
        cls,
        message: str,
        *,
        data: dict[str, Any] | None = None,
        artifacts: list[ArtifactRef] | None = None,
        fix: str | None = None,
    ) -> ToolResult:
        """An error result (still a value, never an exception), optionally with a ``fix``."""
        return cls(output=message, is_error=True, data=data, artifacts=artifacts or [], fix=fix)


class Tool(ABC):
    """Base class for a built-in tool: a :attr:`spec` plus an async :meth:`execute`."""

    #: Declarative metadata. Concrete tools set this as a class attribute.
    spec: ToolSpec

    @property
    def name(self) -> str:
        """The tool's registered name (from its spec)."""
        return self.spec.name

    @abstractmethod
    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """Run the tool. Implementations MUST NOT raise — wrap failures in
        :meth:`ToolResult.error`."""
        raise NotImplementedError


class ToolRegistry:
    """A name-keyed collection of tools with aliasing and one canonical dispatch path."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._aliases: dict[str, str] = {}
        # Canonical names currently EXPOSED to the model (their schemas go into
        # ``definitions()``). A tool can be registered inactive (lazy) so it is
        # dispatchable but its schema stays out of the prompt until activated — the
        # mechanism behind MCP's lazy tool discovery / tool budget (M5). Tools
        # registered ``active=True`` (the default) behave exactly as before.
        self._active: set[str] = set()
        # Per-task tool-exposure filter (self-remediation Step 4): operator-set glob
        # allow/deny over canonical names. Empty = no restriction (default). It narrows the
        # model-facing surface only — see :meth:`set_exposure_filter`.
        self._exposure_allow: list[str] = []
        self._exposure_deny: list[str] = []

    def register(
        self, tool: Tool, *, aliases: list[str] | None = None, active: bool = True
    ) -> None:
        """Add a tool to the registry, optionally under friendly aliases.

        ``active=False`` registers the tool as **dispatchable but not exposed**: it
        will not appear in :meth:`definitions` until :meth:`activate` is called. The
        default keeps every tool exposed, so existing callers are unaffected.
        """
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name!r}")
        # Symmetric collision guard: a new tool's canonical name must not equal an existing
        # alias that points elsewhere. ``_canonical`` resolves aliases FIRST, so such a tool
        # would register but be permanently unreachable by its own name (every call silently
        # dispatched to the alias target). Fail loud instead. (review: collision asymmetry)
        shadowing_alias = self._aliases.get(tool.name)
        if shadowing_alias is not None and shadowing_alias != tool.name:
            raise ValueError(
                f"tool name {tool.name!r} collides with an existing alias for {shadowing_alias!r}"
            )
        self._tools[tool.name] = tool
        for alias in aliases or []:
            # Collision guard (cheap insurance as the alias set grows and plugins/MCP can
            # register too): an alias must not shadow another tool's canonical name or an
            # existing alias that points elsewhere. A self-alias (== tool.name) is a harmless
            # no-op.
            if alias != tool.name and alias in self._tools:
                raise ValueError(
                    f"alias {alias!r} for {tool.name!r} collides with a registered tool name"
                )
            existing = self._aliases.get(alias)
            if existing is not None and existing != tool.name:
                raise ValueError(f"alias {alias!r} already maps to {existing!r}")
            self._aliases[alias] = tool.name
        if active:
            self._active.add(tool.name)

    def _canonical(self, name: str) -> str:
        return self._aliases.get(name, name)

    def get(self, name: str) -> Tool | None:
        """Look up a tool by name or alias (``None`` if unknown)."""
        return self._tools.get(self._canonical(name))

    def names(self) -> list[str]:
        """All registered canonical tool names, in registration order."""
        return list(self._tools)

    def active_names(self) -> list[str]:
        """Canonical names currently exposed to the model, in registration order."""
        return [name for name in self._tools if name in self._active]

    def is_active(self, name: str) -> bool:
        """Whether ``name`` (canonical or alias) is currently exposed."""
        return self._canonical(name) in self._active

    def activate(self, name: str) -> bool:
        """Expose a registered tool. Returns ``True`` if it exists, else ``False``."""
        canonical = self._canonical(name)
        if canonical not in self._tools:
            return False
        self._active.add(canonical)
        return True

    def deactivate(self, name: str) -> bool:
        """Hide a registered tool from the model. Returns ``True`` if it existed."""
        canonical = self._canonical(name)
        if canonical not in self._tools:
            return False
        self._active.discard(canonical)
        return True

    def set_exposure_filter(
        self, allow: list[str] | None = None, deny: list[str] | None = None
    ) -> None:
        """Restrict which tools are EXPOSED to the model (least-privilege; self-remediation Step 4).

        ``allow`` — glob patterns over canonical tool names; when non-empty, ONLY tools matching
        a pattern are advertised. ``deny`` — globs whose matches are NEVER advertised (deny wins
        over allow). Both default to no restriction. This narrows the model-facing surface (the
        schemas in :meth:`definitions` AND the system-prompt list) so a tool the task does not
        need is never offered — the most effective single prompt-injection defense (AgentDojo):
        a tool the model can neither see nor (with the loop's execution guard, via
        :meth:`exposure_allows`) invoke cannot be hijacked by injected content. Exposure-only:
        it never loosens the permission gate, and trusted internal callers of :meth:`execute`
        are unaffected. Patterns are case-sensitive globs (``mcp__*``, ``web_*``) matched against
        canonical names, so a pattern covers a tool's aliases too. Set before a task runs.
        """
        self._exposure_allow = [p for p in (allow or []) if isinstance(p, str) and p.strip()]
        self._exposure_deny = [p for p in (deny or []) if isinstance(p, str) and p.strip()]

    def exposure_allows(self, name: str) -> bool:
        """Whether ``name`` (canonical/alias) passes the exposure filter (allow/deny globs).

        Independent of active state — the loop's execution seam uses this to reject a model call
        to a filtered-out tool even if the model named it from prior knowledge. No filter set
        (the default) → always True.
        """
        canonical = self._canonical(name)
        if any(fnmatchcase(canonical, pat) for pat in self._exposure_deny):
            return False
        if not self._exposure_allow:
            return True
        return any(fnmatchcase(canonical, pat) for pat in self._exposure_allow)

    def exposed_names(self) -> list[str]:
        """Canonical names actually offered to the model now: active AND passing the filter."""
        return [name for name in self.active_names() if self.exposure_allows(name)]

    def definitions(self, allowed: list[str] | None = None) -> list[dict[str, Any]]:
        """OpenAI-shaped tool definitions to send to the model.

        ``allowed`` (canonical names or aliases) restricts the exposed set; ``None``
        exposes the currently **active** tools (every tool, unless some were
        registered inactive for lazy discovery — see :meth:`register`). The
        operator's exposure filter (:meth:`set_exposure_filter`) is applied on top of
        either, so a filtered-out tool is never advertised regardless of ``allowed``/active.
        """
        if allowed is None:
            chosen = [
                t
                for name, t in self._tools.items()
                if name in self._active and self.exposure_allows(name)
            ]
        else:
            wanted = {self._canonical(a) for a in allowed}
            chosen = [
                t
                for name, t in self._tools.items()
                if name in wanted and self.exposure_allows(name)
            ]
        return [t.spec.to_openai() for t in chosen]

    async def execute(self, name: str, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """Dispatch a tool call by name/alias.

        An unknown tool yields an error :class:`ToolResult` (never an exception), and any
        exception escaping a handler is caught and wrapped — the loop must always get a
        value back.
        """
        tool = self.get(name)
        if tool is None:
            # Skill-heavy deployments (Claude-Mind, etc.) list dozens of skills in the
            # system prompt. Open-weights models frequently emit a skill NAME as a bare
            # tool call instead of routing through `use_skill`. When the unknown name
            # matches a discovered skill, return the correct invocation path instead of a
            # dead-end error the model cannot recover from — mirrors the fix-hint the
            # use_skill tool already returns for an unknown skill name.
            try:
                resolver = getattr(ctx, "skill_resolver", None)
                if resolver is not None:
                    match = _match_skill_name(name, resolver.names())
                    if match is not None:
                        return ToolResult.error(
                            f"{name!r} is a skill, not a tool.",
                            fix=f'Run it with use_skill(name="{match}").',
                        )
            except Exception:  # noqa: BLE001 — the skill hint is best-effort; a broken
                # resolver must never turn a clean unknown-tool error into a crash
                # (execute() must always return a value, never raise).
                logger.debug("skill-name hint lookup failed for %r", name, exc_info=True)
            return ToolResult.error(f"unknown tool: {name!r}")
        try:
            return await tool.execute(args, ctx)
        except Exception as exc:  # noqa: BLE001 — handlers must never crash the loop
            # The model sees the wrapped error and adapts; the OPERATOR gets the real
            # traceback here — previously this was the codebase's largest silent
            # swallow (audit P1-5: instrument the bare except handlers).
            logger.exception("tool %r raised (wrapped into an error ToolResult)", name)
            return ToolResult.error(f"{type(exc).__name__}: {exc}")

    def subset(self, names: list[str]) -> ToolRegistry:
        """Return a new registry exposing only ``names`` (canonical names or aliases).

        The returned registry shares the *same* tool instances (tools are stateless
        handlers) but exposes a restricted set — the mechanism behind a sub-agent's
        "filtered tool access" (M4). Unknown names are silently skipped, so a caller
        can pass an over-broad allow-list without error; an alias resolves to its
        canonical tool, and that tool's original aliases are preserved in the subset.
        Order follows this registry's registration order, not the ``names`` order.
        """
        wanted = {self._canonical(n) for n in names}
        sub = ToolRegistry()
        for canonical, tool in self._tools.items():
            if canonical in wanted:
                aliases = [a for a, target in self._aliases.items() if target == canonical]
                sub.register(tool, aliases=aliases)
        # Carry the operator's exposure filter into the child: a tool the operator denied for the
        # session must stay hidden from a delegated sub-agent too (a sub-agent reads untrusted
        # content as well). The child's own allowed_tools (the ``names`` here) and this filter
        # compose — both must permit a tool for it to be offered.
        sub.set_exposure_filter(self._exposure_allow, self._exposure_deny)
        return sub


__all__ = [
    "ConcurrencyClass",
    "ToolSpec",
    "ToolContext",
    "SubAgentSpawner",
    "Sampler",
    "SkillLoad",
    "SkillResolver",
    "ToolResult",
    "Tool",
    "ToolRegistry",
]
