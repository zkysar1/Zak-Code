"""Tool contract and registry.

A tool is **data plus a handler**: a declarative :class:`ToolSpec` (name, description,
JSON-schema parameters, required permission tier, concurrency class) kept separate from an
async :meth:`Tool.execute`. The agent loop discovers tools via :meth:`ToolRegistry.definitions`
(the schemas sent to the model) and invokes them via :meth:`ToolRegistry.execute` — it never
knows a tool's internals.

Design rules (see ``docs/ARCHITECTURE.md`` / ``docs/GUARDRAILS.md``):

* **Handlers never raise.** Failures are wrapped into an error :class:`ToolResult` so a bad
  tool call can never crash the loop — the model sees the error and can recover.
* **Structured I/O.** Input is a validated ``dict``; results carry optional structured
  ``data``. No lossy round-trips through strings.
* **Least privilege.** Every spec declares the narrowest :class:`~zakcode.config.PermissionTier`
  it needs and a :class:`ConcurrencyClass` describing how it may be parallelized.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from zakcode.config import PermissionTier

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


class ToolContext(BaseModel):
    """Ambient state handed to a tool at execution time.

    Carries the workspace root so file tools can scope and validate paths against it
    (see ``docs/GUARDRAILS.md`` §4) and, when delegation is enabled, a
    :class:`SubAgentSpawner` the ``task`` tool uses to launch sub-agents. ``spawner``
    is ``None`` for ordinary turns and for every sub-agent (one-level nesting).
    """

    model_config = {"arbitrary_types_allowed": True}

    workspace_root: Path
    spawner: SubAgentSpawner | None = None


class ToolResult(BaseModel):
    """The outcome of a tool invocation.

    ``output`` is the text the model sees; ``data`` optionally carries structured results
    losslessly alongside it.
    """

    output: str = ""
    is_error: bool = False
    data: dict[str, Any] | None = None

    @classmethod
    def ok(cls, output: str, *, data: dict[str, Any] | None = None) -> ToolResult:
        """A successful result."""
        return cls(output=output, is_error=False, data=data)

    @classmethod
    def error(cls, message: str, *, data: dict[str, Any] | None = None) -> ToolResult:
        """An error result (still a value, never an exception)."""
        return cls(output=message, is_error=True, data=data)


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

    def register(self, tool: Tool, *, aliases: list[str] | None = None) -> None:
        """Add a tool to the registry, optionally under friendly aliases."""
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name!r}")
        self._tools[tool.name] = tool
        for alias in aliases or []:
            self._aliases[alias] = tool.name

    def _canonical(self, name: str) -> str:
        return self._aliases.get(name, name)

    def get(self, name: str) -> Tool | None:
        """Look up a tool by name or alias (``None`` if unknown)."""
        return self._tools.get(self._canonical(name))

    def names(self) -> list[str]:
        """All registered canonical tool names, in registration order."""
        return list(self._tools)

    def definitions(self, allowed: list[str] | None = None) -> list[dict[str, Any]]:
        """OpenAI-shaped tool definitions to send to the model.

        ``allowed`` (canonical names or aliases) optionally restricts the exposed set;
        ``None`` exposes everything registered.
        """
        if allowed is None:
            chosen = list(self._tools.values())
        else:
            wanted = {self._canonical(a) for a in allowed}
            chosen = [t for name, t in self._tools.items() if name in wanted]
        return [t.spec.to_openai() for t in chosen]

    async def execute(self, name: str, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """Dispatch a tool call by name/alias.

        An unknown tool yields an error :class:`ToolResult` (never an exception), and any
        exception escaping a handler is caught and wrapped — the loop must always get a
        value back.
        """
        tool = self.get(name)
        if tool is None:
            return ToolResult.error(f"unknown tool: {name!r}")
        try:
            return await tool.execute(args, ctx)
        except Exception as exc:  # noqa: BLE001 — handlers must never crash the loop
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
        return sub


__all__ = [
    "ConcurrencyClass",
    "ToolSpec",
    "ToolContext",
    "SubAgentSpawner",
    "ToolResult",
    "Tool",
    "ToolRegistry",
]
