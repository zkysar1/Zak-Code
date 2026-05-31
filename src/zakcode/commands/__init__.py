"""Slash-command registry (M6) — a small, UI-agnostic command table.

A *command* is a name, a one-line description, and a handler that takes the
command's argument string and returns a :class:`CommandResult` (text to show plus
an error flag). The core never prints — a client (the CLI) renders the result — so
this table stays free of terminal concerns and is reused across surfaces.

Built-in interactive commands (``/help``, ``/model``, …) currently live in the CLI
REPL; this registry is the seam **plugins** register their own commands through
(`ctx.register_command(...)`), and the CLI consults it for any slash command it does
not handle itself. Handlers must not raise — :meth:`CommandRegistry.run` wraps a
failure into an error result so a bad command can never crash the client.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel

#: A command handler: given the argument string (everything after the command name),
#: return a :class:`CommandResult`. Handlers should be synchronous and quick.
CommandHandler = Callable[[str], "CommandResult"]


class CommandResult(BaseModel):
    """The outcome of running a command — text for the client to display."""

    output: str = ""
    is_error: bool = False

    @classmethod
    def ok(cls, output: str = "") -> CommandResult:
        return cls(output=output, is_error=False)

    @classmethod
    def error(cls, message: str) -> CommandResult:
        return cls(output=message, is_error=True)


class Command(BaseModel):
    """A registered command: a name (without the leading slash), help, and handler."""

    model_config = {"arbitrary_types_allowed": True}

    name: str
    description: str = ""
    handler: CommandHandler
    #: Where the command came from (e.g. a plugin name) — for provenance in ``/help``.
    source: str = ""


class CommandRegistry:
    """A name-keyed table of :class:`Command` s with one safe dispatch path."""

    def __init__(self) -> None:
        self._commands: dict[str, Command] = {}

    @staticmethod
    def _normalize(name: str) -> str:
        """Canonical key: lowercase, no leading slash."""
        return name.lstrip("/").lower()

    def register(
        self,
        name: str,
        handler: CommandHandler,
        *,
        description: str = "",
        source: str = "",
        replace: bool = False,
    ) -> None:
        """Register a command. Refuses to clobber an existing name unless ``replace``."""
        key = self._normalize(name)
        if not key:
            raise ValueError("command name must be non-empty")
        if key in self._commands and not replace:
            raise ValueError(f"command already registered: {key!r}")
        self._commands[key] = Command(
            name=key, description=description, handler=handler, source=source
        )

    def get(self, name: str) -> Command | None:
        """Look up a command by name (with or without the leading slash)."""
        return self._commands.get(self._normalize(name))

    def has(self, name: str) -> bool:
        return self._normalize(name) in self._commands

    def names(self) -> list[str]:
        """All registered command names, in registration order."""
        return list(self._commands)

    def definitions(self) -> list[Command]:
        """All registered commands (for a ``/help`` listing)."""
        return list(self._commands.values())

    def run(self, name: str, arg: str = "") -> CommandResult | None:
        """Dispatch ``name`` with its argument string.

        Returns the command's :class:`CommandResult`, or ``None`` if no such command
        is registered (so the caller can fall through to its own handling). A handler
        that raises is caught and returned as an error result — never propagated.
        """
        command = self.get(name)
        if command is None:
            return None
        try:
            return command.handler(arg)
        except Exception as exc:  # noqa: BLE001 — a command must never crash the client
            return CommandResult.error(f"command {name!r} failed: {type(exc).__name__}: {exc}")


__all__ = [
    "CommandHandler",
    "CommandResult",
    "Command",
    "CommandRegistry",
]
