"""The Zak Code terminal client (a thin client of the core engine).

The CLI owns presentation only: it builds one :class:`~zakcode.Agent`, reads a
line, hands it to ``agent.run_turn``, and renders the returned
:class:`~zakcode.agent.loop.TurnResult`. All agent/tool logic lives in the core
engine — this module never decides what to do, only how to show it.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from typing import TYPE_CHECKING, Any

import typer
from rich.console import Console
from rich.table import Table

from zakcode.cli.render import StreamRenderer
from zakcode.config import Settings, load_settings
from zakcode.providers.base import ProviderError
from zakcode.version import __version__

if TYPE_CHECKING:
    from zakcode import Agent

app = typer.Typer(
    name="zakcode",
    help="Zak Code — a clean-room, vendor-agnostic, API-first agentic coding tool.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

# Provider API keys we report the *presence* of (never the value).
_PROVIDER_KEY_ENV = ["OPENAI_API_KEY"]


def _provider_key_status() -> dict[str, bool]:
    """Map each known provider key env-var to whether it is set (no values read out)."""
    return {name: bool(os.getenv(name)) for name in _PROVIDER_KEY_ENV}


def build_info_lines(settings: Settings) -> list[tuple[str, str]]:
    """Build the (label, value) rows shown by ``info``.

    Secret-safe: provider keys are reported as ``set`` / ``not set`` only.
    """
    rows: list[tuple[str, str]] = [
        ("Model", settings.default_model),
        ("Provider", settings.provider),
        ("API base", settings.api_base or "(default for provider)"),
        ("Fallback model", settings.fallback_model or "(none)"),
        ("Temperature", str(settings.temperature)),
        ("Ollama base URL", settings.ollama_base_url),
        ("Permission mode", settings.permission_mode),
        ("Max iterations", str(settings.max_iterations)),
        ("Workspace root", str(settings.workspace_root)),
    ]
    for name, present in _provider_key_status().items():
        rows.append((name, "set" if present else "not set"))
    return rows


@app.command()
def version() -> None:
    """Print the Zak Code version."""
    typer.echo(f"zakcode {__version__}")


@app.command()
def info() -> None:
    """Show resolved configuration and detected providers (never prints secrets)."""
    settings = load_settings()
    table = Table(title=f"Zak Code {__version__}", show_header=False)
    for label, value in build_info_lines(settings):
        table.add_row(label, value)
    console.print(table)
    console.print("[dim]Start the interactive agent with [bold]zakcode chat[/bold].[/dim]")


# ── chat: interactive REPL (thin client) ──────────────────────────────────────

_CHAT_HELP = """\
[bold]Slash commands[/bold]
  /help          show this help
  /model         show the active model
  /cost          show cumulative token usage and cost this session
  /clear         start a fresh session (clears the transcript)
  /exit, /quit   leave the chat
Anything else is sent to the agent as a turn.\
"""


def _print_banner(console: Console, agent: Agent) -> None:
    """Print the one-shot session banner (model, provider, workspace, perms)."""
    settings = agent.settings
    console.print(f"[bold]Zak Code[/bold] {__version__} — interactive chat")
    console.print(f"[dim]model[/dim]      {settings.default_model}")
    console.print(f"[dim]provider[/dim]   {settings.provider}")
    console.print(f"[dim]workspace[/dim]  {settings.workspace_root}")
    console.print(f"[dim]perms[/dim]      {settings.permission_mode}")
    console.print(f"[dim]session[/dim]    {agent.session.id}")
    console.print(
        "[dim]Tools: edit_file (+ read/write/glob/grep/bash). Ctrl-C interrupts a reply.[/dim]"
    )
    console.print("[dim]Type /help for commands, /exit to quit.[/dim]\n")


async def _drive_stream(agent: Agent, line: str, renderer: StreamRenderer) -> None:
    """Drive one streamed turn through the renderer so tokens appear live.

    The agent's :meth:`astream_turn` yields :class:`~zakcode.events.AgentEvent`s;
    the renderer writes them to the console incrementally (text deltas, tool
    lines, footer). Kept tiny and rendering-free so the core stays UI-agnostic.
    """
    await renderer.render(agent.astream_turn(line))


def _run_streamed_turn(console: Console, agent: Agent, line: str, renderer: StreamRenderer) -> bool:
    """Run a streamed turn on a private event loop; cancellable via Ctrl-C.

    Returns ``True`` if the turn completed, ``False`` if the user interrupted it.
    On interrupt the in-flight task is cancelled cleanly (so the session is left
    consistent — the loop persists at message boundaries), a dim ``interrupted``
    notice is printed, and control returns to the REPL prompt without exiting.
    """
    loop = asyncio.new_event_loop()
    try:
        task = loop.create_task(_drive_stream(agent, line, renderer))
        try:
            loop.run_until_complete(task)
            return True
        except KeyboardInterrupt:
            task.cancel()
            # Pump the loop so the task observes the cancellation and unwinds
            # (its CancelledError handler persists state and re-raises).
            with contextlib.suppress(asyncio.CancelledError, KeyboardInterrupt):
                loop.run_until_complete(task)
            console.print("\n[dim]interrupted[/dim]")
            return False
    finally:
        loop.close()
        asyncio.set_event_loop(None)


@app.command()
def chat(
    model: str = typer.Option(None, "--model", "-m", help="Override the model id."),
    provider: str = typer.Option(  # noqa: ARG001 — derived from the model; reserved for clarity
        None, "--provider", help="Hint the provider family (informational)."
    ),
    session: str = typer.Option(  # noqa: ARG001 — resume hook; sessions wired via the store later
        None, "--session", "-s", help="Resume a saved session by id."
    ),
    workspace: str = typer.Option(
        None, "--workspace", "-w", help="Workspace root for tools and the session."
    ),
) -> None:
    """Start an interactive agent session (a thin REPL over the core engine)."""
    from zakcode import Agent

    overrides: dict[str, Any] = {}
    if model:
        overrides["default_model"] = model
    if workspace:
        overrides["workspace_root"] = workspace

    agent = Agent(**overrides)
    _print_banner(console, agent)

    while True:
        try:
            line = console.input("[bold cyan]›[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/dim]")
            break

        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("/"):
            command = stripped.split(maxsplit=1)[0].lower()
            if command in ("/exit", "/quit"):
                console.print("[dim]Bye.[/dim]")
                break
            if command == "/help":
                console.print(_CHAT_HELP)
                continue
            if command == "/model":
                console.print(agent.settings.default_model)
                continue
            if command == "/clear":
                agent = Agent(**overrides)
                console.print("[dim]Started a fresh session.[/dim]")
                continue
            if command == "/cost":
                usage = agent.session.cumulative_usage()
                console.print(
                    f"[dim]prompt={usage.prompt_tokens} "
                    f"completion={usage.completion_tokens} "
                    f"total={usage.total_tokens} "
                    f"cost=${usage.cost_usd:.4f}[/dim]"
                )
                continue
            console.print(f"[dim]{command} is not yet supported.[/dim]")
            continue

        # Stream the model response token by token through the renderer. A fresh
        # renderer per turn keeps the text/usage buffers from leaking across turns.
        renderer = StreamRenderer(console=console)
        try:
            _run_streamed_turn(console, agent, stripped, renderer)
        except ProviderError as exc:
            console.print(f"[red]Provider error:[/red] {exc}")
            continue


def main() -> None:
    """Console-script entry point (see ``[project.scripts]`` in pyproject.toml)."""
    app()


if __name__ == "__main__":
    main()
