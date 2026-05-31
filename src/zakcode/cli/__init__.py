"""The Zak Code terminal client (a thin client of the core engine).

The CLI owns presentation only: it builds one :class:`~zakcode.Agent`, reads a
line, hands it to ``agent.run_turn``, and renders the returned
:class:`~zakcode.agent.loop.TurnResult`. All agent/tool logic lives in the core
engine — this module never decides what to do, only how to show it.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import os
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

import typer
from rich.console import Console
from rich.table import Table

from zakcode.cli.render import StreamRenderer
from zakcode.config import Settings, load_settings
from zakcode.permissions import PermissionOutcome, PermissionRequest
from zakcode.providers.base import ProviderError
from zakcode.version import __version__

if TYPE_CHECKING:
    from zakcode import Agent
    from zakcode.events import AgentEvent

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
  /permissions   show the permission mode and session grants
  /hooks         list configured lifecycle hooks
  /cost          show cumulative token usage and cost this session
  /clear         start a fresh session (clears the transcript)
  /exit, /quit   leave the chat
Anything else is sent to the agent as a turn.\
"""


def _abbrev(value: object, *, limit: int = 80) -> str:
    """One-line, length-capped rendering of an argument value for a prompt."""
    text = str(value).replace("\n", " ").replace("\r", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


class ConsolePermissionPrompter:
    """Asks the operator to confirm an escalated tool call at the terminal.

    Implements the :class:`~zakcode.permissions.PermissionPrompter` protocol. It
    shows the tool, the *exact* arguments (so the operator approves the real action,
    not a summary — see ``docs/GUARDRAILS.md`` §3), and the reason it was escalated,
    then offers allow-once / allow-session / deny. A non-interactive or
    unrecognized answer defaults to **deny** (fail toward safe).
    """

    def __init__(self, console: Console) -> None:
        self.console = console

    async def confirm(self, request: PermissionRequest) -> PermissionOutcome:
        self.console.print()
        self.console.print(
            f"[yellow]⚠ permission required[/yellow] for "
            f"[bold]{request.tool_name}[/bold] [dim]({request.tier.name})[/dim]"
        )
        if request.reason:
            self.console.print(f"  [dim]{request.reason}[/dim]")
        for key, val in request.arguments.items():
            self.console.print(f"  [dim]{key}=[/dim]{_abbrev(val)}")
        self.console.print("  [dim]allow once [y] · allow for session [a] · deny [n][/dim]")
        try:
            answer = self.console.input("  [bold cyan]permit?[/bold cyan] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            self.console.print("  [dim]denied[/dim]")
            return PermissionOutcome.DENY_ONCE

        if answer in ("y", "yes"):
            return PermissionOutcome.ALLOW_ONCE
        if answer in ("a", "always", "session"):
            return PermissionOutcome.ALLOW_SESSION
        return PermissionOutcome.DENY_ONCE


def _render_permissions(console: Console, agent: Agent) -> None:
    """Render the active permission mode and any session grants (the /permissions cmd)."""
    policy = agent.permission_policy
    console.print(f"[dim]permission mode[/dim]  {policy.mode.value}")
    allow = sorted(policy._session_allow)
    deny = sorted(policy._session_deny)
    console.print(f"[dim]session allow[/dim]    {len(allow)} grant(s)")
    for key in allow:
        console.print(f"    [green]+[/green] {key}")
    console.print(f"[dim]session deny[/dim]     {len(deny)} block(s)")
    for key in deny:
        console.print(f"    [red]-[/red] {key}")


def _render_hooks(console: Console, agent: Agent) -> None:
    """List configured lifecycle hooks (the /hooks cmd)."""
    manager = agent.hook_manager
    shell = manager.shell_hooks
    in_proc = sum(len(v) for v in manager.in_process.values())
    if not shell and not in_proc:
        console.print("[dim]no hooks configured.[/dim]")
        return
    for spec in shell:
        console.print(f"[dim]{spec.event.value}[/dim] [{spec.matcher}] -> {' '.join(spec.command)}")
    if in_proc:
        console.print(f"[dim]{in_proc} in-process hook(s) registered.[/dim]")


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


#: A factory that produces one turn's event stream. The CLI is agnostic to where
#: the stream comes from — a local in-process ``Agent`` or a remote ``ServerClient``
#: — so the same cancellable runner and renderer drive both.
StreamFactory = Callable[[], "AsyncIterator[AgentEvent]"]


async def _drive_stream(make_stream: StreamFactory, renderer: StreamRenderer) -> None:
    """Render one streamed turn live.

    ``make_stream()`` yields :class:`~zakcode.events.AgentEvent`s (from a local
    agent or a remote server); the renderer writes them to the console
    incrementally. The stream is created here, inside the task's event loop, so a
    loop-bound transport (e.g. httpx) is constructed on the right loop.
    """
    await renderer.render(make_stream())


def _run_streamed_turn(
    console: Console, make_stream: StreamFactory, renderer: StreamRenderer
) -> bool:
    """Run a streamed turn on a private event loop; cancellable via Ctrl-C.

    Returns ``True`` if the turn completed, ``False`` if the user interrupted it.
    On interrupt the in-flight task is cancelled cleanly (so the session is left
    consistent — the loop persists at message boundaries), a dim ``interrupted``
    notice is printed, and control returns to the REPL prompt without exiting.
    """
    loop = asyncio.new_event_loop()
    try:
        task = loop.create_task(_drive_stream(make_stream, renderer))
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


async def _server_turn_stream(
    base_url: str, message: str, session_id: str | None
) -> AsyncIterator[AgentEvent]:
    """Stream one turn from a remote server, creating + closing a client per turn.

    A fresh :class:`~zakcode.server.client.ServerClient` (and its httpx client) is
    built inside the consuming event loop and closed when the turn ends, which
    keeps the loop-bound httpx client valid across the REPL's per-turn loops.
    """
    from zakcode.server.client import ServerClient

    client = ServerClient(base_url)
    try:
        async for event in client.astream_turn(message, session_id):
            yield event
    finally:
        await client.aclose()


def _create_remote_session(base_url: str) -> str:
    """Create a session on the remote server and return its id (one-shot)."""

    async def _go() -> str:
        from zakcode.server.client import ServerClient

        client = ServerClient(base_url)
        try:
            return await client.create_session()
        finally:
            await client.aclose()

    return asyncio.run(_go())


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
    server: str = typer.Option(
        None,
        "--server",
        help="Drive a remote zakcode server (e.g. http://127.0.0.1:8000) instead of "
        "running the engine in-process. Proves the client/server boundary.",
    ),
) -> None:
    """Start an interactive agent session.

    By default the engine runs **in-process**. With ``--server <url>`` the CLI is a
    thin client of a remote server, streaming the same ``AgentEvent``s over SSE — the
    same renderer displays either. (Server mode is headless: the server runs turns
    without an interactive permission prompter, so ``ask`` mode fails closed there;
    use the WebSocket channel for interactive approval.)
    """
    if server:
        _run_server_chat(server, model)
        return

    from zakcode import Agent

    overrides: dict[str, Any] = {}
    if model:
        overrides["default_model"] = model
    if workspace:
        overrides["workspace_root"] = workspace

    # A console prompter lets the in-core permission gate escalate to the operator,
    # so 'ask' mode is usable interactively (rather than failing closed). The gate
    # itself still lives in the core; the CLI only renders the prompt.
    prompter = ConsolePermissionPrompter(console)
    agent = Agent(prompter=prompter, **overrides)
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
            if command == "/permissions":
                _render_permissions(console, agent)
                continue
            if command == "/hooks":
                _render_hooks(console, agent)
                continue
            if command == "/clear":
                agent = Agent(prompter=prompter, **overrides)
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
            _run_streamed_turn(console, functools.partial(agent.astream_turn, stripped), renderer)
        except ProviderError as exc:
            console.print(f"[red]Provider error:[/red] {exc}")
            continue


def _run_server_chat(base_url: str, model: str | None) -> None:
    """REPL that drives a remote server over SSE (the ``chat --server`` path)."""
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        console.print(
            "[red]The server extra is not installed.[/red] "
            "Install it with: [bold]pip install 'zakcode[server]'[/bold]"
        )
        raise typer.Exit(code=1) from exc

    try:
        session_id = _create_remote_session(base_url)
    except (httpx.HTTPError, OSError) as exc:
        console.print(f"[red]Could not reach server at {base_url}:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[bold]Zak Code[/bold] {__version__} — connected to {base_url}")
    console.print(f"[dim]session[/dim]  {session_id}")
    if model:
        console.print(f"[dim]model[/dim]    {model} [dim](per-request override)[/dim]")
    console.print("[dim]Type /exit to quit. (Server mode: turns run headless.)[/dim]\n")

    while True:
        try:
            line = console.input("[bold cyan]›[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/dim]")
            break
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower() in ("/exit", "/quit"):
            console.print("[dim]Bye.[/dim]")
            break

        renderer = StreamRenderer(console=console)
        try:
            _run_streamed_turn(
                console,
                functools.partial(_server_turn_stream, base_url, stripped, session_id),
                renderer,
            )
        except httpx.HTTPError as exc:
            console.print(f"[red]Server error:[/red] {exc}")
            continue


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address."),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port."),
) -> None:
    """Run the Zak Code HTTP API server (FastAPI over the same core).

    Exposes REST + SSE + a WebSocket channel — see ``docs/ARCHITECTURE.md``. Requires
    the ``server`` extra (``pip install 'zakcode[server]'``).
    """
    try:
        import uvicorn

        from zakcode.server.app import create_app
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        console.print(
            "[red]The server extra is not installed.[/red] "
            "Install it with: [bold]pip install 'zakcode[server]'[/bold]"
        )
        raise typer.Exit(code=1) from exc

    console.print(f"[bold]Zak Code[/bold] {__version__} — serving on http://{host}:{port}")
    uvicorn.run(create_app(), host=host, port=port)


def main() -> None:
    """Console-script entry point (see ``[project.scripts]`` in pyproject.toml)."""
    app()


if __name__ == "__main__":
    main()
