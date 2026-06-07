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
import ipaddress
import os
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import TYPE_CHECKING, Any

import typer
from rich.console import Console, Group, RenderableType
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from zakcode.cli._glyphs import enable_utf8, resolve_glyphs
from zakcode.cli._layout import (
    kv_table,
    margin,
    notice_error,
    notice_info,
    notice_warn,
    read_prompt,
)
from zakcode.cli._theme import ZAK_THEME
from zakcode.cli.render import StreamRenderer
from zakcode.config import Settings, load_settings
from zakcode.permissions import PermissionOutcome, PermissionRequest
from zakcode.providers.base import ProviderError
from zakcode.secrets import strip_url_credentials
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
# Upgrade the console to UTF-8 where possible, then build it with the Zak theme and
# highlight=False (so rich never auto-colors our metadata). GLYPHS resolves to ASCII
# fallbacks on a cp1252 console.
enable_utf8()
console = Console(theme=ZAK_THEME, highlight=False)
GLYPHS = resolve_glyphs(console)

# Provider / service API keys we report the *presence* of (never the value).
_PROVIDER_KEY_ENV = ["OPENAI_API_KEY", "TAVILY_API_KEY"]


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
        ("API base", strip_url_credentials(settings.api_base) or "(default for provider)"),
        ("Fallback model", settings.fallback_model or "(none)"),
        ("Temperature", str(settings.temperature)),
        ("Ollama base URL", settings.ollama_base_url),
        ("Permission mode", settings.permission_mode),
        ("Max iterations", str(settings.max_iterations)),
        ("Workspace root", str(settings.workspace_root)),
        ("Search backend", settings.search_backend),
        (
            "web_fetch egress",
            (
                ", ".join(settings.web_allowed_domains)
                if settings.web_allowed_domains
                else "any public host"
            )
            + (" (confirm per call)" if settings.web_fetch_confirm else ""),
        ),
    ]
    if settings.search_backend == "searxng":
        rows.append(("SearXNG URL", strip_url_credentials(settings.searxng_url) or "(not set)"))
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


@app.command(name="eval")
def eval_(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show each probe's detail/error line."
    ),
) -> None:
    """Run the behavioral eval suite against the agent (offline, scripted providers).

    Drives the real agent loop with deterministic no-network providers and checks the
    invariants the project must never regress (completion detection, safety rejection,
    plan-mode read-only, doom-loop halting, partial-failure recovery, long-horizon
    compaction). Exits non-zero if any probe fails, so it can gate CI.
    """
    import tempfile

    from zakcode.evals import hermetic_env, run_evals_sync
    from zakcode.evals.probes import build_default_suite

    with tempfile.TemporaryDirectory(prefix="zakcode-eval-") as workspace, hermetic_env():
        report = run_evals_sync(build_default_suite(workspace))

    table = Table(title="Zak Code - behavioral evals", show_header=True)
    table.add_column("probe")
    table.add_column("result")
    if verbose:
        table.add_column("detail")
    for r in report.results:
        mark = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
        row = [r.name, mark]
        if verbose:
            row.append(r.detail if r.passed else f"[red]{r.error}[/red]")
        table.add_row(*row)
    console.print(table)
    # ASCII-only summary line: the Windows console default (cp1252) cannot encode
    # marks like U+2713, which would crash rendering on a plain terminal.
    summary = f"{report.passed}/{report.total} passed"
    if report.ok:
        console.print(f"[green]OK: {summary}[/green]")
    else:
        console.print(f"[red]FAIL: {summary} - {report.failed} failed[/red]")
        raise typer.Exit(code=1)


# ── chat: interactive REPL (thin client) ──────────────────────────────────────

_CHAT_HELP = """\
[bold]Slash commands[/bold]
  /help          show this help
  /model         show the active model
  /permissions   show the permission mode and session grants
  /hooks         list configured lifecycle hooks
  /cost          show cumulative token usage and cost this session
  /agents        list the sub-agent types available for delegation
  /plan <task>   draft a plan with the read-only planner (does not execute)
  /mcp [connect] list MCP servers, or connect them and register their tools
  /plugins       list discovered plugins (loaded / skipped / failed)
  /skills        list discovered skills (invoke one with /<skill-name>)
  /compact       summarize older history now to free up context
  /clear         start a fresh session (clears the transcript)
  /exit, /quit   leave the chat
Anything else is sent to the agent as a turn.\
"""


def _abbrev(value: object, *, limit: int = 80) -> str:
    """One-line, length-capped rendering of an argument value for a prompt."""
    text = str(value).replace("\n", " ").replace("\r", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _parse_permission_answer(answer: str) -> PermissionOutcome | None:
    """Map a typed permission answer to an outcome, or ``None`` if unrecognized.

    Accepts the single-key hints (``y`` / ``a`` / ``n``) and the spelled-out phrases
    shown in the prompt (``allow once`` / ``allow for session`` / ``deny``) plus common
    synonyms, so an operator who types the words they see is understood rather than
    silently denied (the original cause of this prompt's confusion).
    """
    a = answer.strip().lower()
    if a in (
        "a",
        "always",
        "session",
        "allow session",
        "allow for session",
        "allow for the session",
        "allow always",
    ):
        return PermissionOutcome.ALLOW_SESSION
    if a in ("y", "yes", "o", "once", "allow", "allow once"):
        return PermissionOutcome.ALLOW_ONCE
    if a in ("n", "no", "d", "deny"):
        return PermissionOutcome.DENY_ONCE
    return None


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
        g = resolve_glyphs(self.console)
        args = Table(show_header=False, box=None, padding=(0, 2), pad_edge=False)
        args.add_column(style="arg.key", min_width=8, no_wrap=True)
        args.add_column(style="arg.value", overflow="fold")
        for key, val in request.arguments.items():
            args.add_row(key, Text(_abbrev(val)))

        body: list[RenderableType] = [
            Text.assemble(
                (request.tool_name + "  ", "tool.target"),
                ("(" + request.tier.name + ")", "perm.tier"),
            ),
        ]
        if request.reason:
            body.append(Text(request.reason, style="notice.dim"))
        body.append(Text(""))
        body.append(args)
        body.append(Text(""))
        # Keys shown in (parens), never [brackets]: rich parses "[y]" as a markup tag
        # and drops it, so the operator never sees the keys (how a typed "allow for
        # session" once fell through to deny). Show the keys; accept the spelled words.
        body.append(
            Text(
                f"allow once (y) {g['dot']} allow for session (a) {g['dot']} deny (n)",
                style="notice.dim",
            )
        )
        panel = Panel(
            Group(*body),
            title="permission required",
            title_align="left",
            border_style="perm.border",
            padding=(1, 2),
        )
        self.console.print()
        self.console.print(margin(panel))

        prompt = f"  permit? (y/a/n) [prompt.marker]{g['prompt']}[/prompt.marker] "
        for _ in range(3):
            try:
                # Offload the blocking read so concurrent sub-agents (TaskTool runs children
                # via asyncio.gather, sharing this one prompter) don't freeze the event loop
                # — matching bash.py/powershell.py's asyncio.to_thread pattern. (audit2 #11)
                answer = await asyncio.to_thread(self.console.input, prompt)
            except (EOFError, KeyboardInterrupt):
                self.console.print(margin(Text("denied", style="notice.dim")))
                return PermissionOutcome.DENY_ONCE
            decision = _parse_permission_answer(answer)
            if decision is not None:
                return decision
            self.console.print(
                margin(
                    Text(
                        "please answer y (allow once), a (allow for session), or n (deny)",
                        style="notice.dim",
                    )
                )
            )
        self.console.print(margin(Text("no clear answer - denied", style="notice.dim")))
        return PermissionOutcome.DENY_ONCE


def _render_permissions(console: Console, agent: Agent) -> None:
    """Render the active permission mode and any session grants (the /permissions cmd)."""
    policy = agent.permission_policy
    allow = policy.session_allow()
    deny = policy.session_deny()
    console.print(
        margin(
            Text.assemble(("permission mode  ", "banner.label"), (policy.mode.value, "arg.value"))
        )
    )
    console.print(margin(Text(f"session allow    {len(allow)} grant(s)", style="notice.dim")))
    for name in allow:
        console.print(
            Padding(Text.assemble((GLYPHS["add"] + " ", "ok"), (name, "arg.value")), (0, 0, 0, 4))
        )
    console.print(margin(Text(f"session deny     {len(deny)} block(s)", style="notice.dim")))
    for name in deny:
        console.print(
            Padding(Text.assemble((GLYPHS["del"] + " ", "err"), (name, "arg.value")), (0, 0, 0, 4))
        )


def _render_hooks(console: Console, agent: Agent) -> None:
    """List configured lifecycle hooks (the /hooks cmd)."""
    manager = agent.hook_manager
    shell = manager.shell_hooks
    in_proc = sum(len(v) for v in manager.in_process.values())
    if not shell and not in_proc:
        console.print("[dim]no hooks configured.[/dim]")
        return
    for spec in shell:
        line = Text(spec.event.value, style="notice.dim")
        line.append(f" [{spec.matcher}] -> {' '.join(spec.command)}")
        console.print(line)
    if in_proc:
        console.print(Text(f"{in_proc} in-process hook(s) registered.", style="notice.dim"))


def _render_agents(console: Console, agent: Agent) -> None:
    """List the sub-agent types available for delegation (the /agents cmd)."""
    spawner = agent.loop.spawner
    if spawner is None:
        console.print("[dim]delegation is not enabled for this session.[/dim]")
        return
    default = spawner.default_type()
    for name in spawner.available_types():
        marker = " [dim](default)[/dim]" if name == default else ""
        console.print(f"  [bold]{name}[/bold]{marker}")


def _run_plan(console: Console, agent: Agent, task: str) -> None:
    """Run the read-only planner sub-agent and print its plan (the /plan cmd).

    Plan Mode never edits: the planner's tool schema omits write tools. The plan is
    printed for the operator to review; it is NOT auto-executed.
    """
    spawner = agent.loop.spawner
    if spawner is None:
        console.print("[dim]delegation is not enabled for this session.[/dim]")
        return
    if not task:
        console.print("[dim]usage: /plan <what you want planned>[/dim]")
        return
    console.print("[dim]planning (read-only)...[/dim]")
    try:
        result = _run_async(spawner.spawn(type_name="plan", prompt=task))
    except ProviderError as exc:
        notice_error(console, "provider error", str(exc))
        return
    console.print(result.summary or "[dim](the planner produced no plan)[/dim]")


def _render_mcp(console: Console, agent: Agent, arg: str) -> None:
    """Show or connect MCP servers (the /mcp command).

    ``/mcp`` (or ``/mcp list``) lists configured servers and any config errors.
    ``/mcp connect`` spawns the servers and registers their tools into the live
    session — gated, so nothing connects until the operator asks. MCP is opt-in;
    with none configured this is a no-op notice.
    """
    manager = agent.extension_manager
    if manager is None:
        console.print("[dim]MCP is not enabled for this session.[/dim]")
        return
    action = (arg.strip().split(maxsplit=1) or ["list"])[0] if arg.strip() else "list"
    if action == "connect":
        console.print("[dim]connecting MCP servers...[/dim]")
        report = _run_async(agent.connect_mcp())
        if report is None:
            console.print("[dim]MCP is not enabled.[/dim]")
            return
        if report.registered:
            console.print(f"[green]registered {len(report.registered)} tool(s):[/green]")
            for name in report.registered:
                console.print(f"    [green]+[/green] {name}")
        if report.deferred:
            console.print(
                f"[dim]{len(report.deferred)} more tool(s) registered but hidden "
                "(use tool_search to surface them).[/dim]"
            )
        for server, err in report.failed.items():
            console.print(f"    [red]x[/red] {server}: {err}")
        if not report.registered and not report.deferred and not report.failed:
            console.print("[dim]no MCP tools discovered.[/dim]")
        return
    names = manager.server_names
    if not names and not agent.mcp_config_errors:
        console.print("[dim]no MCP servers configured.[/dim]")
        return
    for name in names:
        console.print(f"  [bold]{name}[/bold]")
    for server, err in agent.mcp_config_errors.items():
        console.print(f"  [red]{server}[/red]: {err}")
    console.print("[dim]use /mcp connect to spawn servers and register their tools.[/dim]")


def _render_plugins(console: Console, agent: Agent) -> None:
    """List discovered plugins: what loaded, was skipped, or failed (the /plugins cmd)."""
    report = agent.plugin_report
    errors = agent.plugin_discovery_errors
    if report is None:
        console.print("[dim]plugins are not enabled for this session.[/dim]")
        return
    if not report.loaded and not report.skipped and not report.failed and not errors:
        console.print("[dim]no plugins discovered.[/dim]")
        return
    # Built with Text + console-resolved glyphs (never f-string markup) so a cp1252
    # console can't crash on a glyph and a plugin-supplied name/reason can't inject or
    # drop rich markup.
    g = resolve_glyphs(console)

    def _status_line(glyph: str, glyph_style: str, name: str, detail: str) -> Text:
        line = Text.assemble(("  ", ""), (glyph + " ", glyph_style), (name, "arg.value"))
        if detail:
            line.append(f" ({detail})", style="notice.dim")
        return line

    for name in report.loaded:
        contrib = report.contributions.get(name, {})
        parts = [
            f"{len(contrib.get(kind, []))} {kind}"
            for kind in ("tools", "commands", "hooks")
            if contrib.get(kind)
        ]
        console.print(_status_line(g["ok"], "ok", name, ", ".join(parts)))
    for name, reason in report.skipped.items():
        console.print(_status_line(g["dash"], "warn", name, reason))
    for name, err in report.failed.items():
        console.print(_status_line(g["fail"], "err", name, err))
    for name, err in errors.items():
        console.print(_status_line(g["fail"], "err", name, f"discovery: {err}"))


def _render_skills(console: Console, agent: Agent) -> None:
    """List discovered skills + any discovery errors (the /skills cmd)."""
    registry = getattr(agent, "skill_registry", None)
    g = resolve_glyphs(console)
    if registry is None or len(registry) == 0:
        console.print("[dim]no skills discovered.[/dim]")
    else:
        for name, desc in registry.catalog():
            line = Text.assemble(("  ", ""), (name, "bold"))
            if desc:
                line.append(f" {g['dash']} {desc}", style="notice.dim")
            console.print(line)
        console.print(
            Text("invoke a skill with /<name> to load its instructions.", style="notice.dim")
        )
    for name, err in getattr(agent, "skill_errors", {}).items():
        line = Text.assemble(("  ", ""), (name, "err"))
        line.append(f" ({err})", style="notice.dim")
        console.print(line)


def _invoke_skill(console: Console, agent: Agent, name: str) -> bool:
    """If ``name`` is a skill, load its body into the session and return True.

    Delegates to the CORE :meth:`Agent.invoke_skill` (which injects the body lazily and fires
    the observe-only skill-selection signal); this function only renders the outcome. The body
    is ephemeral, cache-safe context the next turn naturally includes.
    """
    # The live agent may be any AgentLike (a thin/remote client) with no skills surface; a
    # missing invoke_skill just means "not a skill here" — fall through to other command paths.
    invoke = getattr(agent, "invoke_skill", None)
    if invoke is None:
        return False
    result = _run_async(invoke(name))
    if not result.invoked:
        return False  # not a skill — let the caller try other command paths
    if result.error:
        # The file may have changed/vanished since discovery; report, don't crash the REPL.
        notice_error(console, "could not load skill", f"{result.name}: {result.error}")
        return True
    console.print(
        Text.assemble(
            ("loaded skill ", "notice.dim"),
            (result.name, "bold"),
            ("; describe your task and it will apply.", "notice.dim"),
        )
    )
    return True


def _print_banner(console: Console, agent: Agent) -> None:
    """Print the one-shot session banner (model, provider, workspace, perms)."""
    settings = agent.settings
    g = GLYPHS
    dot = f" {g['dot']} "
    title = Text.assemble(("Zak Code", "banner.title"), ("  ", ""), (__version__, "banner.version"))
    facts = kv_table(
        [
            ("model", settings.default_model),
            ("provider", settings.provider),
            ("workspace", str(settings.workspace_root)),
            ("perms", settings.permission_mode),
            ("session", agent.session.id),
        ]
    )
    hints = Text.assemble(
        ("tools  edit (+ read write glob grep bash)", "notice.dim"),
        (dot, "notice.dim"),
        ("Ctrl-C interrupts a reply", "notice.dim"),
    )
    cmds = Text.assemble(
        ("/help for commands", "notice.dim"),
        (dot, "notice.dim"),
        ("/exit to quit", "notice.dim"),
    )
    block = Group(
        title,
        Text("interactive chat", style="banner.version"),
        Text(""),
        facts,
        Text(""),
        hints,
        cmds,
    )
    console.print(Padding(block, (1, 0, 0, 2)))
    # Surface a self.md that was present but failed to load (unreadable / empty after
    # frontmatter) — otherwise the operator silently gets the default identity with no signal.
    identity_error = getattr(agent, "identity_error", None)
    if identity_error:
        console.print(
            Padding(
                Text(f"self.md present but not loaded: {identity_error}", style="warn"),
                (0, 0, 0, 2),
            )
        )


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
    stream = make_stream()
    try:
        await renderer.render(stream)
    finally:
        # The renderer breaks out of the stream on AgentDone, leaving the underlying
        # async generator suspended; close it now (if it supports aclose) so it does
        # not pile up across turns until session shutdown. AsyncIterator does not
        # guarantee aclose(), but our producers (agent/server astream) are async gens.
        aclose = getattr(stream, "aclose", None)
        if aclose is not None:
            await aclose()


# ── session event loop (one per REPL, never one per turn) ──────────────────────
# The whole REPL session runs on a single event loop. This keeps background tasks
# that libraries spawn — notably litellm's logging worker — bound to a *live* loop,
# instead of being orphaned when a per-turn loop closes (which surfaced as
# "RuntimeError: Event loop is closed", "Task was destroyed but it is pending", and
# "coroutine ... was never awaited" floods after a couple of turns). See
# docs/ARCHITECTURE.md: "One async event loop for the whole session."
_SESSION_LOOP: asyncio.AbstractEventLoop | None = None


def _run_async(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run ``coro`` to completion on the active session loop, or a fresh loop if none.

    Inside an interactive REPL the session loop (installed by ``chat`` /
    ``_run_server_chat``) is reused so library background tasks stay valid across
    calls; outside one — e.g. a unit test calling a handler directly — this falls
    back to ``asyncio.run``.
    """
    loop = _SESSION_LOOP
    if loop is not None and not loop.is_closed():
        return loop.run_until_complete(coro)
    return asyncio.run(coro)


def _shutdown_session_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Drain and close the REPL's session loop (best-effort; never raises).

    Closes async generators still suspended on the loop (the renderer breaks out of
    each turn's stream on ``AgentDone``), then cancels any tasks still pending —
    notably background workers libraries spawn (e.g. litellm's logging worker) — and
    lets them unwind on the still-open loop, so nothing is destroyed on a closed
    loop. Finally closes the loop and clears the session reference.
    """
    global _SESSION_LOOP
    with contextlib.suppress(Exception):
        loop.run_until_complete(loop.shutdown_asyncgens())
    pending = asyncio.all_tasks(loop)
    for task in pending:
        task.cancel()
    if pending:
        with contextlib.suppress(Exception):
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    with contextlib.suppress(Exception):
        loop.close()
    _SESSION_LOOP = None
    asyncio.set_event_loop(None)


def _run_streamed_turn(
    console: Console, make_stream: StreamFactory, renderer: StreamRenderer
) -> bool:
    """Run a streamed turn on the session loop; cancellable via Ctrl-C.

    Uses the REPL's session-wide loop (see :data:`_SESSION_LOOP`) — never a loop per
    turn — so library background tasks stay bound to a live loop across turns.
    Returns ``True`` if the turn completed, ``False`` if the user interrupted it
    (Ctrl-C): the in-flight task is cancelled cleanly (the loop persists state at
    message boundaries), a dim ``interrupted`` notice is printed, and control
    returns to the REPL prompt without exiting.
    """
    loop = _SESSION_LOOP
    if loop is None:  # pragma: no cover — chat()/_run_server_chat always install one
        raise RuntimeError("session loop not initialized")
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
        notice_warn(console, "interrupted - turn stopped, returning to prompt")
        return False


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


def _build_chat_agent(
    prompter: ConsolePermissionPrompter,
    overrides: dict[str, Any],
    *,
    enable_memory: bool = True,
    enable_rules: bool = True,
    extra_skill_dirs: list[str] | None = None,
    extra_workspace_roots: list[str] | None = None,
) -> Agent:
    """Build the in-process chat Agent with every interactive feature enabled.

    One builder for both the initial session and ``/clear`` so they never drift.
    ``enable_memory`` / ``enable_rules`` mirror the ``--no-memory`` / ``--no-rules``
    chat flags (on by default). Trusted plugins come from ``ZAKCODE_TRUSTED_PLUGINS``
    (comma-separated names); a discovered plugin runs only if it is named there (else
    it is listed by ``/plugins`` as skipped/untrusted).
    """
    from zakcode import Agent

    trusted = [
        n.strip() for n in os.environ.get("ZAKCODE_TRUSTED_PLUGINS", "").split(",") if n.strip()
    ]
    return Agent(
        prompter=prompter,
        enable_subagents=True,
        enable_mcp=True,
        enable_plugins=True,
        trusted_plugins=trusted,
        enable_skills=True,
        extra_skill_dirs=extra_skill_dirs,
        extra_workspace_roots=extra_workspace_roots,
        enable_rules=enable_rules,
        enable_memory=enable_memory,
        enable_compaction=True,
        **overrides,
    )


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
    no_memory: bool = typer.Option(
        False,
        "--no-memory",
        help="Disable cross-session memory (remember/recall tools + relevant-memory injection).",
    ),
    no_rules: bool = typer.Option(
        False, "--no-rules", help="Disable always-on rules (.zakcode/rules, .claude/rules)."
    ),
    skill_dir: list[str] | None = typer.Option(  # noqa: B008 — typer convention
        None,
        "--skill-dir",
        help=(
            "Extra directory to scan for SKILL.md skills (one subdir per skill). "
            "May be repeated. Later directories shadow earlier same-named skills. "
            "The skill directory's owning repo root (and any external paths from "
            "its local-paths.conf) are automatically added as extra workspace roots."
        ),
    ),
    extra_root: list[str] | None = typer.Option(  # noqa: B008 — typer convention
        None,
        "--extra-root",
        help=(
            "Additional trusted workspace root for file tools. May be repeated. "
            "File reads/writes under any extra root are allowed alongside the "
            "primary workspace root."
        ),
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

    overrides: dict[str, Any] = {}
    if model:
        overrides["default_model"] = model
    if workspace:
        overrides["workspace_root"] = workspace

    # A console prompter lets the in-core permission gate escalate to the operator,
    # so 'ask' mode is usable interactively (rather than failing closed). The gate
    # itself still lives in the core; the CLI only renders the prompt.
    extra_skill_dirs = skill_dir if skill_dir else None
    extra_roots = extra_root if extra_root else None
    prompter = ConsolePermissionPrompter(console)
    agent = _build_chat_agent(
        prompter,
        overrides,
        enable_memory=not no_memory,
        enable_rules=not no_rules,
        extra_skill_dirs=extra_skill_dirs,
        extra_workspace_roots=extra_roots,
    )
    _print_banner(console, agent)

    # One event loop for the whole session (never one per turn) — see
    # _SESSION_LOOP / _shutdown_session_loop.
    global _SESSION_LOOP
    _SESSION_LOOP = loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    while True:
        try:
            line = read_prompt(console)
        except (EOFError, KeyboardInterrupt):
            notice_info(console, "bye")
            break

        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("/"):
            command = stripped.split(maxsplit=1)[0].lower()
            if command in ("/exit", "/quit"):
                notice_info(console, "bye")
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
                agent = _build_chat_agent(
                    prompter,
                    overrides,
                    enable_memory=not no_memory,
                    enable_rules=not no_rules,
                    extra_skill_dirs=extra_skill_dirs,
                    extra_workspace_roots=extra_roots,
                )
                notice_info(console, "started a fresh session")
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
            if command == "/agents":
                _render_agents(console, agent)
                continue
            if command == "/plan":
                _run_plan(console, agent, stripped[len("/plan") :].strip())
                continue
            if command == "/mcp":
                _render_mcp(console, agent, stripped[len("/mcp") :].strip())
                continue
            if command == "/plugins":
                _render_plugins(console, agent)
                continue
            if command == "/skills":
                _render_skills(console, agent)
                continue
            if command == "/compact":
                did = _run_async(agent.loop.compact_now())
                console.print(
                    "[dim]compacted older history into a summary.[/dim]"
                    if did
                    else "[dim]nothing to compact yet.[/dim]"
                )
                continue
            # A bare /<skill-name> invokes a discovered skill (loads its body).
            if _invoke_skill(console, agent, command.lstrip("/")):
                continue
            # Fall through to plugin-registered commands before giving up. ``getattr``
            # because the live agent may be any AgentLike (a thin/remote one without a
            # command registry); a missing registry just means no plugin commands.
            registry = getattr(agent, "command_registry", None)
            cmd_result = (
                registry.run(command, stripped[len(command) :].strip())
                if registry is not None
                else None
            )
            if cmd_result is not None:
                # Opaque plugin output: render as plain Text so a bare [/] can't raise
                # MarkupError (crashing the REPL) and style tags can't drop the literal.
                console.print(Text(cmd_result.output, style="err" if cmd_result.is_error else ""))
                continue
            console.print(f"[dim]{command} is not yet supported.[/dim]")
            continue

        # Stream the model response token by token through the renderer. A fresh
        # renderer per turn keeps the text/usage buffers from leaking across turns.
        renderer = StreamRenderer(console=console)
        try:
            _run_streamed_turn(console, functools.partial(agent.astream_turn, stripped), renderer)
        except ProviderError as exc:
            notice_error(console, "provider error", str(exc))
            continue

    _shutdown_session_loop(loop)


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
        notice_error(console, f"could not reach server at {base_url}", str(exc))
        raise typer.Exit(code=1) from exc

    console.print(
        Text.assemble(
            ("Zak Code", "banner.title"),
            (f" {__version__} ", "banner.version"),
            (f"{GLYPHS['dash']} connected to {base_url}", "notice.dim"),
        )
    )
    console.print(f"[dim]session[/dim]  {session_id}")
    if model:
        console.print(f"[dim]model[/dim]    {model} [dim](per-request override)[/dim]")
    console.print("[dim]Type /exit to quit. (Server mode: turns run headless.)[/dim]\n")

    # One event loop for the whole session (never one per turn) — see
    # _SESSION_LOOP / _shutdown_session_loop.
    global _SESSION_LOOP
    _SESSION_LOOP = loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    while True:
        try:
            line = read_prompt(console)
        except (EOFError, KeyboardInterrupt):
            notice_info(console, "bye")
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
            notice_error(console, "server error", str(exc))
            continue

    _shutdown_session_loop(loop)


def _is_loopback_host(host: str) -> bool:
    """True if ``host`` only reaches the local machine (so an unauthenticated bind is safe).

    Parses the address with :mod:`ipaddress` so the whole ``127.0.0.0/8`` range and every IPv6
    loopback spelling are accepted, while ``0.0.0.0`` / ``::`` (all interfaces) and lookalike
    hostnames (``127.com``, ``127.0.0.1.evil.com``) are correctly NOT loopback. A non-IP host
    is treated as non-loopback — fail-safe: the bind guard then demands a token or ``--insecure``.
    """
    if host == "localhost":
        return True
    candidate = host[1:-1] if host.startswith("[") and host.endswith("]") else host
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address."),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port."),
    workspace: str | None = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace root the served mind loads identity/rules/memory/skills from.",
    ),
    insecure: bool = typer.Option(
        False,
        "--insecure",
        help="Allow binding a non-loopback host with NO auth token (unauthenticated exposure).",
    ),
) -> None:
    """Run the Zak Code HTTP API server (FastAPI over the same core).

    Exposes REST + SSE + a WebSocket channel — see ``docs/ARCHITECTURE.md``. Requires
    the ``server`` extra (``pip install 'zakcode[server]'``). ``--workspace`` points the
    served mind at one customer env (one container per env); without it the server uses the
    configured workspace root (``ZAKCODE_WORKSPACE_ROOT`` / cwd). It is a pointer, not a
    behavior toggle.

    Auth: set ``ZAKCODE_AUTH_TOKEN`` to require ``Authorization: Bearer <token>`` on every
    request (browsers authenticate the WS via the ``Sec-WebSocket-Protocol: bearer, <token>``
    subprotocol). Without a token the server is unauthenticated, so binding a non-loopback
    ``--host`` is refused unless you pass ``--insecure`` (acknowledging the exposure).
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

    # Resolve settings once (also drives the bind guard); only PASS them to create_app when a
    # workspace was given, so --workspace stays a pointer and the no-workspace path resolves
    # settings inside the server from env (unchanged contract).
    resolved_settings = (
        load_settings(workspace_root=workspace) if workspace is not None else load_settings()
    )
    if not _is_loopback_host(host) and not resolved_settings.auth_token and not insecure:
        console.print(
            f"[red]Refusing to bind non-loopback host {host!r} without authentication.[/red]\n"
            "Set [bold]ZAKCODE_AUTH_TOKEN[/bold] to require a bearer token, or pass "
            "[bold]--insecure[/bold] to expose the server unauthenticated (not recommended)."
        )
        raise typer.Exit(code=1)

    fastapi_app = create_app(settings=resolved_settings) if workspace is not None else create_app()
    auth_note = " [auth: on]" if resolved_settings.auth_token else ""
    where = f" (mind workspace: {workspace})" if workspace is not None else ""
    console.print(
        f"[bold]Zak Code[/bold] {__version__} — serving on http://{host}:{port}{where}{auth_note}"
    )
    uvicorn.run(fastapi_app, host=host, port=port)


def main() -> None:
    """Console-script entry point (see ``[project.scripts]`` in pyproject.toml)."""
    app()


if __name__ == "__main__":
    main()
