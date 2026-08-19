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
import random
import signal
import sys
import time
from collections.abc import AsyncIterator, Callable, Coroutine
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import typer
from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.live import Live
from rich.markup import escape
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from zakcode import __version__
from zakcode.cli._glyphs import enable_utf8, resolve_glyphs
from zakcode.cli._layout import (
    heading,
    kv_table,
    margin,
    notice_error,
    notice_info,
    notice_warn,
    panel,
    read_prompt,
)
from zakcode.cli._theme import ZAK_THEME
from zakcode.cli.render import StreamRenderer, display_call
from zakcode.config import PermissionTier, Settings, env_source, load_settings
from zakcode.events import AgentDone, AgentToolCall, AgentToolResult
from zakcode.permissions import PermissionOutcome, PermissionRequest
from zakcode.providers.base import ProviderError
from zakcode.secrets import strip_url_credentials
from zakcode.workspace_trust import ADOPT_ALWAYS, ADOPT_NEVER, ADOPT_SESSION

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

# Quiet third-party startup noise the operator can do nothing about at the prompt:
# the HuggingFace tokenizer-cache symlink warning on Windows and litellm's
# import-time WARNING chatter (missing optional backends). Both must be set via env
# BEFORE the libraries import — litellm reconfigures its own "LiteLLM" logger at
# import time, so a setLevel from here is overwritten; LITELLM_LOG is the supported
# knob it reads first. Real errors still surface via the provider error taxonomy.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("LITELLM_LOG", "ERROR")

# Provider / service API keys we report the *presence* of (never the value).
_PROVIDER_KEY_ENV = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY", "TAVILY_API_KEY"]


def _provider_key_status() -> dict[str, str]:
    """Map each known provider key env-var to its provenance (never its value).

    Sources: ``env`` / ``workspace .env`` / ``user .env`` (``~/.zakcode/.env``) /
    ``not set`` — the debugging story for "why is it using that key on this
    machine" (D20). Accurate after ``load_settings()`` has run.
    """
    return {name: env_source(name) for name in _PROVIDER_KEY_ENV}


def build_info_lines(settings: Settings) -> list[tuple[str, str]]:
    """Build the (label, value) rows shown by ``info``.

    Secret-safe: provider keys are reported as ``set (<source>)`` / ``not set``
    only — the source names where the value came from, never what it is.
    """
    if settings.default_model in ("auto", "zakpick"):
        # Resolve live (read-only probes) so the panel answers "which model WOULD run here,
        # and why". For `zakpick` this is a friendly per-task table; for `auto`, the single pick.
        from zakcode.providers.resolve import describe_resolution

        model_row = describe_resolution(settings)
    else:
        model_row = settings.default_model
    rows: list[tuple[str, str]] = [
        ("Model", model_row),
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
    if settings.egress_proxy:
        rows.append(
            (
                "Subprocess egress",
                "proxy -> " + (", ".join(settings.egress_allowed_domains) or "deny all"),
            )
        )
    for name, source in _provider_key_status().items():
        rows.append((name, "not set" if source == "not set" else f"set ({source})"))
    return rows


@app.command()
def version() -> None:
    """Print the Zak Code version."""
    typer.echo(f"zakcode {__version__}")


@app.command()
def info() -> None:
    """Show resolved configuration and detected providers (never prints secrets)."""
    settings = load_settings()
    g = GLYPHS
    console.print(
        margin(
            Text.assemble(
                (g["spark"] + " ", "brand"),
                ("Zak Code ", "banner.title"),
                (__version__, "banner.version"),
            )
        )
    )
    console.print()
    console.print(margin(kv_table(build_info_lines(settings))))
    console.print()
    console.print(
        margin(
            Text.assemble(
                ("start the interactive agent with ", "banner.hint"),
                ("zakcode chat", "banner.title"),
            )
        )
    )


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

    table = Table(
        title="Zak Code - behavioral evals",
        show_header=True,
        box=None,
        header_style="banner.label",
    )
    table.add_column("probe")
    table.add_column("result")
    if verbose:
        table.add_column("detail")
    for r in report.results:
        mark = Text("PASS", style="ok") if r.passed else Text("FAIL", style="err")
        row: list[RenderableType] = [Text(r.name), mark]
        if verbose:
            row.append(Text(r.detail) if r.passed else Text(r.error or "", style="err"))
        table.add_row(*row)
    console.print(table)
    # ASCII-only summary line: the Windows console default (cp1252) cannot encode
    # marks like U+2713, which would crash rendering on a plain terminal.
    summary = f"{report.passed}/{report.total} passed"
    if report.ok:
        console.print(Text(f"OK: {summary}", style="ok"))
    else:
        console.print(Text(f"FAIL: {summary} - {report.failed} failed", style="err"))
        raise typer.Exit(code=1)


# ── chat: interactive REPL (thin client) ──────────────────────────────────────

#: Rotating banner tips (one per day, never random — stable for eyeball diffs).
_TIPS: tuple[str, ...] = (
    "mention a file by path and Zak reads it before answering",
    "/plan <task> drafts a read-only plan before anything runs",
    "/compact summarizes older history to free up context",
    "ctrl-c interrupts a running reply without leaving the chat",
    "/skills lists what Zak can load; invoke one with /<name>",
)

#: Grouped /help sections (section heading, (command, description) rows) — the
#: full current command set, grouped session / agent / integrations.
_HELP_SECTIONS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "session",
        (
            ("/help", "this list"),
            ("/model", "show the active model"),
            ("/cost", "session token usage and cost"),
            ("/compact", "summarize older history to free context"),
            ("/clear", "start a fresh session"),
            ("/exit /quit", "leave the chat"),
        ),
    ),
    (
        "agent",
        (
            ("/permissions", "permission mode and session grants"),
            ("/hooks", "configured lifecycle hooks"),
            ("/agents", "sub-agent types available for delegation"),
            ("/plan <task>", "draft a plan (read-only, never executes)"),
        ),
    ),
    (
        "integrations",
        (
            ("/mcp [connect]", "list or connect MCP servers"),
            ("/plugins", "discovered plugins (loaded / skipped / failed)"),
            ("/skills", "discovered skills (invoke one with /<name>)"),
        ),
    ),
)


def _print_help(console: Console) -> None:
    """Print the grouped /help layout: spark header, three sections, dim hint."""
    g = resolve_glyphs(console)
    console.print(
        margin(
            Text.assemble(
                (g["spark"] + " ", "brand"),
                ("Zak Code ", "banner.title"),
                (__version__, "banner.version"),
            )
        )
    )
    for name, rows in _HELP_SECTIONS:
        console.print()
        console.print(Padding(heading(name), (0, 0, 0, 4)))
        # escape(): table cells parse markup, and "/mcp [connect]" must render
        # its brackets literally (the same trap as the old "[y]" key hints).
        console.print(Padding(kv_table([(escape(k), escape(v)) for k, v in rows]), (0, 0, 0, 6)))
    console.print()
    console.print(
        Padding(
            Text.assemble(
                ("ctrl-c interrupts a running reply", "banner.hint"),
                (f" {g['dot']} ", "banner.hint"),
                ("anything else talks to Zak", "banner.hint"),
            ),
            (0, 0, 0, 4),
        )
    )


def _dim(console: Console, msg: str) -> None:
    """One dim line in the shared document margin (the slash-command notice style)."""
    console.print(margin(Text(msg, style="notice.dim")))


def _abbrev(value: object, *, limit: int = 80) -> str:
    """One-line, length-capped rendering of an argument value for a prompt."""
    text = str(value).replace("\n", " ").replace("\r", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _parse_permission_answer(answer: str) -> PermissionOutcome | None:
    """Map a typed permission answer to an outcome, or ``None`` if unrecognized.

    Accepts the option numbers (``1`` / ``2`` / ``3``), the single-key hints
    (``y`` / ``a`` / ``n``), and the spelled-out phrases shown in the prompt
    (``allow once`` / ``allow for session`` / ``deny``) plus common synonyms, so an
    operator who types the words they see is understood rather than silently denied
    (the original cause of this prompt's confusion).
    """
    a = answer.strip().lower()
    if a in (
        "2",
        "a",
        "always",
        "session",
        "allow session",
        "allow for session",
        "allow for the session",
        "allow always",
    ):
        return PermissionOutcome.ALLOW_SESSION
    if a in ("1", "y", "yes", "o", "once", "allow", "allow once"):
        return PermissionOutcome.ALLOW_ONCE
    if a in ("3", "n", "no", "d", "deny"):
        return PermissionOutcome.DENY_ONCE
    return None


#: Humanized blast-radius labels (the raw enum name never prints). ``{dash}``
#: resolves per console so the label stays glyph-table-clean in ASCII mode.
_TIER_LABEL: dict[PermissionTier, str] = {
    PermissionTier.READ_ONLY: "read-only",
    PermissionTier.WORKSPACE_WRITE: "writes inside the workspace",
    PermissionTier.DANGER_FULL_ACCESS: "full access {dash} touches paths outside the workspace",
}


class ConsolePermissionPrompter:
    """Asks the operator to confirm an escalated tool call at the terminal.

    Implements the :class:`~zakcode.permissions.PermissionPrompter` protocol. It
    shows the tool call headline (the renderer's own ``display_call``), the
    humanized blast radius, the *exact* arguments (so the operator approves the
    real action, not a summary — see ``docs/GUARDRAILS.md`` §3), and the reason it
    was escalated, then offers the numbered options 1/2/3 with their y/a/n keys.
    A non-interactive or unrecognized answer defaults to **deny** (fail toward
    safe). The prompter owns its own spacing: one blank before the panel, one
    after the decision line — it never touches renderer state.
    """

    def __init__(self, console: Console) -> None:
        self.console = console

    async def confirm(self, request: PermissionRequest) -> PermissionOutcome:
        # The REPL's wait line repaints the bottom row; pause it while the panel
        # and the blocking input prompt own the screen, resume once decided.
        wait = _ACTIVE_WAIT
        if wait is not None:
            wait.pause()
        try:
            return await self._confirm(request)
        finally:
            if wait is not None:
                wait.resume()

    async def _confirm(self, request: PermissionRequest) -> PermissionOutcome:
        g = resolve_glyphs(self.console)
        dash = g["dash"]
        tier_label = _TIER_LABEL.get(
            request.tier, request.tier.name.lower().replace("_", " ")
        ).format(dash=dash)

        args = Table(show_header=False, box=None, padding=(0, 2), pad_edge=False)
        args.add_column(style="arg.key", min_width=8, no_wrap=True)
        args.add_column(style="arg.value", overflow="fold")
        for key, val in request.arguments.items():
            # Both cells as Text: a model-supplied argument NAME containing rich
            # markup (e.g. "[/]") must print literally, never parse — a MarkupError
            # here would unwind the whole REPL mid-consent.
            args.add_row(Text(key), Text(_abbrev(val)))

        options = Table(show_header=False, box=None, padding=(0, 2), pad_edge=False, expand=True)
        options.add_column(style="perm.key", no_wrap=True)
        options.add_column(style="perm.option", overflow="fold", ratio=1)
        options.add_column(style="perm.key", justify="right", no_wrap=True)
        options.add_row("1", "allow once", "y")
        options.add_row("2", "allow for this session", "a")
        options.add_row("3", f"deny {dash} tell Zak what to do instead", "n")

        body: list[RenderableType] = [
            display_call(request.tool_name, request.arguments, glyphs=g),
            Text(tier_label, style="perm.reason"),
        ]
        if request.reason:
            body.append(Text(request.reason, style="notice.dim"))
        body.append(Text(""))
        body.append(args)
        body.append(Text(""))
        body.append(options)
        self.console.print()
        self.console.print(
            panel(
                self.console,
                "[perm.title]permission[/perm.title]",
                Group(*body),
                border_style="perm.border",
            )
        )

        prompt = f"  permit (1-3 or y/a/n) [prompt.marker]{g['prompt']}[/prompt.marker] "
        for _ in range(3):
            try:
                # Offload the blocking read so concurrent sub-agents (TaskTool runs children
                # via asyncio.gather, sharing this one prompter) don't freeze the event loop
                # — matching bash.py/powershell.py's asyncio.to_thread pattern. (audit2 #11)
                answer = await asyncio.to_thread(self.console.input, prompt)
            except (EOFError, KeyboardInterrupt):
                self.console.print(margin(Text("denied", style="notice.dim")))
                self.console.print()
                return PermissionOutcome.DENY_ONCE
            decision = _parse_permission_answer(answer)
            if decision is not None:
                self.console.print()
                return decision
            self.console.print(
                margin(
                    Text(
                        "please answer 1-3, or y (allow once) / a (allow for session) / n (deny)",
                        style="notice.dim",
                    )
                )
            )
        self.console.print(margin(Text("no clear answer - denied", style="notice.dim")))
        self.console.print()
        return PermissionOutcome.DENY_ONCE


def _parse_adoption_answer(answer: str) -> str | None:
    """Map a typed folder-trust answer to an adoption decision (None = not understood).

    ``1/y`` → always for this workspace, ``2/o`` → this session only, ``3/n`` → never for
    this workspace. Mirrors :func:`_parse_permission_answer`'s shape so the two prompts
    stay predictable together.
    """
    a = answer.strip().lower()
    if a in ("1", "y", "yes", "always"):
        return ADOPT_ALWAYS
    if a in ("2", "o", "once", "session"):
        return ADOPT_SESSION
    if a in ("3", "n", "no", "never"):
        return ADOPT_NEVER
    return None


def _ask_hooks_adoption(console: Console, summary: dict[str, int]) -> str | None:
    """Render the one-time folder-trust question for workspace hooks.

    The policy half lives in core (:func:`zakcode.workspace_trust.resolve_hooks_adoption`);
    this renders the question and returns ``"always"`` / ``"session"`` / ``"never"``, or None
    when dismissed (EOF/interrupt/no clear answer) — the resolver treats None as "off, ask
    again next session". Runs at startup before any event loop exists, so the blocking
    ``console.input`` is fine here (contrast the mid-turn permission prompt's to_thread).
    """
    g = resolve_glyphs(console)
    label = ", ".join(f"{name} x{count}" for name, count in summary.items())
    options = Table(show_header=False, box=None, padding=(0, 2), pad_edge=False, expand=True)
    options.add_column(style="perm.key", no_wrap=True)
    options.add_column(style="perm.option", overflow="fold", ratio=1)
    options.add_column(style="perm.key", justify="right", no_wrap=True)
    options.add_row("1", "yes, always for this workspace", "y")
    options.add_row("2", "yes, this session only", "o")
    options.add_row("3", f"no {g['dash']} never for this workspace", "n")
    body: list[RenderableType] = [
        Text.assemble(
            ("this workspace defines Claude Code hooks: ", "notice.dim"),
            (label, "arg.value"),
        ),
        Text(
            "hooks are shell commands from the workspace; they run on zak code's seams "
            "(turn end, tool calls, session start).",
            style="notice.dim",
        ),
        Text(""),
        options,
    ]
    console.print()
    console.print(
        panel(
            console,
            "[perm.title]workspace hooks[/perm.title]",
            Group(*body),
            border_style="perm.border",
        )
    )
    prompt = f"  adopt (1-3 or y/o/n) [prompt.marker]{g['prompt']}[/prompt.marker] "
    for _ in range(3):
        try:
            answer = console.input(prompt)
        except (EOFError, KeyboardInterrupt):
            return None
        decision = _parse_adoption_answer(answer)
        if decision is not None:
            console.print()
            return decision
        console.print(
            margin(
                Text(
                    "please answer 1-3, or y (always) / o (this session) / n (never)",
                    style="notice.dim",
                )
            )
        )
    return None


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
        _dim(console, "no hooks configured.")
        return
    for spec in shell:
        line = Text(spec.event.value, style="notice.dim")
        line.append(f" [{spec.matcher}] -> {' '.join(spec.command)}")
        console.print(margin(line))
    if in_proc:
        _dim(console, f"{in_proc} in-process hook(s) registered.")


def _render_agents(console: Console, agent: Agent) -> None:
    """List the sub-agent types available for delegation (the /agents cmd)."""
    spawner = agent.loop.spawner
    if spawner is None:
        _dim(console, "delegation is not enabled for this session.")
        return
    default = spawner.default_type()
    for name in spawner.available_types():
        line = Text.assemble(("  ", ""), (name, "bold"))
        if name == default:
            line.append(" (default)", style="notice.dim")
        console.print(line)


def _run_plan(console: Console, agent: Agent, task: str) -> None:
    """Run the read-only planner sub-agent and print its plan (the /plan cmd).

    Plan Mode never edits: the planner's tool schema omits write tools. The plan is
    printed for the operator to review; it is NOT auto-executed.
    """
    spawner = agent.loop.spawner
    if spawner is None:
        _dim(console, "delegation is not enabled for this session.")
        return
    if not task:
        _dim(console, "usage: /plan <what you want planned>")
        return
    _dim(console, "planning (read-only)...")
    try:
        result = _run_async(spawner.spawn(type_name="plan", prompt=task))
    except ProviderError as exc:
        notice_error(console, "provider error", str(exc))
        return
    if result.summary:
        console.print(margin(Text(result.summary)))
    else:
        _dim(console, "(the planner produced no plan)")


def _render_mcp(console: Console, agent: Agent, arg: str) -> None:
    """Show or connect MCP servers (the /mcp command).

    ``/mcp`` (or ``/mcp list``) lists configured servers and any config errors.
    ``/mcp connect`` spawns the servers and registers their tools into the live
    session — gated, so nothing connects until the operator asks. MCP is opt-in;
    with none configured this is a no-op notice.
    """
    manager = agent.extension_manager
    if manager is None:
        _dim(console, "MCP is not enabled for this session.")
        return
    g = resolve_glyphs(console)
    action = (arg.strip().split(maxsplit=1) or ["list"])[0] if arg.strip() else "list"
    if action == "connect":
        _dim(console, "connecting MCP servers...")
        report = _run_async(agent.connect_mcp())
        if report is None:
            _dim(console, "MCP is not enabled.")
            return
        if report.registered:
            console.print(margin(Text(f"registered {len(report.registered)} tool(s):", style="ok")))
            for name in report.registered:
                console.print(
                    Padding(
                        Text.assemble((g["add"] + " ", "ok"), (name, "arg.value")), (0, 0, 0, 4)
                    )
                )
        if report.deferred:
            _dim(
                console,
                f"{len(report.deferred)} more tool(s) registered but hidden "
                "(use tool_search to surface them).",
            )
        for server, err in report.failed.items():
            console.print(
                Padding(
                    Text.assemble((g["fail"] + " ", "err"), (f"{server}: {err}", "arg.value")),
                    (0, 0, 0, 4),
                )
            )
        if not report.registered and not report.deferred and not report.failed:
            _dim(console, "no MCP tools discovered.")
        return
    names = manager.server_names
    if not names and not agent.mcp_config_errors:
        _dim(console, "no MCP servers configured.")
        return
    for name in names:
        console.print(Text.assemble(("  ", ""), (name, "bold")))
    for server, err in agent.mcp_config_errors.items():
        console.print(Text.assemble(("  ", ""), (server, "err"), (f": {err}", "notice.dim")))
    _dim(console, "use /mcp connect to spawn servers and register their tools.")


def _render_plugins(console: Console, agent: Agent) -> None:
    """List discovered plugins: what loaded, was skipped, or failed (the /plugins cmd)."""
    report = agent.plugin_report
    errors = agent.plugin_discovery_errors
    if report is None:
        _dim(console, "plugins are not enabled for this session.")
        return
    if not report.loaded and not report.skipped and not report.failed and not errors:
        _dim(console, "no plugins discovered.")
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
        _dim(console, "no skills discovered.")
    else:
        for name, desc in registry.catalog():
            line = Text.assemble(("  ", ""), (name, "bold"))
            if desc:
                line.append(f" {g['dash']} {desc}", style="notice.dim")
            console.print(line)
        _dim(console, f"run a skill with /<name> [args] {g['dash']} it executes as this turn.")
        invoked = getattr(agent, "skill_invocations_this_session", 0)
        if invoked:
            _dim(console, f"the model has invoked skills {invoked}x this session (use_skill).")
    for name, err in getattr(agent, "skill_errors", {}).items():
        line = Text.assemble(("  ", ""), (name, "err"))
        line.append(f" ({err})", style="notice.dim")
        console.print(line)


class _SkillCommandOutcome(NamedTuple):
    """What the REPL should do with a ``/<token>`` that may be a skill."""

    handled: bool  # True → the token matched a discovered skill; stop command fallthrough
    turn_text: str | None  # set → run this text as the current turn's user message


def _skill_command_turn(
    console: Console, agent: Agent, name: str, args: str = ""
) -> _SkillCommandOutcome:
    """If ``name`` is a skill, compose its body as THIS turn's input (Claude Code semantics).

    Delegates to the CORE :meth:`Agent.compose_skill_turn` (which loads, defangs, and fires
    the observe-only skill-selection signal); this function only renders the outcome. On
    success the REPL streams ``turn_text`` through the same path as any typed message — the
    slash command IS the turn, so ``/start sera`` runs now instead of waiting for a second
    "describe your task" message.
    """
    # The live agent may be any AgentLike (a thin/remote client) with no skills surface; a
    # missing compose_skill_turn just means "not a skill here" — fall through to other paths.
    compose = getattr(agent, "compose_skill_turn", None)
    if compose is None:
        return _SkillCommandOutcome(False, None)
    result = _run_async(compose(name, args))
    if not result.invoked:
        return _SkillCommandOutcome(False, None)  # not a skill — try other command paths
    if result.error:
        # The file may have changed/vanished since discovery; report, don't crash the REPL.
        notice_error(console, "could not load skill", f"{result.name}: {result.error}")
        return _SkillCommandOutcome(True, None)
    if result.denied_reason:
        # A policy refusal (e.g. user-invocable: false), not a failure — a friendly notice.
        notice_info(console, result.denied_reason)
        return _SkillCommandOutcome(True, None)
    console.print(
        margin(
            Text.assemble(
                ("running skill ", "notice.dim"),
                (result.name, "bold"),
                ((f" {GLYPHS['dot']} {args}" if args else ""), "notice.dim"),
            )
        )
    )
    return _SkillCommandOutcome(True, result.turn_text)


def _trim_kv_value(value: str, limit: int, g: dict[str, str]) -> str:
    """Left-truncate an over-long banner value so its tail (the filename) survives."""
    if len(value) <= limit:
        return value
    ellipsis = g["ellipsis"]
    return ellipsis + value[-(max(1, limit - len(ellipsis))) :]


def _welcome_panel(console: Console, rows: list[tuple[str, str]], hint: Text) -> Padding:
    """The welcome box: spark + bold title, dim kv rows, dim hint line."""
    g = resolve_glyphs(console)
    # Mirror panel()'s width clamp, then truncate against the box INTERIOR
    # (borders + padding cost 6 columns) so long values never fold mid-row —
    # a left-truncated Windows path keeps its tail on one line.
    width = max(24, min(console.width - 4, 60))
    limit = max(1, (width - 6) - 20)
    # Trim first, then escape: cells parse markup, and a real path may contain
    # literal brackets that must survive (markup immunity, not just truncation).
    trimmed = [(key, escape(_trim_kv_value(value, limit, g))) for key, value in rows]
    body = Group(
        Text.assemble((g["spark"] + " ", "brand"), ("Zak Code", "banner.title")),
        Text(""),
        Padding(kv_table(trimmed), (0, 0, 0, 2)),
        Text(""),
        Padding(hint, (0, 0, 0, 2)),
    )
    return panel(console, "", body, border_style="banner.border")


def _print_banner(console: Console, agent: Agent) -> None:
    """Print the one-shot welcome box (model, workspace, perms, session) + tip."""
    settings = agent.settings
    g = resolve_glyphs(console)
    # "claude-sonnet-4-5 · anthropic", not "anthropic/claude-… · anthropic": the
    # provider half of the row already names the prefix. Under zakpick the model varies per
    # task category, so the row names the mode (the per-task table is on /model + the info panel).
    if getattr(agent, "_zakpick", False):
        model_cell = f"zakpick {g['dot']} picks a model per task"
    else:
        model_id = settings.default_model.removeprefix(settings.provider + "/")
        model_cell = f"{model_id} {g['dot']} {settings.provider}"
    rows = [
        ("model", model_cell),
        ("workspace", str(settings.workspace_root)),
        ("permissions", settings.permission_mode),
        ("session", agent.session.id),
    ]
    hint = Text.assemble(
        ("/help for commands", "banner.hint"),
        (f" {g['dot']} ", "banner.hint"),
        ("/exit to quit", "banner.hint"),
    )
    console.print(_welcome_panel(console, rows, hint))
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
    # The daily tip: date-ordinal rotation (deterministic within a day) and only on a
    # real terminal — hermetic StringIO tests never see it.
    if console.is_terminal:
        tip = _TIPS[date.today().toordinal() % len(_TIPS)]
        console.print(
            margin(Text.assemble((g["spark_soft"] + " ", "brand.soft"), (f"tip: {tip}", "tip")))
        )


#: A factory that produces one turn's event stream. The CLI is agnostic to where
#: the stream comes from — a local in-process ``Agent`` or a remote ``ServerClient``
#: — so the same cancellable runner and renderer drive both.
StreamFactory = Callable[[], "AsyncIterator[AgentEvent]"]


# ── the wait line (REPL-owned; the renderer never animates) ────────────────────

#: The gerund corpus the wait line cycles through between tool calls. All
#: randomness lives here, never in StreamRenderer (which stays deterministic).
_GERUNDS: tuple[str, ...] = (
    "Thinking",
    "Brewing",
    "Wiring",
    "Sketching",
    "Reading",
    "Tinkering",
    "Untangling",
    "Stitching",
    "Pondering",
    "Polishing",
)


class _WaitLine:
    """The transient wait-line renderable: spark frame + verb + dim elapsed hint.

    A plain renderable hosted in ``rich.live.Live`` — never registered into rich's
    spinner machinery. Frames are always built via ``Text`` (never markup-parsed),
    so the ASCII frames ``\\`` and ``|`` are safe literals.
    """

    def __init__(self, console: Console) -> None:
        self._g = resolve_glyphs(console)
        self._start = time.monotonic()
        self.verb: str = random.choice(_GERUNDS)

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        g = self._g
        elapsed = time.monotonic() - self._start
        frames = (g["spin1"], g["spin2"], g["spin3"], g["spin4"])
        line = Text("  ")
        line.append(frames[int(elapsed // 0.12) % 4], style="spinner")
        line.append(f" {self.verb}{g['ellipsis']} ")
        line.append(f"(ctrl-c to interrupt {g['dot']} {int(elapsed)}s)", style="notice.dim")
        yield line


class _WaitHandle:
    """One turn's live wait line, with the pause handle the prompter needs.

    The verb goes concrete (``Running…``) while any tool call is outstanding and
    returns to a fresh gerund once results drain; ``AgentDone`` stops the line so
    the footer prints onto a clean bottom row.
    """

    def __init__(self, console: Console) -> None:
        self.console = console
        self.line = _WaitLine(console)
        self._live = Live(self.line, console=console, transient=True, refresh_per_second=8)
        self._outstanding = 0
        self._active = False
        #: Outstanding pause() calls. Concurrent permission prompts share this one
        #: handle (sub-agents gather on a single prompter), so resume() may only
        #: restart the live region when the LAST pauser releases — else the first
        #: decision repaints the spinner over a prompt that still owns stdin.
        self._pauses = 0
        #: stop() is terminal: a straggling resume() (e.g. a prompt unwinding after
        #: the turn was cancelled) must never restart a stopped live region.
        self._stopped = False

    def start(self) -> None:
        if self._stopped:
            return
        if not self._active:
            self._live.start()
            self._active = True

    def pause(self) -> None:
        """Stop the live region (transient — it erases) so a prompt can own the tty."""
        self._pauses += 1
        if self._active:
            self._live.stop()
            self._active = False

    def resume(self) -> None:
        self._pauses = max(0, self._pauses - 1)
        if self._pauses == 0:
            self.start()

    def stop(self) -> None:
        global _ACTIVE_WAIT
        self._stopped = True
        if self._active:
            self._live.stop()
            self._active = False
        if _ACTIVE_WAIT is self:
            _ACTIVE_WAIT = None

    def observe(self, event: AgentEvent) -> None:
        if isinstance(event, AgentToolCall):
            self._outstanding += 1
            self.line.verb = "Running"
        elif isinstance(event, AgentToolResult):
            self._outstanding = max(0, self._outstanding - 1)
            if self._outstanding == 0:
                self.line.verb = random.choice(_GERUNDS)
        elif isinstance(event, AgentDone):
            self.stop()


#: The live wait line for the in-flight turn, if any — the permission prompter
#: pauses it before printing its panel/reading input and resumes after.
_ACTIVE_WAIT: _WaitHandle | None = None


def _start_wait(console: Console) -> _WaitHandle | None:
    """Start the wait line for a turn, or ``None`` where it must not exist.

    Auto-disabled on legacy conhost (no VT control), off-tty (hermetic tests,
    pipes), and under ``ZAKCODE_NO_SPINNER``; the next printed ``●`` block is the
    fallback narrative.
    """
    global _ACTIVE_WAIT
    if console.legacy_windows or not console.is_terminal:
        return None
    if os.environ.get("ZAKCODE_NO_SPINNER"):
        return None
    handle = _WaitHandle(console)
    _ACTIVE_WAIT = handle
    handle.start()
    return handle


async def _watch_stream(
    stream: AsyncIterator[AgentEvent], wait: _WaitHandle
) -> AsyncIterator[AgentEvent]:
    """Feed the wait line from each event without disturbing the stream."""
    try:
        async for event in stream:
            wait.observe(event)
            yield event
    finally:
        aclose = getattr(stream, "aclose", None)
        if aclose is not None:
            await aclose()


async def _drive_stream(make_stream: StreamFactory, renderer: StreamRenderer) -> None:
    """Render one streamed turn live.

    ``make_stream()`` yields :class:`~zakcode.events.AgentEvent`s (from a local
    agent or a remote server); the renderer writes them to the console
    incrementally. The stream is created here, inside the task's event loop, so a
    loop-bound transport (e.g. httpx) is constructed on the right loop. Where the
    console supports it, the turn hosts the transient wait line (renderer prints
    flow above the shared live region).
    """
    stream = make_stream()
    wait = _start_wait(renderer.console)
    if wait is not None:
        stream = _watch_stream(stream, wait)
    try:
        await renderer.render(stream)
    finally:
        if wait is not None:
            wait.stop()
        # The renderer breaks out of the stream on AgentDone, leaving the underlying
        # async generator suspended; close it now (if it supports aclose) so it does
        # not pile up across turns until session shutdown. AsyncIterator does not
        # guarantee aclose(), but our producers (agent/server astream) are async gens
        # — and the _watch_stream wrapper closes its inner stream on its way out.
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


#: How many clean, un-escalated deep_code turns before the one-time "your deep coder wasn't
#: needed" advisory fires (zakpick only, once per session — a gentle nudge, never naggy).
_ZAKPICK_ADVISORY_AFTER = 3


def _maybe_zakpick_advisory(console: Console, done: AgentDone | None, state: list[int]) -> None:
    """Once per session, gently note that the user's deep coder handled routine work without
    needing to — so they can reconsider their deep_code assignment. Inspired by "Intelligence
    per Watt" (most turns are easy enough for a smaller model).

    ``state`` is ``[count, shown]`` carried across turns. No-ops unless zakpick is active
    (``done.routed_category`` is set), the turn ended cleanly on ``deep_code``, and the soft
    latch never fired (``routed_escalated`` False) — i.e. the strong model wasn't actually
    exercised. Fires at most once, after :data:`_ZAKPICK_ADVISORY_AFTER` such turns.
    """
    if done is None or state[1]:
        return
    if (
        done.routed_category != "deep_code"
        or done.routed_escalated
        or done.stop_reason != "completed"
    ):
        return
    state[0] += 1
    if state[0] < _ZAKPICK_ADVISORY_AFTER:
        return
    state[1] = 1
    console.print(
        margin(
            Text(
                "tip: your deep coder (deep_code) has handled several turns this session "
                "without ever hitting a struggle signal — if those felt routine, a cheaper "
                "model in deep_code may keep up. See per-model spend with /cost.",
                style="tip",
            )
        )
    )


def _maybe_render_status_line(console: Console, agent: Agent) -> None:
    """Render the configured Claude Code statusLine as a dim line after a turn (cosmetic).

    No-ops unless this Agent has a statusLine configured (``status_line_spec``) — which is
    itself gated on the ``status_line`` opt-in. Builds the CC-shaped status JSON from the
    session/usage/model/cwd snapshot, runs the command (env-scrubbed, short timeout), and
    prints its first stdout line dimmed. FAIL-SAFE end to end: ``render_status_line`` swallows
    every error and returns ``None``, and this wrapper guards the whole thing — a broken status
    script prints nothing and NEVER disturbs the REPL or the turn that just finished.
    """
    spec = getattr(agent, "status_line_spec", None)
    if spec is None:
        return
    try:
        from zakcode.status_line import build_status_input, render_status_line

        usage = agent.session.cumulative_usage()
        status = build_status_input(
            session_id=agent.session.id,
            cwd=str(agent.settings.workspace_root),
            model_id=agent.settings.default_model,
            cost_usd=usage.cost_usd,
            total_tokens=usage.total_tokens,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            version=__version__,
        )
        line = _run_async(render_status_line(spec, status))
    except Exception:  # noqa: BLE001 — a cosmetic status line never disturbs the REPL
        return
    if line:
        # Plain Text (never markup) so a status script's output can't inject rich markup
        # or crash a cp1252 console; dimmed in the shared document margin.
        console.print(margin(Text(line, style="notice.dim")))


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
        try:
            # Pump the loop so the task observes the cancellation and unwinds
            # (its CancelledError handler persists state and re-raises).
            with contextlib.suppress(asyncio.CancelledError, KeyboardInterrupt):
                loop.run_until_complete(task)
        finally:
            # A SECOND Ctrl-C during the drain aborts the pump before the task's
            # finally ran — stop the wait line unconditionally here, or its refresh
            # thread keeps repainting over the next REPL prompt.
            if _ACTIVE_WAIT is not None:
                _ACTIVE_WAIT.stop()
        notice_warn(console, f"interrupted {GLYPHS['dash']} turn stopped, returning to prompt")
        return False


async def _server_turn_stream(
    base_url: str, message: str, session_id: str | None, model: str | None = None
) -> AsyncIterator[AgentEvent]:
    """Stream one turn from a remote server, creating + closing a client per turn.

    A fresh :class:`~zakcode.server.client.ServerClient` (and its httpx client) is
    built inside the consuming event loop and closed when the turn ends, which
    keeps the loop-bound httpx client valid across the REPL's per-turn loops.
    """
    from zakcode.server.client import ServerClient

    client = ServerClient(base_url, auth_token=os.environ.get("ZAKCODE_AUTH_TOKEN"))
    try:
        async for event in client.astream_turn(message, session_id, model):
            yield event
    finally:
        await client.aclose()


def _create_remote_session(base_url: str) -> str:
    """Create a session on the remote server and return its id (one-shot)."""

    async def _go() -> str:
        from zakcode.server.client import ServerClient

        client = ServerClient(base_url, auth_token=os.environ.get("ZAKCODE_AUTH_TOKEN"))
        try:
            return await client.create_session()
        finally:
            await client.aclose()

    return asyncio.run(_go())


def _build_chat_agent(
    prompter: ConsolePermissionPrompter,
    overrides: dict[str, Any],
    *,
    enable_rules: bool = True,
    extra_skill_dirs: list[str] | None = None,
    extra_workspace_roots: list[str] | None = None,
    session_id: str | None = None,
) -> Agent:
    """Build the in-process chat Agent with every interactive feature enabled.

    One builder for both the initial session and ``/clear`` so they never drift.
    ``enable_rules`` mirrors the ``--no-rules`` chat flag (on by default). Trusted plugins
    come from ``ZAKCODE_TRUSTED_PLUGINS`` (comma-separated names); a discovered plugin runs
    only if it is named there (else it is listed by ``/plugins`` as skipped/untrusted).
    """
    from zakcode import Agent
    from zakcode.session.store import SessionStore

    trusted = [
        n.strip() for n in os.environ.get("ZAKCODE_TRUSTED_PLUGINS", "").split(",") if n.strip()
    ]
    # The transcript store (~/.zakcode/sessions): persists every turn so -s can resume it later.
    store = SessionStore()
    session = store.load(session_id) if session_id else None
    return Agent(
        prompter=prompter,
        session=session,
        session_store=store,
        enable_subagents=True,
        enable_mcp=True,
        enable_plugins=True,
        trusted_plugins=trusted,
        enable_skills=True,
        extra_skill_dirs=extra_skill_dirs,
        extra_workspace_roots=extra_workspace_roots,
        enable_rules=enable_rules,
        enable_compaction=True,
        **overrides,
    )


def _run_one_shot(
    loop: asyncio.AbstractEventLoop,
    stream_factory: Callable[[], AsyncIterator[AgentEvent]],
) -> int:
    """Run ONE streamed turn and return a shell exit code: 0 if it completed cleanly, 1 otherwise
    (max-iterations, budget, provider/server error, stuck, ...). The loop is torn down before
    returning. Used by ``chat -p`` for non-interactive / scripted runs; a one-shot must yield a
    code, never a traceback.
    """
    renderer = StreamRenderer(console=console)
    try:
        _run_streamed_turn(console, stream_factory, renderer)
    except Exception as exc:  # noqa: BLE001 — a one-shot returns a code, never crashes a script
        notice_error(console, "turn failed", str(exc))
        return 1
    finally:
        _shutdown_session_loop(loop)
    done = renderer.last_done
    return 0 if done is not None and done.stop_reason == "completed" else 1


@app.command()
def chat(
    model: str = typer.Option(None, "--model", "-m", help="Override the model id."),
    prompt: str = typer.Option(
        None,
        "--prompt",
        "-p",
        help="Run a SINGLE task non-interactively and exit (no REPL). Exit code is 0 if the task "
        "completed, non-zero otherwise, so it composes in scripts. Unattended runs want "
        "ZAKCODE_PERMISSION_MODE=allow (or autonomous) -- 'ask' fails closed with no terminal.",
    ),
    provider: str = typer.Option(  # noqa: ARG001 — derived from the model; reserved for clarity
        None, "--provider", help="Hint the provider family (informational)."
    ),
    session: str = typer.Option(None, "--session", "-s", help="Resume a saved session by id."),
    workspace: str = typer.Option(
        None, "--workspace", "-w", help="Workspace root for tools and the session."
    ),
    server: str = typer.Option(
        None,
        "--server",
        help="Drive a remote zakcode server (e.g. http://127.0.0.1:8000) instead of "
        "running the engine in-process. Proves the client/server boundary.",
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
    trace: bool = typer.Option(
        False,
        "--trace",
        help="Write a structured per-turn decision trace (JSONL, one file per turn) to "
        ".zakcode/traces/ in the workspace: routing, gate firings, tool calls, and why each turn "
        "ended. For a custom location, set ZAKCODE_TRACE_DIR instead.",
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
        ignored = [
            flag
            for flag, val in (
                ("--workspace", workspace),
                ("--no-rules", no_rules),
                ("--skill-dir", skill_dir),
                ("--extra-root", extra_root),
                ("--trace", trace),
                ("--session/-s", session),
            )
            if val
        ]
        if ignored:
            notice_warn(
                console,
                f"server mode ignores {', '.join(ignored)} "
                "(these configure the in-process engine, not the remote server)",
            )
        if prompt is not None:
            notice_error(
                console,
                "--prompt is not supported with --server yet",
                "run the task in-process (drop --server), or open an interactive --server session.",
            )
            raise typer.Exit(code=2)
        _run_server_chat(server, model)
        return

    if prompt is not None and not prompt.strip():
        notice_error(console, "--prompt was empty", 'pass a task, e.g. -p "fix the tests".')
        raise typer.Exit(code=2)

    overrides: dict[str, Any] = {}
    if model:
        overrides["default_model"] = model
    if workspace:
        overrides["workspace_root"] = workspace
    if trace:
        trace_dir = (Path(workspace).resolve() if workspace else Path.cwd()) / ".zakcode" / "traces"
        overrides["trace_dir"] = str(trace_dir)
        console.print(f"[dim]decision trace -> {trace_dir}[/dim]")

    # A console prompter lets the in-core permission gate escalate to the operator,
    # so 'ask' mode is usable interactively (rather than failing closed). The gate
    # itself still lives in the core; the CLI only renders the prompt.
    extra_skill_dirs = skill_dir if skill_dir else None
    extra_roots = extra_root if extra_root else None
    prompter = ConsolePermissionPrompter(console)

    # Workspace hook adoption (Claude Code folder-trust semantics). The policy lives in
    # core (zakcode.workspace_trust); this only supplies the ask-UI and applies the result.
    # Only the UNSET tri-state consults the per-workspace memory or asks — an explicit
    # ZAKCODE_SETTINGS_HOOKS (env or .env) is the operator's standing answer. Non-interactive
    # runs (-p, no tty) never prompt: they get a one-line pointer and stay off, so headless
    # behavior is deterministic. The silent failure this replaces: a Claude-Code workspace's
    # hooks block was ignored with NOTHING said, and the miss surfaced layers away.
    from zakcode.hooks.settings_loader import summarize_settings_hooks
    from zakcode.workspace_trust import (
        hooks_decision,
        remember_hooks_decision,
        resolve_hooks_adoption,
    )

    _probe = Settings(**{k: v for k, v in overrides.items() if k in Settings.model_fields})
    adoption = resolve_hooks_adoption(
        configured=_probe.settings_hooks,
        summary=summarize_settings_hooks(_probe.workspace_root),
        decision=hooks_decision(_probe.workspace_root),
        interactive=prompt is None and sys.stdin.isatty(),
        ask=functools.partial(_ask_hooks_adoption, console),
    )
    if adoption.enable:
        overrides["enable_settings_hooks"] = True  # rides into Agent(**overrides) + /clear
    if adoption.remember is not None:
        remember_hooks_decision(_probe.workspace_root, adoption.remember)
    if adoption.notice:
        _dim(console, adoption.notice)
    from zakcode.session.store import SessionError

    try:
        agent = _build_chat_agent(
            prompter,
            overrides,
            enable_rules=not no_rules,
            extra_skill_dirs=extra_skill_dirs,
            extra_workspace_roots=extra_roots,
            session_id=session,
        )
    except ProviderError as exc:
        # The loud startup failure for default_model='auto' with nothing viable:
        # the diagnosis (per-source reasons + key provenance) IS the message.
        notice_error(console, "no model available", str(exc))
        raise typer.Exit(code=1) from exc
    except SessionError as exc:
        # A bad/missing/corrupt --session id: name the id, don't dump a traceback.
        notice_error(console, f"could not resume session {session!r}", str(exc))
        raise typer.Exit(code=1) from exc
    if prompt is None:
        _print_banner(console, agent)

    # One event loop for the whole session (never one per turn) — see
    # _SESSION_LOOP / _shutdown_session_loop.
    global _SESSION_LOOP
    _SESSION_LOOP = loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Headless one-shot (`-p/--prompt`): run a single task, no REPL, and exit with a code that
    # reflects the outcome (0 = completed cleanly, non-zero otherwise) so it composes in scripts.
    if prompt is not None:
        raise typer.Exit(code=_run_one_shot(loop, functools.partial(agent.astream_turn, prompt)))

    # zakpick advisory state for this session: [count of clean un-escalated deep_code turns,
    # 1 once the one-time nudge has fired]. See _maybe_zakpick_advisory.
    zakpick_advisory = [0, 0]

    while True:
        try:
            line = read_prompt(console)
        except (EOFError, KeyboardInterrupt):
            notice_info(console, f"session closed {GLYPHS['dash']} goodbye")
            break

        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("/"):
            command = stripped.split(maxsplit=1)[0].lower()
            if command in ("/exit", "/quit"):
                notice_info(console, f"session closed {GLYPHS['dash']} goodbye")
                break
            if command != "/clear":
                # One blank under the echoed prompt before command output. /clear is
                # excluded because its notice_info (like the close bookend) prints
                # its own leading blank; streamed turns get theirs from the
                # renderer's first _gap().
                console.print()
            if command == "/help":
                _print_help(console)
                continue
            if command == "/model":
                if getattr(agent, "_zakpick", False):
                    # default_model was rewritten to the concrete startup model; show the live
                    # per-task table + the one lever instead of a single slug.
                    from zakcode.providers.routing import describe_zakpick

                    console.print(margin(Text(describe_zakpick(agent.settings))))
                    console.print(
                        margin(
                            Text(
                                "assign a model per category with ZAKCODE_ZAKPICK_MODELS "
                                '(JSON, e.g. {"deep_code":{"model":"qwen3:32b","source":"local"}})',
                                style="dim",
                            )
                        )
                    )
                else:
                    console.print(margin(Text(agent.settings.default_model)))
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
                    enable_rules=not no_rules,
                    extra_skill_dirs=extra_skill_dirs,
                    extra_workspace_roots=extra_roots,
                )
                # A fresh session resets the per-session zakpick advisory (count + once-shown
                # latch), so it never leaks across the documented session boundary.
                zakpick_advisory[:] = [0, 0]
                notice_info(console, "started a fresh session")
                continue
            if command == "/cost":
                usage = agent.session.cumulative_usage()
                dot = f" {GLYPHS['dot']} "
                # Surface cached prompt tokens when the backend reports them (a subset
                # of prompt_tokens billed at the cache discount) — the visible signal
                # that prompt caching is actually hitting (Anthropic explicit; OpenAI/
                # Groq automatic). Absent => 0 => no annotation, no clutter.
                cached_note = (
                    f", {usage.cache_read_tokens} cached" if usage.cache_read_tokens else ""
                )
                console.print(
                    margin(
                        Text.assemble(
                            (f"total={usage.total_tokens} tok", "arg.value"),
                            (
                                f"  (prompt {usage.prompt_tokens}{cached_note} / "
                                f"completion {usage.completion_tokens})",
                                "notice.dim",
                            ),
                            (dot, "notice.dim"),
                            (f"${usage.cost_usd:.4f}", "arg.value"),
                        )
                    )
                )
                # Per-model attribution: only worth showing once a session actually spanned more
                # than one model (e.g. zakpick routed easy vs hard turns differently). Untagged
                # entries (empty key) are dropped from the breakdown.
                by_model = {m: u for m, u in agent.session.usage_by_model().items() if m}
                if len(by_model) >= 2:
                    console.print(margin(Text("by model:", style="notice.dim")))
                    for model_name, model_usage in by_model.items():
                        console.print(
                            margin(
                                Text.assemble(
                                    (f"  {model_name}", "arg.value"),
                                    (dot, "notice.dim"),
                                    (f"{model_usage.total_tokens} tok", "notice.dim"),
                                    (dot, "notice.dim"),
                                    (f"${model_usage.cost_usd:.4f}", "arg.value"),
                                )
                            )
                        )
                    if getattr(agent, "_zakpick", False):
                        console.print(
                            margin(
                                Text(
                                    "zakpick routes per task; compaction & sub-agent costs are "
                                    "not broken out here. (a 'vs all-deep' savings estimate lands "
                                    "with the cost-metadata seam)",
                                    style="notice.dim",
                                )
                            )
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
                _dim(
                    console,
                    "compacted older history into a summary." if did else "nothing to compact yet.",
                )
                continue
            # A bare /<skill-name> RUNS the skill (Claude Code slash semantics): the body —
            # plus the [arguments: …] frame when trailing text was given — becomes THIS
            # turn's user message, and control falls through to the shared streaming path
            # below. No second "describe your task" message needed.
            skill_args = stripped[len(command) :].strip()
            skill = _skill_command_turn(console, agent, command.lstrip("/"), args=skill_args)
            if skill.handled:
                if skill.turn_text is None:
                    continue  # denied / unreadable — the notice already rendered
                stripped = skill.turn_text  # the skill body is the turn; stream it below
            else:
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
                    console.print(
                        margin(Text(cmd_result.output, style="err" if cmd_result.is_error else ""))
                    )
                    continue
                _dim(console, f"{command} is not yet supported.")
                continue

        # Stream the model response token by token through the renderer. A fresh
        # renderer per turn keeps the text/usage buffers from leaking across turns.
        renderer = StreamRenderer(console=console)
        try:
            _run_streamed_turn(console, functools.partial(agent.astream_turn, stripped), renderer)
        except ProviderError as exc:
            notice_error(console, "provider error", str(exc))
            continue
        _maybe_zakpick_advisory(console, renderer.last_done, zakpick_advisory)
        # Cosmetic Claude Code statusLine (opt-in; no-op unless configured). After the turn,
        # in-process only — it never gated or slowed the turn that just ran.
        _maybe_render_status_line(console, agent)

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

    g = GLYPHS
    rows = [
        ("server", base_url),
        ("model", model if model else "(server default)"),
        ("session", session_id),
    ]
    hint = Text.assemble(
        (f"server mode: turns run headless {g['dot']} ", "banner.hint"),
        ("/exit to quit", "banner.hint"),
    )
    console.print(_welcome_panel(console, rows, hint))

    # One event loop for the whole session (never one per turn) — see
    # _SESSION_LOOP / _shutdown_session_loop.
    global _SESSION_LOOP
    _SESSION_LOOP = loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    while True:
        try:
            line = read_prompt(console)
        except (EOFError, KeyboardInterrupt):
            notice_info(console, f"session closed {g['dash']} goodbye")
            break
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower() in ("/exit", "/quit"):
            notice_info(console, f"session closed {g['dash']} goodbye")
            break

        renderer = StreamRenderer(console=console)
        try:
            _run_streamed_turn(
                console,
                functools.partial(_server_turn_stream, base_url, stripped, session_id, model),
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
        help="Workspace root the served mind loads identity/rules/skills from.",
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


@app.command()
def drive(
    workspace: str = typer.Option(
        ...,
        "--workspace",
        "-w",
        help="Served-mind workspace root — where .current-session (the watch 'current' "
        "alias source) is written. Must match the workspace 'serve' was pointed at.",
    ),
    base_url: str = typer.Option(
        "http://127.0.0.1:8000",
        "--base-url",
        help="Base URL of the running zakcode serve daemon to drive.",
    ),
    boot_message: str = typer.Option(
        "Continue.",
        "--boot-message",
        help="First message that opens the autonomous loop (e.g. the mind's boot cue).",
    ),
    continue_message: str = typer.Option(
        "Continue.",
        "--continue-message",
        envvar="ZAKCODE_CONTINUE_MESSAGE",
        help="Message sent on every turn AFTER the first (the perpetual continuation cue). "
        "Defaults to 'Continue.'. A weak mind that no-ops on a bare 'Continue.' needs a "
        "concrete per-turn directive here — this is the one lever that shapes what each "
        "driven turn actually does. Also settable via the ZAKCODE_CONTINUE_MESSAGE env var.",
    ),
    resume_message: str | None = typer.Option(
        None,
        "--resume-message",
        help="Message for the turn after a provider_error stop (default: a built-in cue "
        "to re-check and resume the interrupted work; '{error}' expands to the redacted "
        "provider detail).",
    ),
    model: str | None = typer.Option(
        None, "--model", "-m", help="Model override applied to every driven turn."
    ),
    max_turns: int | None = typer.Option(
        None, "--max-turns", help="Stop after N turns (default: perpetual, until signalled)."
    ),
    nudge_file: str | None = typer.Option(
        None,
        "--nudge-file",
        help="Viewer-suggestion file (relative to the workspace, e.g. .nudge) the driver "
        "folds once into the next turn's preamble, then deletes. Off when unset.",
    ),
    inter_turn_delay: float = typer.Option(
        0.0,
        "--inter-turn-delay",
        envvar="ZAKCODE_INTER_TURN_DELAY",
        help="Seconds to pause between turns to pace the agent under a provider rate limit. "
        "0 = no pacing (fastest, but may hit the provider's tokens/requests-per-minute limit); "
        "higher spreads token usage over time and reduces rate-limit stalls. A per-agent 'speed' "
        "setting. Also settable via the ZAKCODE_INTER_TURN_DELAY environment variable.",
    ),
) -> None:
    """Drive an autonomous, watchable mind against a running ``zakcode serve`` daemon.

    The autonomous half of ``serve``: it creates a session, records it in
    ``<workspace>/.current-session``, then keeps a turn always running so a ``/watch``
    client sees a live stream. Each turn self-continues via the served mind's own
    ``TURN_END``/Stop-hook veto; the driver only supervises liveness (backoff on a
    daemon hiccup, recreate the session if the daemon was restarted) — it never decides
    what the mind does. The bearer token, if any, is read from ``ZAKCODE_AUTH_TOKEN``,
    the same source ``serve`` uses.
    """
    try:
        from zakcode.server.client import ServerClient
        from zakcode.server.driver import ServeDriver
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        console.print(
            "[red]The server extra is not installed.[/red] "
            "Install it with: [bold]pip install 'zakcode[server]'[/bold]"
        )
        raise typer.Exit(code=1) from exc

    auth_token = os.environ.get("ZAKCODE_AUTH_TOKEN") or None
    client = ServerClient(base_url, auth_token=auth_token)
    driver = ServeDriver(
        client,
        workspace,
        boot_message=boot_message,
        continue_message=continue_message,
        model=model,
        max_turns=max_turns,
        nudge_file=nudge_file,
        inter_turn_delay=max(0.0, inter_turn_delay),
    )
    if resume_message is not None:
        # Override the driver's built-in resume cue only when the flag was given — the
        # default lives on ServeDriver (not importable at module scope: optional extra).
        driver.resume_message = resume_message
    console.print(
        f"[bold]Zak Code[/bold] {__version__} — driving watched mind at {base_url} "
        f"(workspace: {workspace})"
    )

    async def _runner() -> None:
        loop = asyncio.get_running_loop()
        # Graceful stop on SIGTERM/SIGINT (systemd sends SIGTERM). Not implemented on
        # Windows event loops — harmless there; Ctrl+C still unwinds via KeyboardInterrupt.
        with contextlib.suppress(NotImplementedError):
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, driver.request_stop)
        try:
            await driver.run()
        finally:
            await client.aclose()

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_runner())


def main() -> None:
    """Console-script entry point (see ``[project.scripts]`` in pyproject.toml)."""
    app()


if __name__ == "__main__":
    main()
