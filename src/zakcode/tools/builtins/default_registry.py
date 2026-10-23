"""Factory for the default tool registry (all built-in tools, pre-registered)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from zakcode.tools.base import ToolRegistry
from zakcode.tools.builtins._secrets import SecretsProvider
from zakcode.tools.builtins.await_user import AwaitUserTool
from zakcode.tools.builtins.bash import BashTool
from zakcode.tools.builtins.deep_think import DeepThinkTool
from zakcode.tools.builtins.edit import EditFileTool
from zakcode.tools.builtins.glob import GlobTool
from zakcode.tools.builtins.grep import GrepTool
from zakcode.tools.builtins.images import CreateChartImageTool, InspectImageTool, SaveImageTool
from zakcode.tools.builtins.list_dir import ListDirTool
from zakcode.tools.builtins.office import CreateDocxTool, CreateXlsxTool, ReadDocxTool, ReadXlsxTool
from zakcode.tools.builtins.pdf import CreatePdfTool, ReadPdfTool
from zakcode.tools.builtins.plan_recall import PlanRecallTool
from zakcode.tools.builtins.powershell import PowerShellTool
from zakcode.tools.builtins.read_file import ReadFileTool
from zakcode.tools.builtins.schedule_wakeup import ScheduleWakeupTool
from zakcode.tools.builtins.secret_names import SecretNamesTool
from zakcode.tools.builtins.task_output import TaskOutputTool
from zakcode.tools.builtins.task_stop import TaskStopTool
from zakcode.tools.builtins.update_plan import UpdatePlanTool
from zakcode.tools.builtins.web_fetch import WebFetchTool
from zakcode.tools.builtins.web_search import WebSearchTool
from zakcode.tools.builtins.write_file import WriteFileTool

if TYPE_CHECKING:
    from zakcode.config import Settings

#: Built-ins a session rarely needs. The main agent keeps them out of the model's tool list
#: until it loads one with ``tool_search`` (ADR-0228). On 2026-09-23 their schemas were 5,063
#: of the built-ins' 26,970 JSON characters, sent on every model call, and none of them had
#: been called in a served workspace's CLI log (17 MB) or in the worker session logs:
#: they appeared only on two lines listing tool names. The creators are here, and so are the
#: document readers (``read_docx`` / ``read_xlsx`` / ``read_pdf``): small models confuse a
#: document reader with ``read_file`` and burn a turn reaching for ``read_docx`` on a ``.py``
#: file, and a coding session rarely has an Office file to read. Hiding a tool only drops it
#: from the advertised list — a hidden tool stays DISPATCHABLE when named, so the server's
#: upload prompt (which names its reader via ``_reader_tool_for_artifact``) still works, and
#: ``tool_search`` surfaces a reader on demand for an unannounced file. The image reader
#: (``inspect_image``) stays exposed — it is not confusable with ``read_file``. This factory
#: still registers every tool active, so a sub-agent's registry, which has no ``tool_search``,
#: keeps them all.
ON_REQUEST_TOOLS: tuple[str, ...] = (
    "create_docx",
    "create_xlsx",
    "create_pdf",
    "save_image",
    "create_chart_image",
    "read_docx",
    "read_xlsx",
    "read_pdf",
)


#: Extension -> the on-request document reader that handles it. A hidden reader (above) is
#: re-exposed by :func:`workspace_doc_readers` when the workspace actually holds that type, so a
#: session with a real document keeps its reader in the list while a session with none keeps the
#: smaller, less-confusable menu. (``.pdf`` has no server upload-prompt mapping, but a ``.pdf``
#: present in the workspace still re-exposes ``read_pdf`` for symmetry.)
DOC_READER_BY_SUFFIX: dict[str, str] = {
    ".docx": "read_docx",
    ".xlsx": "read_xlsx",
    ".xlsm": "read_xlsx",
    ".pdf": "read_pdf",
}

#: Directories :func:`workspace_doc_readers` never descends into -- vendored or generated trees
#: that would not hold a user's document and can be huge. Dot-directories are pruned separately.
_SCAN_PRUNE_DIRS: frozenset[str] = frozenset(
    {"node_modules", "venv", "dist", "build", "target", "__pycache__", "site-packages"}
)


def workspace_doc_readers(workspace_root: str | Path, *, max_entries: int = 10000) -> set[str]:
    """The document readers to EXPOSE because the workspace already holds that file type.

    The main agent hides the document readers by default (see :data:`ON_REQUEST_TOOLS`): a small
    model otherwise reaches for ``read_docx`` on a ``.py`` file and loops. This bounded walk
    (dot-directories and :data:`_SCAN_PRUNE_DIRS` pruned, a hard ``max_entries`` cap, early exit
    once every reader is matched) re-exposes a reader the moment its document type is present, so
    capability is never removed -- only deferred until it is relevant. A hidden reader also stays
    dispatchable when named and loadable via ``tool_search``, so a miss here is recoverable.
    """
    targets = set(DOC_READER_BY_SUFFIX.values())
    found: set[str] = set()
    seen = 0
    try:
        for _dirpath, dirnames, filenames in os.walk(Path(workspace_root), followlinks=False):
            dirnames[:] = [
                d for d in dirnames if not d.startswith(".") and d not in _SCAN_PRUNE_DIRS
            ]
            for filename in filenames:
                reader = DOC_READER_BY_SUFFIX.get(Path(filename).suffix.lower())
                if reader is not None:
                    found.add(reader)
                    if found >= targets:
                        return found
            seen += len(filenames)
            if seen >= max_entries:
                break
    except OSError:
        # Agent init must never crash on an unreadable workspace; expose what was found so far
        # (a miss is recoverable via tool_search), never raise.
        pass
    return found


def default_registry(settings: Settings | None = None) -> ToolRegistry:
    """Build a :class:`ToolRegistry` populated with every built-in tool.

    Each tool is registered under a small set of guessable aliases (POSIX muscle-memory like
    ``cat`` / ``ls`` / ``grep`` aliases) so a model that guesses a sibling name still resolves
    to the right tool ("learn one, guess the rest"). Aliases are a SILENT fallback — they are
    not advertised in the prompt (that would cost tokens and dilute the one-canonical-name
    signal); the registry just routes them. See ToolRegistry.register's collision guard.

    ``settings`` selects the web-search backend (``ZAKCODE_SEARCH_BACKEND``, default DuckDuckGo);
    ``None`` builds the zero-config default so existing no-arg callers keep working. The web
    tools are always registered — a missing optional dep surfaces as a clean tool error, never
    an import failure here.
    """
    from zakcode.search import make_search_backend

    registry = ToolRegistry()
    # ADR-0190: the canonical names are Claude Code's (Read, Write, Edit, LS, Glob, Grep, Bash,
    # WebFetch, WebSearch, Skill, ScheduleWakeup) — the names every Claude-Code-shaped prompt,
    # skill and hook already uses, so they run here verbatim. The pre-0190 snake_case names
    # stay as SILENT aliases (an older prompt, skill body or config still resolves).
    registry.register(ReadFileTool(), aliases=["read_file", "read", "cat"])
    registry.register(WriteFileTool(), aliases=["write_file", "write"])
    registry.register(EditFileTool(), aliases=["edit_file", "edit"])
    registry.register(ListDirTool(), aliases=["list_dir", "ls", "dir"])
    registry.register(GlobTool(), aliases=["glob", "find"])
    registry.register(GrepTool(), aliases=["grep", "search", "rg"])
    registry.register(ReadDocxTool(), aliases=["read_word"])
    registry.register(ReadXlsxTool(), aliases=["read_excel"])
    registry.register(CreateDocxTool(), aliases=["docx", "word"])
    registry.register(CreateXlsxTool(), aliases=["xlsx", "excel"])
    registry.register(ReadPdfTool(), aliases=["readpdf"])
    registry.register(CreatePdfTool(), aliases=["pdf", "makepdf"])
    registry.register(InspectImageTool(), aliases=["image_info"])
    registry.register(SaveImageTool(), aliases=["image"])
    registry.register(CreateChartImageTool(), aliases=["chart"])
    # TodoWrite / TodoRead: Claude Code's names for the plan pair; the tools accept its shape.
    registry.register(UpdatePlanTool(), aliases=["plan", "todo", "TodoWrite"])
    # No bare "recall" alias: that name is reserved by the persistence boundary (the harness
    # ships no cross-session memory tool; a host framework attaches its own recall
    # through the seams —
    # docs/PERSISTENCE-BOUNDARY.md). plan_recall reads THIS plan's record only.
    registry.register(PlanRecallTool(), aliases=["plan_history", "TodoRead"])
    registry.register(AwaitUserTool(), aliases=["ask_user", "wait_for_user"])
    registry.register(ScheduleWakeupTool(), aliases=["schedule_wakeup", "schedulewakeup", "wakeup"])
    registry.register(BashTool(), aliases=["bash", "sh", "shell"])
    # Background commands (ADR-0191): Bash(run_in_background=true) starts one; these read it
    # back and kill it — Claude Code's names, since they ARE Claude Code's tools.
    registry.register(TaskOutputTool(), aliases=["task_output"])
    registry.register(TaskStopTool(), aliases=["task_stop"])
    registry.register(PowerShellTool(), aliases=["pwsh"])
    web_allowlist = settings.web_allowed_domains if settings else None
    # One provider instance shared by every secrets-aware tool, so web_fetch's
    # substitution, web_search's query screen, and secret_names' listing can never
    # disagree about the source.
    secrets = SecretsProvider(
        settings.secrets_file if settings else None,
        usage_path=settings.secrets_usage_file if settings else None,
    )
    registry.register(
        WebSearchTool(make_search_backend(settings), secrets=secrets),
        aliases=["web_search", "websearch"],
    )
    registry.register(
        WebFetchTool(allowed_domains=web_allowlist, secrets=secrets),
        aliases=["web_fetch", "fetch", "webfetch"],
    )
    registry.register(SecretNamesTool(secrets), aliases=["secrets", "list_secrets"])
    # Opt-in deliberation: only WORKS when a Sampler is wired (the main Agent loop); a bare or
    # delegated loop exposes it but it returns a clean "unavailable" error.
    registry.register(DeepThinkTool(), aliases=["deliberate"])
    return registry
