"""Factory for the default tool registry (all built-in tools, pre-registered)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from zakcode.tools.base import ToolRegistry
from zakcode.tools.builtins._secrets import SecretsProvider
from zakcode.tools.builtins.bash import BashTool
from zakcode.tools.builtins.deep_think import DeepThinkTool
from zakcode.tools.builtins.edit import EditFileTool
from zakcode.tools.builtins.glob import GlobTool
from zakcode.tools.builtins.grep import GrepTool
from zakcode.tools.builtins.images import CreateChartImageTool, InspectImageTool, SaveImageTool
from zakcode.tools.builtins.list_dir import ListDirTool
from zakcode.tools.builtins.office import CreateDocxTool, CreateXlsxTool, ReadDocxTool, ReadXlsxTool
from zakcode.tools.builtins.pdf import CreatePdfTool, ReadPdfTool
from zakcode.tools.builtins.powershell import PowerShellTool
from zakcode.tools.builtins.read_file import ReadFileTool
from zakcode.tools.builtins.schedule_wakeup import ScheduleWakeupTool
from zakcode.tools.builtins.secret_names import SecretNamesTool
from zakcode.tools.builtins.update_plan import UpdatePlanTool
from zakcode.tools.builtins.web_fetch import WebFetchTool
from zakcode.tools.builtins.web_search import WebSearchTool
from zakcode.tools.builtins.write_file import WriteFileTool

if TYPE_CHECKING:
    from zakcode.config import Settings


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
    registry.register(ReadFileTool(), aliases=["read", "cat"])
    registry.register(WriteFileTool(), aliases=["write"])
    registry.register(EditFileTool(), aliases=["edit"])
    registry.register(ListDirTool(), aliases=["ls", "dir"])
    registry.register(GlobTool(), aliases=["find"])
    registry.register(GrepTool(), aliases=["search", "rg"])
    registry.register(ReadDocxTool(), aliases=["read_word"])
    registry.register(ReadXlsxTool(), aliases=["read_excel"])
    registry.register(CreateDocxTool(), aliases=["docx", "word"])
    registry.register(CreateXlsxTool(), aliases=["xlsx", "excel"])
    registry.register(ReadPdfTool(), aliases=["readpdf"])
    registry.register(CreatePdfTool(), aliases=["pdf", "makepdf"])
    registry.register(InspectImageTool(), aliases=["image_info"])
    registry.register(SaveImageTool(), aliases=["image"])
    registry.register(CreateChartImageTool(), aliases=["chart"])
    registry.register(UpdatePlanTool(), aliases=["plan", "todo"])
    # Claude Code's name too (ADR-0094): a Mind's loop calls ScheduleWakeup by that name.
    registry.register(ScheduleWakeupTool(), aliases=["ScheduleWakeup", "schedulewakeup", "wakeup"])
    registry.register(BashTool(), aliases=["sh", "shell"])
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
        WebSearchTool(make_search_backend(settings), secrets=secrets), aliases=["websearch"]
    )
    registry.register(
        WebFetchTool(allowed_domains=web_allowlist, secrets=secrets),
        aliases=["fetch", "webfetch"],
    )
    registry.register(SecretNamesTool(secrets), aliases=["secrets", "list_secrets"])
    # Opt-in deliberation: only WORKS when a Sampler is wired (the main Agent loop); a bare or
    # delegated loop exposes it but it returns a clean "unavailable" error.
    registry.register(DeepThinkTool(), aliases=["deliberate"])
    return registry
