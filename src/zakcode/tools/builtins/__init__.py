"""Built-in tools shipped with Zak Code (filesystem, shell, search).

Each tool is a small :class:`~zakcode.tools.base.Tool` subclass; :func:`default_registry`
assembles them into a ready-to-use :class:`~zakcode.tools.base.ToolRegistry`.
See ``docs/ARCHITECTURE.md``.
"""

from zakcode.tools.builtins.bash import BashTool
from zakcode.tools.builtins.default_registry import default_registry
from zakcode.tools.builtins.glob import GlobTool
from zakcode.tools.builtins.grep import GrepTool
from zakcode.tools.builtins.list_dir import ListDirTool
from zakcode.tools.builtins.read_file import ReadFileTool
from zakcode.tools.builtins.web_fetch import WebFetchTool
from zakcode.tools.builtins.web_search import WebSearchTool
from zakcode.tools.builtins.write_file import WriteFileTool

__all__ = [
    "BashTool",
    "GlobTool",
    "GrepTool",
    "ListDirTool",
    "ReadFileTool",
    "WebFetchTool",
    "WebSearchTool",
    "WriteFileTool",
    "default_registry",
]
