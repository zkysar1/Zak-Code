"""Zak Code — a clean-room, vendor-agnostic, API-first agentic coding tool.

The :mod:`zakcode` package is the *core engine*: an importable library exposing the
agent loop, tools, sessions, context management, providers, and extension surfaces.
The CLI (:mod:`zakcode.cli`) and HTTP server (:mod:`zakcode.server`) are thin clients
of this core. See ``docs/ARCHITECTURE.md``.
"""

from zakcode.version import __version__

__all__ = ["__version__"]
