"""Tool registry and built-in tools.

A small, sharp, composable tool set the agent uses to act on the workspace (read/write
files, run commands, search code). ``base.py`` defines the Tool contract, JSON-schema
exposure, and the registry; concrete tools live in ``builtins/``.

Status: scaffolded; the first tools land in milestone M0. See ``docs/ARCHITECTURE.md``.
"""
