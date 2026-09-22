"""The workspace's settings ``env`` block, handed to every child a session starts there (ADR-0212).

Claude Code's settings files carry an ``env`` object, and its contract is one sentence: the
variables are set "for every session and its subprocesses", the settings value beats a value
exported by the shell, and a per-machine ``settings.local.json`` beats the shared file. A
framework written for Claude Code keeps its own switches there because that block is the one
place that travels with the repository. Until this module Zak Code had no reader for it at
all, so every such switch was OFF in every shell call, hook and status line of a hosted
workspace, and nothing said so.

What this module is:

* ONE reader (:func:`settings_env`) for the same three files, in the same order, that the
  hook loader reads, so "which settings files count" has a single answer.
* Applied to CHILDREN only. The block is overlaid on the environment each child is given
  (:func:`zakcode.tools.builtins._proc.child_environment`, :func:`zakcode.hooks._hook_env`,
  and the framework-script calls in :mod:`zakcode.session.framework_signal`). The product's
  own process environment is never written, so a repository's settings file cannot reconfigure
  the harness that is hosting it, and two workspaces served by one process never see each
  other's variables.
* Below the product's own per-child variables and below the credential scrub. Each builder
  overlays the block first and then sets what it owns (``CLAUDE_PROJECT_DIR``, the no-colour
  pair, the egress proxy, ``BASH_ENV``) and removes provider keys LAST, exactly as before, so
  a block can neither point a child around the egress sandbox nor hand it a model credential.
* Fresh without a restart. The merged block is cached per workspace under the settings
  files' ``(path, mtime, size)`` signature, so a saved edit reaches the very next child for
  the price of three ``stat`` calls, as Claude Code applies a saved change to a running
  session.

What it is not: a trust decision. A workspace's settings file may already register commands
that run unconditionally (ADR-0025); a variable adds nothing a registered hook could not
``export`` for itself. Values are never logged, only names, because a block may hold a token.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
from pathlib import Path

logger = logging.getLogger("zakcode.workspace_env")

#: A portable environment variable name. Anything else in the block is reported and skipped:
#: a name with ``=`` or a NUL in it cannot be put in a child's environment at all.
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_Signature = tuple[tuple[str, int, int], ...]

#: workspace root -> (signature the block was read under, the merged block). Guarded by
#: ``_lock``: tools spawn from the event loop, the framework-signal calls from worker threads.
_cache: dict[str, tuple[_Signature, dict[str, str]]] = {}
_lock = threading.Lock()


def _read_blocks(workspace_root: Path) -> tuple[dict[str, str], list[str], bool]:
    """Merge the ``env`` objects of the workspace's settings files, later files winning.

    Returns ``(block, complaints, parse_failed)``. ``complaints`` name what was skipped and
    why, without any value. ``parse_failed`` is True when a file that exists could not be
    read as JSON, which the caller treats differently from a block that is merely absent.
    """
    # Imported here, not at module top: ``zakcode.hooks`` imports this module for its child
    # environment, and the loader lives under that package.
    from zakcode.hooks.settings_loader import _settings_candidates

    block: dict[str, str] = {}
    complaints: list[str] = []
    parse_failed = False
    for path in _settings_candidates(workspace_root):
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            complaints.append(f"{path}: not readable as JSON ({type(exc).__name__})")
            parse_failed = True
            continue
        env = data.get("env") if isinstance(data, dict) else None
        if env is None:
            continue
        if not isinstance(env, dict):
            complaints.append(f"{path}: `env` is not an object; ignored")
            continue
        for name, value in env.items():
            if not isinstance(name, str) or not _NAME.match(name):
                complaints.append(f"{path}: `{name}` is not a variable name; skipped")
            elif not isinstance(value, str) or "\x00" in value:
                # Claude Code's schema is string to string. A number or a boolean here is a
                # typo whose intended spelling ("1"? "true"?) this reader must not guess.
                complaints.append(f"{path}: the value of `{name}` is not a string; skipped")
            else:
                # Windows environment names are case-insensitive and ``os.environ`` holds them
                # upper-cased; a second spelling of one name would make the winner arbitrary.
                block[name.upper() if sys.platform == "win32" else name] = value
    return block, complaints, parse_failed


def settings_env(workspace_root: str | os.PathLike[str]) -> dict[str, str]:
    """The workspace's merged settings ``env`` block (a fresh dict; empty when there is none).

    A settings file that stops parsing keeps the LAST GOOD block in force, as the hook loader
    keeps the last good hooks (ADR-0079): a bad edit must not silently switch a framework's
    configuration off in the middle of a run. On a first read with nothing to fall back on,
    the files that do parse still count.
    """
    from zakcode.hooks.settings_loader import settings_hooks_signature

    root = Path(workspace_root)
    key = str(root)
    signature = settings_hooks_signature(root)
    with _lock:
        cached = _cache.get(key)
        if cached is not None and cached[0] == signature:
            return dict(cached[1])
    block, complaints, parse_failed = _read_blocks(root)
    with _lock:
        previous = _cache.get(key)
        if parse_failed and previous is not None:
            block = dict(previous[1])
        # Stored under the NEW signature either way, so a broken file is reported once and
        # not re-parsed for every child until someone saves it again.
        _cache[key] = (signature, block)
    for complaint in complaints:
        logger.warning("settings env: %s", complaint)
    if parse_failed and previous is not None:
        logger.warning("settings env: keeping the last good block for %s", key)
    if block:
        logger.info(
            "settings env: %d variable(s) for children of %s: %s",
            len(block),
            key,
            ", ".join(sorted(block)),
        )
    return dict(block)
