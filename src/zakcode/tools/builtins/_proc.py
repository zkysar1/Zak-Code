"""Shared subprocess runner for the shell tools (bash / powershell).

Spawns a child via :mod:`asyncio` so that BOTH a cancelled turn (a WebSocket interrupt or
client disconnect that cancels ``astream_turn``) and a timeout tear down the **whole
process tree** instead of orphaning grandchildren that hold ports / file locks. The
previous ``subprocess.run``-via-``asyncio.to_thread`` approach kept no process handle: a
``to_thread`` worker can't be interrupted (so a cancel just abandons the thread while the
child runs on), and ``subprocess.run``'s own timeout terminates only the *parent*, leaving
the tree alive. The group-spawn + tree-teardown primitives are shared with the hook runners
and the MCP transport via :mod:`zakcode._subprocess`. (audit3 #3 / audit4 #2 / #3)
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from collections.abc import Iterable
from typing import Any

from zakcode._subprocess import CommandTimeout, new_group_kwargs, terminate_process_tree

__all__ = ["CommandTimeout", "run_capturing"]


async def run_capturing(
    *,
    argv: list[str] | None = None,
    shell_command: str | None = None,
    cwd: str,
    timeout: float,
    stdin_text: str | None = None,
    extra_env: dict[str, str] | None = None,
    drop_env: Iterable[str] | None = None,
) -> tuple[str, int]:
    """Run a child to completion, capturing combined stdout+stderr; enforce ``timeout``.

    Exactly one of ``argv`` (direct exec) or ``shell_command`` (platform shell) must be
    given. Returns ``(combined_output, exit_code)``. On timeout raises
    :class:`CommandTimeout`; on cancellation re-raises ``CancelledError`` — in BOTH cases
    the child's entire process tree is killed first, never orphaned. ``extra_env`` is overlaid
    on the inherited environment (e.g. ``HTTP(S)_PROXY`` for the egress sandbox);
    ``drop_env`` names are then REMOVED (the provider-key scrub — applied last so an
    overlay can never resurrect a scrubbed credential).
    """
    if (argv is None) == (shell_command is None):
        raise ValueError("exactly one of argv or shell_command is required")

    stdin = subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL
    # Suppress child-emitted ANSI color: the combined stdout+stderr is fed straight to the
    # model, and raw escape codes are token-noise it can't use. Inherit the full parent env
    # (unchanged behavior) and add the standard no-color signals most CLIs honor. (#5)
    child_env = {**os.environ, "NO_COLOR": "1", "TERM": "dumb", **(extra_env or {})}
    for name in drop_env or ():
        child_env.pop(name, None)
    spawn_kwargs: dict[str, Any] = {
        "cwd": cwd,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "stdin": stdin,
        "env": child_env,
        **new_group_kwargs(),
    }
    if shell_command is not None:
        proc = await asyncio.create_subprocess_shell(shell_command, **spawn_kwargs)
    else:
        assert argv is not None
        proc = await asyncio.create_subprocess_exec(*argv, **spawn_kwargs)

    input_bytes = stdin_text.encode("utf-8") if stdin_text is not None else None
    try:
        out_bytes, _ = await asyncio.wait_for(proc.communicate(input=input_bytes), timeout=timeout)
    except (TimeoutError, asyncio.CancelledError) as exc:
        # wait_for cancelled communicate() but the child is still running — kill the tree.
        await terminate_process_tree(proc)
        if isinstance(exc, asyncio.CancelledError):
            raise  # a turn cancel: propagate after teardown
        raise CommandTimeout from exc
    output = out_bytes.decode("utf-8", errors="replace") if out_bytes else ""
    return output, proc.returncode if proc.returncode is not None else -1
