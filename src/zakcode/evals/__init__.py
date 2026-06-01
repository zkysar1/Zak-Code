"""Evaluation harness (M9) — test the agent like software.

Zak Code is exercised by **behavioral** probes that drive the *real* agent loop with
deterministic, no-network :class:`ScriptedProvider`s and assert end-to-end outcomes
(completion detection, safety rejection, plan-mode read-only enforcement, doom-loop
halting, partial-failure recovery, long-horizon compaction). The harness is importable
(`from zakcode.evals import ...`), runnable (`zakcode eval`), and itself covered by
pytest — so it can gate the project's own changes.

This package is intentionally dependency-light and core-respecting: it consumes the
same public contracts every client uses (the :class:`~zakcode.Agent` facade, the
:class:`~zakcode.providers.base.Provider` ABC, the frozen message/tool types) and adds
no new agent behavior of its own.
"""

from __future__ import annotations

from zakcode.evals.harness import (
    EvalCase,
    EvalReport,
    EvalResult,
    ScriptedProvider,
    call_tool,
    hermetic_env,
    make_agent,
    reply,
    run_case,
    run_evals,
    run_evals_sync,
)

__all__ = [
    "EvalCase",
    "EvalReport",
    "EvalResult",
    "ScriptedProvider",
    "call_tool",
    "hermetic_env",
    "make_agent",
    "reply",
    "run_case",
    "run_evals",
    "run_evals_sync",
]
