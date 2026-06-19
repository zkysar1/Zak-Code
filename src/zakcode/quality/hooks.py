"""A pluggable judge hook — the seam for nudging the engine's judgments later.

Register a global hook once (e.g. at config time) and every criteria-based judge — ``binary_judge``,
``pairwise_judge``, and everything built on them (the votes, the tournament, plan selection,
the completion critic) — runs it over its CRITERIA before judging. Use it to fold in an oracle's
result, project conventions, or a domain rubric WITHOUT touching the judging primitives. It is a
small, explicit surface (criteria in → criteria out) so a future nudge is a one-liner.

Fail-safe: a missing hook, a hook that raises, or a hook that returns a non-string is treated as the
IDENTITY (the criteria pass through unchanged) — a bad nudge can never break judging.
"""

from __future__ import annotations

from collections.abc import Callable

#: A nudge: given the criteria a judge is about to use, return augmented criteria.
JudgeHook = Callable[[str], str]

_hook: JudgeHook | None = None


def set_judge_hook(hook: JudgeHook | None) -> None:
    """Set the global judge hook (run over every judge's criteria); ``None`` clears it."""
    global _hook
    _hook = hook


def get_judge_hook() -> JudgeHook | None:
    """The currently-registered judge hook, or ``None``."""
    return _hook


def apply_judge_hook(criteria: str) -> str:
    """Run the registered hook over ``criteria`` — identity if none is set, it raises, or it returns
    a non-string (fail-safe)."""
    hook = _hook
    if hook is None:
        return criteria
    try:
        result = hook(criteria)
    except Exception:  # noqa: BLE001 — a bad nudge must never break judging
        return criteria
    return result if isinstance(result, str) else criteria


__all__ = ["JudgeHook", "apply_judge_hook", "get_judge_hook", "set_judge_hook"]
