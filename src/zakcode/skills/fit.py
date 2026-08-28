"""Skill fit (ADR-0066): can each skill body sit in this model's context window at all?

The startup half of the loud block. The in-turn half is the loop's ``_verbatim_overflow``
— same arithmetic, so what ``zakcode info`` and the chat banner flag at start is exactly
what would end a turn as ``skill_too_large`` later: a body that, beside the system prompt,
leaves less than the answer room. The 50 % flag is a warning short of that: such a skill
loads, but only with most of the transcript compacted away.

Pure functions over ``(name, body)`` pairs and a token counter — no provider, no I/O —
so the verdicts are unit-testable and the same for every caller.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

#: Share of the window past which a skill is flagged: it loads, but crowds out the work.
FLAG_FRACTION = 0.5


@dataclass(frozen=True)
class SkillFit:
    """One skill measured against one window."""

    name: str
    tokens: int
    window: int
    #: ``ok`` | ``large`` (over half the window) | ``too_large`` (cannot load at all).
    verdict: str

    @property
    def share(self) -> float:
        """Body tokens as a fraction of the window (1.52 = 152 %)."""
        return self.tokens / self.window if self.window else 0.0

    def describe(self) -> str:
        """``/aspirations-precheck 49.7k tokens (152%) — cannot load on this model``."""
        size = f"{self.tokens / 1000:.1f}k" if self.tokens >= 1000 else str(self.tokens)
        line = f"/{self.name} {size} tokens ({self.share:.0%} of the window)"
        if self.verdict == "too_large":
            return line + " — cannot load on this model"
        if self.verdict == "large":
            return line + " — over half the window; loads only with the transcript compacted"
        return line


def measure_skill_fit(
    skills: Iterable[tuple[str, str]],
    *,
    count_tokens: Callable[[str], int],
    window: int,
    system_tokens: int = 0,
    reserve: int = 0,
) -> list[SkillFit]:
    """Measure every ``(name, body)`` against ``window``; worst first, then by name.

    ``too_large`` when ``tokens + system_tokens + reserve > window`` — the loop's own
    in-turn test, so start and turn never disagree; ``large`` when the body alone is over
    :data:`FLAG_FRACTION` of the window; else ``ok``. A body that cannot be counted is
    skipped rather than guessed at.
    """
    fits: list[SkillFit] = []
    for name, body in skills:
        try:
            tokens = int(count_tokens(body))
        except Exception:  # noqa: BLE001 — never block a start on a tokenizer hiccup
            continue
        if tokens + system_tokens + reserve > window:
            verdict = "too_large"
        elif tokens > window * FLAG_FRACTION:
            verdict = "large"
        else:
            verdict = "ok"
        fits.append(SkillFit(name=name, tokens=tokens, window=window, verdict=verdict))
    rank = {"too_large": 0, "large": 1, "ok": 2}
    fits.sort(key=lambda f: (rank[f.verdict], -f.tokens, f.name))
    return fits


def flagged(fits: Iterable[SkillFit]) -> list[SkillFit]:
    """The entries worth printing: everything that is not plainly ``ok``."""
    return [f for f in fits if f.verdict != "ok"]


__all__ = ["FLAG_FRACTION", "SkillFit", "flagged", "measure_skill_fit"]
