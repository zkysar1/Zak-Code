"""LLM-as-judge primitives for the quality engine.

The bet: several cheap, INDEPENDENT small-model judgments beat one (small *or* large). Quality from
a fixed-capability model comes from STRUCTURE, not a bigger model — so this layer offers the
primitives a quality loop composes:

* :func:`binary_judge` — one judge: does an artifact satisfy the criteria? → ``{approved, issues}``.
* :func:`vote_binary` — N INDEPENDENT judges (sampled for diversity), STRICT-majority vote. The
  self-consistency lever: a panel of cheap judges is more robust than one.
* :func:`pairwise_judge` — one judge: which of A / B better satisfies the criteria? Pairwise is far
  more reliable than absolute 1–10 scoring (RankPrompt), so it is the basis of SELECTION.
* :func:`best_of` — the best of N candidates by a pairwise ROUND-ROBIN (most wins). The selection
  primitive a best-of-N generator will stand on.

By design these are **LLM judges** (the project's call: a judge assesses QUALITY — completeness,
design, requirement coverage — which a deterministic oracle cannot). An oracle's result (tests
passed, lints clean) can later be folded in as additional ``criteria`` context, but the verdict is
the model's. All are vendor-agnostic, **fail-safe** (a judge error abstains — it never crashes the
caller), and use **json_object** mode so they sidestep litellm's Groq ``json_schema``→tool-calling
trap (the verdict is validated locally). Each returns its :class:`~zakcode.usage.Usage` so the
caller accounts the spend on its own session / shared budget.
"""

from __future__ import annotations

import asyncio
from typing import Literal

from pydantic import BaseModel

from zakcode.messages import Message
from zakcode.providers.base import Provider
from zakcode.providers.structured import coerce_structured, make_response_format
from zakcode.usage import Usage


class BinaryVerdict(BaseModel):
    """An approve/reject judgment; ``issues`` lists the unmet requirements when not approved."""

    approved: bool
    issues: str = ""


class PairwiseVerdict(BaseModel):
    """Which of two candidates (``"a"`` / ``"b"``) wins, or ``"tie"``; ``reason`` is one line."""

    winner: Literal["a", "b", "tie"]
    reason: str = ""


_BINARY_SCHEMA = {
    "type": "object",
    "properties": {"approved": {"type": "boolean"}, "issues": {"type": "string"}},
    "required": ["approved"],
    "additionalProperties": False,
}
_PAIRWISE_SCHEMA = {
    "type": "object",
    "properties": {
        "winner": {"type": "string", "enum": ["a", "b", "tie"]},
        "reason": {"type": "string"},
    },
    "required": ["winner"],
    "additionalProperties": False,
}

_BINARY_SYSTEM = (
    "You are an INDEPENDENT reviewer. You are given CRITERIA and an ARTIFACT. Judge ONLY whether "
    "the artifact satisfies EVERY part of the criteria — catch requirements that are missing, "
    "unaddressed, or only half-done; do NOT nitpick style or demand more than the criteria asks. "
    'Reply with ONLY a JSON object: {"approved": <true if it satisfies the criteria, else false>, '
    '"issues": "<the specific unmet requirements if not approved; else empty>"}. No prose, no '
    "markdown fence."
)
_PAIRWISE_SYSTEM = (
    "You are an INDEPENDENT reviewer comparing two candidate artifacts, A and B, against CRITERIA. "
    "Decide which one better satisfies the criteria AS A WHOLE — prefer the one that more "
    "completely and correctly meets the requirements; do not reward length or style. Reply with "
    'ONLY a JSON object: {"winner": "a" | "b" | "tie", "reason": "<one short sentence>"}. Use '
    '"tie" only when they are genuinely indistinguishable. No prose, no markdown fence.'
)

#: Per-side input cap — judging is about requirement COVERAGE, not re-reading every byte; a bound
#: keeps each judge call cheap no matter how large the artifact grew.
_MAX_CHARS = 4000


def _clip(text: str) -> str:
    text = text or ""
    return text if len(text) <= _MAX_CHARS else text[:_MAX_CHARS] + "\n…[truncated]"


async def binary_judge(
    provider: Provider,
    *,
    criteria: str,
    artifact: str,
    system: str = _BINARY_SYSTEM,
    temperature: float = 0.0,
) -> tuple[BinaryVerdict, Usage]:
    """One judge: does ``artifact`` satisfy ``criteria``? Returns ``(verdict, usage)``.

    A FRESH context (just the criteria + artifact, never a conversation transcript), json_object
    mode, validated locally. FAIL-OPEN: any error (provider failure, unparseable output) returns
    ``approved=True`` — a judge must never trap its caller.
    """
    payload = (
        f"<criteria>\n{_clip(criteria)}\n</criteria>\n\n<artifact>\n{_clip(artifact)}\n</artifact>"
    )
    try:
        result = await provider.acomplete(
            [Message.user(payload)],
            system=system,
            response_format=make_response_format(None),
            temperature=temperature,
        )
        data = coerce_structured(result.text, schema=_BINARY_SCHEMA)
        verdict = BinaryVerdict(
            approved=bool(data.get("approved", True)), issues=str(data.get("issues", ""))
        )
        return verdict, result.usage
    except Exception:  # noqa: BLE001 — fail-open: a judge failure must never trap the caller
        return BinaryVerdict(approved=True), Usage()


async def pairwise_judge(
    provider: Provider,
    *,
    criteria: str,
    a: str,
    b: str,
    system: str = _PAIRWISE_SYSTEM,
    temperature: float = 0.0,
) -> tuple[PairwiseVerdict, Usage]:
    """One judge: does ``a`` or ``b`` better satisfy ``criteria``? Returns ``(verdict, usage)``.

    Pairwise comparison (more reliable than absolute scoring). FAIL-SAFE: any error returns
    ``"tie"`` (no opinion), so a flaky judge never derails a selection.
    """
    payload = (
        f"<criteria>\n{_clip(criteria)}\n</criteria>\n\n"
        f"<candidate_a>\n{_clip(a)}\n</candidate_a>\n\n"
        f"<candidate_b>\n{_clip(b)}\n</candidate_b>"
    )
    try:
        result = await provider.acomplete(
            [Message.user(payload)],
            system=system,
            response_format=make_response_format(None),
            temperature=temperature,
        )
        data = coerce_structured(result.text, schema=_PAIRWISE_SCHEMA)
        winner = data.get("winner")
        if winner not in ("a", "b", "tie"):
            winner = "tie"
        return PairwiseVerdict(winner=winner, reason=str(data.get("reason", ""))), result.usage
    except Exception:  # noqa: BLE001 — fail-safe: an error is "no opinion" (tie)
        return PairwiseVerdict(winner="tie"), Usage()


async def vote_binary(
    provider: Provider,
    *,
    criteria: str,
    artifact: str,
    n: int = 3,
    temperature: float = 0.7,
) -> tuple[BinaryVerdict, Usage]:
    """``n`` INDEPENDENT binary judges (sampled at ``temperature`` for diversity), STRICT-majority
    vote on ``approved``; the dissenters' ``issues`` are unioned (deduped, order-preserved) so the
    caller sees every concern raised. Returns ``(verdict, total_usage)``. ``n`` is clamped to ≥ 1
    (``n=1`` is one judge, no vote)."""
    n = max(1, n)
    results = await asyncio.gather(
        *(
            binary_judge(provider, criteria=criteria, artifact=artifact, temperature=temperature)
            for _ in range(n)
        )
    )
    verdicts = [v for v, _ in results]
    usage = sum((u for _, u in results), Usage())
    approvals = sum(1 for v in verdicts if v.approved)
    approved = approvals * 2 > n  # strict majority approves
    issues = "; ".join(
        dict.fromkeys(v.issues.strip() for v in verdicts if not v.approved and v.issues.strip())
    )
    return BinaryVerdict(approved=approved, issues=issues), usage


async def best_of(
    provider: Provider,
    *,
    criteria: str,
    candidates: list[str],
    temperature: float = 0.0,
) -> tuple[int, Usage]:
    """The INDEX of the best candidate by a pairwise ROUND-ROBIN against ``criteria``.

    Every unordered pair is judged once; a win scores 1, a tie scores 0.5 to both; the candidate
    with the most wins is chosen, ties broken toward the EARLIEST index (stable). Pairwise, not
    absolute scoring, per the research. With ≤ 1 candidate, returns ``0`` and makes no calls.
    Returns ``(winning_index, total_usage)``.
    """
    if len(candidates) <= 1:
        return 0, Usage()
    pairs = [(i, j) for i in range(len(candidates)) for j in range(i + 1, len(candidates))]
    results = await asyncio.gather(
        *(
            pairwise_judge(
                provider,
                criteria=criteria,
                a=candidates[i],
                b=candidates[j],
                temperature=temperature,
            )
            for i, j in pairs
        )
    )
    wins = [0.0] * len(candidates)
    usage = Usage()
    for (i, j), (verdict, u) in zip(pairs, results, strict=True):
        usage = usage + u
        if verdict.winner == "a":
            wins[i] += 1.0
        elif verdict.winner == "b":
            wins[j] += 1.0
        else:
            wins[i] += 0.5
            wins[j] += 0.5
    best = max(range(len(candidates)), key=lambda k: (wins[k], -k))  # most wins, earliest index
    return best, usage


__all__ = [
    "BinaryVerdict",
    "PairwiseVerdict",
    "best_of",
    "binary_judge",
    "pairwise_judge",
    "vote_binary",
]
