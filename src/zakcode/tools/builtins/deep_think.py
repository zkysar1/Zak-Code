"""``deep_think`` — deliberate on ONE hard question via best-of-N self-fusion.

Sample several *independent* answers to a single hard question from a capable model, then make
one more pass that reads them all and writes the single best, combined answer. This is "self
fusion": the lift comes from sampling diversity **and** the synthesis step itself (OpenRouter's
*Fusion beats Frontier* reports a +6.7-point jump from pairing a model with *itself*), and it is
the "generate → critique → synthesize" pattern in one bounded tool.

It is the model's **opt-in** way to spend more compute on a genuinely hard sub-problem — the
agent decides when a question is worth the extra calls (architecture decisions, a subtle bug's
root cause, a correctness-critical answer). It owns no escalation policy and never fires
automatically. It uses the agent's strongest configured model (under zakpick, the ``deep_code``
category) via the :class:`~zakcode.tools.base.Sampler` seam, and its spend is attributed in
``/cost`` and counted against the turn budget like any model call — so the cost is visible and
bounded, never a hidden surprise.
"""

from __future__ import annotations

import asyncio
from typing import Any

from zakcode.config import PermissionTier
from zakcode.tools.base import (
    ConcurrencyClass,
    SampleCutOff,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)

#: Default / max independent answers to synthesize. Bounded so a single call can't fan out
#: unboundedly; the synthesis step needs ≥2 to have anything to combine.
_DEFAULT_SAMPLES = 3
_MAX_SAMPLES = 5
#: Candidates are sampled with temperature for diversity; the synthesis is deterministic.
_CANDIDATE_TEMP = 0.7

_CANDIDATE_SYSTEM = (
    "You are deliberating on one hard question. Reason carefully and independently, then give "
    "your single best, complete, and correct answer. Be concrete and specific; do not hedge or "
    "ask for clarification."
)
_SYNTH_SYSTEM = (
    "You synthesize several independent answers to one question into the single best answer. "
    "You are rigorous: you weigh the evidence, not the wording."
)


def _synthesis_prompt(question: str, candidates: list[str]) -> str:
    """The prompt that fuses ``candidates`` into one best answer."""
    blocks = "\n\n".join(f"--- ANSWER {i + 1} ---\n{c}" for i, c in enumerate(candidates))
    return (
        f"QUESTION:\n{question}\n\n"
        f"Here are {len(candidates)} independent answers to that question:\n\n{blocks}\n\n"
        "Analyze them: where do they agree (likely correct), where do they disagree or "
        "contradict (decide who is right and why), and what unique insight or error does each "
        "contain? Then write the SINGLE best, most complete and correct answer to the question, "
        "incorporating the strongest reasoning and discarding the mistakes. Output ONLY that "
        "final answer — no meta-commentary about the other answers."
    )


#: Appended to a candidate that was cut off at the output limit, so the synthesis step (and the
#: model, when that candidate is what comes back) weighs it as the fragment it is.
_CUT_OFF_MARK = "\n\n[This answer was cut off at the output limit and is incomplete.]"


def _fullest(candidates: list[str]) -> str:
    """The fallback answer when synthesis does not deliver: the longest candidate that FINISHED,
    and a cut-off one only when none did. A long fragment is still a fragment."""
    return max(candidates, key=lambda c: (not c.endswith(_CUT_OFF_MARK), len(c)))


def _cut_off_note(cut_off: int, n: int) -> str:
    """Hint suffix saying how many candidates ran out of output room ("" when none did)."""
    if not cut_off:
        return ""
    return f" Note: {cut_off} of {n} candidate answers were cut off at the output limit."


class DeepThinkTool(Tool):
    """Best-of-N self-fusion over the agent's strongest model (see the module docstring)."""

    spec = ToolSpec(
        name="deep_think",
        description=(
            "Deliberate hard on ONE difficult question: sample several independent answers from a "
            "capable model and synthesize the single best one (best-of-N self-fusion). Use it "
            "SPARINGLY — only for a genuinely hard, high-stakes sub-problem (a tricky design "
            "decision, a subtle bug's root cause, a correctness-critical answer) where the extra "
            "deliberation is worth extra time and cost. It makes SEVERAL model calls, so it is "
            "EXPENSIVE — never use it for routine work or simple lookups. Ask one specific, "
            "self-contained question: the deliberation sees only that question, not the "
            "conversation, so include the context it needs."
        ),
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "The single hard question to deliberate on. Specific and self-contained "
                        "(the deliberation does NOT see the conversation — include needed context)."
                    ),
                },
                "samples": {
                    "type": "integer",
                    "description": (
                        f"How many independent answers to synthesize ({1}–{_MAX_SAMPLES}, default "
                        f"{_DEFAULT_SAMPLES}). More = better but costlier."
                    ),
                    "minimum": 1,
                    "maximum": _MAX_SAMPLES,
                },
            },
            "required": ["question"],
        },
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        question = str(args.get("question") or "").strip()
        if not question:
            return ToolResult.error(
                "deep_think needs a non-empty 'question'.",
                fix="Pass the one specific, self-contained hard question to deliberate on.",
            )
        if ctx.sampler is None:
            return ToolResult.error(
                "deep_think is unavailable here — no model sampler is wired (e.g. a bare or "
                "delegated loop).",
                fix="Answer directly; deep_think runs only in the main agent loop.",
            )
        n = args.get("samples", _DEFAULT_SAMPLES)
        if not isinstance(n, int) or n < 1:
            n = _DEFAULT_SAMPLES
        n = min(n, _MAX_SAMPLES)

        # 1) Sample N independent candidates in parallel (diverse temperature). A failed sample
        # is tolerated — we synthesize from whatever succeeded. A sample cut off at the output
        # limit is kept only when some answer text arrived, and is marked incomplete; one cut off
        # before it wrote anything (a reasoning model can spend the whole limit thinking) adds
        # nothing but is counted, so the result says why it is thin.
        results = await asyncio.gather(
            *(
                ctx.sampler(question, system=_CANDIDATE_SYSTEM, temperature=_CANDIDATE_TEMP)
                for _ in range(n)
            ),
            return_exceptions=True,
        )
        candidates: list[str] = []
        cut_off = 0
        for r in results:
            if isinstance(r, SampleCutOff):
                cut_off += 1
                if r.text.strip():
                    candidates.append(r.text.strip() + _CUT_OFF_MARK)
            elif isinstance(r, str) and r.strip():
                candidates.append(r.strip())
        extra: dict[str, Any] = {"cut_off": cut_off} if cut_off else {}
        note = _cut_off_note(cut_off, n)
        if not candidates:
            if cut_off == n:
                # Every candidate ran out of room before answering. Retrying the same question
                # spends the same time to the same end, so the error says what would change it.
                return ToolResult.error(
                    f"deep_think produced no answer: all {n} candidates were cut off at the "
                    "output limit before writing one. The model's reasoning counts against that "
                    "limit and used all of it, so the same question will end the same way.",
                    fix="Ask a narrower question (one part of the problem at a time), or reason "
                    "through it directly.",
                )
            reason = next(
                (str(r) for r in results if isinstance(r, BaseException)), "no answers produced"
            )
            return ToolResult.error(
                f"deep_think produced no candidate answers ({reason}).",
                fix="Try again, or reason through the question directly.",
            )
        if len(candidates) == 1:
            # Nothing to fuse across — return the single deliberated answer (still useful).
            return ToolResult.ok(
                candidates[0],
                data={"samples": len(candidates), "synthesized": False, **extra},
                hint="A single deliberated answer — use it to proceed." + note,
            )

        # 2) Synthesize the best combined answer. If the synthesis call fails, fall back to the
        # fullest candidate rather than losing the work. A synthesis cut off at the output limit
        # is a failure here too: a finished candidate beats a fragment of a fusion.
        try:
            answer = (
                await ctx.sampler(
                    _synthesis_prompt(question, candidates), system=_SYNTH_SYSTEM, temperature=0.0
                )
            ).strip()
        except Exception as exc:  # noqa: BLE001 — a handler must never raise; degrade gracefully
            return ToolResult.ok(
                _fullest(candidates),
                data={
                    "samples": len(candidates),
                    "synthesized": False,
                    "synthesis_error": f"{type(exc).__name__}: {exc}",
                    **extra,
                },
                hint="Deliberated answer (the synthesis step failed; returned the fullest sample)."
                + note,
            )
        if not answer:
            # Synthesis succeeded but produced nothing usable — fall back to the fullest candidate
            # and label it honestly (synthesized=False), mirroring the exception path above.
            return ToolResult.ok(
                _fullest(candidates),
                data={
                    "samples": len(candidates),
                    "synthesized": False,
                    "synthesis_error": "empty",
                    **extra,
                },
                hint="Deliberated answer (the synthesis step returned nothing; fullest sample)."
                + note,
            )
        return ToolResult.ok(
            answer,
            data={"samples": len(candidates), "synthesized": True, **extra},
            hint="Synthesized best-of-N answer — use it to proceed." + note,
        )


__all__ = ["DeepThinkTool"]
