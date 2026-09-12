# External-validity notes for the choosability eval — measured 2026-09-12

Recorded BEFORE the non-inferiority run's verdict, so neither note can be shaped by it.

## 1. The eval's rendering is not production's

`SkillRegistry.render_catalog()` emits `- use_skill(name="X") — <desc>`; the eval emits
`- X: <desc>`. Measured over the same 145 skills:

```
arm      eval shape   prod shape     delta
NAMES         3,250        5,860    +2,610
FIRST        33,695       36,450    +2,755
FULL         97,907      100,662    +2,755
```

The `use_skill(...)` wrapper is a CONSTANT per-entry cost, independent of which description arm is
rendered, so the absolute saving is IDENTICAL in both shapes: 64,212 chars = 15,038 tokens. Only
the percentage moves, because the denominator grows: **64% of the production catalogue, 66% of the
eval's.** Cite 15,038 tokens (exact for production) rather than the percentage where precision
matters.

## 2. The eval's TASK is not production's — this is the real limit

The eval is an isolated multiple-choice question: one catalogue, one query, "reply with the skill
name only". Production is a `use_skill` tool call chosen mid-conversation, with the rest of the
system prompt, the conversation history, and the user's actual request all present, and with the
classifier (`providers/routing.py`) having already had a say.

The ARM COMPARISON is internally valid — all three arms face the identical task, identical queries
and identical scoring, so the difference between them is real. What does NOT follow automatically is
that the same difference appears in production's selection behaviour. That is an assumption, and it
is the same class of gap ADR-0146 and ADR-0154 already record for this bench: the instrument and the
product exercise overlapping but distinct slices.

Do not restate the arm scores as production skill-selection accuracy. The transferable claims are
the RATIO (marginal value of description bytes) and the ORDERING (names << first-sentence <= full),
not the absolute percentages.

## 3. The existing tests would not catch a first-sentence regression

`tests/test_skills.py` builds skills with single-word descriptions ("alpha", "beta"). A
first-sentence truncation is a no-op on a string with no sentence boundary, so every current
assertion passes whether the truncation works, silently does nothing, or is wrong. Any
implementation of the shortened catalogue MUST add a multi-sentence-description case, or the change
ships with a test suite that cannot see it.
