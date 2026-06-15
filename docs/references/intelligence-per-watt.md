# Intelligence per Watt — research digest (zakpick evidence)

**Paper:** Saad-Falcon, Narayan, Akengin, Griffin, Shandilya, Lafuente, Goel, Joseph, Natarajan,
Guha, Zhu, Athiwaratkun, Hennessy, Mirhoseini, Ré — *Intelligence per Watt: Measuring
Intelligence Efficiency of Local AI* (arXiv **2511.07885**, 2025; Stanford Hazy Research et al.).

Clean-room study note — a distillation of a public paper that informs zakpick's design and the
triggers for its deferred seams. No proprietary material; see [`../GUARDRAILS.md`](../GUARDRAILS.md).

## What the paper measures

An empirical study of when small/local models suffice: **20+ local LMs, 8 accelerators, ~1.04M
real-world queries** (Wildchat chat, NaturalReasoning, MMLU-Pro, SuperGPQA). It introduces
**Intelligence per Watt (IPW)** — task accuracy divided by average inference power
(`APW = E[acc] / E[power]`), plus a per-joule variant that folds in latency. It is single-turn
chat/reasoning only — **no coding, tool-use, or agentic tasks** — so it validates a *thesis*, not
a method, for our domain.

## Findings that bear on zakpick

- **Most prompts don't need the frontier model.** Best single local model answers ~88.9% of chat
  / ~64.9% of hard-reasoning queries; **routing across several local models** reaches **93.4%** on
  MMLU-Pro vs 80.4% for the best single model. → direct support for zakpick's core bet
  (many-models-by-task beats one-model-for-everything).
- **`gpt-oss-120b` is named the best individual local model (80.4% MMLU-Pro)** — exactly our
  `deep_code` / `delegate` default in `DEFAULT_CATEGORY_MODELS`. External validation of the pick.
- **Accuracy varies by domain the way our categories assume:** >90% on creative/writing, dropping
  to ~68% technical and ~60% engineering. → small models are fine for `summarize` / `classify`;
  capable models for `deep_code` / `plan` (our default ladder already reflects this).
- **An imperfect router still pays — the yardstick.** Routing scenarios: oracle ≈ **80.4% energy
  savings**; an **80%-accurate router** captures ~80% of that (**64.3% energy, ~59% cost**); a
  **60%-accurate router** still gets ~48%. → even a crude difficulty classifier is worth having,
  and the *ceiling* on improving it is bounded.
- **No verifier/confidence escalation** — the paper's hybrid just blindly retries misrouted
  queries on cloud. zakpick's soft latch escalates on *real* signals (verify-gate failure, stuck,
  doom-loop), so our escalation trigger is, if anything, ahead of this baseline.

## Implication for the deferred classifier-model seam (ADR-0009)

Our `classify_main_turn` is a length/context heuristic. The paper supplies the decision rule for
whether to upgrade it to a learned/embedding classifier (the `classify`-category model seam): an
offline eval that measures the heuristic's routing accuracy. If it sits well below ~80% on a
representative coding-task set — i.e. it is leaving a meaningful slice of the ~59% cost headroom on
the table — the model classifier earns its keep; if it is already near that, the heuristic stays.
The "rejected" auto local-vs-cloud router from the paper remains rejected (ADR-0009): we route to
the user's per-category models, never an availability-chosen substitute.
