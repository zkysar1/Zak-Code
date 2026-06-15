# Fusion beats Frontier — research digest (deep_think evidence)

**Source:** OpenRouter, *Fusion beats Frontier* (announcement, 2026),
<https://openrouter.ai/blog/announcements/fusion-beats-frontier/>.

Clean-room study note — a distillation of a public announcement that informs the `deep_think`
tool. No proprietary material; see [`../GUARDRAILS.md`](../GUARDRAILS.md).

## What Fusion is

OpenRouter's "Fusion" sends one request to a **panel of models in parallel**, then a **judge**
model writes a structured analysis (consensus, contradictions, partial coverage, unique insights,
blind spots), and a **synthesizer** writes the final answer grounded in that analysis. It is
ensemble-synthesis, **not** routing or mixture-of-experts, and it is a server-side pipeline behind
a single API call (`model: openrouter/fusion`).

## Findings that bear on `deep_think`

- **Fusion exceeds any single model.** On DRACO (100 deep-research tasks): a fused Fable 5 + GPT-5.5
  scored **69.0%** vs 65.3% / 60.0% solo.
- **The synthesis step alone is a big chunk of the lift.** Opus 4.8 paired with **itself**
  ("self-fusion") jumped to **65.5% from 58.8% solo — +6.7 points**. So you do not need multiple
  providers to benefit: sampling one model several times and synthesizing already helps. This is
  the core insight `deep_think` builds on.
- **Cheap panels rival frontier.** A budget panel (3 cheap models) landed within ~1% of a single
  frontier model at ~half the cost — a third data point (with *Intelligence per Watt*) that
  small/cheap models, used well, rival frontier.
- **Cost/latency.** Fusion is **2–3× a normal call**. OpenRouter's own guidance: it is **not** a
  coding replacement; a coding agent should **invoke it selectively, when the model decides a
  question is worth the extra time and money** (e.g. architecture decisions).

## How `deep_think` adapts it (ADR-0010)

`deep_think` is **self-fusion as an opt-in tool**, scoped to Zak Code's constraints:

- **One model, the user's own.** It samples the agent's strongest *configured* model (under zakpick
  the `deep_code` category, else `default_model`) several times and synthesizes — it never reaches
  for a model the user did not assign, so it owns no provider tradeoff (the line ADR-0009 draws).
- **The model decides, never automatic** (OpenRouter's own recommendation): it is a tool the agent
  invokes on a genuinely hard sub-problem, visible in the transcript, gated by permissions.
- **Cost is visible and bounded.** Its extra calls are attributed in `/cost` (per-model) and folded
  into the turn budget — the 2–3× is never hidden.
- We deliberately did **not** build the multi-provider panel + separate judge: the self-fusion
  finding says the synthesis step carries most of the lift, and a single-model best-of-N keeps the
  feature small, vendor-agnostic, and free of a cross-model orchestration layer (a possible future
  seam if a measured need appears).
