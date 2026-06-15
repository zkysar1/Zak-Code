# Reference digests

Architecture study notes mined from prior art by the `zakcode-foundation` workflow (2026-05-30). These are **clean-room study notes** — distillations of architecture and public documentation that informed Zak Code's own design. No proprietary or leaked source is reproduced here. See [`../GUARDRAILS.md`](../GUARDRAILS.md).

- [`claw-code-rust.md`](claw-code-rust.md) — claw-code (Rust) architecture digest
- [`claude-code-parity.md`](claude-code-parity.md) — Claude Code parity surface
- [`hermes.md`](hermes.md) — Hermes architecture digest
- [`goose.md`](goose.md) — goose architecture digest
- [`agentic-best-practices.md`](agentic-best-practices.md) — Agentic coding best-practices digest
- [`litellm.md`](litellm.md) — zak/llm/provider.py
- [`intelligence-per-watt.md`](intelligence-per-watt.md) — *Intelligence per Watt* (arXiv 2511.07885): evidence that small/local models handle most prompts, and the router-accuracy→savings yardstick for zakpick's deferred classifier-model seam
- [`fusion-beats-frontier.md`](fusion-beats-frontier.md) — OpenRouter *Fusion beats Frontier*: ensemble + synthesis beats a single model, and self-fusion (one model, best-of-N + synthesize) carries most of the lift — the basis for the `deep_think` tool
