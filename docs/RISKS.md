# Risk Register

> **The full register is being authored by the `zakcode-foundation` workflow**
> (run `wf_efd14b18-b4c`, launched 2026-05-30). Seed entries below.

| Risk | Category | Likelihood | Impact | Mitigation | Status |
| --- | --- | --- | --- | --- | --- |
| Accidentally copying leaked/proprietary code | Legal/IP | M | H | Clean-room rule; reference material kept outside repo & gitignored; review diffs | Open |
| API key leakage via logs/commits | Security | M | H | `.env` gitignored; never print key values; report presence only | Open |
| RCE / data loss via file & shell tools | Security | M | H | Permission model; confirm destructive ops; path scoping | Open |
| Prompt injection via tool output / web content | Security | M | M | Treat tool/web output as untrusted; don't auto-execute embedded instructions | Open |
| Local models lacking native tool-calling | Reliability | H | M | Provider-layer fallback (prompted tool protocol) for such models | Open |
| Provider/API drift (litellm/OpenAI/Ollama) | Dependency | M | M | Thin provider layer isolates changes; pin versions | Open |
| Scope creep / never-ending parity chase | Delivery | H | M | Milestones with exit criteria; tiered parity matrix | Open |
| Context/cost blowups during long sessions | Cost | M | M | Aggressive context compaction; budgets; observability | Open |
