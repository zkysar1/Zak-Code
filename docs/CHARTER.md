# Zak Code — Project Charter

## Vision

A coding agent we **own end-to-end**: a single reusable core engine that any
interface — CLI, web app, IDE, or automation — can drive, free of lock-in to any one
model vendor, and built to be understood, extended, and taken in new directions.

## Mission for this build

Build Zak Code from scratch (clean-room), reaching a daily-driver agentic coding tool
with a clear path to Claude Code-level capability, while learning the architecture
deeply along the way.

## Goals

- **G1 — API-first core.** A provider-agnostic engine usable as a library and over HTTP,
  with the CLI as the first client (the Claude Code SDK shape).
- **G2 — Vendor-agnostic.** Run on local (Ollama) and cloud (OpenAI) day one; swap to any
  litellm provider by config.
- **G3 — Capable agent loop.** Read/write files, run commands, search code, plan and
  execute multi-step tasks reliably, with good context management.
- **G4 — Extensible.** Tools, MCP extensions, plugins, hooks, and skills.
- **G5 — Safe & legible.** Strong permission/guardrail model; clean, documented, tested code.
- **G6 — Learnable.** The codebase and docs teach how an agentic harness works.

## Non-goals (for now)

- Training or hosting our own models (we orchestrate existing ones).
- A polished commercial GUI in the first phases (web client comes after the core is solid).
- 1:1 byte-parity with Claude Code's source — we target *capability* parity, our own way.
- Cloud multi-tenant SaaS / billing infrastructure.

## Principles

1. **Clean-room.** Patterns, not proprietary code. Re-express every idea in our design.
2. **Vendor-agnostic by construction.** No provider-specific assumptions in the core loop.
3. **Separation of concerns.** Core engine ↔ server ↔ clients are cleanly divided.
4. **Few sharp tools.** Prefer a small set of reliable, composable tools (CLI-first).
5. **Context is the budget.** Engineer context deliberately; compact aggressively.
6. **Safe by default.** Confirm destructive actions; never leak secrets.
7. **Documentation-driven.** Docs are the contract; they change with the code.
8. **Small, verifiable steps.** Every milestone is testable and demoable.

## Success criteria

- Zak Code can complete a real multi-file coding task end-to-end, on **both** a local
  Ollama model and OpenAI, driven from the CLI **and** via the HTTP API.
- A new contributor can read the docs and understand/extend the system in an afternoon.
- The parity matrix ([`PARITY.md`](PARITY.md)) shows steady progress toward daily-driver
  and then full capability.

## Stakeholders

- **Owner / product direction:** Zachary Kysar (Zak Data Solutions).
- **Lead builder / orchestrator:** the agent team, coordinated via
  [`WORKFLOW.md`](WORKFLOW.md).

## Roadmap at a glance

The full milestone plan lives in [`ROADMAP.md`](ROADMAP.md). In short: **M0** runnable core
loop (Ollama + OpenAI, sharp tool set, `zakcode chat`, sessions) → **M1** streaming/TUI →
**M2** permissions/hooks → **M3** FastAPI server (SSE/WS) → **M4** sub-agents → **M5** MCP →
**M6** plugins → **M7** skills → **M8** advanced compaction → **M9** eval harness →
**M10+** web client & deferred surfaces. Capability parity vs. Claude Code is tracked in
[`PARITY.md`](PARITY.md).
