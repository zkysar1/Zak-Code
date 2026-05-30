# How the Zak Code build is orchestrated

Zak Code is built by an **orchestrated team of AI agents**, coordinated by a lead
orchestrator. This document is the operating manual: how we plan, fan out work, verify it,
and keep documentation coherent across a long, high-token initiative.

## Roles

- **Orchestrator (lead).** Owns the plan, the docs, and the git history. Decides phase
  scope, launches workflows, reviews returned work, writes/commits the canonical files.
- **Build subagents.** Spawned per phase to implement, research, or verify within a tightly
  scoped boundary. They return their work to the orchestrator; they do not own the repo.

## Documentation control (important)

The orchestrator owns the canonical docs in `docs/`. Subagents may *draft* content and
*return* it, but the orchestrator integrates and commits it. This keeps a single coherent
voice and prevents parallel agents from clobbering each other's files. Rule of thumb:
**agents return content; the orchestrator writes files.**

When behavior changes, the matching doc changes in the same commit. Stale docs are bugs.

## The phase model

Work proceeds in **phases**, each producing a milestone from [`ROADMAP.md`](ROADMAP.md):

1. **Understand** — parallel readers map the problem / prior art → structured digests.
2. **Design** — synthesize digests into specs (architecture, parity, interfaces).
3. **Implement** — per-subsystem pipelines build the milestone, isolated by module.
4. **Verify** — adversarial review + tests run/pass; the milestone's exit criteria are met.
5. **Integrate** — orchestrator merges, updates docs, commits, and demos.

## Orchestration patterns we use

- **Freeze shared contracts before fan-out** — when a milestone's modules are tightly
  coupled through shared types (e.g. M0's `Message`/`ToolResult`/`Provider` ABC are imported
  by providers, agent, session, *and* tools), the orchestrator hand-writes and commits the
  shared contracts first ("Phase A"), then fans out implementation agents against those
  frozen interfaces. This stops parallel agents from each inventing an incompatible
  vocabulary and turning integration into a rewrite. Builders validate only their own files;
  the orchestrator runs the full integration sweep. Builders never edit the frozen contract
  files except additively, flagged loudly.
- **Fan-out research** — many readers, each on a different source/subsystem; barrier;
  synthesize. (Used for the foundation phase.)
- **Per-subsystem implementation pipeline** — each subsystem flows build → self-test →
  adversarial review independently (no needless barriers).
- **Adversarial verification** — independent skeptics try to refute a change before it's
  accepted; tests are the tie-breaker.
- **Worktree isolation** — when multiple agents mutate files in parallel, each runs in its
  own git worktree to avoid conflicts; the orchestrator integrates.
- **Loop-until-dry / completeness critic** — for audits and parity sweeps, keep going until
  no new gaps surface.
- **Orchestrator re-verifies, never trusts self-reports** — build agents report their own
  test results, but the orchestrator independently re-runs the full gate (`ruff`/`mypy`/
  `pytest`), diffs frozen contract files against HEAD to catch any accidental overwrite, greps
  for guardrail violations (e.g. vendor imports out of place), and runs an end-to-end smoke
  before committing. (In the M0 run a builder did a stray `git restore` mid-flight; this
  re-verification is what confirmed the repair was clean.)

## Verification discipline

A milestone is **not done** until, from a clean checkout:

```bash
uv run ruff check .
uv run mypy
uv run pytest
```

all pass, **and** the milestone's exit criteria in [`ROADMAP.md`](ROADMAP.md) are
demonstrably met (a real task completed end-to-end where applicable).

## Guardrails for the team

All subagents inherit the rules in [`../CLAUDE.md`](../CLAUDE.md) and
[`GUARDRAILS.md`](GUARDRAILS.md): clean-room, vendor-agnostic, core/interface separation,
no secret leakage, docs-with-code. Work outside your assigned module boundary only by
escalating to the orchestrator.

## Run log

A short, append-only record of orchestration runs (newest at bottom).

| Date | Workflow | Run ID | Purpose | Outcome |
| --- | --- | --- | --- | --- |
| 2026-05-30 | `zakcode-foundation` | `wf_efd14b18-b4c` | Mine prior art (claw-code/Hermes/goose/litellm/best-practices) → draft ARCHITECTURE/ROADMAP/PARITY/GUARDRAILS+RISKS | ✅ Done — 10 agents, ~995K tokens. Wrote full ARCHITECTURE, ROADMAP, PARITY, GUARDRAILS, RISKS + 6 reference digests in `docs/references/`. |
| 2026-05-30 | _(orchestrator, manual)_ | — | M0 Phase A: hand-write & freeze the shared contracts (messages, usage, provider ABC, tool registry, PermissionTier) before fan-out | ✅ Done — commit `e2db020`; 18 tests green; ruff + mypy clean. |
| 2026-05-30 | `zakcode-m0` | `wf_c46105ce-6ea` | Build M0 against the frozen contracts: leaf modules (provider, builtins, session, prompt) in parallel → integrate loop + public `Agent` API → CLI `chat` → adversarial review | ✅ Done — 7 agents, ~931K tokens. Committed `0d4b9fd`. Orchestrator independently verified: contracts byte-intact, vendor-agnostic holds, ruff+mypy clean, 93 tests, end-to-end smoke wrote a real file via the loop. One in-run `git restore` mishap by a builder was detected and fully repaired (no scar). |
| 2026-05-30 | `zakcode-m0-harden` | `wf_e462e9a9-a43` | Harden M0 (no new surface): edge-case robustness + tests for tools / loop / provider / session, in parallel on disjoint files → integrate → adversarial review. Adds a doom-loop guard, regex/timeout/corruption/error-mapping coverage. | ✅ Done — committed `8bfb23c`. Orchestrator independently re-verified: contracts byte-identical to HEAD, vendor-agnostic holds, ruff+format+mypy clean (32 src files), **216 tests pass + 1 skipped** (was 93). Found+fixed a real `glob` path-escape leak (`../*`/absolute patterns). Review rated hardening "strong". No git mishaps this run. |
