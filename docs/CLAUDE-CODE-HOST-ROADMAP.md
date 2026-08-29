# Zak Code as a Claude-Code-Compatible Host — Strategic Roadmap

*Companion to [`CLAUDE-MIND-COMPAT.md`](CLAUDE-MIND-COMPAT.md) (the gap map). This is the plan: the
architecture, the rule that keeps it clean, the phases (with a live run early), and who builds what.*

> **Status (2026-06-22): Phases 0–3 SHIPPED, and the ecosystem proof landed.** The CC-compat edge
> exists, every contract area has a generic conformance test that never names a plug-in, and a
> complete third-party-shaped plug-in runs end-to-end on one Agent. The remaining items are a
> short, deliberately-deferred robustness/cosmetic tail — see **Deferred (documented, with
> rationale)** below. Each phase row is annotated **✅ shipped** inline.

## North star

Zak Code is a **generic, behavior-free host that speaks the Claude Code extension contract.** It has
no memory, no goals, no opinions. The *behavior* comes from whatever you plug in:

- Plug in **claude-mind** → it comes to life as a self-directed research mind.
- Plug in a **different skill system** → it comes to life another way.
- Plug in **someone else's Claude-Code skill / hook / tool** → it just works.

The strategic prize is the last one: because the host speaks the **general** Claude-Code language (not
"claude-mind's needs"), **anything built for Claude Code runs on Zak Code unmodified** — and
claude-mind stays an upstream project we pull updates from, never a fork we babysit.

## Architecture: generic core · CC-compat edge · plug-ins

```
   plug-ins      │  claude-mind   ·   other skill systems   ·   3rd-party CC tools
  ───────────────┼──────────────────────────────────────────────────────────────
   CC-compat edge │  translates Claude Code's language ⇄ generic seams
                  │  (skills · commands · hooks · settings · transcript · env)
  ───────────────┼──────────────────────────────────────────────────────────────
   generic core   │  hook seams · tool registry · skills loader · SessionStore · context
                  │  (knows nothing about Claude Code OR claude-mind)
```

The core already exists and stays CC-agnostic. The **CC-compat edge** is the thing we build out. Part
of it already exists and proves the pattern: Zak Code didn't add a "claude-mind loop" — it added a
generic `TURN_END` event plus a thin translator (`"Stop" → TURN_END`). The Mind plugs into the
generic event. **Every item in this plan follows that template.**

## The boundary test (run on every step)

> **1. Generic, not specific.** *Am I teaching the host a GENERIC Claude-Code capability any plug-in
> can use — or leaking claude-mind-specific behavior into the host?* If the latter, redesign until the
> host stays generic and the plug-in-specific part lives on the plug-in's side.
>
> **2. Within reason.** *Is this a real Claude-Code contract worth speaking — or a CC implementation
> quirk we should NOT replicate (and expose more cleanly instead)?* See "What we deliberately don't
> replicate."

Where a step involves a judgment call, the plan flags it inline as **⟂ boundary call**.

## The Claude Code contract — the surface we're implementing

"Speaking the language" means these nine areas. Each is built as a generic capability; claude-mind is
just the first consumer that proves it.

1. **Skills** — `.claude/skills/*/SKILL.md` discovery, tolerant frontmatter, invocation, args, chaining.
2. **Commands** — `triggers:`-based and `.claude/commands/*.md`; dispatch, args, `user-invocable`.
3. **Hooks — events** — PreToolUse, PostToolUse, Stop, SessionStart, PreCompact, StopFailure, UserPrompt*.
4. **Hooks — payload** — the stdin JSON shape + decision protocols (block/continue, deny, updatedInput, additionalContext).
5. **Settings** — `settings.json` + `settings.local.json` layering and dispatch.
6. **Transcript** — a Claude-Code-shaped view exposed via `transcript_path`.
7. **Environment** — `$CLAUDE_PROJECT_DIR`, a user-level config location.
8. **Permissions** — deny-beats-allow, the `Tool(pattern)` gesture grammar.
9. **Presentation** — statusLine, output-styles.

---

## Phase 0 — Name the layer + build the guardian (foundation) — ✅ DONE

**Shipped:** the CC-compat layer is named and the conformance guardian
(`tests/test_cc_conformance.py`) is live; the stale `INTEGRATIONS.md` "deferred" section was corrected.

The strategic backbone. Without this, the abstraction rots into hacks within two PRs.

| Item | What / why | Owner | Effort |
|---|---|---|---|
| **Name the CC-compat layer** | Gather the scattered translators (`settings_loader` Stop-map, tool-name map, `$CLAUDE_PROJECT_DIR`) under one named, documented surface (`compat/claude_code/`). Make "the edge" a real place, not a habit. | dev | S |
| **Conformance suite** | A test suite that asserts each contract area independently of claude-mind ("a Stop hook returning `{decision:block}` re-enters the loop", "a skill with `triggers:[/x]` is invocable as `/x`"). **This is how we keep the abstraction honest forever** — no contract piece is "done" without a CC-generic test that never names the Mind. | dev | M |
| **Fix stale docs** | `INTEGRATIONS.md` "deferred" section wrongly says Stop-continuation + settings ingestion aren't built — they are. Correct it; publish the CC-compat contract as the public integration surface. | dev | S |

## Phase 1 — Minimum viable host → **LIVE "ALIENS" RUN** 🛸 — ✅ DONE

**Shipped:** slash dispatch on `triggers:`, command/`use_skill` args, and `user-invocable`
enforcement, plus `settings.local.json` layering — the minimum viable generic host surface.

The proof-of-life. Almost entirely **dev-side surface work** (CLI + skills + settings reader), so it
moves fast with no hard omni dependency. Each item is a generic capability.

| Item | Generic capability (not "make Mind work") | Owner | Effort |
|---|---|---|---|
| **Trigger dispatch** | Route `/<tok>` on a skill's `triggers:` frontmatter, not just its folder name. | dev | S |
| **Command args** | Thread the text after the command into the skill body, so `/x foo --bar` keeps `foo --bar`. | dev | S |
| **`use_skill` args** | Optional `args` param so skill→skill chaining (`Skill('x') with args='loop'`) survives. | dev | S |
| **`user-invocable`** | Honor the frontmatter flag: gate user-typed `/x` vs. model-invoked `use_skill` on it (a generic safety contract, not a Mind rule). | dev | S |
| **`settings.local.json`** | Read it alongside `settings.json`, local-over-project precedence. | dev / ⟂ shared | S |
| **Daemon startup** | **⟂ boundary call:** the host does **not** manage claude-mind's `mind_api` daemon. The Mind starts it via a `SessionStart` hook the host **already fires**. Host work = none; Mind config = one settings entry. | Mind-side config | — |
| **Tolerate transcript gap** | First run is always "fresh", so the transcript view (Phase 2) isn't on the critical path; the Mind's stop-hook already gets `last_assistant_message` directly. Fail-open. | — | — |

**Milestone:** `zakcode cli` in a fresh `aliens/` folder → `/start aliens` → the Mind boots, picks a
goal, researches, and the loop re-enters itself. We watch it actually do research. **Proof the engine
works on Zak Code.**

## Phase 2 — Transcript & lifecycle fidelity (mostly omni seam-domain) — ✅ DONE

**Shipped:** a Claude-Code-shaped `transcript_path` view, SessionStart `source`, PreCompact `trigger`
at the payload top level, and PostToolUse `additionalContext` (StopFailure + UserPromptExpansion
events deferred — see below).

Make the host *record and signal* like Claude Code, so the Mind's full machinery (recovery, resume,
consolidation) works — and so do other CC tools that read transcripts/lifecycle.

| Item | Generic capability | Owner | Effort |
|---|---|---|---|
| **Transcript view** | **⟂ boundary call:** keep `SessionStore` as the clean source of truth; expose a Claude-Code-shaped `.jsonl` *view* at the edge, handed to hooks as `transcript_path`. Core stays CC-agnostic; the edge does the projection. | omni (store) + dev (projection) | M |
| **SessionStart `source`** | Add `startup`/`resume`/`compact` to the lifecycle payload (resume-vs-fresh branching). | omni | S |
| **PreCompact `trigger`** | Surface `trigger` at the stdin top level, matching the contract. | omni | S |
| **PostToolUse `additionalContext`** | Honor it (a hook injecting post-tool context). ~5 lines. | omni | S |
| **`StopFailure` + `UserPromptExpansion` events** | Fire these generic events in the loop (crash-recovery + prompt telemetry). | omni | M |

## Phase 3 — Settings, permissions & presentation (the parity subsystems) — ✅ DONE

**Shipped:** `permissions.{allow,deny}` ingestion into the deny-first policy (tighten-only; a bare
whole-tool deny binds even read-only tools), a statusLine subsystem, and output-styles → system
prompt — all off by default.

The genuinely-new generic subsystems. All three are **already "Planned" in Zak Code's own roadmap**,
so they serve broad Claude-Code parity, not just the Mind.

| Item | Generic capability | Owner | Effort |
|---|---|---|---|
| **Permission ingestion** | Read CC `permissions.{allow,deny}` `Tool(glob)` gestures from settings(.local).json and translate them into Zak Code's (stronger) deny-first policy. **⟂** Translate the *gesture grammar*; don't weaken the catastrophic floor. | dev (translator) + omni (policy seam) | M |
| **statusLine** | A generic statusLine subsystem: read the `statusLine` command from settings, feed it session JSON per turn, render it. | dev | M–L |
| **output-styles** | A generic output-style subsystem (named styles that shape generation + persist in settings). | dev | M–L |

## Phase 4 — Ecosystem proof + lock

Prove the *bonus*, then freeze the contract.

| Item | What | Owner | Effort |
|---|---|---|---|
| **Second-plug-in proof** ✅ shipped | A **complete, generic** Claude-Code plug-in (a slash-triggered skill + a `settings.json` Stop hook + permission denies + an output style + an always-on rule, naming no framework) runs end-to-end on one Agent: `tests/test_cc_ecosystem.py`. Turns "should work" into "proven." | dev | M |
| **Conformance lock** ✅ shipped | Every contract area has a passing CC-generic test, grouped under the `cc_conformance` marker so the guardian is runnable on its own (`pytest -m cc_conformance`) and runs as part of the full `poe check` suite — a future change can't silently break the abstraction. | dev | S |
| **Public contract docs** ✅ shipped | The supported CC-compat surface + opt-in flags are documented in [`docs/INTEGRATIONS.md`](INTEGRATIONS.md) ("Claude Code compatibility"). | dev | S |

### Deferred (documented, with rationale)

These were scoped out of the shipped work *on purpose*. They are a small robustness/cosmetic tail —
none is a loop-blocker, and each is recognised-and-handled today (never silently dropped):

- **`StopFailure` + `UserPromptExpansion` hook events** — real CC events, deferred for **scope**: a
  non-loop-blocking robustness tail, and firing `StopFailure` would thread the critical
  turn-finalize path. Both are recognised and skipped **with a warning** today, not silently dropped.
- **statusLine: cap the command's stdout read** — the status command's output is bounded only by the
  5s timeout today; add an explicit byte cap.
- **statusLine: the status JSON's model id uses `default_model`** — cosmetic under zakpick/failover
  (the rendered id can lag the model actually used for the turn).
- **Protected-path deny reason wording** — the deny reason says "write to a protected path" even for
  a denied *read*; cosmetic wording only.
- **output-styles: an off-default-byte-identical test that ALSO has rules present** — the current
  off-test uses a no-rules baseline; the with-rules case is safe by inspection, but deserves its own
  regression test.

---

## What we deliberately DON'T replicate (the "within reason" list)

Faithful ≠ slavish. These are Claude-Code *quirks*, not contracts:

- **Literal `~/.claude/` home** — honor the *concept* (a user-level config dir) at the edge; keep Zak Code's own `~/.zakcode/`. ⟂
- **The `mind_api` daemon** — claude-mind's own process; the host never manages a plug-in's background services. ⟂
- **`ScheduleWakeup` tool** — not on the heartbeat path (the loop re-enters via skill chaining); build only if a real plug-in needs timed self-wake.
- **`CLAUDE_CODE_AUTO_COMPACT_WINDOW` envs** — plug-ins degrade gracefully without them.

If faithfully speaking the contract ever forces a genuinely bad design into the core, that's a signal
to expose the capability more cleanly at the edge — and I'll flag it, per "within reason."

## Ownership at a glance

- **Dev surface (build via PR, omni reviews):** the named compat layer, conformance suite, skill/command
  dispatch + args + `user-invocable`, settings reading/dispatch, the permission-gesture translator,
  statusLine + output-styles, the transcript *projection*, docs, ecosystem proof.
- **Omni seam-domain (design-sensitive internals):** new hook *events* (StopFailure, UserPromptExpansion),
  hook *payload* fields (SessionStart `source`, PreCompact `trigger`, PostToolUse `additionalContext`),
  the SessionStore side of the transcript view, the permission-policy core seam.
- **Mind-side (config, not host):** starting `mind_api` via a SessionStart hook; any Mind-specific paths.

> Phase 1 is almost entirely dev-surface, so the **live aliens run isn't blocked on omni.** Phases 2–3
> lean into omni's seam-domain — sequence those with omni in the loop.

## How we keep the abstraction honest

The **conformance suite is the guardian.** Rule: *no Claude-Code contract piece is "done" until it has
a test that proves it generically — without ever naming claude-mind.* If a test can only be written by
referencing the Mind, the boundary has been crossed and the design is wrong.
