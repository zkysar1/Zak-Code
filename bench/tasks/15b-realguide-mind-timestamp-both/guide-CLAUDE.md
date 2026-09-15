# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Domain-agnostic continual learning base agent. The system forms hypotheses, tracks their outcomes, builds memory of what worked and what failed, and self-evolves its reasoning capabilities over time. It serves as a reusable foundation for any domain where an autonomous agent needs to learn, reflect, and improve through experience.

## Architecture

This is a **Claude-native data repository** — no traditional source code or build tools. Configuration and state live in YAML, JSONL, and Markdown files that Claude reads, reasons over, and updates autonomously.

### Core Design Principle: No Terminal State

The system is a perpetual loop. Completion of one thing seeds the next. `/aspirations loop` is the heartbeat — it never exits, it always has work to create.

*(Full rules in `core/config/modes/autonomous.md`)*

### Core Design Principle: Consolidate Before Expand

Depth over breadth. Completion of existing work takes priority over starting new.
An aspiration 90% complete has more gravitational pull than a brand-new aspiration.
New directions require healthy existing completion rates (>25% average) or explicit
justification (user directive, critical blocker, all existing work blocked).

*(Full rules in `.claude/rules/consolidate-before-expand.md`)*

### Mode System

The framework has three operational modes. Mode is the single user-facing control — state and persona are derived automatically.

| Mode | State | Persona | Capabilities |
|------|-------|---------|-------------|
| `reader` (safe floor) | IDLE | ON (light) | Read knowledge, prime, answer questions. No writes. Opt-in via `/stop <agent-name> --reader`. |
| `assistant` (post-stop default) | IDLE | ON (full) | Reader + write to tree, remember things, research when asked, accept directives. No loop. |
| `autonomous` | RUNNING | ON (full) | Everything. Self-directed perpetual learning loop. |

Reader is the safe floor, not the routine default. `/stop <agent-name>` lands in assistant mode (reconciliation-ready) so the user can mark goals complete, edit tree nodes, or add guardrails without a mode-switch ceremony. Pass `/stop <agent-name> --reader` to drop to read-only (walking-away case). Agent name is REQUIRED on `/stop` — bare `/stop` is refused (prevents the cross-session wrong-agent stop, 2026-04-24 incident). Disk default (absence of `agent-mode`) is still reader — that handles passive/crashed sessions.

Mode-specific behavioral rules live in `core/config/modes/{mode}.md` — loaded on demand at session start.
Mode signal file: `agents/<agent>/session/agent-mode` (plain text: reader, assistant, autonomous).
Scripts: `session-mode-get.sh`, `session-mode-set.sh` (only /start and /stop may write).

### Cognitive Primitives

Four goal types the agent can create anytime via `aspirations-add-goal.sh`:
- **Unblock** (`"Unblock: ..."`, HIGH) — created by CREATE_BLOCKER protocol when a problem can't be fixed inline
- **Investigate** (`"Investigate: ..."`, MEDIUM) — diagnostic, something seems off
- **Idea** (`"Idea: ..."`, MEDIUM) — creative insight, improvement opportunity
- **Maintain** (`"Maintain: ..."`, MEDIUM) — in-flight framework correction the agent JUST performed inline; filed with `status: completed` so the standard encoding pipeline fires

Not mutually exclusive. A single event can spawn all four. See `aspirations-execute/SKILL.md` Cognitive Primitives section (Cross-Agent Insight is also described there but produces a board post, not a goal).

## Universal Conventions

### File Formats
- **YAML** (`.yaml`) for structured data: config, indexes
- **JSONL** (`.jsonl`) for lifecycle records: aspirations, pipeline, experiences, reasoning bank, guardrails, pattern signatures, spark questions, journal index
- **JSON** (`.json`) for metadata: aspirations-meta, pipeline-meta, experience-meta
- **Markdown** (`.md`) with YAML front matter for knowledge articles and journal entries

### Domain-Free Cognitive Core
Everything in `world/` is collective domain state (shared across agents). Everything in `<agent>/` is per-agent private state. Everything in `meta/` is domain-agnostic improvement strategy. Everything in `core/` and `.claude/` is immutable framework.
The cognitive core (base skills, rules, `core/`) describes INTENT, never domain-specific
implementation. Domain knowledge lives in `world/`: conventions (`world/conventions/*.md`),
guardrails, reasoning bank, knowledge tree, forged skills (`world/forged-skills.yaml`). Agent-specific state lives in `<agent>/`: experience, journal, session.

### Naming Rules
- All filenames: **lowercase, kebab-case** (hyphens, no spaces, no underscores except pipeline/experience record IDs)
- ISO 8601 dates everywhere. Timestamps: naive format (no zone suffix) via `$(date +%Y-%m-%dT%H:%M:%S)`, in **UTC wall time on every box** — enforced by `.claude/settings.json` env `TZ=UTC` (all boxes) plus box TZ=Etc/UTC where the OS allows (Linux). "Local system time" and UTC converged by fiat 2026-07 (g-115-2546): a multi-box fleet comparing naive stamps (board `--since`, `last_active` staleness, LWW merges) needs one shared wall clock, and mixed domains silently corrupt every comparison. Long-lived processes keep the TZ env they started with — after changing TZ posture, restart daemons or stamps stay in the old zone.

### ID Formats
- Aspirations: `asp-NNN` | Goals: `g-NNN-NN` (supports 2-4 digit: `g-NNN-NNNN`; expanded 2026-05-19 after asp-115 hit g-115-999) | Prep tasks: `pt-NNN`
- Guardrails: `guard-NNN` | Reasoning bank: `rb-NNN` | Beliefs: `bel-NNN`
- Transitions: `trans-NNN` | Spark questions: `sq-NNN`, candidates: `sq-cNN`
- Pattern signatures: `sig-NNN` | Strategy archive: `sa-NNN`
- Experiences: `exp-{source-id-or-slug}` | Pipeline: `YYYY-MM-DD_slug`

### Priority Values
- `HIGH`, `MEDIUM`, `LOW` (uppercase)

### Status Values

Goals: `pending`, `in-progress`, `completed`, `blocked`, `skipped`, `expired`, `decomposed`, `superseded` | Pipeline: `discovered`, `active`, `resolved`, `archived` | Aspirations: `active`, `completed`, `paused`, `retired`. Full per-entity status lists: see convention files.

### Pipeline Rules
- **Never delete** pipeline records — move via `pipeline-move.sh`
- Journal entries are **append-only**
- Hypothesis horizons: `micro`, `session`, `short`, `long`
- Hypothesis types: `high-conviction`, `calibration`, `exploration`, `contrarian`

### Python Invocation (Windows)
Direct `python3 -c "..."` from a Bash tool call hits a Microsoft Store stub on this machine (Exit code 49: "Python was not found"). The shim in `core/scripts/.python-shim/` plus the `bash-agent-inject.sh` PreToolUse hook defend against this, but the hook fails open on timeout — when it does, the raw command fails. **Rule**: prefer `bash core/scripts/<wrapper>.sh` or `py -3 -c "..."` for direct Python from Bash. Use `python3` only inside `.sh` scripts that source `_paths.sh`. Full detail: `core/config/conventions/python-invocation.md`.

### Daemon-Only Architecture
As of 2026-05-14, 35 wrappers are daemon-only — no Python CLI fallback. See `.claude/rules/no-python-cli-fallback.md` for the behavioral rule and recovery procedures.

### Self File Format and The Program

The shared purpose lives in `world/program.md` (The Program). Each agent's identity lives in `agents/<agent>/self.md` (YAML front matter + markdown body). Schema and maintenance: `.claude/rules/self.md`.

### Skill Invocation Rules
- **Control skills** (/start, /stop, /open-questions): user-invocable only — Claude MUST NOT invoke these
- **Mode control**: `/start <agent-name> --mode <mode>` to enter a mode, `/stop <agent-name>` to return to assistant (or `/stop <agent-name> --reader` for read-only). Agent name is REQUIRED on `/stop`.
- **Hybrid skills** (/agent-completion-report, /backlog-report, /forge-skill, /priority-review, /sprint-planning, /verify-learning, /generate-domain-goals, /update-framework): user-invocable AND agent-callable
- **Internal skills**: `user-invocable: false` — invoked by agent during RUNNING state
- **No blocking on user input in RUNNING state** — skills must never wait for, request, or depend on user input during autonomous execution

### Code Change Verification (MANDATORY)
After ANY code change: read the project's CLAUDE.md, run tests, fix errors. Never declare ready until build passes.

### Knowledge Reconciliation
After any action that changes the world, check if knowledge tree nodes need updating. Detail: `core/config/conventions/infrastructure.md`.

## Knowledge Retrieval (All States)

When persona is active, the agent MUST consult its knowledge before answering domain questions.
Follow the retrieval escalation convention (`core/config/conventions/retrieval-escalation.md`):

1. **Tier 1 — Knowledge Tree**: `retrieve.sh --category {category} --depth medium` or intelligent retrieval protocol
2. **Tier 2 — Codebase Exploration**: Grep/Glob/Read on the primary workspace (from `agents/<agent>/self.md`)
3. **Tier 2.5 — Peer Worlds**: `peer-retrieve.sh` — only `status: empty` at `completeness: complete` licenses a negative; rc=3 / `partial` means a lane was blind, not empty
4. **Tier 3 — Web Search**: WebSearch/WebFetch (assistant/autonomous mode only)

Stop at the first tier that provides sufficient knowledge. Never say "I don't have context"
without attempting all eligible tiers.

### Enforcement Rules

1. Claude MUST NOT invoke /start, /stop, or /open-questions.
2. Claude MUST NOT invoke boot or start the aspirations loop without RUNNING state and autonomous mode.
3. In reader mode: read-only assistant. May read state but MUST NOT execute write operations or workflow skills.
4. In assistant mode: user-directed assistant. May read and write when asked but MUST NOT self-initiate or run the loop.
5. In autonomous mode (RUNNING state): autonomous via aspirations loop.
6. Auto-resume after autocompact is handled by the stop hook (unconditional BLOCK + LOOP_CONTINUE), NOT by the Session Start Protocol. A new session that finds RUNNING state must show the error (or start an observer session if `--mode reader|assistant` is requested), not auto-resume.

### Autonomous Loop Rules

See `core/config/modes/autonomous.md` (loaded on demand in autonomous mode).
[... CLAUDE.md: 47511 characters, 24 of 39 sections kept within the 12288-character fold (sections that name or emphasize a rule or mandate are kept first); omitted: Framework vs State Split (4-Tier Architecture); Core Systems; Agent-dir Resolution; Session Binding (Phase 2.6); Convention Index; Tool Usage + Write Permissions; Session Start Protocol; Agent-Session Binding; External Knowledge Hubs; User Control Commands; Auto-Session Continuation; Compact Checkpoint Protocol; +3 more; read the file for the omitted sections]