# Claude-Mind ↔ Zak Code — Compatibility Gap Map

*Empirical, code-grounded (read both repos, 2026-06-22). Input to a build plan, not a commitment.*

## Headline

Zak Code was **deliberately engineered toward claude-mind compatibility** — far more than a clean-room
harness would be by accident. The bridge is roughly **70–80% built**, not half. The hard part — the
**perpetual-loop engine** — already works: claude-mind's `Stop` hook maps to Zak Code's `TURN_END`
seam, which honors the `{"decision":"block","reason":…}` continuation contract and re-enters the loop.
Remaining gaps are **bounded and mostly small**; the genuinely-missing pieces (statusLine,
output-styles) are *degradations*, not loop-blockers.

> A commit `ebc1816` ("honor the Claude Code hook contract") built the hook bridge on purpose.
> Zak Code's own `docs/INTEGRATIONS.md` "deferred" section is now **stale** — it still says
> Stop-continuation + settings.json ingestion aren't done, but they are. (Fix per docs-travel-with-code.)

## How claude-mind actually works (so the map reads correctly)

- Its "slash commands" are **SKILL.md skills** (`.claude/skills/<name>/` with `triggers: ["/start"]`),
  whose bodies are **prompt templates the model executes** turn-by-turn (calling `core/scripts/*.sh`).
- Its **autonomy** is the **`Stop` hook** (`stop-hook.sh`) that BLOCKs turn-end and re-injects
  `/aspirations loop` — *that* is the perpetual loop, not the slash command.
- State lives in a **`mind_api` localhost daemon** (Python HTTP on 127.0.0.1) the framework spawns
  itself — harness-agnostic.

## ✅ Already works (no work)

- `.claude/skills/*/SKILL.md` discovery + **tolerant cognitive frontmatter** (user-invocable, triggers,
  execution_history… all preserved).
- `use_skill` invocation; skill bodies injected as the model's next instruction (same execution model
  as Claude Code).
- **The loop engine**: `Stop → TURN_END`, `{"decision":"block","reason"}` veto + continuation, bounded
  by `turn_end_veto_budget`.
- PreToolUse **deny** (`permissionDecision:"deny"`) + **`updatedInput`** rewrite (the `MIND_SID` inject).
- Hook stdin shape: `session_id`, `tool_input` (aliased), `cwd`, `tool_response`.
- `$CLAUDE_PROJECT_DIR` expansion + child-env. Claude-Code **tool-name matchers** (`Skill`→use_skill…).
- `settings.json` **hook** ingestion (gated behind `ZAKCODE_SETTINGS_HOOKS`, default off).

## 🟡 Small gaps (hours each)

| Gap | Why it matters | Fix |
|---|---|---|
| Slash dispatch on skill **name** only, not `triggers:`; sub-args dropped | `/aspirations loop`, `/start alpha --mode reader` lose args | route `/<tok>` on `extras["triggers"]`; thread the remainder into `invoke_skill` |
| `user-invocable: false` **unenforced** | model can call `/boot`/`/start` itself — loop hazard | gate `/<name>` (CLI) vs `use_skill` (model) on the flag |
| `use_skill` has no **`args`** param | `Skill('aspirations') with args='loop'` has nowhere to land | add optional `args` passthrough |
| No SessionStart **`source`** (startup/resume/compact) | can't tell compact-resume from fresh boot | add `source` to the lifecycle payload |
| PreCompact `trigger` nested under `data` (Mind reads top-level) | precompact script misreads | surface `trigger` at stdin top level |
| PostToolUse **`additionalContext`** dropped | iteration-close reminders silently lost | ~5 lines in `_parse_stdout` |
| `.claude/settings.local.json` **not read** (only settings.json) | the per-machine anchor file is ignored | add to candidate list, local-over-project precedence |

## 🟠 Medium gaps (days)

- **`StopFailure` + `UserPromptExpansion` events** unimplemented (skipped) → lose crash-recovery +
  prompt telemetry. Need new events fired in the loop.
- **`transcript_path` / CC `.jsonl` transcript** missing — Zak Code's SessionStore is its own schema.
  *Mitigated*: the load-bearing consumer (trailing-text detector) gets `last_assistant_message` directly,
  so the loop survives; only the full-transcript walk + `aspirations-rejection-audit` go dark. Fix =
  write a CC-shaped JSONL alongside SessionStore, or rewrite those scripts against `last_assistant_message`.
- **Permission ingestion** — Zak Code never reads Claude-Code `permissions.{allow,deny}` `Tool(glob)`
  gestures from settings(.local).json. It HAS a *stronger* deny model (tighten-only, catastrophic
  blocklist, `.claude/` protected) but no bridge to ingest Mind's rules and can't express
  `Read(glob)`/`AskUserQuestion(*)`/per-tool globs. Fix = a translator into PermissionPolicy's `extra_*` seams.
- **mind_api daemon startup** — decide who launches it (a packaged SessionStart hook running `mind-api-start.sh`).

## 🔴 Genuinely missing (larger; *degradations*, not loop-blockers)

- **statusLine** — no seam; Mind's context-budget render goes dark. (Zak PARITY: "Planned".)
- **output-styles** — no concept; Mind's `autonomous+explanatory` guard has no source of truth.
  (Zak has an `autonomous` *permission mode* but no output styles; the collision can't even arise yet.)

## ⚪ Not load-bearing (skip)

- `ScheduleWakeup` — the heartbeat uses `Skill('aspirations')` re-entry, not timed wake.
- `~/.claude` mirror + `~/.claude/projects` — relocatable/advisory.
- `CLAUDE_CODE_AUTO_COMPACT_WINDOW` — graceful fallback exists.

## ➡️ Minimum path to a LIVE alien-research run

The loop engine already works, so "Mind boots + runs its research loop on Zak Code" needs mostly the
**Small** items:

1. Slash dispatch on `triggers` + sub-args → `/start aliens`, `/aspirations loop` route.
2. `use_skill` `args` passthrough → loop re-entry works.
3. SessionStart `source` (+ start `mind_api` via a SessionStart hook).
4. Tolerate the `transcript_path` gap (fail-open). + enforce `user-invocable` for safety.

≈ a few days of small, contained work → then run the actual alien test. The Medium/Large items
(StopFailure, permission ingestion, statusLine, output-styles, full transcript) are a *robustness +
parity* tail, **not** loop-blockers — sequence them after the MVP run proves the concept.
