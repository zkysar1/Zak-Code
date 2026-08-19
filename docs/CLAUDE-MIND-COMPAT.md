# Claude-Mind ↔ Zak Code — Compatibility Map

*Empirical, code-grounded. **Last verified against `main` 2026-08-19** — every ✅ below names the
code or test that proves it; re-verify the date before planning against this file.*

> **Why the date is the first thing you read:** the 2026-06-22 revision of this map listed seven
> gaps that had ALL since been closed in code. Planning a live Claude-Mind deployment against it
> cost a working session designing around fixed problems, and its "sub-args dropped" row nearly
> got re-reported as a live bug. A compatibility map is a **claim about code at a moment**; when
> the date is stale, read `docs/INTEGRATIONS.md` (kept current with code by the docs-travel-with-
> code rule) and `tests/test_cc_conformance.py` (the executable form of this contract) instead.

## Headline

**Claude-Mind runs on Zak Code.** The bridge is built and generically tested: skills, the
perpetual loop, hooks, permissions, statusLine, output-styles, and the transcript projection all
carry over. Deploying a Mind is a **configuration exercise, not an adapter project**:
clone the Mind repo, run `zakcode chat` inside it, answer the folder-trust prompt once, and the
`Stop → TURN_END` loop engine does the rest. First proven live on a GCE deployment 2026-08-19
(Vertex AI Gemini via ADC — no Anthropic dependency anywhere in the stack).

## How claude-mind actually works (so the map reads correctly)

- Its "slash commands" are **SKILL.md skills** (`.claude/skills/<name>/` with
  `triggers: ["/start"]`), whose bodies are **prompt templates the model executes**
  turn-by-turn (calling `core/scripts/*.sh`).
- Its **autonomy** is the **`Stop` hook** (`stop-hook.sh`) that BLOCKs turn-end and re-injects
  `/aspirations loop` — *that* is the perpetual loop, not the slash command.
- State lives in a **`mind_api` localhost daemon** (Python HTTP on 127.0.0.1) the framework
  spawns itself — harness-agnostic.

## ✅ The supported surface (all shipped; all conformance-tested)

| Mind needs | Zak Code has | Proof |
|---|---|---|
| SKILL.md discovery + tolerant cognitive frontmatter | `.claude/skills/*/SKILL.md` discovery; unknown keys (`execution_history`, …) preserved in `extras` | `test_cc_conformance.py` frontmatter tests |
| `/start sera --mode autonomous` routes AND runs | Slash dispatch by name OR `triggers:` token; sub-args threaded as `[arguments: …]`; **the CLI runs the skill as that turn** (`Agent.compose_skill_turn`, 2026-08-19 — previously it loaded lazily and waited for a second message) | `test_slash_command_composes_an_immediate_turn`; ADR-0012 follow-ups |
| `Skill('aspirations') with args='loop'` from the model | `use_skill(name, args=…)` with the same `[arguments: …]` frame | `test_use_skill_tool_passes_arguments_to_the_body` |
| `user-invocable: false` control skills | Enforced on the human `/<name>` path; model→skill chaining still allowed (Mind's own rule) | `test_user_invocable_false_blocks_human_path_not_model_chaining` |
| **The perpetual loop** (`stop-hook.sh` BLOCK + continue) | `Stop` → `TURN_END`; `{"decision":"block","reason":…}` veto + continuation, bounded by `turn_end_veto_budget` | INTEGRATIONS "Turn-end continuation" |
| PreToolUse deny / rewrite (the `MIND_SID` inject) | `permissionDecision:"deny"` + `updatedInput`; Claude-Code stdin shape (`session_id`, `tool_input`, `cwd`, `tool_response`, `transcript_path`); CC tool-name matchers (`Skill` → `use_skill`, …) | `test_claude_code_hook_contract.py` |
| `settings.json` + `settings.local.json` hooks | Both read (local over project, plus `.zakcode/settings.json`); `$CLAUDE_PROJECT_DIR` expanded; commands danger-scanned; provider keys scrubbed | `test_settings_loader.py` |
| **Hooks actually load without env gymnastics** | **Folder trust (2026-08-19, ADR-0013):** unset `ZAKCODE_SETTINGS_HOOKS` + interactive CLI ⇒ ask once per workspace, remember in `~/.zakcode/workspace-trust.json`. The silent-drop failure (hooks ignored, `MIND_SID` missing four layers later) is gone | `test_workspace_trust.py` |
| SessionStart `source` / PreCompact `trigger` (top-level) / PostToolUse `additionalContext` | All in the hook payloads | conformance lifecycle tests |
| `permissions.{allow,deny}` gestures | Ingested into the deny-first policy (tighten-only; the catastrophic floor survives any CC allow) — `ZAKCODE_SETTINGS_PERMISSIONS` / `Agent(enable_settings_permissions=True)` | `permissions_settings.py` + its tests |
| statusLine | Cosmetic, fail-safe, danger-scanned — `ZAKCODE_STATUS_LINE` | `status_line.py` |
| output-styles | `.claude/output-styles/<name>.md` via `outputStyle`, folded into the cacheable prompt tier — `ZAKCODE_OUTPUT_STYLE` | `output_styles.py` |
| CC-shaped transcript for hook consumers | `transcript_path` projection handed to hooks (`hooks/transcript.py` writes the `sessionId` JSONL shape Mind's detectors read) | `hooks/transcript.py` |
| Vertex/Gemini inference for a Google-cloud Mind | `zakcode[google]` extra + ADC (metadata-server auth, no key file); missing-extra failures name the install command | CONFIG.md "Recipe: Vertex AI" |

## 🟡 Real remaining gaps

- **`StopFailure` + `UserPromptExpansion` events** — recognised and skipped LOUDLY
  (`_SKIP_EVENTS` in `hooks/settings_loader.py`; the skip lands in the errors dict, never a
  silent drop). Cost to a Mind: no crash breadcrumb on a provider-error turn end, and no
  human-typed-slash telemetry distinct from `ON_SKILL_SELECTED`. Deferred until those firing
  points are designed — see the roadmap.
- **Headless one-shot slash dispatch** — `zakcode chat -p "/start sera"` sends the text to the
  model instead of dispatching the skill (the REPL dispatches; the one-shot path does not).
  Matters for cron/systemd Mind boots; interactive tmux sessions (the normal deployment shape)
  are unaffected.

## ⚪ Not load-bearing (deliberately skipped)

- `ScheduleWakeup` — Mind's heartbeat is `Skill('aspirations')` re-entry, not timed wake.
- `~/.claude` mirror + `~/.claude/projects` — relocatable/advisory.
- `CLAUDE_CODE_AUTO_COMPACT_WINDOW` — graceful fallback exists.

## ➡️ Deploying a Mind on Zak Code (the live checklist)

1. `uv tool install 'zakcode[google]'` (or plain `zakcode` for non-Vertex providers) — the tool
   is installed ONCE, globally; the Mind clone is a **workspace**, never the tool's source tree.
2. Clone the Mind repo into its own directory; `cd` there; `zakcode chat`.
3. Answer the **workspace hooks** folder-trust prompt (`1` = always) — that single keystroke is
   what used to be the silent `ZAKCODE_SETTINGS_HOOKS` failure.
4. `/start <agent-name> …` — the skill runs immediately as that turn.
5. For unattended/autonomous operation mind the permission mode: the default `ask` blocks on a
   human at every gated tool call. `ZAKCODE_PERMISSION_MODE` is the lever; dangerous hook
   commands are hard-denied in autonomous mode by the settings loader.

## Provenance

2026-06-22: original gap map (7 gaps listed). Between then and 2026-08-19 every one of them
shipped: triggers routing, args threading, `use_skill` args, `user-invocable` enforcement,
`settings.local.json`, `additionalContext`, SessionStart `source`, PreCompact `trigger`,
permissions ingestion, statusLine, output-styles, and the transcript projection. 2026-08-19
(Serene dogfooding engagement): immediate slash-turn execution (#144), hook folder-trust
adoption (#145), and the `google` extra + actionable Vertex error (#146) landed from the first
live deployment's findings.
