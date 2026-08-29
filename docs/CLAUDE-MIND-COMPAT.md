# Claude-Mind ↔ Zak Code — Compatibility Map

*Empirical, code-grounded. **Last verified against `main` 2026-08-29** — every ✅ below names the
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
clone the Mind repo, run `zakcode cli` inside it, answer the folder-trust prompt once, and the
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
| `/start sera --mode autonomous` routes AND runs | Slash dispatch by name OR `triggers:` token; **the CLI runs the skill as that turn** (`Agent.compose_skill_turn`, 2026-08-19 — previously it loaded lazily and waited for a second message); sub-args threaded as `<command-args>` inside the frame | `test_slash_command_composes_an_immediate_turn`; ADR-0012 follow-ups |
| **Control skills run when the USER types them** ("Claude MUST NOT invoke /start" — Mind's own rule) | The composed slash turn opens with Claude Code's command-expansion frame (`<command-message>`/`<command-name>`/`<command-args>`) and the system prompt states the contract: a message BEGINNING with `<command-name>` means the human typed it, so user-invocable-only rules are satisfied. Without this the first live boot's model refused `/start sera` as self-invocation ("user-only command, run it yourself in the terminal" — typed from the terminal, 2026-08-19). `use_skill` loads stay `[arguments: …]`-framed — the asymmetry IS the provenance signal, pinned in both directions | `test_slash_frame_echoes_the_typed_command_under_triggers_routing`; `test_render_catalog_states_user_provenance_contract` |
| `Skill('aspirations') with args='loop'` from the model | `use_skill(name, args=…)` with the `[arguments: …]` frame — deliberately NOT the human path's `<command-args>` shape, so provenance stays legible | `test_use_skill_tool_passes_arguments_to_the_body` |
| `user-invocable: false` control skills | Enforced on the human `/<name>` path; model→skill chaining still allowed (Mind's own rule) | `test_user_invocable_false_blocks_human_path_not_model_chaining` |
| **The perpetual loop** (`stop-hook.sh` BLOCK + continue) | `Stop` → `TURN_END`; `{"decision":"block","reason":…}` veto + continuation — always on for the main loop when a hook is registered | INTEGRATIONS "Turn-end continuation" |
| **The re-entry after a BLOCK gets its skill body** (`Skill('aspirations') loop` right after the Stop hook blocked) | A TURN_END veto opens a fresh per-turn skill state (ADR-0048): the `use_skill` reload dedup and invocation budget reset the moment a hook vetoes, so the mandated re-entry is answered with the body, never an `[already loaded]` pointer. Claude Code never dedups a Skill call — the Mind's own PreToolUse gate does, and it exempts `aspirations|aspirations-*|worker-loop`. Measured 2026-08-26 on a live Mind (coach, zc-03): four vetoes each answered by the pointer, then the loop died and stayed dark ~29h | `test_a_stop_hook_veto_opens_a_fresh_skill_turn`; `test_turn_end_veto_calls_the_turn_reset` |
| PreToolUse deny / rewrite (the `MIND_SID` inject) | `permissionDecision:"deny"` + `updatedInput`; Claude-Code stdin shape (`session_id`, `tool_input`, `cwd`, `tool_response`, `transcript_path`); CC tool-name matchers (`Skill` → `use_skill`, …); the stdin names the tool as CC does (`write_file` → `Write`) with the file tools' `path` sent as workspace-resolved `file_path`, and an `updatedInput` `file_path` mapped back onto `path` (ADR-0071) | `test_claude_code_hook_contract.py` |
| `settings.json` + `settings.local.json` hooks | Both read (local over project, plus `.zakcode/settings.json`); `$CLAUDE_PROJECT_DIR` expanded; commands danger-scanned; provider keys scrubbed | `test_settings_loader.py` |
| **Hooks actually load without env gymnastics** | **Always on (ADR-0025):** declared hooks load unconditionally — no flag, no prompt, no trust file. The silent-drop failure (hooks ignored, `MIND_SID` missing four layers later) is structurally gone | `test_settings_loader.py`, `test_cc_ecosystem.py` |
| SessionStart `source` / PreCompact `trigger` (top-level) / PostToolUse `additionalContext` | All in the hook payloads | conformance lifecycle tests |
| `permissions.{allow,deny}` gestures | **Always on (ADR-0029)** — ingested into the deny-first policy (tighten-only; the catastrophic floor survives any CC allow). The sole authority over `.claude/`: the engine hardcodes no agent-config protection, so Mind's constitutional-anchor deny in `settings.local.json` is enforced exactly as written | `permissions_settings.py` + its tests |
| statusLine | Cosmetic, fail-safe, danger-scanned — `ZAKCODE_STATUS_LINE` | `status_line.py` |
| output-styles | `.claude/output-styles/<name>.md` via `outputStyle`, folded into the cacheable prompt tier — `ZAKCODE_OUTPUT_STYLE` | `output_styles.py` |
| CC-shaped transcript for hook consumers | `transcript_path` projection handed to hooks (`hooks/transcript.py` writes the `sessionId` JSONL shape Mind’s detectors read); each line stamped with the message’s EVENT time (ADR-0049), so a Mind audit that dates turns — or a human dating a loop death — reads real times, not one render instant | `hooks/transcript.py` |
| Vertex/Gemini inference for a Google-cloud Mind | `zakcode[google]` extra + ADC (metadata-server auth, no key file); missing-extra failures name the install command | CONFIG.md "Recipe: Vertex AI" |
| Block-form frontmatter (`triggers:` + `- "/start"` lines) | The YAML spelling real Mind skills actually use — 60 of 78 in a live tree — parses as lists (2026-08-20; previously silently empty, degrading trigger routing and the extras-preservation promise) | `test_parse_frontmatter_block_style_lists`; `test_block_style_triggers_route_the_slash` |
| **Served** boots — `/start <agent> --mode assistant` written to the say inbox (or POSTed to `/chat`, `/chat/stream`) | Every server door dispatches a leading slash through the SAME compose path + provenance frame as the CLI (2026-08-27, ADR-0037). Denied/unreadable runs NO turn (`/chat` 403/500; stream + watch bus get `status` + `done(skill_refused)`); unknown `/token` stays prose. Before this the served doors passed raw text, so a headless Mind (systemd `mind-serve@`, no terminal) could never run its own boot command. Once the ceremony turn ENDS the stored message keeps the frame and drops the ~23k-token body (ADR-0045), so a resumed session does not re-pay every boot it ever ran | `test_server_slash_dispatch.py`; `test_skill_turn_body_is_elided_once_the_turn_ends` |
| Cron/systemd one-shot boots (`zakcode cli -p "/start sera"`) | The one-shot path dispatches slash skills through the SAME compose path + provenance frame as the REPL (2026-08-20, closes #148). A denied/unreadable skill exits 1 — a scripted boot fails loudly, never silently sends its slash line to the model as prose; an unknown `/token` (or a thin `--server` agent) falls through as plain text. Verified live 2026-08-29 on a Mind (coach, zc-03): `nohup zakcode cli -w <mind> --dangerously-skip-permissions -p "/start coach --mode reader"` bound an observer session, ran the nested `/prime`, printed the priming summary and EXITED with the turn (11m 28s, 24 iterations, 93% cached) — a cron line is the whole scheduler. The docs said `zakcode chat` for this until then; the command has been `cli` since #204 | `test_chat_headless_slash_dispatches_the_skill` + denied/fallthrough siblings |
| **A runner's single 142-minute turn survives its own context** | Auto-compaction checks the threshold before EVERY call, not only at turn start, and the overflow compact-then-retry bound is per CALL (ADR-0074) — the per-turn count of two was spent by the third overflow of coach's first loop turn. A paged skill's dropped sections come back WITH an explanation and stop coming back after two restores (ADR-0075) — the silent restore/collapse loop that pushed that turn over its 131k window | `test_resilience_context.py` (per-call bound, mid-turn compaction, both twins), `test_skill_paging.py` (the rail with no page to turn; the third drop) |
| **A provider blip is a backoff, never a dead loop** | `provider_error` is the one turn end a Stop hook cannot veto, so every transient the provider mapping classifies as terminal kills the perpetual loop outright. A mid-stream `RateLimitError` (RESOURCE_EXHAUSTED) is unwrapped and retried (ADR-0070); a 502 `BadGatewayError` — which litellm does NOT derive from `ServiceUnavailableError` — and ANY 5xx by status code retry under the same 15-minute backoff horizon (ADR-0076). Measured 2026-08-28: coach's reducer died at /boot page 22 on the pod engine's restart, healthy again within the minute | `test_provider.py` (`test_map_error_retries_a_502_however_litellm_names_it`), `test_provider_edge.py` (the mid-stream unwrap) |
| **The compaction check sees the real prompt size, not a chars/4 guess** | A Mind's transcript is id-dense tool output (goal ids, shas, YAML, JSONL) that tokenizes ~1.6x denser than the local estimate, so ADR-0074's pre-call check stayed silent while the provider reported 129,251 of 131,072 tokens (coach, 2026-08-28) — the turn before had died at 131,297. The check is now floored by the provider's last REPORTED prompt size plus the estimate of only what was appended since, persisted for a resume, forgotten on compaction (ADR-0077) — ~25k tokens of headroom back on this content | `test_resilience_context.py` (`test_the_compaction_check_is_floored_by_what_the_provider_last_measured` + the streaming twin, the floor/forget/round-trip units) |
| **What you type at a Body reaches THAT Body** | A reducer and its worker Bodies are several sessions in ONE checkout, and ADR-0073 routed a mid-turn keystroke through the workspace say slot — one file, consumed by whichever session's loop polled first. Measured 2026-08-29 on coach: the reducer consumed an instruction typed at a worker and executed a goal it did not hold; another worker's line vanished. The keyboard is now in-process (ADR-0078): the line is handed to that session's own agent and folded in at its next boundary; the say slot remains the door for `zakcode say` / `POST /say` | `test_cli_chat.py` (`test_typed_line_mid_turn_is_injected_into_this_process_agent`), `test_loop_say.py` (the sibling-loop isolation test) |
| **A hook that lands on disk mid-session fires from the next turn** | The Mind promotes gates into `.claude/settings.json` and pulls them onto live deployments whose sessions run for hours; the hook list was read once at `Agent` construction, so a gate promoted 2026-08-29 was invisible to all four running coach sessions until their next restart. ADR-0079: `Agent.refresh_settings_hooks()` stats the settings files at every turn boundary and re-reads on change, replacing only the settings-sourced slice; a broken edit keeps the previous hooks | `test_settings_hooks_refresh.py` |
| **A worker's loop skill comes back after a compaction** | A worker Body re-enters `worker-loop` after every unit inside one turn; after a `119 → 7` compaction the re-entry returned the ADR-0063 "[already loaded] … continue from where you are" pointer with the instructions gone, and the Body hand-wrote its close (coach, 2026-08-29). ADR-0080: a compaction forgets the per-turn reload dedup (and refills the invocation budget), so the next `use_skill` delivers the body | `test_use_skill.py` (`test_compaction_forgets_the_reload_dedup`) |
| **A cut-off tool call is reported as cut off** | A 27B writing a long module in one `write_file` had its JSON truncated by the output limit; the tool answered "'path' is required" three times before the model guessed and used a heredoc (coach, 2026-08-29). ADR-0081: the loop recognizes `{"_raw": …}` arguments and says the JSON was invalid, whether it stopped mid-value, and to write the file in pieces | `test_small_model_containment.py` (the two `test_undecodable_*` tests) |
| **A compaction summary can be resumed from** | After a `154 → 7` compaction the 27B reducer's "summary" was its own last reply plus a text-format `<tool_call>` block, and the kept tail still showed `/boot` page 17's hint, so it re-ran `/boot` from page 17 — an hour of a five-agent run (coach, 2026-08-29). ADR-0082: the transcript reaches the summarizer as one user message of role-labeled text (never raw messages), model markup is stripped, and the harness appends a generated position note — the plan's current step and each paged skill's current section ("do not re-load the skill") | `test_compact_loop.py` (`test_summarize_sends_the_rendered_transcript_as_one_user_message`, `test_summary_drops_the_model_s_tool_call_and_thinking_markup`, `test_summary_carries_the_harness_position_note`) |
| **An overflow is always recoverable, and a failed compaction says why** | A worker Body overflowed at 137,486 tokens, the summarizer failed silently (the `zakcode` logger has a `NullHandler`), and every "continue" re-died the same way — the last tool result was an 87 KB skill load no summary could get under the window (coach, 2026-08-29). ADR-0083: recovery is a two-rung ladder — summarize (eliding old tool outputs if the summarizer fails), then a model-free elision of every long tool output, tail included — and each outcome is written on `last_compaction`, the status line, the terminal error and `/compact`'s reply | `test_compact_loop.py` (`test_compact_now_elides_old_tool_outputs_when_the_summarizer_fails`, `test_elide_now_reaches_the_tail`), `test_resilience_context.py` (`test_a_second_overflow_elides_without_the_model`, `test_a_transcript_nothing_can_summarize_is_elided_within_the_same_overflow`) |
| **The Mind's largest skills arrive a section at a time** | Nine of coach's biggest skills were delivered whole — `/worker-loop` 84 KB (one fenced block of `# Phase N` comments, on every worker unit), `/aspirations-spark` 116 KB (every goal), `/aspirations-evolve` 88 KB, the loop 54 KB, `/review-hypotheses` 51 KB (steps under `## Mode` headings) — because the pager cut only at step-like `##` headings (coach, 2026-08-29). ADR-0084: one outline read to a 12,000-char page budget — sections cut at their deeper markers (`###`/bold/fenced `# Phase`), then headings, then paragraphs; a documentation section holding markers is a container. worker-loop → 23 pages, spark → 27, first delivery 19 KB / 2.9 KB | `test_skill_paging.py` (`test_bold_lead_ins_and_fenced_phase_comments_page_too`, `test_a_section_over_the_budget_is_cut_at_its_markers_then_headings_then_paragraphs`, `test_a_documentation_section_holding_steps_is_a_container`) |
| **An agent can clean its own absolute paths** | A worker's `rm -rf /opt/coach-mind/yahoo/__pycache__` was hard-denied as a "recursive remove of a root or home path" — the floor flagged every absolute path, and every path a Mind agent cleans is absolute; the worker spent its goal investigating the refusal (coach, 2026-08-29). ADR-0085: the floor names the root, top-level directories and homes (`/`, `/*`, `/etc`, `~`, `~/*`, `$HOME`, `/home/<user>`, `/opt/..`); deeper paths follow the ordinary tier/mode gate | `test_permissions.py` (`test_dangerous_command_escalates_in_allow_mode`, `test_relative_recursive_rm_not_flagged`) |
| **A section closed unseen comes back with its page** | A worker's paged `/start` was closed nine steps at a time — the RUNNING branch marked done unseen, every later branch cancelled — and the worker sat idle for an hour "waiting for the next /start page" (coach-w3, 2026-08-29). ADR-0086: done-before-held is reopened and its page delivered with a rail; cancelled stays closed; a cancelled later step is not "moved past"; the pages held live in the session (`skill_pages_delivered`), not the transcript | `test_skill_paging.py` (`test_a_section_marked_done_unseen_is_reopened_and_its_page_arrives`, `test_cancelling_later_sections_does_not_finish_an_unseen_one`, `test_held_pages_survive_a_restart_that_lost_their_headers`) |

## 🟡 Real remaining gaps

- **`StopFailure` + `UserPromptExpansion` events** — recognised and skipped LOUDLY
  (`_SKIP_EVENTS` in `hooks/settings_loader.py`; the skip lands in the errors dict, never a
  silent drop). Cost to a Mind: no crash breadcrumb on a provider-error turn end, and no
  human-typed-slash telemetry distinct from `ON_SKILL_SELECTED`. Deferred until those firing
  points are designed — see the roadmap.

## ⚪ Not load-bearing (deliberately skipped)

- `ScheduleWakeup` — Mind's heartbeat is `Skill('aspirations')` re-entry, not timed wake.
- `~/.claude` mirror + `~/.claude/projects` — relocatable/advisory.
- `CLAUDE_CODE_AUTO_COMPACT_WINDOW` — graceful fallback exists.

## ➡️ Deploying a Mind on Zak Code (the live checklist)

1. `uv tool install 'zakcode[google]'` (or plain `zakcode` for non-Vertex providers) — the tool
   is installed ONCE, globally; the Mind clone is a **workspace**, never the tool's source tree.
2. Clone the Mind repo into its own directory; `cd` there; `zakcode cli`.
3. Answer the **workspace hooks** folder-trust prompt (`1` = always) — that single keystroke is
   what used to be the silent `ZAKCODE_SETTINGS_HOOKS` failure.
4. `/start <agent-name> …` — the skill runs immediately as that turn.
5. For unattended/autonomous operation THREE levers matter, and every default assumes a
   supervised human turn (measured 2026-08-20):
   - `ZAKCODE_PERMISSION_MODE` — default `ask` blocks on a human at every gated tool call
     (and fails closed with no terminal); `autonomous` is the unattended posture, with
     dangerous hook commands still hard-denied by the settings loader.
   Both former sizing knobs are GONE (2026-08-25 no-knobs ruling — their silent-off
   defaults cost a full autonomous-loop death): the Stop-hook seam is structurally
   ALWAYS ON for the main loop (a registered `Stop` hook fires at every vetoable turn
   end, vetoes unbounded — the hook stands down, the cost budget is the hard bound),
   and iterations are ALWAYS unlimited (a hard cap exists only as an SDK constructor
   arg for tests/evals). Recommended sequence: first sprout supervised
   (`/start <agent> --mode assistant`, approve prompts as they come), then flip to
   autonomous — nothing to size.

## Provenance

2026-06-22: original gap map (7 gaps listed). Between then and 2026-08-19 every one of them
shipped: triggers routing, args threading, `use_skill` args, `user-invocable` enforcement,
`settings.local.json`, `additionalContext`, SessionStart `source`, PreCompact `trigger`,
permissions ingestion, statusLine, output-styles, and the transcript projection. 2026-08-19
(Serene dogfooding engagement): immediate slash-turn execution (#144), hook folder-trust
adoption (#145), and the `google` extra + actionable Vertex error (#146) landed from the first
live deployment's findings.
