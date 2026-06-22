"""Build the canonical feature-audit tracker (docs/FEATURE_AUDIT.csv).

One row per distinct, user-triggerable feature: a user story + the expected behavior grounded in
the code (from the Phase-1 inventory). Status starts 'Spec'd'; Phase 2 fills Test Result + Notes.
Rebuild any time: `python scripts/build_audit.py`.
"""

# ruff: noqa: E501 -- data rows (user stories + behaviors) are intentionally long, not wrapped.
from __future__ import annotations

import csv
import pathlib

HEADER = [
    "ID",
    "Area",
    "Feature",
    "User Story",
    "Expected Behavior (from code)",
    "Status",
    "Test Result",
    "Notes",
]

ROWS: list[tuple[str, str, str, str, str]] = []


def add(area: str, prefix: str, items: list[tuple[str, str, str]]) -> None:
    for i, (feat, story, behavior) in enumerate(items, 1):
        ROWS.append((f"{prefix}-{i:02d}", area, feat, story, behavior))


# ---------------------------------------------------------------- Agent loop & turn control
add(
    "Agent loop",
    "LOOP",
    [
        (
            "ReAct loop",
            "As the agent, I iterate model->tools->results until there are no tool calls, so I drive a multi-step task to completion.",
            "Each iteration calls the provider, executes requested tools, feeds results back; ends when the model emits no tool calls.",
        ),
        (
            "Max-iterations cap",
            "As an operator, I cap a turn's iterations, so a runaway turn can't loop forever.",
            "max_iterations (default 50) bounds a turn; exhausting it stops with stop_reason=max_iterations.",
        ),
        (
            "Shared delegation budget",
            "As an operator delegating to sub-agents, I want one shared iteration pool, so the whole tree is bounded.",
            "An IterationBudget shared by parent + sub-agents; each iteration draws one unit; empty pool stops the turn; also caps child count.",
        ),
        (
            "Cost/token ceiling",
            "As an operator, I set a USD/token ceiling, so spend can't exceed my limit.",
            "max_cost_usd / max_tokens accumulate across the tree; crossing a ceiling stops with stop_reason=budget_exhausted (non-vetoable).",
        ),
        (
            "Doom-loop detection",
            "As an operator, I want identical repeated tool batches stopped, so the agent doesn't burn budget repeating itself.",
            "Same tool batch >=3x in a row stops with doom_loop; fires before execution so exact repeats never run again; one recovery nudge first if budget allows.",
        ),
        (
            "Stuck recovery ladder",
            "As an operator, I want a struggling agent to escalate nudge->narrow->stop, so it investigates before giving up.",
            "Multi-signal stuck detector (repeated batch / all-errors / repeated-failure / no-progress) escalates: nudge, then read-only narrow, then stop with stop_reason=stuck.",
        ),
        (
            "Auto-compaction",
            "As a user on a long session, I want history auto-summarized near the context limit, so the turn never overflows.",
            "Before a turn, if the session exceeds ~80% of the window, older messages are replaced by a model summary keeping >=6 recent verbatim; tool pairs kept intact.",
        ),
        (
            "Context-window recovery",
            "As a user, I want a context-overflow to self-heal, so a too-long prompt doesn't crash the turn.",
            "A ContextWindowExceeded error forces a compaction and retries the same call in place (bounded), else ends as provider_error.",
        ),
        (
            "Cache-stable system prompt",
            "As an operator, I want a stable cacheable prompt prefix, so prompt-cache hits stay high across turns.",
            "Two-tier prompt: stable (identity, tool cheat-sheet, safety) + dynamic (env, context) split by a boundary marker so context changes don't invalidate the cache.",
        ),
        (
            "Operator identity (self.md)",
            "As an operator, I author self.md, so the agent adopts my persona/framing.",
            "A self.md in the workspace replaces the default identity line at the top of the stable prompt tier.",
        ),
        (
            "Project context discovery",
            "As a user, I want AGENTS.md/CLAUDE.md/ZAK.md and README folded in, so the agent knows my project's conventions.",
            "Walks workspace->repo root loading agent guides (all) + workspace README; content-hash deduped; per-file ~8KB and total ~32KB caps.",
        ),
        (
            "Tool cheat-sheet",
            "As the model, I want a terse grouped tool summary, so I pick the right tool without probing.",
            "Stable prompt lists one line per tool grouped by permission tier; full JSON schema still available at call time.",
        ),
        (
            "Tool exposure filter",
            "As an operator, I restrict which tools the model sees, so I can least-privilege a task.",
            "tool_exposure_allow/deny globs filter the advertised tool set; the execution seam also rejects filtered-out tools.",
        ),
        (
            "PreToolUse veto/rewrite",
            "As an integrator, I want a hook that can block or rewrite a tool call, so I can add runtime guardrails.",
            "PRE_TOOL_USE hooks run before execution; can veto (block) or rewrite args; rewrites re-check the never-waivable floors.",
        ),
        (
            "PostToolUse observe",
            "As an integrator, I want a post-execution hook, so I can audit/annotate results.",
            "POST_TOOL_USE hooks run after execution (observe-only, no veto); their message is appended to the tool result.",
        ),
        (
            "Parallel read-only tools",
            "As a user, I want independent read-only calls to run together, so multi-file reads are fast.",
            "A batch where every call is READ_ONLY_SAFE + READ_ONLY tier runs concurrently; anything else runs sequentially; result order preserved.",
        ),
        (
            "Write-grounding",
            "As a user on a weak model, I want files re-read after a write, so the model sees ground truth not its own claim.",
            "After a successful write/edit the harness reads the file back and injects the real content (with a .py syntax check), defanged.",
        ),
        (
            "Recipe cursor",
            "As a user, I want a created runnable script verified to run, so 'done' means it actually executes.",
            "Once a runnable file is written, the turn can't end completed until that file runs successfully (harness auto-run or model nudge); bounded, else recipe_stalled.",
        ),
        (
            "Project verification gate",
            "As an operator, I set verify_command, so a code-changing turn can't finish until my checks pass.",
            "When verify_command is set and code changed, the turn can't end completed until it runs successfully; bounded, else verification_failed.",
        ),
        (
            "Plan-finish gate",
            "As a user, I want open plan steps finished before completion, so the agent doesn't abandon a plan.",
            "At completion, an unfinished plan triggers a nudge to finish/mark steps; bounded; past the cap the turn completes flagged degraded.",
        ),
        (
            "Plan idle detection",
            "As a user, I want an abandoned plan auto-dropped, so a stale plan stops nagging.",
            "A plan byte-identical across >=3 turn-starts (never advanced) is dropped as abandoned; signature resets on any edit.",
        ),
        (
            "Plan-first gate",
            "As an operator, I require a plan before mutations, so the agent thinks before it writes.",
            "With require_plan on and no plan yet, the first mutating batch is withheld with a plan-first nudge; read-only never gated; fail-open past the cap.",
        ),
        (
            "Length auto-continuation",
            "As a user, I want a truncated final answer auto-continued, so a long answer isn't cut off.",
            "A final answer cut at the token cap (finish_reason length) is auto-continued, bounded; the turn is flagged degraded.",
        ),
        (
            "Completion-review critic",
            "As a user, I want an independent fresh-context judge before finishing, so a code-changing turn really covers the request.",
            "When completion_review_attempts>0, a separate no-transcript judge scores the request vs the claimed result; gaps send the agent back; fail-open.",
        ),
        (
            "Quality gate",
            "As an operator, I want passing-but-weak output refined, so quality clears a bar before shipping.",
            "With quality_gate on, after the verifier passes the result+written code is scored on a rubric; below threshold sends the agent back; bounded; fail-open.",
        ),
        (
            "Turn-end veto hook",
            "As an integrator, I want a Stop-hook that can keep the turn going, so an autonomous driver can continue.",
            "TURN_END hooks at a vetoable stop (completed/doom_loop/stuck) may veto and re-enter with a continuation, up to turn_end_veto_budget.",
        ),
        (
            "Turn-end observer",
            "As an integrator, I want an observe-only turn-end hook, so logging fires every turn regardless of veto budget.",
            "Observe-only TURN_END hooks fire on every turn end (any stop reason, any budget); errors fail open.",
        ),
        (
            "Streaming turn",
            "As a user, I want tokens streamed live, so I see the answer as it forms.",
            "astream_turn emits incremental AgentEvents (text deltas, tool calls/results, usage, done) with identical stop/iteration semantics to the buffered path.",
        ),
        (
            "Mid-stream tool-rejection retry",
            "As a user on Groq, I want a malformed tool call retried mid-stream, so a tool_use_failed doesn't kill the turn.",
            "A mid-stream ModelOutputRejected (tool_use_failed) is retried (resampled at raised temperature), bounded by provider_max_retries.",
        ),
        (
            "zakpick main-turn routing",
            "As an operator, I route each turn to a model by task difficulty, so easy turns stay cheap and hard turns get a strong model.",
            "With default_model=zakpick, the main generator is picked per classified category (quick/deep); re-selected only on category change.",
        ),
        (
            "Difficulty classifier",
            "As an operator, I want scope (not length) to decide cheap-vs-deep, so a short hard task still gets the strong model.",
            "For ambiguous short turns, one cheap classify call judges scope (quick/deep); fails up to deep; overrides the length heuristic.",
        ),
        (
            "Escalation latch",
            "As a user, I want a struggling turn promoted to the deep model, so it can recover.",
            "A struggle signal (doom/stuck/verify-failed) latches the turn to the deep_code model for the rest of the turn.",
        ),
        (
            "Runtime model failover",
            "As an operator, I want a failed model swapped out, so a provider outage doesn't end the turn.",
            "On a non-rate-limit provider failure, a model_failover callback supplies a replacement once per turn (before any stream output).",
        ),
        (
            "Rate-limit retry",
            "As a user, I want 429s retried with backoff, so transient rate limits don't fail my turn.",
            "RateLimited calls retry up to provider_max_retries with exponential backoff honoring Retry-After.",
        ),
        (
            "Turn-boundary persistence",
            "As a user, I want the session saved at message boundaries, so a crash/cancel never leaves inconsistent state.",
            "The session persists after each message add (never mid-iteration); a cancelled turn is saved before raising.",
        ),
        (
            "Per-model cost accounting",
            "As an operator, I want spend attributed per model, so /cost shows quick-vs-deep breakdown.",
            "Every call's Usage is tagged with the model id and folded into session + budget; /cost can group by model.",
        ),
        (
            "Decision trace",
            "As an operator debugging, I want a per-turn decision trace, so I can see every routing/recovery move.",
            "With trace_dir set, an append-only JSONL of routing + gate/recovery interventions + the final stop is written per turn; best-effort.",
        ),
        (
            "Subprocess key scrubbing",
            "As an operator, I want provider keys hidden from shell children, so a script can't leak my keys.",
            "Provider API-key env vars are scrubbed from subprocess children unless subprocess_inherit_provider_keys is set.",
        ),
        (
            "Egress sandbox",
            "As an operator, I want subprocess network egress allowlisted, so tools can't reach arbitrary hosts.",
            "With egress_proxy on, subprocess traffic routes through a localhost allowlisting proxy (egress_allowed_domains); empty list denies all.",
        ),
        (
            "Best-of-attempts on stall",
            "As a user, I want N isolated retries on a stalled turn, so a flaky task still lands a verified result.",
            "With best_of_attempts>1 + verify_command, a stalled turn copies the workspace N ways, runs the verifier in each, and applies the first verified diff back.",
        ),
    ],
)


# ---------------------------------------------------------------- Permissions
add(
    "Permissions",
    "PERM",
    [
        (
            "Permission modes",
            "As an operator, I pick a posture (deny/ask/acceptEdits/allow/autonomous), so I control how much auto-runs.",
            "Five modes order how much auto-allows; ask is the safe default; autonomous auto-allows but hard-denies dangerous patterns without prompting.",
        ),
        (
            "Deny-first authorization",
            "As an operator, I want every privileged call authorized first, so nothing dangerous runs unchecked.",
            "Each tool declares a tier; the policy checks tier-vs-mode, then dangerous patterns, dependency gate, protected paths -- checks only tighten.",
        ),
        (
            "Dangerous-command blocklist",
            "As an operator, I want catastrophic commands blocked, so 'rm -rf /' / sudo / fork bombs can't run.",
            "14 built-in regexes escalate a match to ask (hard-deny in autonomous/deny); operator regexes can only tighten.",
        ),
        (
            "Protected-path floor",
            "As an operator, I want writes to .git/.env/.venv/.claude blocked, so config/secrets aren't clobbered.",
            "A write whose path matches a protected pattern never auto-allows -- escalates to ask (hard-deny in autonomous).",
        ),
        (
            "Undeclared-install gate",
            "As an operator, I want installs of undeclared packages flagged, so typosquatting can't sneak in.",
            "When enabled, a pip/npm install of a package not in the manifests escalates (hard-deny in autonomous).",
        ),
        (
            "Session grants",
            "As a user, I want 'allow for session' to persist, so I'm not re-asked every call.",
            "ALLOW_SESSION grants persist for the conversation (honored only if mode is at least as loose as when granted); never waive the floors.",
        ),
        (
            "Confirm-on-every-call",
            "As an operator, I force a re-prompt for a named tool (e.g. web_fetch), so a sensitive tool always asks.",
            "Tools in confirm_tools escalate to ask every call regardless of tier; session-grantable; can only tighten.",
        ),
        (
            "Per-tool mode overrides",
            "As an operator, I trust/distrust a specific tool, so I can loosen or tighten it independently.",
            "tool_trust_overrides map a tool to a different mode; in autonomous a dangerous match is still a hard deny.",
        ),
    ],
)

# ---------------------------------------------------------------- Built-in tools
add(
    "Tools",
    "TOOL",
    [
        (
            "read_file",
            "As the agent, I read a workspace file (optionally line-sliced), so I can inspect code.",
            "Reads a UTF-8 file within the workspace, optional 1-based offset/limit, ~100KB cap with truncation notes; errors cleanly out-of-workspace.",
        ),
        (
            "write_file",
            "As the agent, I create/overwrite a file atomically, so I can author code safely.",
            "Atomic temp+replace; refuses shell-command-as-content and unparseable .py; returns byte count + artifact.",
        ),
        (
            "edit_file",
            "As the agent, I replace an exact string in a file, so I can make targeted edits.",
            "Replaces old_string (one or all); idempotent no-op if already applied (distinctive new_string present); same write firewalls.",
        ),
        (
            "bash",
            "As the agent, I run a shell command in the workspace, so I can build/test/run.",
            "Runs with workspace cwd, combined stdout+stderr ~64KB, 60s timeout; non-zero exit is an error; cmd.exe fallback on Windows without Git Bash.",
        ),
        (
            "powershell",
            "As the agent on Windows, I run PowerShell, so I can use PS cmdlets.",
            "Prefers pwsh, falls back to powershell.exe; same cwd/timeout/cap; same dangerous-pattern gating.",
        ),
        (
            "list_dir",
            "As the agent, I list a directory, so I can explore the tree.",
            "Lists entries (dirs suffixed /), up to 1000 shown; respects .git/.gitignore/.zakcodeignore unless include_ignored.",
        ),
        (
            "glob",
            "As the agent, I match files by glob, so I can find files by name.",
            "pathlib glob within the workspace (** recursive), up to 1000 sorted relative paths; rejects absolute patterns.",
        ),
        (
            "grep",
            "As the agent, I search file contents by regex, so I can find code.",
            "Regex over the workspace (or a path/glob scope), skips binaries+ignored, up to 1000 file:line:text rows.",
        ),
        (
            "web_search",
            "As the agent, I search the web, so I can find current info.",
            "Queries a swappable backend (ddgs/tavily/searxng), up to 10 results, all strings defanged.",
        ),
        (
            "web_fetch",
            "As the agent, I fetch a URL's text, so I can read a page.",
            "SSRF-pinned fetch (validate+pin host, re-check redirects), bounded, defanged, up to 50K chars; optional confirm gate.",
        ),
        (
            "office docs",
            "As the agent, I read/create Excel and Word files, so I can handle office documents.",
            "read_xlsx/read_docx extract data; create_xlsx/create_docx build documents atomically; downloadable artifacts; size-capped.",
        ),
        (
            "pdf",
            "As the agent, I read/create PDFs, so I can handle PDFs.",
            "read_pdf extracts text page-by-page; create_pdf flows text across pages; capped; artifacts downloadable.",
        ),
        (
            "images",
            "As the agent, I inspect/save images and create charts, so I can handle visuals.",
            "inspect_image returns metadata; save_image writes base64/data-url; create_chart renders a bar/line/pie PNG; artifacts.",
        ),
        (
            "task (delegation)",
            "As the agent, I delegate subtasks to sub-agents, so I parallelize independent work.",
            "Spawns isolated concurrent sub-agents with fresh context + shared budget; returns summaries; one-level nesting; needs a spawner.",
        ),
        (
            "update_plan",
            "As the agent, I lay out/track a plan, so I keep multi-step work organized.",
            "Full-replace HTN; loop re-numbers + derives invariants (compound status from children, one in_progress primitive); re-injected each iteration.",
        ),
        (
            "use_skill",
            "As the model, I invoke a discovered skill by name, so I load a procedure on demand.",
            "Loads the skill body via the resolver, fires ON_SKILL_SELECTED; body returned as the tool result; per-turn budget; errors if disabled/missing.",
        ),
        (
            "save_skill",
            "As the model, I save a reusable procedure, so it's available next session.",
            "Writes a kebab-case SKILL.md under .zakcode/skills (name validated so it can't escape); available next session.",
        ),
        (
            "tool_search",
            "As the model, I search+activate hidden tools, so I find MCP/extra tools on demand.",
            "Searches inactive tool names/descriptions and activates matches up to a budget.",
        ),
        (
            "deep_think",
            "As the model, I deliberate on one hard question, so I get a better answer via self-fusion.",
            "Samples N answers (default 3) from the strongest model at temp 0.7 then synthesizes one; spend counted; errors if no sampler.",
        ),
        (
            "Path-escape guard",
            "As an operator, I want file tools confined to the workspace, so the agent can't read/write outside.",
            "resolve_path rejects .. traversal, outside-absolute, and symlinks escaping the root(s); raises a clean error.",
        ),
        (
            "Write firewall",
            "As an operator, I want bad writes refused before they land, so I don't get shell-injection or broken .py.",
            "write/edit refuse literal shell-command content and .py content that won't parse, before the atomic write.",
        ),
        (
            "Ignore rules",
            "As a user, I want build/vcs noise hidden, so search/list results stay relevant.",
            "list_dir/glob/grep honor .gitignore + .zakcodeignore by default (.git always hidden); include_ignored bypasses.",
        ),
        (
            "MCP tools",
            "As an integrator, I want MCP servers' tools registered, so the model can call them.",
            "ExtensionManager discovers MCP tools, registers them as mcp__server__tool (DANGER tier, serial), failures logged not fatal.",
        ),
    ],
)


# ---------------------------------------------------------------- Providers & routing
add(
    "Providers",
    "PROV",
    [
        (
            "Vendor-agnostic provider",
            "As a developer, I want all model access behind one interface, so I can switch vendors via config.",
            "The loop depends only on Provider/LLMResult/error taxonomy; litellm is isolated to one module.",
        ),
        (
            "litellm backend",
            "As an operator, I point at any litellm model, so I can use OpenAI/Anthropic/Groq/Ollama.",
            "LiteLLMProvider maps canonical messages to wire format, calls litellm, normalizes back, maps vendor errors to the taxonomy.",
        ),
        (
            "Text tool-calling protocol",
            "As a user on a local model, I want tools to work without native function-calling, so weak models can still call tools.",
            "auto/native/text modes: text injects a <tool_call> protocol and parses it back; auto falls back to text for unreliable-native models.",
        ),
        (
            "Tool-call salvage",
            "As a user, I want stray text tool calls recovered, so a model that emits <tool_call> anyway still works.",
            "parse_text_tool_calls brace-balances and recovers <tool_call>/fenced blocks (even truncated) into canonical ToolCalls.",
        ),
        (
            "Protocol defense",
            "As an operator, I want untrusted output unable to forge tool frames, so file/web content can't inject a protocol.",
            "defang inserts zero-width breaks into frame markers + chat-template tokens; forged frames stripped from residual text.",
        ),
        (
            "Auto model resolution",
            "As a user, I set model=auto, so the engine picks an available model for me.",
            "Read-only probes pick local Ollama, then preferred external providers; tool-unreliable models skipped when tools needed; clear diagnosis if none.",
        ),
        (
            "zakpick category routing",
            "As an operator, I assign a model per task category, so I tune cost/capability per workload.",
            "model=zakpick routes quick_code/deep_code/summarize/plan/delegate/classify to configured (model,source); unset use Groq defaults.",
        ),
        (
            "Structured output",
            "As an API client, I request JSON to a schema, so I get validated structured data.",
            "json_object mode + local jsonschema validation; complete_structured repairs up to N times then returns valid=False (never raises on validation).",
        ),
        (
            "Capability registry",
            "As the engine, I look up a model's window/tools/caching, so routing and budgeting are correct.",
            "Static table -> litellm metadata -> safe default; never raises; Ollama tag matching; conservative Anthropic window pin.",
        ),
        (
            "Error taxonomy",
            "As the loop, I want normalized provider errors, so I can retry/compact/fail correctly.",
            "Vendor exceptions map to Auth/ContextWindowExceeded/RateLimited/ModelOutputRejected/RequestFailed; messages secret-redacted.",
        ),
        (
            "Groq tool_use_failed handling",
            "As a user on Groq, I want a rejected tool call retried, so a malformed native call self-corrects.",
            "A Groq tool_use_failed maps to ModelOutputRejected (a RateLimited subclass) signaling retry.",
        ),
        (
            "Usage & cost accounting",
            "As an operator, I want tokens+cost per call, so /cost is accurate.",
            "Usage carries prompt/completion/total tokens, cost, cache tokens, model; additive; Groq fallback pricing when litellm returns $0.",
        ),
        (
            "Prompt caching",
            "As an operator, I want the stable prefix cached, so repeat turns are cheaper.",
            "Anthropic gets an explicit cache_control breakpoint at the boundary marker; OpenAI auto-caches; cache_read tokens tracked.",
        ),
        (
            "True token streaming",
            "As a user, I want real token streaming, so output appears live.",
            "LiteLLMProvider.astream requests stream+usage and yields normalized deltas; base default re-emits acomplete for providers without streaming.",
        ),
        (
            "Ollama context lifting",
            "As a local user, I want Ollama's context raised, so my prompt isn't silently truncated.",
            "num_ctx is lifted to the model's real window (capped 16K) and the capability window clamped to match so compaction fires correctly.",
        ),
        (
            "Generic API base routing",
            "As a local user, I point api_base at a local server, so I can use llama.cpp/vLLM.",
            "api_base is forwarded only to generic OpenAI prefixes, never to named clouds (so it can't reroute a groq/ call to localhost).",
        ),
    ],
)

# ---------------------------------------------------------------- Quality engine
add(
    "Quality engine",
    "QUAL",
    [
        (
            "Binary judge",
            "As a developer, I want an artifact judged pass/fail on criteria, so I can gate quality.",
            "One LLM judge returns approved+issues; json_object; fail-open (errors approve).",
        ),
        (
            "Pairwise judge",
            "As a developer, I want two candidates compared, so I can pick the better one.",
            "One judge returns winner a/b/tie; fail-safe to tie; inputs clipped.",
        ),
        (
            "Judge voting",
            "As a developer, I want N independent judges to vote, so a verdict is robust.",
            "N judges at temp 0.7, strict-majority vote, deduped union of dissent issues.",
        ),
        (
            "Best-of-N selection",
            "As a developer, I want N attempts and the best picked, so quality improves via fan-out.",
            "Spawn N generations, tolerate failures, run a pairwise round-robin tournament, return the winner + usage.",
        ),
        (
            "Oracle-filter select",
            "As a developer, I want the judge to rank only candidates that actually work, so it can't pick a failing one.",
            "An oracle (tests pass) filters survivors, then the judge ranks them; if none pass, ranks all (best effort).",
        ),
        (
            "Rubric scoring",
            "As a developer, I want an artifact scored on weighted dimensions, so I can gate on a threshold.",
            "One call scores each dimension [0,1]; aggregated multiplicatively with compensation; fail-safe to 0 (keep iterating).",
        ),
        (
            "Score-and-ship gate",
            "As an operator, I want to ship at a quality bar or when out of budget, so iteration is bounded.",
            "Pennywise ships when overall>=threshold or can't afford another round, else iterates.",
        ),
        (
            "Generate-score-refine",
            "As a developer, I want a bounded refine loop, so weak output is improved then shipped.",
            "Generate->score->if weak, re-enter targeting weak dimensions; bounded; returns the best artifact seen (no regression).",
        ),
        (
            "Plan quality",
            "As a developer, I want decompositions judged/scored, so over/under-planning is caught.",
            "judge_plan runs a pairwise tournament over candidate decompositions; score_plan scores one on the plan rubric.",
        ),
        (
            "Judge-criteria hook",
            "As an integrator, I want to augment judge criteria globally, so I can fold in oracle/domain rubrics.",
            "A pluggable hook rewrites every criteria string before judging; identity/fail-safe if unset or bad.",
        ),
    ],
)


# ---------------------------------------------------------------- Context gathering
add(
    "Context gathering",
    "CTX",
    [
        (
            "Deterministic gatherer",
            "As a user, I want relevant context gathered every turn, so the model isn't blind to my workspace.",
            "With enable_context_gathering, a PRE_LLM_CALL hook collects+ranks+budget-fits candidates and injects them every turn (off by default).",
        ),
        (
            "Mentioned-files collector",
            "As a user, I want files I name surfaced, so referencing a file pulls in its content.",
            "Scans the task for path tokens; existing workspace files get cheap_score 1.0 (top priority).",
        ),
        (
            "Recent-files collector",
            "As a user, I want recently-touched relevant files surfaced, so the agent sees what I'm working on.",
            "Walks the workspace (bounded), scores by recency + task name-overlap, returns the top candidates.",
        ),
        (
            "Heuristic ranker",
            "As a user, I want zero-cost ranking by default, so gathering adds no model spend.",
            "Default classifier sorts candidates by their deterministic cheap_score; no model call.",
        ),
        (
            "Small-model ranker",
            "As an operator, I opt into model ranking, so relevance is smarter.",
            "context_classifier=model scores all candidates in one cheap batched call per turn; falls back to heuristic on failure.",
        ),
        (
            "Trained ranker",
            "As an operator, I rank with my own trained model, so relevance is tuned to my workloads.",
            "context_classifier_weights loads a RelevanceModel (logistic regression) scoring candidates; guards a feature-set mismatch; fail-soft.",
        ),
        (
            "Signal logger",
            "As an operator, I log offered-vs-used context, so I can train a ranker.",
            "context_signal_log appends per-turn JSONL of offered candidates + whether each was used; observe-only turn-end hook; best-effort.",
        ),
        (
            "Reference used-detector",
            "As an operator, I want 'used' labelled cheaply by default, so logging adds no spend.",
            "Default: a ref is used if it appears (path-normalized) in a tool-call arg or the assistant text; blind to in-context use.",
        ),
        (
            "Model used-judge",
            "As an operator, I want 'used' to catch in-context use, so the label isn't biased against injected content.",
            "context_signal_judge labels used via a cheap model judging which offered items the response drew on; unions the proxy; fail-soft.",
        ),
        (
            "Train-your-own ranker",
            "As an operator, I train a ranker from my log, so I can deploy it.",
            "train_relevance fits a pure-Python logistic regression on the signal log (no ML deps); empty/garbage log -> default prior.",
        ),
    ],
)

# ---------------------------------------------------------------- Extension seams
add(
    "Extension seams",
    "SEAM",
    [
        (
            "Lifecycle hooks",
            "As an integrator, I want session-boundary hooks, so I can prime/serialize/encode my framework state.",
            "SESSION_START (once), PRE_COMPACT, SESSION_END, ON_SKILL_SELECTED fire observe-only; failures isolated.",
        ),
        (
            "Context-injection seam",
            "As an integrator, I register a recall hook, so my memory is folded into each turn.",
            "register_context hooks return text injected as a fenced untrusted ephemeral tail message (prompt-cache safe).",
        ),
        (
            "Settings-hooks loader",
            "As an operator, I configure shell hooks in .claude/settings.json, so I reuse Claude-Code-style hooks.",
            "With settings_hooks on, hook specs load from .claude/.zakcode settings.json (event-name mapped); dangerous commands gated; provider keys scrubbed.",
        ),
        (
            "Shell hook protocol",
            "As an integrator, I write a shell hook, so a script can gate/observe.",
            "Shell hooks get a JSON payload on stdin; exit 0 allow / 2 block; stdout JSON may rewrite args (Zak-native or Claude-Code shape).",
        ),
        (
            "Plugin discovery + trust",
            "As an operator, I load trusted plugins, so I extend the agent without editing core.",
            "Plugins discovered from entry-points/user/workspace; loaded only if enabled AND trusted; untrusted code not imported until the gate passes.",
        ),
        (
            "Plugin contribution API",
            "As a plugin author, I register tools/commands/hooks, so I extend the agent through a narrow surface.",
            "PluginContext.register_tool/command/hook; contributions tracked per plugin; a report records loaded/skipped/failed.",
        ),
        (
            "Skill discovery",
            "As a user, I drop a SKILL.md, so a reusable procedure is available.",
            "Discovers SKILL.md from bundled/user/.zakcode/.claude (later overrides), parses L0 metadata eagerly, lazy-loads the body; malformed recorded not fatal.",
        ),
        (
            "Skill catalog",
            "As the model, I want a skills catalog in the prompt, so I know what I can invoke.",
            "name+description rendered into the stable (cached) prompt tier; the body loads on demand via use_skill.",
        ),
        (
            "Rules",
            "As an operator, I author always-on rules, so the agent follows my standing guidance.",
            "*.md rules from bundled/user/.zakcode/.claude render into the stable prompt tier; per-file 8KB + total 32KB caps.",
        ),
        (
            "Multi-root workspace",
            "As a user, I add extra trusted roots, so file tools can reach cross-repo paths.",
            "extra_workspace_roots widen the file-tool sandbox; path-escape guard checks all roots.",
        ),
    ],
)


# ---------------------------------------------------------------- CLI & commands
add(
    "CLI",
    "CLI",
    [
        (
            "zakcode chat",
            "As a user, I run `zakcode chat`, so I get an interactive agent session.",
            "Starts the REPL: welcome banner, daily tip, `>` prompt; streams responses; supports slash commands + free text; Ctrl-C interrupts.",
        ),
        (
            "zakcode info",
            "As a user, I run `zakcode info`, so I see my resolved config + which keys are present.",
            "Prints model/provider/workspace/permissions and provider-key presence (never the key values).",
        ),
        (
            "zakcode version",
            "As a user, I run `zakcode version`, so I know the installed version.",
            "Prints the installed version string.",
        ),
        (
            "zakcode eval",
            "As a developer, I run `zakcode eval`, so I verify behavioral invariants offline.",
            "Runs the hermetic probe suite, prints PASS/FAIL per probe, exits non-zero on any failure; -v shows detail.",
        ),
        (
            "zakcode serve",
            "As an operator, I run `zakcode serve`, so I expose the HTTP/SSE/WS API.",
            "Starts the FastAPI server on --host/--port; requires auth for non-loopback unless --insecure; shows a startup banner.",
        ),
        (
            "--model / -m",
            "As a user, I pass --model, so I choose the model (incl. auto/zakpick).",
            "Overrides default_model; shown in the banner.",
        ),
        (
            "--workspace / -w",
            "As a user, I pass --workspace, so the agent operates on a chosen repo.",
            "Overrides the workspace root for tools + session ops.",
        ),
        (
            "--session / -s (resume)",
            "As a user, I pass --session, so I resume a prior conversation.",
            "Loads the saved session by id and continues it.",
        ),
        (
            "--server (remote)",
            "As a user, I pass --server, so the CLI drives a remote engine.",
            "The CLI becomes a thin SSE client of a remote zakcode server; same renderer; headless permission prompts fail closed.",
        ),
        (
            "--no-rules",
            "As a user, I pass --no-rules, so I disable always-on rules for a run.",
            "Turns off rules discovery/injection (default on).",
        ),
        (
            "--skill-dir",
            "As a user, I pass --skill-dir, so I add a skills directory.",
            "Adds a directory scanned for SKILL.md (repeatable); the dir's repo root + local-paths.conf paths become extra roots.",
        ),
        (
            "--extra-root",
            "As a user, I pass --extra-root, so file tools can reach another path.",
            "Adds a trusted workspace root (repeatable).",
        ),
        (
            "--trace",
            "As a developer, I pass --trace, so I capture per-turn decision traces.",
            "Writes JSONL traces to .zakcode/traces (or ZAKCODE_TRACE_DIR); shows the trace dir on startup.",
        ),
        (
            "/help",
            "As a user, I type /help, so I see the available commands.",
            "Prints grouped slash-command reference; notes Ctrl-C interrupts and free text talks to the agent.",
        ),
        (
            "/model",
            "As a user, I type /model, so I see the active model / routing.",
            "Shows the model id, or the zakpick routing table + how to assign models.",
        ),
        (
            "/cost",
            "As a user, I type /cost, so I see token+cost usage.",
            "Shows cumulative tokens (total/prompt/completion), USD cost, cached tokens, and a per-model breakdown when multiple were used.",
        ),
        (
            "/permissions",
            "As a user, I type /permissions, so I see my posture + grants.",
            "Shows the active mode + session allow/deny grants.",
        ),
        (
            "/hooks",
            "As a user, I type /hooks, so I see configured hooks.",
            "Lists shell hooks (event + command) and a count of in-process hooks; dim notice if none.",
        ),
        (
            "/agents",
            "As a user, I type /agents, so I see delegatable sub-agent types.",
            "Lists sub-agent types (default marked); dim notice if delegation off.",
        ),
        (
            "/plan <task>",
            "As a user, I type /plan, so I get a read-only plan without execution.",
            "Spawns a read-only planner sub-agent (no write tools); prints the plan for review; not auto-executed.",
        ),
        (
            "/mcp",
            "As a user, I type /mcp [connect], so I list/connect MCP servers.",
            "Lists configured MCP servers + errors; /mcp connect spawns them and registers tools (gated).",
        ),
        (
            "/plugins",
            "As a user, I type /plugins, so I see plugin status.",
            "Lists loaded (with contribution counts) / skipped (reason) / failed (error) plugins.",
        ),
        (
            "/skills",
            "As a user, I type /skills, so I see discovered skills.",
            "Lists skills + descriptions, hint to invoke via /<name>, invocation count, discovery errors.",
        ),
        (
            "/compact",
            "As a user, I type /compact, so I free up context.",
            "Summarizes older history into a summary; notice if nothing to compact.",
        ),
        (
            "/clear",
            "As a user, I type /clear, so I start fresh.",
            "New Agent + session id, reset counters, same model/workspace/settings.",
        ),
        (
            "/exit",
            "As a user, I type /exit, so I leave cleanly.",
            "Closes the session and exits the REPL.",
        ),
        (
            "/<skill-name>",
            "As a user, I type /<skill>, so I load a skill's instructions.",
            "Loads the skill body into the session as ephemeral context for the next turn; error notice if it fails to load.",
        ),
        (
            "Streaming render",
            "As a user, I want a readable streamed layout, so output is legible.",
            "Hanging-indent grid: block markers, receipt connectors with durations, rail bars (red on error); code fences held until complete.",
        ),
        (
            "Wait line / spinner",
            "As a user, I want a thinking indicator, so I know it's working.",
            "A transient spinner with a rotating gerund + elapsed time; disabled off-tty / legacy console / ZAKCODE_NO_SPINNER; pauses for prompts.",
        ),
        (
            "Permission prompt",
            "As a user, I want a clear approve/deny prompt, so I control escalated calls.",
            "Shows tool+args+blast-radius+reason and 3 options (allow once/session, deny); accepts numbers/letters/phrases; unknown/non-interactive denies.",
        ),
        (
            "Welcome banner",
            "As a user, I want a startup banner, so I see model/workspace/permissions/session.",
            "One-shot box: brand, model (or 'zakpick - picks per task'), workspace, permissions, session id, help hint, daily tip.",
        ),
        (
            "Glyph fallback",
            "As a user on a constrained terminal, I want ASCII fallback, so output isn't mojibake.",
            "Tries UTF-8; falls back to ASCII glyphs on cp1252; ZAKCODE_ASCII forces ASCII.",
        ),
    ],
)

# ---------------------------------------------------------------- Server, sessions & config
add(
    "Server & sessions",
    "SRV",
    [
        (
            "GET /health",
            "As an orchestrator, I probe /health, so I know the server is live.",
            "Returns status+version; the only endpoint exempt from bearer auth.",
        ),
        (
            "GET /config",
            "As a client, I GET /config, so I read non-secret settings.",
            "Returns resolved settings with api_key stripped and credentials masked from api_base.",
        ),
        (
            "GET /tools",
            "As a client, I GET /tools, so I discover tool schemas.",
            "Lists tool definitions (name, description, tier, concurrency, params).",
        ),
        (
            "Session CRUD",
            "As a client, I create/list/fetch/delete sessions, so I manage conversations.",
            "POST/GET /sessions, GET/DELETE /sessions/{id}; corrupt sessions skipped in listing; double-delete harmless.",
        ),
        (
            "POST /chat",
            "As a client, I POST /chat, so I run one turn and get the full result.",
            "Runs a buffered turn; returns text/tool_results/usage/cost/stop_reason/iterations; 409 if a turn is inflight on the session.",
        ),
        (
            "POST /chat/stream",
            "As a client, I POST /chat/stream, so I stream a turn over SSE.",
            "Streams AgentEvent frames as SSE; persists even if the client disconnects; 409 if inflight.",
        ),
        (
            "POST /complete",
            "As a client, I POST /complete, so I get a session-less schema-valid completion.",
            "Tool-less, loop-less, session-less; validates the schema first; returns data+text+usage; provider error -> 502.",
        ),
        (
            "WS /ws/{id}",
            "As a web client, I open a WebSocket, so I get bidirectional input/interrupt/approval.",
            "Client sends input/interrupt/approval; server streams events + out-of-band permission requests; one turn at a time; 120s approval timeout.",
        ),
        (
            "Uploads & artifacts",
            "As a web user, I upload a file + download artifacts, so I can work with files.",
            "POST uploads (base64/data-url, <=50MB, sanitized, atomic, workspace-bounded); list + digest-verified download of artifacts.",
        ),
        (
            "GET /schema/events",
            "As a web client, I fetch the event schema, so I stay in lockstep with the wire.",
            "Publishes the AgentEvent JSON Schema from the same adapter that serializes frames.",
        ),
        (
            "Bundled web client",
            "As a user, I open /, so I get a browser UI.",
            "Serves index.html (pure event renderer) mounted after the API routes.",
        ),
        (
            "Bearer auth",
            "As an operator, I set ZAKCODE_AUTH_TOKEN, so the server requires a token.",
            "Bearer header on every request (except /health), constant-time compare; WS via Authorization or Sec-WebSocket-Protocol subprotocol.",
        ),
        (
            "Inflight-turn guard",
            "As an operator, I want one turn per session at a time, so concurrent turns can't corrupt state.",
            "A shared inflight set makes REST + WS refuse 409 when a turn is already running on that session.",
        ),
        (
            "SessionStore persistence",
            "As a user, I want sessions saved durably, so /resume works and a crash is recoverable.",
            "Atomic temp+replace JSON under ~/.zakcode/sessions; load validates the id (no traversal); corruption never deletes/rewrites the file.",
        ),
        (
            "Session versioning",
            "As an operator, I want forward-compatible session files, so an older file still loads.",
            "Accepts schema version <= current; rejects newer as a version error; never destroys on error.",
        ),
        (
            "Settings loading",
            "As an operator, I configure via env/.env, so I control the engine without code.",
            "Precedence: defaults -> ~/.zakcode/.env -> workspace .env -> env -> overrides; .env exported so provider keys reach litellm; secrets excluded from dumps.",
        ),
        (
            "Remote client driver",
            "As the CLI, I drive a remote server, so --server works.",
            "ServerClient health/create_session/astream_turn over HTTP+SSE; same AgentEvent contract as in-process.",
        ),
    ],
)


# Phase 2 outcomes: id -> (status, test_result, notes). Unlisted ids default to ("Tested","PASS","").
RESULTS: dict[str, tuple[str, str, str]] = {
    # --- HIGH ---
    "CLI-08": (
        "Error (high)",
        "FAIL",
        "logistical: --session/-s resume is a dead no-op; the CLI never passes session_store, so no session persists to disk. Resume non-functional despite core + SessionStore being implemented.",
    ),
    "PROV-12": (
        "Error (high)",
        "FAIL",
        "logistical: streaming reports $0 cost for all non-Groq cloud models (litellm nulls response_cost on chunks); breaks /cost and the max_cost_usd ceiling on the default streaming path.",
    ),
    # --- MED ---
    "CLI-06": (
        "Error (med)",
        "FAIL",
        "logistical: --model shown in the banner but ServerClient never sends it; the banner misreports the running model in --server mode.",
    ),
    "CLI-09": (
        "Error (med)",
        "FAIL",
        "ux: --server silently ignores --workspace/--session/--no-rules/--skill-dir/--extra-root/--trace with no warning.",
    ),
    "PERM-03": (
        "Error (med)",
        "FAIL",
        "ux: rm -rf <relative/path> over-blocked (regex matches any slash); hard-deny in autonomous, so an unattended agent cannot delete a relative subdir. Also: 15 patterns not 14.",
    ),
    "TOOL-04": (
        "Error (med)",
        "FAIL",
        "ux: bash 60s timeout is a hard ceiling (schema max 60, no override); a >60s build/test always times out -> false recipe_stalled.",
    ),
    "TOOL-05": ("Error (med)", "FAIL", "ux: powershell 60s timeout hard ceiling, same as bash."),
    "PROV-08": (
        "Error (med)",
        "FAIL",
        "logistical: complete_structured json_object-only path never tells the model the schema (not response_format, not the prompt) -> blind first attempt, wasted repair round.",
    ),
    "PROV-10": (
        "Error (med)",
        "FAIL",
        "logistical: Timeout/APIConnection/503/500 map to RequestFailed (non-retryable), so a transient error kills the turn instead of a bounded backoff-retry; plus a false code comment.",
    ),
    "QUAL-02": (
        "Error (med)",
        "FAIL",
        "logistical: pairwise_judge/best_of never swap A/B -> position bias favours the lowest index; votes>1 does not fix it (correlated). Affects best_of/select/judge_plan.",
    ),
    "SRV-07": (
        "Error (med)",
        "FAIL",
        "secret-leak: redact_secrets does not strip URL userinfo (user:password@host), so a credentialed api_base leaks the password in error bodies/logs and ChatResponse.error.",
    ),
    "SRV-17": (
        "Error (med)",
        "FAIL",
        "logistical: ServerClient has no bearer-token path, so chat --server cannot talk to a server with ZAKCODE_AUTH_TOKEN set (every call 401s).",
    ),
    # --- LOW ---
    "TOOL-08": (
        "Error (low)",
        "FAIL",
        "logistical: grep glob filter ignored when path is a single file (_gather_files returns [root] before applying the filter).",
    ),
    "PROV-03": (
        "Error (low)",
        "FAIL",
        "ux: spec wording -- auto->text fallback only applies to the Ollama prefix; Groq tools_unreliable models stay native (handled by routing instead).",
    ),
    "CTX-02": (
        "Error (low)",
        "FAIL",
        "logistical: no cross-collector dedup; a file both mentioned and recent yields two Candidates with the same ref -> injected twice, double-charges the budget.",
    ),
    "CTX-05": (
        "Error (low)",
        "FAIL",
        "ux: context_classifier is an unvalidated free string; a typo silently falls back to the heuristic with no warning.",
    ),
    "SEAM-08": (
        "Error (low)",
        "FAIL",
        "logistical: skills catalog overloads the extra_instructions slot; an injected prompt_builder that sets it silently drops the catalog while use_skill still advertises skills.",
    ),
    "LOOP-37": (
        "Error (low)",
        "FAIL",
        "ux: docstring says append-only JSONL but _dump_trace uses write_text (whole file); behavior correct (one file per turn), wording misleads.",
    ),
    "LOOP-13": (
        "Error (low)",
        "FAIL",
        "logistical: a wholly tool-exposure-filtered batch (tool_not_exposed) is not in _batch_did_no_work refund set, unlike permission_denied/hook_blocked/step_restricted -> spends a budget unit.",
    ),
    # --- PASS with a note ---
    "CTX-01": (
        "Tested",
        "PASS",
        "note (med/ux, judgment call): the context-gathering subsystem has no CLI/env surface -- Python Agent kwargs only. Intended API-only, or a gap?",
    ),
    "SRV-04": (
        "Tested",
        "PASS",
        "note (low): no test asserts the corrupt-session-skip listing behavior (code is correct).",
    ),
    "SRV-09": (
        "Tested",
        "PASS",
        "note (low): upload dir keyed by session.id[:8]; an 8-hex prefix collision shares a dir (hygiene, not a leak).",
    ),
}


# Phase 3: ids whose error has been fixed (Status -> Fixed). Grows per fix batch.
FIXED: set[str] = {
    "PROV-12",
    "PROV-08",
    "PROV-10",
    "PROV-03",
    "SRV-07",
    "SRV-17",
    "PERM-03",
    "TOOL-04",
    "TOOL-05",
    "TOOL-08",
    "LOOP-13",
    "LOOP-37",
    "CTX-02",
    "CTX-05",
    "SEAM-08",
}


def main() -> None:
    out = pathlib.Path(__file__).resolve().parents[1] / "docs" / "FEATURE_AUDIT.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for fid, area, feat, story, behavior in ROWS:
            status, result, notes = RESULTS.get(fid, ("Tested", "PASS", ""))
            if fid in FIXED:
                status, result = "Fixed", "FIXED"
            w.writerow((fid, area, feat, story, behavior, status, result, notes))
    print(f"wrote {len(ROWS)} rows to {out}")


if __name__ == "__main__":
    main()
