## Agentic coding best-practices digest

A synthesis of 2026 best practices for building terminal/agentic coding agents, organized as actionable principles. Each principle states *what*, *why it matters*, and the *concrete implication for Zak Code*.

---

### 1. The agent loop is a ReAct cycle with explicit, layered stop conditions

- **Principle.** The core is a simple loop: call the model, execute any tool calls it emits, feed results back, repeat. Production-hardened loops wrap this with multiple distinct stop conditions rather than relying on one. Mature implementations structure each iteration into phases: pre-check/compaction → thinking → self-critique → action → tool execution → post-processing.
- **Why it matters.** A naive "stop when no tool calls" loop runs away on failure, burns budget, and enters doom loops (repeating identical calls). Distinguishing *completion* from *failure* is essential to safety and cost control.
- **Implication for Zak Code.** Implement the loop with at least four independent terminators: (a) text response with zero tool calls = natural completion, (b) a configurable iteration cap (15–20), (c) a cost/token budget, (d) a doom-loop detector that halts when the same tool is called with identical args within N iterations and asks the user for guidance. Add an explicit `task_complete` tool so the agent signals success rather than the harness inferring it. Add cooperative cancellation so the user can interrupt mid-run.

---

### 2. Treat context as a finite, actively-managed resource (context engineering > prompt engineering)

- **Principle.** Context engineering owns the entire token lifecycle, not just prompt wording. "Context rot" degrades quality long before the technical token limit is reached — the *effective* window is much smaller than advertised. GPT-class accuracy can collapse (e.g., 98% → 64%) purely from how information is presented. Curate the minimal high-signal token set at every step.
- **Why it matters.** More tokens make agents worse, not better. Long-horizon coding tasks fail from context exhaustion and decay, not model weakness.
- **Implication for Zak Code.** Make context a first-class subsystem. Use just-in-time retrieval (lightweight identifiers + on-demand tool calls) instead of pre-loading files. Per-tool-type result summarization: truncate bash output differently than file reads, offload large outputs to temp files referenced by handle, and always emit truncation hints ("…12KB elided…"). Cache static prompt sections (identity, safety policy, tool schemas) separately from dynamic context for provider-level prompt caching. Build a `CLAUDE.md`-style project memory file loaded every session for conventions and architecture.

---

### 3. Compaction with checkpoints keeps long sessions alive

- **Principle.** When the conversation nears the limit, summarize older history and restart from the summary while preserving recent turns verbatim. Anthropic's context-editing + memory tooling shows ~29–39% performance lift and up to 84% token reduction on long-horizon evals.
- **Why it matters.** Without compaction, long tasks hit a wall; with naive truncation, the agent loses critical state.
- **Implication for Zak Code.** Trigger compaction at ~70–80% of the window. Preserve the last 3–5 iterations fully before compressing. Tune the compaction prompt by maximizing recall first (capture everything relevant), then improving precision. Pair compaction with Git commits as checkpoints with descriptive messages so the agent can recover state via `git log`/`git diff`. Use structured note-taking (external TODO/scratch files) and re-inject the live TODO list at the *end* of context (recent positions resist decay) to counter "instruction fade-out."

---

### 4. Few sharp tools, designed for the model, CLI-first

- **Principle.** Maintain a minimal viable toolset where the right choice is always obvious; bloated tool libraries confuse the model. Tools should be self-contained, unambiguous, return token-sparse output in natural formats (prose/markdown), and never require the model to do precise arithmetic (line counts, offsets). Crucially: prefer composing capability through scripting over hand-built specialty tools (delete `resize_image`/`convert_csv` in favor of letting the agent write a script).
- **Why it matters.** LLMs were trained on millions of CLI examples (`git`, `gh`, `jq`, `kubectl`), so CLI tools get near-100% reliability and Unix pipes enable multi-step work in one model call. Sprawling bespoke tools are the #1 cause of 6-month harness obsolescence.
- **Implication for Zak Code.** Keep a small core registry: file read/edit, shell/bash, code search, and a handful of others. Lean on the shell + standard CLIs for everything that already has a mature CLI. Design each tool with rich docs, graceful informative errors, validate_args/execute separation, enforced timeouts (30–60s), and output caps (5–10MB). Support batch/parallel execution for embarrassingly-parallel ops (read 50 files concurrently). Use background promotion for server-like long-running processes.

---

### 5. CLI for the 80%, MCP for the 20% — hybrid by design

- **Principle.** Benchmarks favor CLI heavily: ~200 tokens/op vs. 32K–82K for MCP (full schemas injected every turn — the "MCP tax"), ~17–35x cheaper, ~100% vs. ~72% raw reliability. MCP earns its place for OAuth/per-user auth, multi-tenant governance, audit trails, runtime discovery, and SaaS tools that lack CLIs (Figma, Notion, Linear). Leading agents ship both.
- **Why it matters.** Protocol choice is a cost/reliability/security tradeoff, not ideology. Picking wrong inflates cost and latency or blocks enterprise use.
- **Implication for Zak Code.** Default to CLI/shell for local dev and known tools. Add MCP only where auth, stateful remote connections, or CLI-less integrations require it. Discover MCP tools *lazily* (keyword `search_tools`, load schema on demand) and gate activation behind user approval so schemas never bloat the base prompt. Write dense (<100-token) tool descriptions. Audit any third-party MCP server before production (supply-chain CVEs are real).

---

### 6. Defense-in-depth permissions: separate reasoning from enforcement, deny-first

- **Principle.** Safety must be multiple independent OS/harness-level layers, never a single mechanism and never the model policing itself. Claude Code's model: deny-first (read-only by default), with enforcement on a *separate code path* from model reasoning — a compromised/prompt-injected model cannot override deny rules. Layers stack: prompt guardrails, schema-level restrictions, runtime approval, tool-level validation (dangerous-pattern blocklist, stale-read detection), and lifecycle hooks (exit-code-2 veto).
- **Why it matters.** Because enforcement is outside model control, even a successful prompt injection stays contained. Single-layer safety fails to catch diverse failure modes.
- **Implication for Zak Code.** Put permission checks in the harness, not the prompt. Each tool gets its own gate running a rule pipeline before execution. Maintain a `DANGEROUS_PATTERNS` blocklist (`rm -rf /`, `sudo`, DB drops, network resets). Add stale-read detection (flag files modified externally since last read). Support user-defined pre-tool hooks (JSON-over-stdin, exit code 2 = veto, may mutate args). Persist approval decisions per session ("allow this pattern for the rest of the conversation") to fight approval fatigue.

---

### 7. Sandbox with both filesystem and network isolation

- **Principle.** Effective sandboxing requires *both* boundaries simultaneously: filesystem isolation (agent confined to designated directories) AND network isolation (egress only to whitelisted domains via a proxy). Either alone is insufficient — filesystem-only allows subprocess credential theft via network; network-only allows tampering with SSH keys/configs. Enforce at OS level (Linux bubblewrap, macOS seatbelt) so subprocesses are covered. Keep credentials *outside* the sandbox, routed through a proxy.
- **Why it matters.** This reduces permission prompts ~84% while *improving* security — the agent runs autonomously inside the box and only prompts on boundary violations. It directly counters approval fatigue (inattentive click-through that defeats the permission system).
- **Implication for Zak Code.** Build a sandboxed execution mode using OS primitives, with configurable allowed directories and an egress allowlist enforced by a domain-whitelisting proxy. Auto-allow trivially-safe commands (`echo`, `cat`) inside the box; only surface a prompt when the agent reaches for something outside the defined boundary. Never place secrets inside the sandbox.

---

### 8. Sub-agents for isolation and parallelism — but only with good parent context

- **Principle.** Converge on a minimal generic pattern (Plan / Execute / Task agents) over hand-crafted domain specialists, which become legacy fast. Sub-agents run with isolated context windows and return condensed 1–2K-token summaries to the parent (Anthropic's research system beat single-agent Opus by ~90% this way). Spawn Task agents dynamically for parallel/isolated subroutines (chunk X of N). Critical caveat: "sub-agents amplify whatever context the parent has — bad context yields parallel wrong answers faster."
- **Why it matters.** Isolation prevents context bleed and lets clean focused contexts tackle subtasks; parallelism cuts wall-clock time. But fan-out multiplies a bad plan.
- **Implication for Zak Code.** Implement sub-agents as lightweight instances sharing the tool registry/LLM infra but with filtered tool access and fresh history (`message_history=None`). Give each only the dependencies it needs (mode, approval, undo managers — not global config). Have them return short summaries, not raw transcripts. Run independent subtasks concurrently and merge. Invest first in getting the parent's plan/context right before fanning out.

---

### 9. Plan/execute separation with explicit human approval

- **Principle.** Separate a read-only Plan Mode from a full-access Execute Mode. Planning should be enforced by *schema* (write tools entirely absent from the planner's toolset), which is stronger than runtime checks. The planner explores, then emits a structured plan (goal, context, files, steps, verification, risks) the human reviews and edits before execution. Studies show experienced developers plan before generating, and planning is the single highest-impact practice.
- **Why it matters.** Planning before coding catches the "80% problem" (agents do visible work but miss cross-cutting changes); approval gates build user trust and accountability — "who owns correctness."
- **Implication for Zak Code.** Add a Plan Mode (via `/plan` or intent detection) backed by a read-only Planner sub-agent. Surface the plan as an editable markdown artifact the user can trim/adjust/save. Require explicit approval to transition to execution. Allow re-entering planning if execution diverges. Persist plans as documentation/context for future runs.

---

### 10. Evaluate the agent like software: end-to-end behavioral tests + CI gating

- **Principle.** Test the agent's *behavior/contracts*, not its internals, so subsystems can be rebuilt without regression. Gate agent output on the same CI checks (lint, type, unit/integration/E2E, security scan) applied to human PRs. After a change, search for other usages of modified symbols — missed references mean the task is incomplete. Treat agent code with junior-developer-level scrutiny.
- **Why it matters.** Tests are the safety net when agents err and the behavioral contract that licenses aggressive rebuilds. The agent's productivity ceiling is set by context/eval rigor, not model quality.
- **Implication for Zak Code.** Build an eval harness covering: long-horizon (50+ turn) sessions to verify compaction; safety probes that attempt dangerous ops and confirm rejection across every layer; completion-detection tests (graceful termination, no iteration-cap hits); plan-mode tests (write tools genuinely unavailable); partial-failure/recovery tests (timeout → recovers, not doom-loops). Gate Zak Code's own self-generated changes through the project's CI before commit.

---

### 11. Observability: structured logs, real-time state, undo

- **Principle.** Pipe all output to terminal *and* log files, consolidating server/test/build logs. Emit structured logs at every phase boundary (thinking, action, dispatch, result). Expose iteration count, token usage, compaction events, model selection, and safety rejections in real time. Provide Git-snapshot-based undo. Track activity by agent/prompt/diff for adoption and audit.
- **Why it matters.** Visibility lets the agent self-monitor and recover, lets users diagnose fade-out and refine safety policies, and provides the audit trail enterprises require.
- **Implication for Zak Code.** Emit structured (JSON) phase-boundary events and surface a live status line (iterations, tokens, compaction triggers, active model/sub-agents, pending approvals). Log every tool invocation, approval, and safety rejection for post-hoc analysis. Implement Git-based snapshots for per-step undo. Log system-reminder injections to diagnose instruction decay.

---

### 12. Eager construction & a clean harness/model boundary

- **Principle.** Build the entire agent (system prompt, tool schemas, sub-agent registry) at construction time, not lazily on first call — lazy building adds first-call latency and races with MCP discovery. Keep the harness (execution, parsing tool_use, permission checks, dispatch, context mgmt) cleanly separated from the model (reasoning only), via dependency injection. Use one concrete agent class parameterized by `allowed_tools` rather than a fragile class hierarchy.
- **Why it matters.** Eager construction guarantees every agent is complete and consistent before input; the harness/model split is exactly what enables tamper-proof enforcement (Principle 6) and is widely cited as the real engineering moat.
- **Implication for Zak Code.** Adopt a factory pattern: discover skills → register sub-agents → construct main agent, validating all config/credentials up front (fail fast). Inject runtime services (approval, undo, mode, context managers) rather than hardcoding. Filter tool *schemas* per agent/mode so the model never sees unavailable tools. Plan to rewrite, not retrofit, the harness periodically — 6-month-old agent architectures are effectively legacy.

---

### 13. Engineering discipline compounds agent value (the human-factor principles)

- **Principle.** From the field-lessons literature: implement to learn (code surfaces hidden decisions; refine specs iteratively); keep specs in sync as living artifacts; document *intent* (the *why*, which neither code nor tests capture); automate everything easy and distill repeated patterns into reusable skills; reserve human attention for the hard stuff (UX, security, resilience); remember "code is cheap, maintenance isn't."
- **Why it matters.** Agentic velocity creates maintenance/security debt; expert framing and intent docs dramatically cut needless agent exploration. Rigor must increase with agents, not decrease.
- **Implication for Zak Code.** Support a living spec/intent file in the project (with metadata: last_updated, owner, scope) that the agent reads and updates. Make skills first-class and reusable (distill recurring workflows into named skills). Encourage tight feedback loops and aggressive refactoring. Bias the agent toward asking precise clarifying questions on under-specified tasks rather than guessing — expert-level prompting is what amplifies results.

---

**Sources:** [arXiv: Building AI Coding Agents for the Terminal (2603.05344)](https://arxiv.org/html/2603.05344v2) · [dbreunig: 10 Lessons for Agentic Coding](https://www.dbreunig.com/2026/05/04/10-lessons-for-agentic-coding.html) · [TeamDay: Complete Guide to Agentic Coding 2026](https://www.teamday.ai/blog/complete-guide-agentic-coding-2026) · [Anthropic: Effective Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) · [Anthropic: Claude Code Sandboxing](https://www.anthropic.com/engineering/claude-code-sandboxing) · [sshh.io: Building Multi-Agent Systems Part 3](https://blog.sshh.io/p/building-multi-agent-systems-part-c0c) · [Firecrawl: MCP vs CLI](https://www.firecrawl.dev/blog/mcp-vs-cli) · [Scalekit: MCP vs CLI Benchmarking](https://www.scalekit.com/blog/mcp-vs-cli-use) · [Sourcegraph: Agentic Coding in 2026](https://sourcegraph.com/blog/agentic-coding) · [Packmind: Context Engineering Best Practices 2026](https://packmind.com/context-engineering-ai-coding/context-engineering-best-practices/) · [Morph: Why More Tokens Makes Agents Worse](https://www.morphllm.com/context-engineering)
