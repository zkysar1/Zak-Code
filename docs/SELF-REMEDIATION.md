# Safe self-remediation — research findings + the design path for Zak Code

**Question.** How should an autonomous agent decide whether an action it wants to take —
especially *self-remediation* (installing a dependency, editing its own config, running a
shell command to fix its own environment) — is safe to do **without** a human, versus when
it must escalate? Is the answer "a better classifier for when an action is bad," or
something else?

**Short answer.** A better classifier is the **weakest and last** layer, not the answer.
The state of the art (2024–2026) converges hard on a **layered design that contains rather
than perfectly predicts**: a deterministic never-waivable floor, an OS-level sandbox that
bounds the blast radius *regardless of any classifier's accuracy*, least-privilege tool
restriction, and only then a risk classifier for the residual "gray zone" — with escalation
to the human when prior consent is unclear. For Zak Code specifically, the single most
actionable next step is a **declared-vs-undeclared dependency gate** (auto-install what is
already in the lockfile/manifest; escalate or block anything new, and `curl | bash`), which
structurally defuses supply-chain and prompt-injection attacks without needing a smarter
model at all.

This doc summarizes the researched evidence (cited) and maps it onto Zak Code's existing
permission model to give a concrete, ordered roadmap. (Source: a fan-out deep-research pass,
2026-06-12, adversarially verified.)

---

## 1. Why "just a better classifier" is the wrong place to put your trust

The empirical case against trusting any single model-as-judge is strong, and it comes from
the people who ship these systems:

- **Anthropic's own production Claude Code "auto mode" classifier** (a dedicated transcript
  classifier on Sonnet 4.6, reasoning-blind to resist manipulation) self-reports a **17%
  false-negative rate** on real overeager actions. Tellingly, *"in the majority of misses…
  the classifier correctly identifies the danger. What it gets wrong is whether a real
  consent signal in the session actually covers THIS action"* — e.g. "Clean up the PR"
  didn't name force-pushing. **The hard problem isn't spotting danger; it's scoping
  consent.** [[claude-code-auto-mode]]
- **R-Judge** (569 multi-turn records, 27 risk scenarios): the best model (GPT-4o) reaches
  only **~74% F1**, and *"no other models significantly exceed random."* [[r-judge]]
- **SABER** (716 executable tasks, scored on final environment state): even the strongest
  frontier model (Claude Opus 4.6) commits harmful safety violations in **>54%** of
  stateful-workspace runs — *"current alignment remains insufficient for realistic project
  environments."* [[saber]]

A classifier whose miss rate is ~17–25% is fine as a *backstop on already-contained
actions* and catastrophic as a *sole gate* — because the cost of one false negative
(a destructive action wrongly approved) is unbounded, and the judge is itself *attackable*
(a prompt-injected agent can argue its own judge into "this is safe"). **The reliability of
the judge is a ceiling you design around, not a number you wait to improve.**

> Direct implication for Zak Code: our judge may be a **small/local model** (a stated
> target). Local models sit well below the GPT-4o ceiling above, so leaning on a classifier
> is *even less* viable for us — which pushes more weight onto the deterministic and sandbox
> layers, not less.

## 2. The convergent layered pattern (what actually ships)

Across Claude Code, OpenAI Codex, and the research systems (Progent, Conseca, GuardAgent),
the same four-layer shape recurs:

### Layer 1 — a deterministic floor that front-runs the model
Claude Code's auto mode applies a **fixed first-match order**: explicit allow/deny rules
resolve immediately **except** writes to protected paths, which route to the safety check
*even when an allow rule matches* (`the safety check runs before Claude Code evaluates allow
rules`); read-only and in-working-dir edits auto-approve; *everything else* goes to the
classifier. Critically, **on entering autonomy, broad allow rules that grant arbitrary code
execution are dropped** — blanket `Bash(*)`, wildcarded interpreters like `Bash(python*)`,
package-manager run commands, sub-agent rules — while *narrow* rules like `Bash(npm test)`
carry over. A pre-approved wildcard can't silently waive the gate. [[permission-modes]]
[[claude-code-auto-mode]]

### Layer 2 — OS-level sandboxing: contain, don't predict
OpenAI Codex makes the split explicit: **"Sandbox mode defines what Codex can do
technically… Approval policy determines when Codex must ask,"** with OS mechanisms that
*"prevent Codex from accessing unintended system resources regardless of model
instructions."* Defaults: **network access OFF**, writes **confined to the workspace**
(cwd + /tmp). Concretely: macOS Seatbelt (`sandbox-exec`); Linux **bubblewrap + seccomp**
(`--unshare-user/--unshare-pid/--unshare-net`, `PR_SET_NO_NEW_PRIVS`, a seccomp filter
blocking `connect`/`bind`/`ptrace`, plus Landlock for a filesystem layer). **This is the
substrate that holds regardless of classifier accuracy** — a wrong call inside the box
can't reach anything that matters. [[codex-security]]

### Layer 3 — least-privilege tool restriction
On **AgentDojo**, a deterministic *tool filter* (restrict which tools the agent may call for
the planned task, decided *before* it sees untrusted data) was the **single most effective
defense**: targeted attack-success fell **57.7% → 6.84%** while *keeping* benign utility at
73% — better on **both** axes than an LLM-based injection detector (which cut attacks but
collapsed utility from 69% → 41.5%, the concrete approval-fatigue failure mode). Its
structural limit: it fails *"when the tools required to solve the task are also sufficient to
carry out the attack"* (17% of cases) — necessary, not sufficient, which is exactly why the
sandbox under it matters. [[agentdojo]]

### Layer 4 — a classifier for the gray zone only, with consent-aware escalation
Only the residual band — between "obviously fine" (read a file) and "never" (the floor) —
reaches a model judge, and when prior consent is ambiguous it **escalates to the human**.
The research systems sharpen this further: **separate fallible LLM policy *authoring* from
deterministic *enforcement*.** Progent has an LLM generate/update a JSON policy DSL while a
**symbolic engine enforces it** (rules sorted by priority, forbid-before-allow, args
validated, *"enforced deterministically"* — no LLM judgment at runtime; an SMT solver
classifies each policy update as a *narrowing* [auto] or *expansion* [needs approval]).
Conseca (Google) likewise generates just-in-time policies but keeps *"enforcement
deterministic, making it impervious to prompt injection."* **The model may *propose* the
rule; only a deterministic engine *applies* it.** [[progent]] [[conseca]]

### The line is drawn at *scope boundaries*, not classifier confidence
Both Codex and Claude Code draw the autonomous-vs-ask line at the **workspace + network
edge** (in-workspace edits auto-run; out-of-workspace or network access asks) — a
*predictable, deterministic* line — not at a model's per-action risk score. [[codex-security]]
[[permission-modes]]

### The self-remediation line: declared vs undeclared
For our exact question, the production-validated rule is **provenance, not the act**: Claude
Code auto mode **allows** *"installing dependencies declared in your lock files or
manifests"* and **blocks** *"downloading and executing code, like `curl | bash`."* An
*undeclared* install — the typosquatting / malicious-dependency vector — is simply **not on
the allow list**. Combined with egress-deny + sandboxing, even a prompt-injected install is
structurally contained to declared, version-pinned packages. [[permission-modes]]

---

## 3. Where Zak Code stands today (layer by layer)

Zak Code is **architecturally well-aligned** with the convergent pattern — it already has
the policy layers; the gap is the *containment substrate*.

| Layer | State of the art | Zak Code today | Gap |
| --- | --- | --- | --- |
| **1. Deterministic floor** | allow/deny front-runs the model; protected-path & arbitrary-exec wildcards never waivable | **Strong.** Deny-first, enforced outside the model's reach (`permissions.py`); never-waivable `DANGEROUS_PATTERNS` blocklist; `autonomous` = hard-deny on a dangerous match; `curl\|bash`/pipe-to-shell already in the blocklist. | (a) No declared-vs-undeclared install gate. (b) A session "allow bash" grant or a per-tool `allow` override can still cover a (non-dangerous) broad command — no "drop arbitrary-exec breadth on entering autonomous." |
| **2. OS sandbox** | bubblewrap+seccomp / Seatbelt / containers; net off, writes workspace-only — *the* enabler | **Weakest layer.** Userspace path guard (`_safety.py`) + a **best-effort** egress proxy that its own docstring says a process can bypass with raw sockets. No namespace/seccomp/container. | The big one. Without a real sandbox, "self-remediation" is only as safe as the floor + classifier — i.e. not safe enough to widen. Already tracked as the parity-review Executor item. |
| **3. Tool restriction** | least-privilege tool filter (AgentDojo's best defense) | **Partial.** Schema-filtered sub-agents (a planner structurally can't edit); per-tool trust tiers; tool budgets. | No *per-task* tool filter that narrows the toolset to the planned work before untrusted data is seen. |
| **4. Gray-zone classifier** | model proposes, deterministic engine enforces; escalate on unclear consent | **None** (deliberately — deterministic-only). | A future option, but **lowest priority** for us, and weakened by the local-model constraint. |

## 4. Recommended roadmap (ordered by value, each independently shippable)

The order is deliberate: **floor → containment → then, only maybe, a classifier.** Each step
makes self-remediation safer *without* trusting a smarter judge.

**Step 1 — Declared-dependency self-remediation gate (S–M; ship first). ✅ SHIPPED.**
A new check on package-install commands (`pip install`, `uv add/pip`, `npm/pnpm/yarn add`,
`poetry add`): **auto-allow only when every named package already appears in the repo's
lockfile/manifest** (`uv.lock`, `pyproject.toml`, `package.json`/lockfile); route any new or
undeclared package to ASK (and hard-deny in `autonomous`, since there's no sandbox yet);
`curl|bash` stays blocked by the existing floor. This is deterministic, needs no model, and
directly fixes the original pain point (it lets the agent autonomously install a *declared*
dep like `ddgs` from the `[web]` extra) while closing the typosquatting/injection vector.
*This is the concrete "safe self-remediation" first move.* Backed by [[permission-modes]].

> **Implemented** in `src/zakcode/deps_gate.py` (a pure parser + manifest reader) and wired
> into `PermissionPolicy.decide` as a tighten-only check after the dangerous-pattern floor;
> toggled by the `dependency_gate` setting (default on). The parser is launcher-aware — it
> sees through `python -m pip install`, `uv pip install`, and the full-interpreter-path form
> the project's own `pip_install_hint` emits — so the self-fix path can't dodge the gate by
> spelling the install differently. `uv sync` / `npm ci` / editable + local installs pass
> through untouched. See `dependency_gate` in [CONFIG.md](CONFIG.md) and `tests/test_deps_gate.py`.

**Step 2 — Autonomy breadth-downgrade + un-waivable protected-path floor (S).**
On entering `autonomous`, drop session grants / per-tool overrides that amount to *arbitrary
code execution* breadth (a blanket "allow bash for the session" should not auto-carry into
autonomous); keep narrow, specific grants. Make writes to protected paths (`.git`, `.env`,
the venv, config) re-check the floor *even under an allow grant* — mirroring Claude Code's
"safety check runs before allow rules." Small change to `PermissionPolicy`; closes the
"pre-approved wildcard silently waives the gate" hole. Backed by [[claude-code-auto-mode]].

**Step 3 — A real Executor sandbox (L–XL; the substrate that unlocks real autonomy).**
The parity review already scoped this (pluggable `Executor` behind `run_capturing`): a
Docker/podman backend and a Linux bubblewrap+seccomp namespace mode (workspace bind-mounted,
tmpfs HOME, net off unless the egress proxy is on), Seatbelt on macOS, a restricted Job
Object/AppContainer on Windows, today's proxy+path-guard as the fallback. **This research
elevates its priority from "nice-to-have" to "the precondition for trustworthy
self-remediation"** — it is the layer that makes a classifier mistake or a missed injection
*survivable*. Report the active isolation level on every shell `ToolResult`. Backed by
[[codex-security]], [[agentdojo]].

**Step 4 — Per-task tool filter (M; after sandbox).**
Narrow the exposed toolset to what the planned task needs *before* untrusted content enters
context (AgentDojo's most effective single defense). Builds on the existing tool-budget /
schema-filter machinery. Backed by [[agentdojo]].

**Step 5 — *(Optional, last)* a gray-zone risk check that proposes, never enforces.**
Only if Steps 1–4 leave a real residual: a check that runs **only** in the narrow gray band,
phrased as *propose-a-policy-the-deterministic-engine-then-enforces* (the Progent/Conseca
shape), never as the runtime decision-maker — so a prompt-injected proposal still can't
escape the symbolic floor. Given our local-model target and the ~74%-F1 ceiling, treat this
as a **convenience to reduce asks, never a security boundary.** Backed by [[progent]],
[[conseca]], [[r-judge]].

## 5. The honest tradeoffs

- **False-negative danger vs approval fatigue.** These trade off, and a classifier improves
  *both* only marginally. Layers 1–3 improve the security axis *without* the utility cost,
  which is why they come first.
- **Classifier reliability is a ceiling, not a roadmap.** ~17–25% miss rates are the current
  reality; design so a miss is *contained*, don't wait for the number to improve.
- **Local-model constraint.** A small judge is materially worse than GPT-4o here. For Zak
  Code this is decisive: **the sandbox (Step 3) is more important than any classifier**,
  because it's the only layer whose guarantee doesn't depend on model quality.
- **Self-modification is its own risk class.** Auto-installing packages is a supply-chain
  surface (typosquatting, malicious deps) and a prompt-injection escalation path. The
  declared-vs-undeclared gate (Step 1) is what turns "let it fix itself" from dangerous into
  bounded.

**Bottom line for the original question:** the "middle" between *always ask the user* and
*let it fix itself* is **not** a smarter classifier. It is: keep the deterministic floor,
gate self-remediation on *declared provenance*, and build the *sandbox* that makes any
mistake survivable — then a classifier is an optional convenience on top, never the thing
you trust.

## Sources

- [claude-code-auto-mode] Anthropic Engineering — *Claude Code auto mode* —
  https://www.anthropic.com/engineering/claude-code-auto-mode
- [permission-modes] Claude Code docs — *Permission modes* —
  https://code.claude.com/docs/en/permission-modes
- [codex-security] OpenAI — *Codex agent approvals & security* —
  https://developers.openai.com/codex/agent-approvals-security
- [r-judge] *R-Judge: Benchmarking Safety Risk Awareness for LLM Agents* —
  https://arxiv.org/abs/2401.10019
- [saber] *SABER* (stateful agent safety benchmark) — https://arxiv.org/abs/2606.01317
- [agentdojo] *AgentDojo* (NeurIPS 2024, ETH Zurich SPY Lab) —
  https://arxiv.org/html/2406.13352v3
- [progent] *Progent: Programmable Privilege Control for LLM Agents* —
  https://arxiv.org/abs/2504.11703v1
- [conseca] *Conseca: Context-Sensitive Capabilities* (Google) —
  https://arxiv.org/abs/2501.17070
- [guardagent] *GuardAgent* (ICML 2025) — https://arxiv.org/abs/2406.09187
