# bench/ — Lane-D model-routing benchmark harness

Headless task runner that drives the zak-code agent to completion in an isolated temp
workspace, then grades it with a **held-out oracle** (`verify.py`). Used to compare model
+ provider + tool-calling-mode combinations for the `deep_code` / `delegate` zakpick tier.

This harness is **excluded from the package gate** (`ruff` via `extend-exclude`, `pytest`
via `testpaths = ["tests"]`, `mypy` via `packages = ["zakcode"]`) — it is experiment code,
not shipped library code. Run it manually; it is not part of CI.

## Layout

```
bench/
  run_task.py            # run ONE task headless, print a JSON result
  run_suite.py           # run several tasks in parallel, print a summary table
  probe_tool_use_failed.py  # diagnostic: inspect a provider's raw text vs tool_calls
  tasks/
    01-wordfreq/         # task.json + held-out verify.py
    02-median-bug/       # + workspace/ seed files (bugfix task)
    03-lru/
    04-todo-cli/         # multi-file CLI stretch task (store.py + cli.py)
  results/               # run artifacts (.log/.json) — gitignored, regenerable
```

A task dir holds `task.json` (`{id, title, prompt, max_iterations?, max_cost_usd?,
verify_timeout_s?}`), an optional `workspace/` seed copied into the temp run dir, and a
`verify.py` that exits 0 on success (run with `cwd` = the temp workspace).

## Running (from the Zak-Code repo root, with the repo venv)

```bash
# one task
./.venv/Scripts/python.exe bench/run_task.py bench/tasks/01-wordfreq

# cheap API smoke (constructs the agent, exercises cost/permission API, NO LLM call)
./.venv/Scripts/python.exe bench/run_task.py --preflight bench/tasks/01-wordfreq

# the suite (see run_suite.py header for flags)
./.venv/Scripts/python.exe bench/run_suite.py
```

## Benchmarking a different model / provider (no code change)

`run_task.py` reads two env vars to swap the `deep_code` + `delegate` category model so
the same tasks run against any litellm-supported supplier. The litellm string is
`<source>/<model>`.

```bash
# default deep tier (gpt-4o-mini, openai) — current main default
./.venv/Scripts/python.exe bench/run_task.py bench/tasks/04-todo-cli

# Groq open model in TEXT tool-calling mode (the Groq-only fork path)
ZBENCH_DEEP_MODEL=llama-3.3-70b-versatile ZBENCH_DEEP_SOURCE=groq \
  ZAKCODE_TOOL_CALLING_MODE=text \
  ./.venv/Scripts/python.exe bench/run_task.py bench/tasks/04-todo-cli

# any other supplier: ZBENCH_DEEP_SOURCE ∈ {openai, gemini, deepseek, fireworks_ai,
#   together_ai, groq, local/ollama, ...}
```

`ZAKCODE_TOOL_CALLING_MODE` ∈ `auto` | `native` | `text` controls the protocol.

## Baseline results (deep set 01-wordfreq / 03-lru / 04-todo-cli, 2026-06-18)

| model (mode) | pass | $/task | cache-hit | notes |
|---|---|---|---|---|
| **openai/gpt-4o-mini (native)** | **3/3** | **$0.0074** | **85%** | reliable native tools incl. multi-file 04; current `deep_code` default |
| groq/openai/gpt-oss-120b (native) | 0/3 | — | — | malformed native tool calls → Groq `tool_use_failed`; reasoning model → text-mode returns empty content (broken both modes) |
| groq/llama-3.3-70b-versatile (text) | 2/3 | ~$0.03 | 0% | works for focused tasks; **04 stalled** (turn-1 prose, no tool call); Groq does not cache it |
| groq/llama-3.3-70b-versatile (native) | 0/2 | $0.022 | 0% | pseudo-XML rejected (`tools_unreliable`) → provider_error / doom_loop |

Upstream confirmation of the Groq tool-call failure mode: pydantic-ai #4350, OpenHands #10187.

## Open testing backlog

See the handoff brief. In short: (1) supplier comparison — Gemini 2.5 Flash / DeepSeek V3 /
Fireworks via `ZBENCH_*`; (2) flag `gpt-oss-120b` `tools_unreliable` in `registry.py`
(ripples into `AvailabilityResolver`); (3) `recipe_stalled` over-fires on successful runs
(01 + 04 pass the oracle but report `stop_reason=recipe_stalled`); (4) `04` stall on
llama-text; (5) broaden task coverage with harder multi-file tasks.

## Constraints

`.env` (gitignored) holds the real `OPENAI_API_KEY` + `GROQ_API_KEY`; litellm reads them
directly. **Never commit/print keys.** You need your own keys for whatever providers you
test. Local model option is `llama.cpp`'s `llama-server` (source `local`), not Ollama.
