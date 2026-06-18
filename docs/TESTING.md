# Testing Zak Code

Testing an agent has two very different costs, so we keep two tiers. **Reach for tier 1 by
default** — it is where almost all iteration should happen.

| | Tier 1 — deterministic behavior tests | Tier 2 — real-model quality bench |
| --- | --- | --- |
| Where | `tests/` (`pytest`) + `src/zakcode/evals/` probes | `bench/` |
| Model | **scripted, in-memory** (no network) | a **real** provider/model (costs money) |
| Speed | the whole suite runs in ~45s | minutes per task; non-deterministic |
| Determinism | total — same input, same result | none — same prompt varies run to run |
| Catches | **control-flow** regressions (does the loop route / gate / recover / stop correctly?) | **output-quality** regressions (did the model actually solve the task well?) |
| Runs in CI | yes, on every push | no (opt-in, key-gated — see `#6` / `docs/ROADMAP.md`) |

The trap to avoid: reaching for a real-model run to answer a question tier 1 can answer
deterministically in milliseconds. "Does a doom loop now recover before it gives up?" is a
control-flow question — script it, don't pay for an API round-trip to find out.

## Tier 1: write a fast behavior test (no API)

The agent loop takes a `Provider`. In tests we pass a **scripted** one, so the "model" is just a
list of canned replies (or a callback). The loop is otherwise 100% real — real tools, real
permission gate, real session — so the test exercises genuine behavior, just deterministically.

Two building blocks:

- **`tests/` unit tests** — `ScriptedProvider` / `LoopingProvider` from `tests/test_loop_edge.py`
  replay `LLMResult`s; assert on the returned `TurnResult` (`stop_reason`, `iterations`,
  `routed_category`, the session messages, files on disk). See `test_loop_edge.py` (doom guard,
  cancellation, `max_iterations`), `test_zakpick.py` (routing), `test_recipe.py` (the
  verify-before-finish gate), `test_loop_stream.py` (the streaming twin).
- **`src/zakcode/evals/probes.py`** — the curated **behavioral probe suite**: each probe drives
  the *real* agent via a scripted provider and asserts one end-to-end behavior the project must
  never regress (completion, fail-closed safety, plan read-only, doom-halt, **doom-recovery**,
  partial-failure recovery, stuck recovery, compaction, plan decomposition). Run them with
  `zakcode eval` or
  `pytest tests/test_evals_probes.py`. CI runs them on every push.

A probe is the clearest template. The `responder` callback scripts a *dynamic* scenario — the
reply can depend on which iteration the loop is on:

```python
async def _probe_doom_loop_recovery(workspace: str) -> str:
    same = call_tool("write_file", {"path": ".../scratch.txt", "content": "x"})

    def responder(messages, system, i):     # i = 0-based loop iteration
        return same if i < 3 else reply("changing approach; done")  # repeat, then break out

    agent = make_agent(ScriptedProvider([reply("unused")], responder=responder),
                       workspace_root=workspace)
    result = await agent.arun_turn("keep writing")
    assert result.stop_reason == "completed"   # the recovery nudge rescued it, not 'doom_loop'
    return "recovered"
```

To add a probe: write the `async def _probe_*` function, then register an `EvalCase` in
`build_default_suite` and a one-line `test_probe_*` in `tests/test_evals_probes.py`.

## Tier 2: the real-model bench

`bench/` runs the agent against a real model on coding tasks and reports pass-rate, cost, and
convergence. Use it to answer *quality* questions ("is gpt-4o-mini good enough for `deep_code`?")
and to compare models/routing — never for control-flow correctness (tier 1 owns that, for free).
It is **not** part of the default gate: it needs API keys and money, and its results are
inherently noisy. See `bench/README.md`.

## The gate

```bash
uv run poe check     # ruff + format-check + mypy + pytest (tier 1) — must be green before a PR
```
