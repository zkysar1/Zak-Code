"""Live decomposition eval — does the planning GUIDANCE actually make a real model author
checkable, decomposed plans?

SKIPPED by default (costs money / needs a model), same gate as the live smoke tests:

    LIVE_TESTS=1 uv run pytest tests/test_live_decomposition.py -v
    ZAKCODE_LIVE_MODEL=openai/gpt-4o-mini    (default; needs OPENAI_API_KEY + quota)
    ZAKCODE_LIVE_MODEL=ollama_chat/llama3.1  (needs `ollama serve` + the model)

A local llama.cpp ``llama-server`` (OpenAI-compatible, started with ``--jinja`` for tool
calling) is the free path — point litellm at it via ``OPENAI_API_BASE`` and a throwaway key:

    OPENAI_API_BASE=http://127.0.0.1:8080/v1 OPENAI_API_KEY=sk-noop \\
      ZAKCODE_LIVE_MODEL=openai/<model.gguf> LIVE_TESTS=1 \\
      uv run pytest tests/test_live_decomposition.py -v

The no-network ``plan-decomposition`` probe (``src/zakcode/evals/probes.py``) proves the
HARNESS runs a well-formed checkable plan to completion. This proves the OTHER half: that,
given the sharpened primitiveness + done-condition guidance in the *system prompt* (the task
below deliberately says nothing about notes or checkability), a real model DECOMPOSES a
multi-step task into several steps and records each step's checkable done-condition in its
``note``. A model that does not plan — or makes a one-step plan — SKIPS rather than fails
(weaker models sometimes won't cooperate): the eval scores the guidance's effect when the
model plans, not the model's raw capability.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_LIVE = os.environ.get("LIVE_TESTS") == "1"
_MODEL = os.environ.get("ZAKCODE_LIVE_MODEL", "openai/gpt-4o-mini")

pytestmark = pytest.mark.skipif(
    not _LIVE, reason="live-provider test; set LIVE_TESTS=1 (and ZAKCODE_LIVE_MODEL) to run"
)

#: A neutral multi-step task. It mentions NOTHING about notes, done-conditions, or checkability —
#: so any done-conditions the model records must come from the system-prompt guidance, not the
#: task text. Three distinct actions (endpoint + test + docs) reliably trip the planning guidance.
_TASK = (
    "I am building a small Python web service. Add a `/health` endpoint that returns the JSON "
    '{"status": "ok"}, add a unit test for it, and document it in the README. '
    "Plan the work first, then start."
)

#: Minimum fraction of the plan's primitive steps that must carry a `note` done-condition for the
#: guidance to count as "landing". Lenient on purpose (this is a manual, model-dependent eval); the
#: failure message dumps the whole plan so a low score is diagnosable, not mysterious.
_MIN_NOTE_COVERAGE = 0.5


def _agent(workspace: Path):  # type: ignore[no-untyped-def]
    from zakcode import Agent
    from zakcode.permissions import PermissionPolicy

    # A modest iteration cap: enough for the model to author (and begin) the plan, not enough to
    # finish the whole task and clear the plan — so the authored plan is still on the session to
    # score after the turn.
    return Agent(
        default_model=_MODEL,
        workspace_root=str(workspace),
        permission_policy=PermissionPolicy("acceptEdits"),  # allow writes, gate shell
        max_iterations=4,
    )


def _skip_on_provider_error(exc: Exception) -> None:
    """Turn an environmental provider failure (no quota / not running) into a SKIP."""
    from zakcode.providers.base import ProviderError

    if isinstance(exc, ProviderError):
        pytest.skip(f"live provider unavailable ({type(exc).__name__}): {exc}")
    raise exc


async def test_live_guidance_yields_checkable_decomposition(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    try:
        result = await agent.arun_turn(_TASK)
    except Exception as exc:  # noqa: BLE001 — classify env failures as skips
        _skip_on_provider_error(exc)
        return

    network = agent.loop.session.task_network
    leaves = network.leaves()
    # The model must have decomposed the task into multiple concrete steps to score it. A model
    # that didn't plan (or made a one-step plan) gets a SKIP, not a failure — same cooperation
    # bias as the live tool-use smoke test.
    if len(leaves) < 2:
        pytest.skip(
            f"model produced no multi-step plan (leaves={len(leaves)}, "
            f"stop={result.stop_reason}); decomposition guidance untested"
        )

    noted = [leaf for leaf in leaves if leaf.note.strip()]
    coverage = len(noted) / len(leaves)
    assert coverage >= _MIN_NOTE_COVERAGE, (
        f"guidance should make the model record per-step done-conditions, but only "
        f"{len(noted)}/{len(leaves)} primitive steps carry a `note` (coverage {coverage:.0%} < "
        f"{_MIN_NOTE_COVERAGE:.0%}).\nPlan:\n{network.render()}"
    )
