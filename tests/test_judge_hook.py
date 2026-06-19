"""Tests for the judge hook (quality engine, increment 5).

The hook is global mutable state, so an autouse fixture clears it after every test (a leak would
silently nudge unrelated judge tests). Covers identity/augment/clear, fail-safety (raising or
non-string), and that ``binary_judge`` actually runs the hook over its criteria.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from zakcode.providers.base import Capabilities, LLMResult, Provider
from zakcode.quality import apply_judge_hook, binary_judge, get_judge_hook, set_judge_hook
from zakcode.usage import Usage


@pytest.fixture(autouse=True)
def _clear_hook():
    yield
    set_judge_hook(None)  # never let a hook leak into another test


class _Approve(Provider):
    async def acomplete(  # noqa: ANN001
        self, messages, *, system=None, tools=None, response_format=None, **kwargs: Any
    ) -> LLMResult:
        return LLMResult(text=json.dumps({"approved": True}), usage=Usage(total_tokens=1))

    def count_tokens(self, messages, *, system=None) -> int:  # noqa: ANN001
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities()

    def model_id(self) -> str:
        return "judge/test"


def test_no_hook_is_identity() -> None:
    assert get_judge_hook() is None
    assert apply_judge_hook("criteria") == "criteria"


def test_set_hook_augments_criteria() -> None:
    set_judge_hook(lambda c: c + " [also weight security]")
    assert apply_judge_hook("do X") == "do X [also weight security]"
    assert get_judge_hook() is not None


def test_clearing_hook_restores_identity() -> None:
    set_judge_hook(lambda c: "nudged")
    set_judge_hook(None)
    assert apply_judge_hook("c") == "c"


def test_raising_hook_is_identity() -> None:
    def boom(c: str) -> str:
        raise RuntimeError("bad nudge")

    set_judge_hook(boom)
    assert apply_judge_hook("c") == "c"  # fail-safe: a bad nudge can't break judging


def test_non_string_hook_result_is_identity() -> None:
    set_judge_hook(lambda c: 123)  # type: ignore[return-value,arg-type]
    assert apply_judge_hook("c") == "c"


async def test_binary_judge_applies_the_hook() -> None:
    seen: list[str] = []

    def hook(c: str) -> str:
        seen.append(c)
        return c + " EXTRA"

    set_judge_hook(hook)
    await binary_judge(_Approve(), criteria="do X", artifact="A")
    assert seen == ["do X"]  # binary_judge ran the hook over its criteria
