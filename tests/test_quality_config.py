"""Tests for the quality-engine config (increment 6a): defaults OFF, the judge role accepted.

6a is inert scaffolding — these pin that the new fields exist, default to today's behavior (every
seam off), accept the new ``judge`` model role, and enforce their bounds. The seams that read them
arrive in 6b–6d.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from zakcode.config import Settings


def test_quality_flags_default_off() -> None:
    s = Settings()
    assert s.quality_gate is False
    assert s.best_of_attempts == 1  # 1 = off
    assert s.quality_gate_threshold == 0.8
    assert s.quality_gate_dimensions is None


def test_judge_role_is_accepted() -> None:
    s = Settings(model_roles={"judge": "groq/qwen/qwen3-32b"})
    assert s.model_roles["judge"] == "groq/qwen/qwen3-32b"


def test_unknown_model_role_is_still_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(model_roles={"planer": "x"})  # a typo must still fail fast


def test_quality_constraints_enforced() -> None:
    with pytest.raises(ValidationError):
        Settings(quality_gate_threshold=1.5)  # > 1.0
    with pytest.raises(ValidationError):
        Settings(best_of_attempts=0)  # < 1
