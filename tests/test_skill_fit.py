"""ADR-0066, the startup half of the loud block: every skill measured against the window.

Pure verdicts (``measure_skill_fit``), the CLI's window rows (``zakcode info``) and the
banner's notice lines — hermetic, no provider, no socket.
"""

from __future__ import annotations

from typing import Any

import pytest

from zakcode.config import Settings
from zakcode.providers import litellm_provider as lp
from zakcode.skills.fit import SkillFit, flagged, measure_skill_fit


def _count(text: str) -> int:
    return len(text) // 4


def test_verdicts_match_the_loops_arithmetic() -> None:
    skills = [
        ("precheck", "x" * 200_000),  # 50,000 tokens: cannot load on a 32k model
        ("start", "x" * 68_000),  # 17,000 tokens: over half of 32,768
        ("prime", "x" * 30_000),  # 7,500 tokens: fine
    ]
    fits = measure_skill_fit(
        skills, count_tokens=_count, window=32_768, system_tokens=3_000, reserve=4_096
    )
    assert [(f.name, f.verdict) for f in fits] == [
        ("precheck", "too_large"),
        ("start", "large"),
        ("prime", "ok"),
    ]
    assert fits[0].share == pytest.approx(50_000 / 32_768)
    assert "cannot load on this model" in fits[0].describe()
    assert "over half the window" in fits[1].describe()
    assert [f.name for f in flagged(fits)] == ["precheck", "start"]


def test_the_reserve_and_system_prompt_count_against_the_body() -> None:
    # 7,000 tokens fits a 8,192 window alone, but not beside a 1,000-token system prompt
    # with 4,096 tokens of answer room — the same sum the loop refuses on in-turn.
    fits = measure_skill_fit(
        [("s", "x" * 28_000)], count_tokens=_count, window=8_192, system_tokens=1_000, reserve=4_096
    )
    assert fits[0].verdict == "too_large"
    assert (
        measure_skill_fit([("s", "x" * 28_000)], count_tokens=_count, window=8_192)[0].verdict
        == "large"
    )


def test_an_uncountable_body_is_skipped_not_guessed() -> None:
    def boom(text: str) -> int:
        raise RuntimeError("tokenizer down")

    assert measure_skill_fit([("s", "body")], count_tokens=boom, window=8_192) == []


def test_worst_first_then_biggest() -> None:
    fits = measure_skill_fit(
        [("a", "x" * 40), ("b", "x" * 400_000), ("c", "x" * 60_000)],
        count_tokens=_count,
        window=32_768,
    )
    assert [f.name for f in fits] == ["b", "c", "a"]


# ── zakcode info rows + the banner lines ────────────────────────────────────────

ZDS = {"data": [{"id": "zds-qwen3.8-27b", "zds": {"ctx_per_engine": 131072}}]}


def test_info_rows_name_the_window_its_source_and_a_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from zakcode.cli import _window_rows

    monkeypatch.setattr(lp, "_fetch_models", lambda base, key, timeout: ZDS)
    settings = Settings(
        default_model="zakpick",
        api_base="http://pod.test:9090/v1",
        zakpick_models={
            "deep_code": {"model": "zds-qwen3.8-27b", "source": "openai", "context_window": 131072},
            "plan": {"model": "zds-qwen3.8-27b", "source": "openai", "context_window": 32768},
            "classify": {"model": "zds-qwen3.8-27b", "source": "openai"},
        },
    )
    rows = dict(_window_rows(settings))
    assert "Context window (default_model)" not in rows  # the sentinel is not a model
    assert rows["Context window (zakpick 'deep_code')"] == (
        "openai/zds-qwen3.8-27b: 131,072 (config), server agrees"
    )
    assert rows["Context window (zakpick 'plan')"] == (
        "openai/zds-qwen3.8-27b: 32,768 (config) — server declares 131,072"
    )
    assert rows["Context window (zakpick 'classify')"].startswith(
        "openai/zds-qwen3.8-27b: unknown — REFUSES TO RUN"
    )
    # A category left at its built-in default resolves through the registry.
    assert rows["Context window (zakpick 'summarize')"].endswith("131,072 (registry)")


def test_banner_lines_carry_warnings_then_the_flagged_skills() -> None:
    from zakcode.cli import _window_notice_lines

    class _Agent:
        window_warnings = ["context window check: plan is configured at 32,768 but …"]

        def skill_fit_report(self) -> list[SkillFit]:
            return [
                SkillFit("precheck", 49_700, 32_768, "too_large"),
                SkillFit("start", 17_000, 32_768, "large"),
                SkillFit("prime", 7_800, 32_768, "ok"),
            ]

    lines = _window_notice_lines(_Agent())  # type: ignore[arg-type]
    assert lines[0].startswith("context window check:")
    assert lines[1] == "skill fit against the 32,768-token window (2 of 3 skills flagged):"
    assert lines[2].startswith("  /precheck 49.7k tokens (152% of the window) — cannot load")
    assert lines[3].startswith("  /start 17.0k tokens (52% of the window) — over half")
    assert len(lines) == 4


def test_banner_is_silent_when_everything_fits() -> None:
    from zakcode.cli import _window_notice_lines

    class _Agent:
        window_warnings: list[str] = []

        def skill_fit_report(self) -> list[Any]:
            return [SkillFit("prime", 7_800, 131_072, "ok")]

    assert _window_notice_lines(_Agent()) == []  # type: ignore[arg-type]
