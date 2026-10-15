"""docs/CONFIG.md carries one short recipe for LiteLLM-gateway users, and what it quotes is
what the code reads (ADR-0309).

Asked for on 2026-10-14: what a user behind a LiteLLM 1.9x gateway in front of Bedrock/Vertex
needs to know, gathered in one short section: the per-route refusals of stop, tools and
temperature, silent fallbacks and ``disable_fallbacks``, ``ZAKCODE_EXTRA_BODY``, the per-entry
tool mode, and the ``unpriced`` display with its effect under ``ZAKCODE_MAX_COST_USD``. Until
now each lived in a setting row or an ADR. These tests hold the recipe to the code: each
refusal it quotes is read by the classifier for that refusal and by no other, the fallback
switch it gives is the JSON it names, the cost words are the renderer's, the stop reason is the
loop's, and every setting it names exists.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from zakcode.cli.render import cost_text
from zakcode.config import Settings
from zakcode.providers.base import (
    stop_field_unsupported,
    stop_length_limit,
    temperature_field_unsupported,
    tool_use_unsupported,
)
from zakcode.usage import Usage

ROOT = Path(__file__).resolve().parents[1]
HEADING = "### Recipe: behind a LiteLLM gateway"


def _doc() -> str:
    return (ROOT / "docs" / "CONFIG.md").read_text(encoding="utf-8")


def _recipe() -> str:
    doc = _doc()
    assert HEADING in doc, "docs/CONFIG.md has no recipe for LiteLLM-gateway users"
    body = doc.split(HEADING, 1)[1]
    return re.split(r"\n#{2,3} ", body, maxsplit=1)[0]


def _classifiers(text: str) -> list[str]:
    """The names of the refusal classifiers that read ``text``."""
    read = {
        "stop_field": stop_field_unsupported(text),
        "stop_length": stop_length_limit(text) is not None,
        "temperature": temperature_field_unsupported(text),
        "tool_use": tool_use_unsupported(text),
    }
    return [name for name, hit in read.items() if hit]


def test_the_recipe_sits_with_the_other_provider_recipes_and_is_short() -> None:
    doc = _doc()
    assert HEADING in doc
    assert doc.index("## Model / provider") < doc.index(HEADING) < doc.index("## Agent behavior")
    assert len(_recipe().strip().splitlines()) <= 60


def test_each_quoted_refusal_is_read_by_its_own_classifier() -> None:
    quoted = re.findall(r"^\* `([^`]+)`", _recipe(), flags=re.MULTILINE)
    assert [_classifiers(q) for q in quoted] == [
        ["stop_field"],
        ["stop_length"],
        ["temperature"],
        ["tool_use"],
    ]
    assert stop_length_limit(quoted[1]) == 10


def test_the_fallback_switch_is_the_json_it_names() -> None:
    match = re.search(r"`ZAKCODE_EXTRA_BODY=(\{[^`]*\})`", _recipe())
    assert match is not None
    assert json.loads(match.group(1)) == {"disable_fallbacks": True}


def test_the_cost_words_are_the_renderers() -> None:
    recipe = _recipe()
    unpriced = cost_text(Usage(cost_usd=0.0, cost_unpriced=True), "$0.00")
    floor = cost_text(Usage(cost_usd=0.25, cost_unpriced=True), "$x")
    assert f"`{unpriced}`" in recipe
    assert f"`{floor}`" in recipe


def test_the_stop_reason_is_the_loops() -> None:
    match = re.search(r'`stop_reason="([a-z_]+)"`', _recipe())
    assert match is not None
    loop = (ROOT / "src" / "zakcode" / "agent" / "loop.py").read_text(encoding="utf-8")
    assert f'stop_reason = "{match.group(1)}"' in loop


def test_every_setting_the_recipe_names_exists() -> None:
    recipe = _recipe()
    fields = set(Settings.model_fields)
    env = set(re.findall(r"ZAKCODE_([A-Z0-9_]+)", recipe))
    assert env
    assert {name.lower() for name in env} <= fields
    named = set(re.findall(r"`([a-z_]+)(?:\[[^\]]*\]\.[a-z_]+)?[`=]", recipe))
    settings = {"tool_calling_mode", "zakpick_models", "default_model", "temperature"}
    assert settings <= named
    assert settings <= fields


def test_the_recipe_names_each_topic() -> None:
    recipe = _recipe()
    for topic in (
        "stopSequences",
        "temperature field",
        "tool use",
        "disable_fallbacks",
        "ZAKCODE_EXTRA_BODY",
        "zakpick_models[...].tool_calling_mode",
        "`unpriced`",
        "ZAKCODE_MAX_COST_USD",
    ):
        assert topic in recipe, topic


def test_the_recipe_names_no_host() -> None:
    hosts = set(re.findall(r"https?://([^/\s`]+)", _recipe()))
    assert hosts == {"gateway.invalid"}
