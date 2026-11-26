"""The plan schema writes each field description once (ADR-0411).

``update_plan`` nests sub-steps, and every level takes the same fields. The schema is sent on every
model call, so the nested levels repeat the shape only (names, types, the status values, what is
required) and the descriptions sit on the outermost step.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from zakcode.tools.builtins.update_plan import _MAX_DEPTH, UpdatePlanTool


def _levels() -> list[dict[str, Any]]:
    """The step object at each nesting level of the schema the model is sent, outermost first."""
    levels = [UpdatePlanTool.spec.parameters["properties"]["tasks"]["items"]]
    while "subtasks" in levels[-1]["properties"]:
        levels.append(levels[-1]["properties"]["subtasks"]["items"])
    return levels


def _keys(node: Any) -> Iterator[str]:
    """Every key anywhere inside a schema node."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _keys(value)
    elif isinstance(node, list):
        for item in node:
            yield from _keys(item)


def test_every_level_takes_the_outer_steps_fields() -> None:
    levels = _levels()
    assert len(levels) == _MAX_DEPTH
    outer = levels[0]["properties"]
    # No level adds or drops a constraint of its own: what a level accepts is its fields.
    for level in levels:
        assert set(level) == {"type", "properties", "required"}
        assert level["type"] == "object" and level["required"] == ["title"]
    assert set(outer["subtasks"]) == {"type", "description", "items"}
    for depth, level in enumerate(levels[1:], start=2):
        # The deepest level is a primitive step: it takes no ``subtasks`` of its own.
        expected = set(outer) - ({"subtasks"} if depth == _MAX_DEPTH else set())
        assert set(level["properties"]) == expected
        for name, field in level["properties"].items():
            if name == "subtasks":
                assert set(field) == {"type", "items"} and field["type"] == "array"
                continue
            # The shape a backend builds a tool-call grammar from is the outer one, prose aside.
            assert field == {k: v for k, v in outer[name].items() if k != "description"}


def test_each_field_description_is_sent_once() -> None:
    outer = _levels()[0]["properties"]
    sent = json.dumps(UpdatePlanTool.spec.parameters)
    for name, field in outer.items():
        assert field.get("description"), name
        assert sent.count(json.dumps(field["description"])) == 1, name
    # No prose anywhere below the outermost step, however deep it would sit.
    assert "description" not in set(_keys(outer["subtasks"]["items"]))
    # The one place a nested level's meaning is said, and how deep a plan goes.
    said = outer["subtasks"]["description"]
    assert "same fields as a step" in said and f"{_MAX_DEPTH} levels at most" in said
