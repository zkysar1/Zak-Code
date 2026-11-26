"""The plan schema writes each field description once (ADR-0411).

``update_plan`` nests sub-steps, and every level takes the same fields. The schema is sent on every
model call, so the nested levels repeat the shape only (names, types, the status values, what is
required) and the descriptions sit on the outermost step.
"""

from __future__ import annotations

import json
from typing import Any

from zakcode.tools.builtins.update_plan import _MAX_DEPTH, UpdatePlanTool


def _levels() -> list[dict[str, Any]]:
    """The step object at each nesting level of the schema the model is sent, outermost first."""
    levels = [UpdatePlanTool.spec.parameters["properties"]["tasks"]["items"]]
    while "subtasks" in levels[-1]["properties"]:
        levels.append(levels[-1]["properties"]["subtasks"]["items"])
    return levels


def test_every_level_takes_the_outer_steps_fields() -> None:
    levels = _levels()
    assert len(levels) == _MAX_DEPTH
    outer = levels[0]["properties"]
    for level in levels:
        assert level["type"] == "object" and level["required"] == ["title"]
    for depth, level in enumerate(levels[1:], start=2):
        # The deepest level is a primitive step: it takes no ``subtasks`` of its own.
        expected = set(outer) - ({"subtasks"} if depth == _MAX_DEPTH else set())
        assert set(level["properties"]) == expected
        for name, field in level["properties"].items():
            if name == "subtasks":
                assert field["type"] == "array"
                continue
            # The shape a backend builds a tool-call grammar from is the outer one, prose aside.
            assert field == {k: v for k, v in outer[name].items() if k != "description"}


def test_each_field_description_is_sent_once() -> None:
    outer, *nested = _levels()
    sent = json.dumps(UpdatePlanTool.spec.parameters)
    for name, field in outer["properties"].items():
        assert field.get("description"), name
        assert sent.count(json.dumps(field["description"])) == 1, name
    for level in nested:
        assert not [n for n, f in level["properties"].items() if "description" in f]
    # The one place a nested level's meaning is said.
    assert "same fields as a step" in outer["properties"]["subtasks"]["description"]
