"""ADR-0420 with ADR-0413: a ``refresh=true`` ask for a rule whose file was edited on disk.

The edited mark goes ahead of the refresh mark, so the copy the next refresh looks for stays
whole, and that refresh gets the pointer instead of the text again.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from zakcode.rules import discover_rules
from zakcode.tools.base import ToolContext
from zakcode.tools.builtins.read_rule import EDITED_HEAD, REFRESH_HEAD, REFRESHED, ReadRuleTool

_OLD = "# Errors\nAfter a remote call, run the check named in the settings file."
_NEW = "# Errors\nAfter a remote call, run: tool check-errors."


def test_a_refresh_of_an_edited_rule_keeps_both_marks_and_the_next_one_points(
    tmp_path: Path,
) -> None:
    rules = tmp_path / ".claude" / "rules"
    rules.mkdir(parents=True)
    path = rules / "error-response.md"
    path.write_text(f"{_OLD}\n", encoding="utf-8")
    registry, errors = discover_rules(tmp_path)
    assert not errors
    path.write_text(f"{_NEW}\n", encoding="utf-8")
    held: list[str] = []
    ctx = ToolContext(
        workspace_root=Path("/tmp"),
        rule_registry=registry,
        in_context=lambda text: any(text in output for output in held),
        last_tool_output=lambda: held[-1] if held else None,
    )

    def ask(**args: object) -> str:
        result = asyncio.run(ReadRuleTool().execute({"name": "error-response", **args}, ctx))
        held.append(result.output)
        return result.output

    name = "error-response"
    text = f"# {name}\n{_NEW}\n[end of rule {name!r}]"
    assert ask() == EDITED_HEAD.format(name=name) + text
    assert (
        ask(refresh=True) == EDITED_HEAD.format(name=name) + REFRESH_HEAD.format(name=name) + text
    )
    assert ask(refresh=True) == REFRESHED.format(n=1, name=name, step="")
