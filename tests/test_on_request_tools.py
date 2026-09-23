"""Rarely-used built-ins start hidden, and tool_search loads them on request (ADR-0228)."""

from __future__ import annotations

from pathlib import Path

import pytest

from zakcode import Agent
from zakcode.artifacts import artifact_from_path
from zakcode.config import Settings
from zakcode.server.app import _reader_tool_for_artifact
from zakcode.tools.base import ToolContext
from zakcode.tools.builtins.default_registry import ON_REQUEST_TOOLS, default_registry


def _agent(tmp_path: Path) -> Agent:
    return Agent(
        settings=Settings(
            default_model="scripted/test", context_window=8192, workspace_root=tmp_path
        )
    )


def _exposed(agent: Agent) -> list[str]:
    return [d["function"]["name"] for d in agent.registry.definitions()]


def _on_request_line(agent: Agent) -> str | None:
    lines = agent.loop._build_system().splitlines()
    return next((line for line in lines if line.startswith("Loaded on request")), None)


def test_the_on_request_tools_start_hidden_and_the_rest_do_not(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    exposed = _exposed(agent)
    for name in ON_REQUEST_TOOLS:
        assert agent.registry.get(name) is not None  # still registered: searchable, dispatchable
        assert name not in exposed
    assert {"Read", "Edit", "Bash", "Grep", "update_plan", "tool_search"} <= set(exposed)


def test_a_sub_agent_registry_keeps_them(tmp_path: Path) -> None:
    # The factory is unchanged: a sub-agent's registry has no tool_search to load them with.
    registry = default_registry()
    assert all(registry.is_active(name) for name in ON_REQUEST_TOOLS)


async def test_tool_search_loads_one_by_name_in_a_session_without_mcp(tmp_path: Path) -> None:
    # Without MCP the budget is 0. It limits MCP tools only; a hidden built-in still loads.
    agent = _agent(tmp_path)
    search = agent.registry.get("tool_search")
    assert search is not None
    result = await search.execute({"query": "create_docx"}, ToolContext(workspace_root=tmp_path))
    assert result.data is not None and result.data["activated"] == ["create_docx"]
    assert "create_docx" in _exposed(agent)


def test_the_prompt_names_them_until_one_is_loaded(tmp_path: Path) -> None:
    agent = _agent(tmp_path)
    line = _on_request_line(agent)
    assert line is not None and "tool_search" in line
    assert all(name in line for name in ON_REQUEST_TOOLS)
    agent.registry.activate("create_docx")
    line = _on_request_line(agent)
    assert line is not None and "create_docx" not in line and "create_xlsx" in line


def test_the_prompt_lists_none_when_nothing_can_load_them(tmp_path: Path) -> None:
    # Without tool_search the list would name tools the model has no way to load.
    agent = _agent(tmp_path)
    agent.registry.deactivate("tool_search")
    assert _on_request_line(agent) is None


@pytest.mark.parametrize(
    "filename", ["report.docx", "sheet.xlsx", "book.xlsm", "photo.png", "notes.txt"]
)
def test_the_reader_an_upload_names_is_exposed(tmp_path: Path, filename: str) -> None:
    # The server's upload prompt tells the model to inspect the file with this tool by name. A
    # file arrives unannounced, so its reader must already be in the list, not on request.
    upload = tmp_path / filename
    upload.write_bytes(b"x")
    reader = _reader_tool_for_artifact(artifact_from_path(upload, workspace_root=tmp_path))
    assert reader
    assert reader in _exposed(_agent(tmp_path))
