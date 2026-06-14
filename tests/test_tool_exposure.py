"""Per-task tool-exposure filter (self-remediation Step 4).

Narrowing the toolset the model can see (and invoke) to a task-appropriate subset is the most
effective single prompt-injection defense (AgentDojo): a tool that is never advertised — and is
rejected at the execution seam if named anyway — cannot be hijacked by injected content. The
filter is operator-set glob allow/deny over canonical tool names, applied at BOTH the schema
seam (``definitions``/the system-prompt list) and the loop's execution seam. Exposure only — it
never loosens the permission gate, and trusted internal callers of ``execute`` are unaffected.
"""

from __future__ import annotations

from pathlib import Path

from zakcode.agent.loop import AgentLoop
from zakcode.config import PermissionTier
from zakcode.messages import ToolResultBlock
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.store import Session
from zakcode.tools.base import (
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from zakcode.usage import Usage


class _FakeTool(Tool):
    def __init__(self, name: str, tier: PermissionTier = PermissionTier.READ_ONLY) -> None:
        # READ_ONLY_SAFE is only valid for a READ_ONLY tier; a writing/dangerous tool must use
        # NEVER_PARALLEL (ToolSpec validates this).
        concurrency = (
            ConcurrencyClass.READ_ONLY_SAFE
            if tier is PermissionTier.READ_ONLY
            else ConcurrencyClass.NEVER_PARALLEL
        )
        self.spec = ToolSpec(
            name=name,
            description=f"{name} tool",
            parameters={"type": "object", "properties": {}},
            required_permission=tier,
            concurrency=concurrency,
        )
        self.calls: list[dict] = []

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        self.calls.append(dict(args))
        return ToolResult.ok(f"ran {self.name}")


def _registry(*names_or_tools) -> ToolRegistry:
    reg = ToolRegistry()
    for item in names_or_tools:
        reg.register(item if isinstance(item, Tool) else _FakeTool(item))
    return reg


def _def_names(reg: ToolRegistry, **kw) -> set[str]:
    return {d["function"]["name"] for d in reg.definitions(**kw)}


# ── 1. registry exposure filter ──────────────────────────────────────────────────


def test_no_filter_exposes_all_active() -> None:
    reg = _registry("read_file", "bash", "web_fetch")
    assert _def_names(reg) == {"read_file", "bash", "web_fetch"}
    assert set(reg.exposed_names()) == {"read_file", "bash", "web_fetch"}


def test_allow_list_restricts_to_matches() -> None:
    reg = _registry("read_file", "grep", "bash", "write_file")
    reg.set_exposure_filter(allow=["read_file", "grep"])
    assert _def_names(reg) == {"read_file", "grep"}
    assert reg.exposure_allows("read_file") is True
    assert reg.exposure_allows("bash") is False


def test_deny_list_hides_matches() -> None:
    reg = _registry("read_file", "bash", "powershell", "web_fetch")
    reg.set_exposure_filter(deny=["bash", "powershell"])
    assert _def_names(reg) == {"read_file", "web_fetch"}
    assert reg.exposure_allows("bash") is False


def test_deny_wins_over_allow() -> None:
    reg = _registry("read_file", "bash")
    reg.set_exposure_filter(allow=["*"], deny=["bash"])
    assert _def_names(reg) == {"read_file"}


def test_glob_patterns() -> None:
    reg = _registry("read_file", "web_fetch", "web_search", "mcp__srv__a", "mcp__srv__b")
    reg.set_exposure_filter(deny=["mcp__*"])
    assert _def_names(reg) == {"read_file", "web_fetch", "web_search"}
    reg.set_exposure_filter(allow=["web_*"])
    assert _def_names(reg) == {"web_fetch", "web_search"}


def test_filter_composes_with_active_set() -> None:
    # the filter narrows the ACTIVE set; an inactive tool stays hidden regardless.
    reg = ToolRegistry()
    reg.register(_FakeTool("read_file"))
    reg.register(_FakeTool("mcp__srv__tool"), active=False)  # lazy / not surfaced
    reg.set_exposure_filter(deny=["bash"])
    assert _def_names(reg) == {"read_file"}  # mcp tool inactive → not exposed
    reg.activate("mcp__srv__tool")
    assert _def_names(reg) == {"read_file", "mcp__srv__tool"}


def test_filter_is_sticky_against_activation() -> None:
    # a DENIED tool that gets activated (e.g. by tool_search) is still not exposed.
    reg = ToolRegistry()
    reg.register(_FakeTool("read_file"))
    reg.register(_FakeTool("mcp__evil__tool"), active=False)
    reg.set_exposure_filter(deny=["mcp__evil__*"])
    reg.activate("mcp__evil__tool")
    assert reg.is_active("mcp__evil__tool") is True  # active...
    assert reg.exposure_allows("mcp__evil__tool") is False  # ...but filtered out
    assert "mcp__evil__tool" not in _def_names(reg)
    assert "mcp__evil__tool" not in reg.exposed_names()


def test_filter_applies_to_allowed_override() -> None:
    # definitions(allowed=...) (the stuck-NARROW path) must also honor the filter.
    reg = _registry("read_file", "grep", "bash")
    reg.set_exposure_filter(deny=["bash"])
    assert _def_names(reg, allowed=["read_file", "bash"]) == {"read_file"}


def test_subset_inherits_the_filter() -> None:
    reg = _registry("read_file", "grep", "bash")
    reg.set_exposure_filter(deny=["bash"])
    sub = reg.subset(["read_file", "grep", "bash"])
    # the child's allowed_tools includes bash, but the inherited deny filter still hides it
    assert _def_names(sub) == {"read_file", "grep"}
    assert sub.exposure_allows("bash") is False


def test_blank_patterns_ignored() -> None:
    reg = _registry("read_file", "bash")
    reg.set_exposure_filter(allow=["", "  ", "read_file"])
    assert _def_names(reg) == {"read_file"}


# ── 2. loop execution-seam guard ──────────────────────────────────────────────────


class _OneToolThenDone(Provider):
    def __init__(self, tool_name: str, arguments: dict) -> None:
        self.tool_name = tool_name
        self.arguments = arguments
        self.calls = 0

    async def acomplete(self, messages, *, system=None, tools=None, **kw) -> LLMResult:
        self.calls += 1
        if self.calls == 1:
            return LLMResult(
                text="",
                tool_calls=[ToolCall(id="c1", name=self.tool_name, arguments=self.arguments)],
                usage=Usage(total_tokens=1),
            )
        return LLMResult(text="done", tool_calls=[], usage=Usage(total_tokens=1))

    def count_tokens(self, messages, *, system=None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities()


def _loop(tmp_path: Path, provider: Provider, registry: ToolRegistry) -> AgentLoop:
    session = Session(cwd=str(tmp_path), model="scripted/test")
    return AgentLoop(provider, registry, session, workspace_root=tmp_path, max_iterations=5)


async def _collect(loop: AgentLoop, text: str) -> list[ToolResultBlock]:
    result = await loop.arun_turn(text)
    return result.tool_results


async def test_execution_guard_rejects_a_filtered_out_tool(tmp_path: Path) -> None:
    # The model names a denied tool from "prior knowledge"; the execution seam must reject it
    # even though it was never advertised — the tool must NOT run.
    bash = _FakeTool("bash", PermissionTier.DANGER_FULL_ACCESS)
    reg = _registry("read_file")
    reg.register(bash)
    reg.set_exposure_filter(deny=["bash"])
    loop = _loop(tmp_path, _OneToolThenDone("bash", {"command": "echo hi"}), reg)

    blocks = await _collect(loop, "go")
    assert blocks[0].is_error
    assert "not available in this task's tool scope" in blocks[0].output
    assert bash.calls == []  # never executed


async def test_execution_guard_allows_an_exposed_tool(tmp_path: Path) -> None:
    read = _FakeTool("read_file")
    reg = ToolRegistry()
    reg.register(read)
    reg.set_exposure_filter(allow=["read_file"])
    loop = _loop(tmp_path, _OneToolThenDone("read_file", {"path": "x"}), reg)

    blocks = await _collect(loop, "go")
    assert not blocks[0].is_error
    assert read.calls == [{"path": "x"}]


# ── 3. facade wiring ──────────────────────────────────────────────────────────────


def test_facade_wires_deny_filter(tmp_path) -> None:
    import zakcode

    agent = zakcode.Agent(workspace_root=tmp_path, tool_exposure_deny=["bash", "powershell"])
    assert agent.registry.exposure_allows("bash") is False
    assert agent.registry.exposure_allows("read_file") is True
    assert "bash" not in {d["function"]["name"] for d in agent.registry.definitions()}


def test_facade_wires_allow_filter(tmp_path) -> None:
    import zakcode

    agent = zakcode.Agent(workspace_root=tmp_path, tool_exposure_allow=["read_file", "grep"])
    exposed = {d["function"]["name"] for d in agent.registry.definitions()}
    assert exposed <= {"read_file", "grep"}
    assert "bash" not in exposed
    assert agent.registry.exposure_allows("write_file") is False


def test_facade_default_no_filter(tmp_path) -> None:
    import zakcode

    agent = zakcode.Agent(workspace_root=tmp_path)
    # default: no restriction — bash (a registered builtin) is exposed
    assert agent.registry.exposure_allows("bash") is True


def test_facade_subagent_registry_inherits_filter(tmp_path) -> None:
    import zakcode

    # A denied tool must NOT be re-exposed to a delegated sub-agent: the child (task-free)
    # registry the SubAgentRunner holds must carry the same exposure filter as the parent.
    agent = zakcode.Agent(
        workspace_root=tmp_path, enable_subagents=True, tool_exposure_deny=["bash"]
    )
    child_registry = agent.loop.spawner._runner.registry
    assert child_registry.exposure_allows("bash") is False
    assert child_registry.exposure_allows("read_file") is True
