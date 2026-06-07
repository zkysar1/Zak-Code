"""Tests for the cross-session memory substrate (provider, store, recall hook, tools)."""

from __future__ import annotations

from pathlib import Path

from zakcode.memory import MemoryProvider, MemoryRecallHook, MemoryRecord
from zakcode.memory.sqlite_store import SqliteMemoryProvider
from zakcode.tools.base import ToolContext
from zakcode.tools.builtins.memory import RecallTool, RememberTool


def _provider(tmp_path: Path) -> SqliteMemoryProvider:
    return SqliteMemoryProvider(tmp_path / "mem.db")


# ── SqliteMemoryProvider ──────────────────────────────────────────────────────


def test_add_and_count(tmp_path: Path) -> None:
    p = _provider(tmp_path)
    assert p.count() == 0
    rec = p.add("the user prefers tabs", kind="preference", tags=["style"])
    assert isinstance(rec, MemoryRecord)
    assert p.count() == 1
    assert rec.id and rec.kind == "preference" and rec.tags == ["style"]


def test_search_finds_relevant(tmp_path: Path) -> None:
    p = _provider(tmp_path)
    p.add("the project uses pytest for testing")
    p.add("the deploy script lives in scripts/deploy.sh")
    hits = p.search("how do we run testing", limit=5)
    assert any("pytest" in h.text for h in hits)
    assert all(isinstance(h, MemoryRecord) for h in hits)


def test_search_empty_query_returns_nothing(tmp_path: Path) -> None:
    p = _provider(tmp_path)
    p.add("something")
    assert p.search("   ", limit=5) == []
    assert p.search("anything", limit=0) == []


def test_recent_orders_newest_first(tmp_path: Path) -> None:
    p = _provider(tmp_path)
    p.add("first", source="s")
    p.add("second", source="s")
    p.add("third", source="s")
    recent = p.recent(limit=2)
    assert [r.text for r in recent] == ["third", "second"]


def test_delete(tmp_path: Path) -> None:
    p = _provider(tmp_path)
    rec = p.add("forget me")
    assert p.delete(rec.id) is True
    assert p.delete(rec.id) is False  # already gone
    assert p.count() == 0


def test_persists_across_instances(tmp_path: Path) -> None:
    db = tmp_path / "mem.db"
    p1 = SqliteMemoryProvider(db)
    p1.add("durable fact about indentation", kind="fact", tags=["x"])
    p1.close()
    p2 = SqliteMemoryProvider(db)  # fresh connection, same file
    assert p2.count() == 1
    hits = p2.search("indentation")
    assert hits and hits[0].kind == "fact" and hits[0].tags == ["x"]


def test_like_fallback_path(tmp_path: Path) -> None:
    # Force the non-FTS LIKE path and confirm search still works.
    p = _provider(tmp_path)
    p._fts = False  # type: ignore[attr-defined]
    p.add("the LIKE_MARKER_TOKEN matters here")
    hits = p.search("LIKE_MARKER_TOKEN")
    assert any("LIKE_MARKER_TOKEN" in h.text for h in hits)


def test_in_memory_db(tmp_path: Path) -> None:
    p = SqliteMemoryProvider(":memory:")
    p.add("ephemeral")
    assert p.count() == 1


def test_multiword_tag_round_trips(tmp_path: Path) -> None:
    p = _provider(tmp_path)
    p.add("note", tags=["code review", "single"])
    got = p.recent(limit=1)[0].tags
    assert got == ["code review", "single"]  # lossless (JSON-stored), not split on space


# ── MemoryRecallHook ──────────────────────────────────────────────────────────


class _CountingProvider(MemoryProvider):
    """A fake provider that counts search calls (to prove per-turn caching)."""

    def __init__(self, records: list[MemoryRecord]) -> None:
        self._records = records
        self.searches = 0

    def add(self, text, *, kind="note", tags=None, source=""):  # noqa: ANN001, ANN201
        rec = MemoryRecord(text=text, kind=kind, tags=tags or [], source=source)
        self._records.append(rec)
        return rec

    def search(self, query, *, limit=5):  # noqa: ANN001, ANN201
        self.searches += 1
        return self._records[:limit]

    def recent(self, *, limit=10):  # noqa: ANN201
        return self._records[:limit]

    def update(self, memory_id, *, text=None, kind=None, tags=None):  # noqa: ANN001, ANN201
        return None

    def delete(self, memory_id):  # noqa: ANN001, ANN201
        return False

    def count(self):  # noqa: ANN201
        return len(self._records)


class _Payload:
    def __init__(self, user_text: str) -> None:
        self.user_text = user_text


def test_recall_hook_renders_relevant_block() -> None:
    # Floor off (min_overlap=0) to test the rendering path in isolation.
    provider = _CountingProvider([MemoryRecord(text="RECALL_BLOCK_MARKER", kind="fact")])
    hook = MemoryRecallHook(provider, limit=3, min_overlap=0)
    out = hook(_Payload("anything"))
    assert out is not None
    assert "RECALL_BLOCK_MARKER" in out
    assert "memories" in out.lower()


def test_recall_hook_none_when_empty() -> None:
    hook = MemoryRecallHook(_CountingProvider([]))
    assert hook(_Payload("query")) is None
    assert hook(_Payload("")) is None  # blank user_text → no recall


def test_recall_hook_caches_per_turn() -> None:
    provider = _CountingProvider([MemoryRecord(text="x")])
    hook = MemoryRecallHook(provider)
    hook(_Payload("same query"))
    hook(_Payload("same query"))  # same turn text → no second search
    assert provider.searches == 1
    hook(_Payload("different query"))
    assert provider.searches == 2


def test_recall_hook_floor_drops_distinctive_word_misses() -> None:
    # The fake provider returns BOTH records (ignores the query); the floor keeps only
    # the one sharing a distinctive (non-stopword) word with the turn.
    provider = _CountingProvider(
        [MemoryRecord(text="pytest is the test runner"), MemoryRecord(text="unrelated note")]
    )
    out = MemoryRecallHook(provider, min_overlap=1)(_Payload("how do we run pytest")) or ""
    assert "pytest is the test runner" in out  # shares "pytest"
    assert "unrelated note" not in out  # shares no distinctive word


def test_recall_hook_floor_returns_none_when_nothing_distinctive_matches() -> None:
    # Query HAS distinctive words, but none appear in the memory → filtered → None.
    provider = _CountingProvider([MemoryRecord(text="some unrelated memory")])
    assert MemoryRecallHook(provider, min_overlap=1)(_Payload("pytest pipeline deployment")) is None


def test_recall_hook_floor_disabled_keeps_all() -> None:
    provider = _CountingProvider([MemoryRecord(text="unrelated note")])
    out = MemoryRecallHook(provider, min_overlap=0)(_Payload("pytest tests")) or ""
    assert "unrelated note" in out  # floor off → inject every match


def test_recall_hook_all_stopword_query_not_overfiltered() -> None:
    # A query with no distinctive words → nothing to match on → don't filter the tail.
    provider = _CountingProvider([MemoryRecord(text="some memory")])
    out = MemoryRecallHook(provider, min_overlap=1)(_Payload("what is it")) or ""
    assert "some memory" in out


def test_recall_hook_floor_works_with_a_single_memory() -> None:
    # The bm25-collapse case: a 1-memory store still recalls a real match under the floor.
    p = SqliteMemoryProvider(":memory:")
    p.add("the user prefers 4-space indentation in Python")
    hook = MemoryRecallHook(p, min_overlap=1)
    out = hook(_Payload("what indentation does the user prefer")) or ""
    assert "indentation" in out  # shares "indentation" → kept despite tiny corpus


# ── remember / recall tools ───────────────────────────────────────────────────


async def test_remember_then_recall_tools(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    ctx = ToolContext(workspace_root=tmp_path)
    remember = RememberTool(provider, source="sess-1")
    recall = RecallTool(provider)

    res = await remember.execute(
        {"text": "the API key lives in the env, never in code", "kind": "fact"}, ctx
    )
    assert not res.is_error
    assert provider.count() == 1

    res = await recall.execute({"query": "where is the API key"}, ctx)
    assert not res.is_error
    assert res.data is not None and res.data["count"] >= 1
    assert "env" in res.output


async def test_remember_rejects_blank(tmp_path: Path) -> None:
    res = await RememberTool(_provider(tmp_path)).execute(
        {"text": "   "}, ToolContext(workspace_root=tmp_path)
    )
    assert res.is_error


async def test_recall_no_results(tmp_path: Path) -> None:
    res = await RecallTool(_provider(tmp_path)).execute(
        {"query": "nothing stored yet"}, ToolContext(workspace_root=tmp_path)
    )
    assert not res.is_error
    assert res.data is not None and res.data["count"] == 0


async def test_remember_redacts_secrets_before_storing(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    ctx = ToolContext(workspace_root=tmp_path)
    res = await RememberTool(provider).execute(
        {"text": "deploy key is sk-ABCDEFGHIJKLMNOP1234567890"}, ctx
    )
    assert not res.is_error
    assert "Redacted" in res.output  # the model is told a secret was scrubbed
    stored = provider.recent(limit=1)[0].text
    assert "sk-ABCDEFGHIJKLMNOP" not in stored  # the secret never hit disk
    assert "[REDACTED]" in stored


def test_recall_hook_redacts_externally_written_secret() -> None:
    # A memory written outside the scrubbing remember tool still gets scrubbed on recall.
    provider = _CountingProvider([MemoryRecord(text="token=ghp_0123456789012345678901")])
    hook = MemoryRecallHook(provider)
    out = hook(_Payload("token"))
    assert out is not None
    assert "ghp_0123456789012345678901" not in out
    assert "[REDACTED]" in out


# ── facade wiring + end-to-end recall ─────────────────────────────────────────


def test_agent_enable_memory_registers_tools(tmp_path: Path) -> None:
    from zakcode import Agent
    from zakcode.evals.harness import ScriptedProvider, reply

    agent = Agent(
        provider=ScriptedProvider(script=[reply("ok")]),
        enable_memory=True,
        default_model="scripted/test",
        workspace_root=str(tmp_path),
        memory_db_path=str(tmp_path / "mem.db"),
    )
    assert agent.memory is not None
    assert agent.registry.get("remember") is not None
    assert agent.registry.get("recall") is not None


async def test_agent_memory_recall_injects_into_turn(tmp_path: Path) -> None:
    from zakcode import Agent
    from zakcode.evals.harness import ScriptedProvider, reply
    from zakcode.messages import Message
    from zakcode.providers.base import LLMResult

    seen: list[list[Message]] = []

    def responder(messages: list[Message], system: str | None, idx: int) -> LLMResult:
        seen.append(list(messages))
        return reply("done")

    seeded = SqliteMemoryProvider(":memory:")
    seeded.add("MEMORY_RECALL_MARKER about testing conventions", kind="fact")

    agent = Agent(
        provider=ScriptedProvider(responder=responder),
        enable_memory=True,
        memory_provider=seeded,
        default_model="scripted/test",
        workspace_root=str(tmp_path),
    )
    await agent.arun_turn("what are our testing conventions")

    injected = seen[0][-1].text
    assert "MEMORY_RECALL_MARKER" in injected
    assert "<injected_context>" in injected  # fenced as untrusted by the loop
