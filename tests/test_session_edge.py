"""Edge-case and corruption hardening tests for session persistence.

All tests are hermetic: they operate purely under ``tmp_path`` and use
``monkeypatch`` to simulate failures. No network or live model is touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zakcode.messages import (
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from zakcode.session.store import (
    CURRENT_SCHEMA_VERSION,
    Session,
    SessionCorruptError,
    SessionNotFound,
    SessionStore,
    SessionVersionError,
)
from zakcode.usage import Usage


def _full_session() -> Session:
    """A session exercising every content-block type plus usage records."""
    session = Session(cwd="/work/project", model="ollama/llama3", id="full")
    session.add_message(Message.user("hello"))
    session.add_message(
        Message(
            role="assistant",
            blocks=[
                ThinkingBlock(text="let me think", signature="sig-1"),
                TextBlock(text="I'll call a tool"),
                ToolUseBlock(id="tu-1", name="search", input={"q": "cats", "n": 3}),
            ],
        )
    )
    session.add_message(
        Message.tool_results(
            [ToolResultBlock(tool_use_id="tu-1", output="found", data={"hits": 2})]
        )
    )
    session.add_message(Message.assistant_text("done"))
    session.add_usage(Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15, cost_usd=0.01))
    session.add_usage(Usage(prompt_tokens=4, completion_tokens=8, total_tokens=12, cost_usd=0.02))
    return session


# --------------------------------------------------------------------------- #
# load(): missing / corrupt / version drift
# --------------------------------------------------------------------------- #


def test_load_missing_id_raises_session_not_found(tmp_path: Path) -> None:
    store = SessionStore(base_dir=tmp_path)
    with pytest.raises(SessionNotFound) as exc:
        store.load("does-not-exist")
    assert exc.value.session_id == "does-not-exist"


_TRAVERSAL_IDS = [
    "../outside/leak",
    "..\\outside\\leak",
    "/etc/passwd",
    "C:/Windows/win",
    "C:secret",  # drive-relative
    "a/b",
    "..",
    "with\x00null",
    "public.txt:secret",  # NTFS alternate data stream
    "",
]


@pytest.mark.parametrize("bad_id", _TRAVERSAL_IDS)
def test_load_rejects_traversal_id_as_not_found(tmp_path: Path, bad_id: str) -> None:
    # audit3 #1: an id that isn't a safe single component must not escape base_dir — load
    # treats it as not-found, never reaching outside the sessions dir.
    store = SessionStore(base_dir=tmp_path)
    # Plant a session-shaped file one level above base_dir to prove it is NOT read.
    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)
    (outside / "leak.json").write_text(Session(cwd="C:/secret", model="m").model_dump_json())
    with pytest.raises(SessionNotFound):
        store.load(bad_id)


@pytest.mark.parametrize("bad_id", _TRAVERSAL_IDS)
def test_save_rejects_traversal_id(tmp_path: Path, bad_id: str) -> None:
    # save() is an arbitrary-write primitive at the library boundary; a traversal id is a
    # caller bug and must be refused before any file is written. (audit3 #1)
    store = SessionStore(base_dir=tmp_path)
    session = Session(cwd=".", model="m", id=bad_id)
    with pytest.raises(ValueError):
        store.save(session)


@pytest.mark.parametrize("bad_id", _TRAVERSAL_IDS)
def test_delete_rejects_traversal_id(tmp_path: Path, bad_id: str) -> None:
    store = SessionStore(base_dir=tmp_path)
    assert store.delete(bad_id) is False  # nothing deleted, no traversal


def test_load_invalid_json_raises_corrupt_and_preserves_file(tmp_path: Path) -> None:
    store = SessionStore(base_dir=tmp_path)
    bad = tmp_path / "broken.json"
    bad.write_text("{ this is not valid json ", encoding="utf-8")
    with pytest.raises(SessionCorruptError):
        store.load("broken")
    # The corrupt file must be left exactly as-is, not deleted or rewritten.
    assert bad.exists()
    assert bad.read_text(encoding="utf-8") == "{ this is not valid json "


def test_load_json_that_is_not_an_object_raises_corrupt(tmp_path: Path) -> None:
    store = SessionStore(base_dir=tmp_path)
    (tmp_path / "arr.json").write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(SessionCorruptError):
        store.load("arr")


def test_load_schema_mismatch_raises_corrupt_and_preserves_file(tmp_path: Path) -> None:
    store = SessionStore(base_dir=tmp_path)
    # Valid JSON object, version ok, but a message block has the wrong shape.
    payload = (
        '{"id": "x", "version": 1, "cwd": "/c", "model": "m", '
        '"messages": [{"role": "assistant", "blocks": [{"type": "bogus"}]}]}'
    )
    path = tmp_path / "x.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(SessionCorruptError):
        store.load("x")
    assert path.read_text(encoding="utf-8") == payload


def test_load_missing_required_field_raises_corrupt(tmp_path: Path) -> None:
    store = SessionStore(base_dir=tmp_path)
    # Missing the required "model" field -> validation error, not a traceback.
    payload = '{"id": "m1", "version": 1, "cwd": "/c", "messages": []}'
    (tmp_path / "m1.json").write_text(payload, encoding="utf-8")
    with pytest.raises(SessionCorruptError):
        store.load("m1")


def test_load_newer_version_raises_version_error(tmp_path: Path) -> None:
    store = SessionStore(base_dir=tmp_path)
    session = Session(cwd="/c", model="m", id="future")
    store.save(session)
    # Bump the on-disk version beyond what this build supports.
    path = tmp_path / "future.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["version"] = CURRENT_SCHEMA_VERSION + 5
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(SessionVersionError) as exc:
        store.load("future")
    assert exc.value.found == CURRENT_SCHEMA_VERSION + 5
    assert exc.value.supported == CURRENT_SCHEMA_VERSION
    # File untouched by the failed load.
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == CURRENT_SCHEMA_VERSION + 5


def test_load_older_or_equal_version_loads_when_compatible(tmp_path: Path) -> None:
    store = SessionStore(base_dir=tmp_path)
    session = _full_session()
    session.version = 0  # an older, structurally-compatible writer
    store.save(session)
    loaded = store.load(session.id)
    assert loaded.version == 0
    assert len(loaded.messages) == len(session.messages)


def test_load_missing_version_defaults_and_loads(tmp_path: Path) -> None:
    store = SessionStore(base_dir=tmp_path)
    # No "version" key at all -> default applies, document still loads.
    payload = {
        "id": "noversion",
        "cwd": "/c",
        "model": "m",
        "messages": [{"role": "user", "blocks": [{"type": "text", "text": "hi"}]}],
    }
    (tmp_path / "noversion.json").write_text(json.dumps(payload), encoding="utf-8")
    loaded = store.load("noversion")
    assert loaded.version == CURRENT_SCHEMA_VERSION
    assert loaded.messages[0].text == "hi"


def test_load_non_integer_version_raises_corrupt(tmp_path: Path) -> None:
    store = SessionStore(base_dir=tmp_path)
    payload = {"id": "badver", "version": "two", "cwd": "/c", "model": "m", "messages": []}
    (tmp_path / "badver.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SessionCorruptError):
        store.load("badver")


# --------------------------------------------------------------------------- #
# save(): atomicity
# --------------------------------------------------------------------------- #


def test_save_leaves_no_temp_files(tmp_path: Path) -> None:
    store = SessionStore(base_dir=tmp_path)
    store.save(Session(cwd="/c", model="m", id="clean"))
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []
    assert (tmp_path / "clean.json").exists()


def test_save_failure_leaves_no_partial_and_preserves_previous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SessionStore(base_dir=tmp_path)
    good = Session(cwd="/c", model="m", id="keep")
    good.add_message(Message.user("v1"))
    store.save(good)
    original = (tmp_path / "keep.json").read_text(encoding="utf-8")

    # Make serialization blow up partway through the save.
    def boom(self: Session, *args: object, **kwargs: object) -> str:
        raise RuntimeError("serialization exploded")

    monkeypatch.setattr(Session, "model_dump_json", boom)

    failing = Session(cwd="/c", model="m", id="keep")
    failing.add_message(Message.user("v2"))
    with pytest.raises(RuntimeError):
        store.save(failing)

    # Previous good file intact, no temp/garbage files left behind.
    assert (tmp_path / "keep.json").read_text(encoding="utf-8") == original
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []


def test_save_failure_during_replace_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import zakcode.session.store as store_mod

    store = SessionStore(base_dir=tmp_path)

    def boom_replace(src: object, dst: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(store_mod.os, "replace", boom_replace)
    with pytest.raises(OSError):
        store.save(Session(cwd="/c", model="m", id="r"))

    # No target file, and the temp artifact must be cleaned up.
    assert not (tmp_path / "r.json").exists()
    assert list(tmp_path.iterdir()) == []


def test_repeated_saves_overwrite_cleanly(tmp_path: Path) -> None:
    store = SessionStore(base_dir=tmp_path)
    for i in range(5):
        session = Session(cwd="/c", model="m", id="loop")
        session.add_message(Message.user(str(i)))
        store.save(session)
    loaded = store.load("loop")
    assert loaded.messages[0].text == "4"
    # Exactly one file for this id, nothing left over.
    assert [p.name for p in tmp_path.iterdir()] == ["loop.json"]


# --------------------------------------------------------------------------- #
# list(): junk tolerance
# --------------------------------------------------------------------------- #


def test_list_ignores_non_session_and_junk_files(tmp_path: Path) -> None:
    store = SessionStore(base_dir=tmp_path)
    store.save(Session(cwd="/c", model="m", id="real"))
    # Junk that must not crash list() nor be returned as a *.json id, plus a
    # directory that matches the glob and a leftover temp artifact.
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "README.md").write_text("docs", encoding="utf-8")
    (tmp_path / "real.json.deadbeef.tmp").write_text("partial", encoding="utf-8")
    (tmp_path / "adir.json").mkdir()
    ids = store.list()
    assert ids == ["real"]


def test_list_empty_dir(tmp_path: Path) -> None:
    store = SessionStore(base_dir=tmp_path)
    assert store.list() == []


# --------------------------------------------------------------------------- #
# round-trip fidelity + cumulative usage
# --------------------------------------------------------------------------- #


def test_full_roundtrip_is_byte_faithful(tmp_path: Path) -> None:
    store = SessionStore(base_dir=tmp_path)
    session = _full_session()
    store.save(session)
    on_disk = (tmp_path / "full.json").read_text(encoding="utf-8")

    loaded = store.load("full")
    # The loaded session re-renders to the exact bytes persisted on disk.
    assert loaded.model_dump_json() == on_disk
    assert loaded == session

    # Every block type survived intact.
    asst = loaded.messages[1]
    assert isinstance(asst.blocks[0], ThinkingBlock)
    assert asst.blocks[0].signature == "sig-1"
    assert isinstance(asst.blocks[2], ToolUseBlock)
    assert asst.blocks[2].input == {"q": "cats", "n": 3}
    tool_result = loaded.messages[2].blocks[0]
    assert isinstance(tool_result, ToolResultBlock)
    assert tool_result.tool_use_id == "tu-1"
    assert tool_result.data == {"hits": 2}


def test_cumulative_usage_after_roundtrip(tmp_path: Path) -> None:
    store = SessionStore(base_dir=tmp_path)
    session = _full_session()
    store.save(session)
    loaded = store.load("full")
    total = loaded.cumulative_usage()
    assert total.prompt_tokens == 14
    assert total.completion_tokens == 13
    assert total.total_tokens == 27
    assert total.cost_usd == pytest.approx(0.03)
