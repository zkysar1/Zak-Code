"""``bench/run_task.py`` request dumps: one ``wire-N.json`` per provider request.

The wire dump used to name its files by the buffered-call counter, which only ``acomplete``
bumps. The streaming main loop never calls ``acomplete``, so under the stream driver every
request of a turn overwrote the same file until the next buffered call moved the counter
(measured 2026-09-25: an eight-turn run left three wire files). Every consumer of the dumps
globs ``wire-*.json`` expecting one file per request. The wire files now count on their own.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest
from pydantic import BaseModel

RUN_TASK = Path(__file__).resolve().parent.parent / "bench" / "run_task.py"


class _Msg(BaseModel):
    role: str
    content: str


@pytest.fixture
def dumps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """The instrument installed over a fake provider and a fake litellm.

    Both fakes go in through ``monkeypatch`` BEFORE the install, so the install wraps the
    fakes and the teardown restores the real attributes over the wrappers.
    """
    import litellm

    from zakcode.providers.litellm_provider import LiteLLMProvider

    sent: list[dict] = []

    async def fake_acompletion(**kw):
        sent.append(kw)
        return {"served": "fake"}

    async def fake_acomplete(self, messages, *, system=None, tools=None, **kw):
        # The buffered path: the provider builds one litellm call per ``acomplete``.
        body = [m.model_dump(mode="json") for m in messages]
        return await litellm.acompletion(model="m", messages=body, stream=False, api_key="k")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(LiteLLMProvider, "acomplete", fake_acomplete)

    spec = importlib.util.spec_from_file_location("run_task_under_test", RUN_TASK)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    target = tmp_path / "dumps"
    mod._install_request_dumps(target)
    return target, litellm, LiteLLMProvider, sent


def _names(target: Path, pattern: str) -> list[str]:
    return sorted(p.name for p in target.glob(pattern))


def _first_content(target: Path, name: str) -> str:
    return json.loads((target / name).read_text(encoding="utf-8"))["messages"][0]["content"]


def test_a_streaming_turn_leaves_one_wire_file_per_request(dumps) -> None:
    """Two main-loop requests straight to litellm, then the buffered classifier: three files."""
    target, litellm, provider, sent = dumps

    async def turn():
        for text in ("one", "two"):
            msgs = [{"role": "user", "content": text}]
            await litellm.acompletion(model="m", messages=msgs, stream=True, api_key="k")
        await provider.acomplete(object(), [_Msg(role="user", content="three")], system="s")

    asyncio.run(turn())
    assert _names(target, "wire-*.json") == ["wire-0001.json", "wire-0002.json", "wire-0003.json"]
    assert [_first_content(target, n) for n in _names(target, "wire-*.json")] == [
        "one",
        "two",
        "three",
    ]
    # ``call``/``echo`` are the buffered path's own record: only the classifier has them.
    assert _names(target, "call-*.json") == ["call-0001.json"]
    assert _names(target, "echo-*.json") == ["echo-0001.json"]
    assert len(sent) == 3


def test_the_buffered_driver_still_pairs_call_n_with_wire_n(dumps) -> None:
    """Positive control: every request through ``acomplete`` keeps the two counters equal."""
    target, _litellm, provider, _sent = dumps

    async def turn():
        for text in ("one", "two"):
            await provider.acomplete(object(), [_Msg(role="user", content=text)], system="s")

    asyncio.run(turn())
    assert _names(target, "call-*.json") == ["call-0001.json", "call-0002.json"]
    assert _names(target, "wire-*.json") == ["wire-0001.json", "wire-0002.json"]
    for n in ("0001", "0002"):
        call = json.loads((target / f"call-{n}.json").read_text(encoding="utf-8"))
        assert call["messages"][0]["content"] == _first_content(target, f"wire-{n}.json")


def test_the_wire_body_never_carries_the_api_key(dumps) -> None:
    target, litellm, _provider, sent = dumps
    msgs = [{"role": "user", "content": "x"}]
    asyncio.run(litellm.acompletion(model="m", messages=msgs, stream=True, api_key="k"))
    assert "api_key" not in json.loads((target / "wire-0001.json").read_text(encoding="utf-8"))
    assert sent[0]["api_key"] == "k"  # the request itself still carries it
