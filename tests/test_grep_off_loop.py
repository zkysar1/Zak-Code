"""Grep's walk and per-file reads run off the event loop.

Walking a tree and reading its files are blocking calls. Made inside ``async def execute`` with no
await, they held the event loop for the whole search, so nothing else the process serves (a timer or
an input route) ran until it ended.

Each case slows one blocking call by a fixed amount per file, so the search spans many heartbeat
periods on any machine, then counts the heartbeats the loop ran while the search was in flight. The
floor is far below the number of periods the search spans, so a coarse system timer still clears it.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from zakcode.tools.base import ToolContext
from zakcode.tools.builtins.grep import GrepTool

_FILES = 40
_STALL_S = 0.01  # per file, so the slowed phase alone spans at least 0.4 s
_BEAT_S = 0.005
_MIN_BEATS = 5


@pytest.mark.parametrize(
    "slowed_call",
    [
        pytest.param("is_symlink", id="the walk"),
        pytest.param("read_bytes", id="the reads"),
    ],
)
async def test_the_event_loop_keeps_running_while_grep_searches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, slowed_call: str
) -> None:
    ws = tmp_path.resolve()
    for i in range(_FILES):
        (ws / f"f{i:02}.txt").write_text("needle\n", encoding="utf-8")
    real = getattr(Path, slowed_call)
    stalled: list[Path] = []

    def slow(self: Path):
        if self.parent == ws and self.suffix == ".txt":
            stalled.append(self)
            time.sleep(_STALL_S)
        return real(self)

    monkeypatch.setattr(Path, slowed_call, slow)
    beats = 0

    async def heartbeat() -> None:
        nonlocal beats
        while True:
            await asyncio.sleep(_BEAT_S)
            beats += 1

    ticker = asyncio.create_task(heartbeat())
    try:
        await asyncio.sleep(0)  # let the heartbeat task start
        before = beats
        result = await GrepTool().execute({"pattern": "needle"}, ToolContext(workspace_root=ws))
        during = beats - before
    finally:
        ticker.cancel()

    assert not result.is_error, result.output
    assert result.data is not None and result.data["count"] == _FILES
    assert len(stalled) >= _FILES  # the slowed call ran for every file, so the case is not vacuous
    assert during >= _MIN_BEATS, (
        f"the loop ran the heartbeat {during} times while {len(stalled)} calls to "
        f"Path.{slowed_call} stalled for {_STALL_S}s each"
    )
