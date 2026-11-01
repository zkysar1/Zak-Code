"""A new chat in a workspace with framework agents offers to start one first (ADR-0351).

An agent's identity and what it has learned load only when ``/start <name>`` starts it, and each new
chat begins without one, so a question typed first is answered without the agent. The server names
the agents a new chat can start (``GET /workspace/agents``), and the page, on a chat with nothing in
it, shows an offer with one button per agent in place of its line and its suggestions. The route is
tested on a real app over tmp workspaces; the page runs its OWN script, cut from the shipped page,
under node, on the DOM the turn-clock tests use, against a scripted server.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.test_framework_agents import mind
from tests.test_webclient_turn_clock import _NODE, _PRELUDE, _page_script
from zakcode.config import Settings
from zakcode.server.app import create_app
from zakcode.session.store import SessionStore

_DOM = _PRELUDE.split("const __STREAMS = ")[0]

#: The scripted server, then a look at the start screen, then the click a scenario asks for.
_SERVER = r"""
const __SCN = JSON.parse(process.argv[2]);
const __later = setTimeout;
const __posted = [];
const __SESSION = { id: "abcdef0123456789", model: "fake-model" };
function __answer(body, status) {
  return Promise.resolve({ ok: status >= 200 && status < 300, status, json: async () => body,
                           headers: { get: () => null } });
}
globalThis.fetch = async (url, opts) => {
  opts = opts || {};
  const u = new URL(url, "http://page.test");
  if (u.pathname === "/sessions/current") return __answer(__SESSION, 200);
  if (u.pathname === "/sessions/" + __SESSION.id + "/transcript") {
    return __answer({ session_id: __SESSION.id, messages: [], message_count: 0, bus_from: null,
                      bus_since: null, running: false }, 200);
  }
  if (u.pathname === "/workspace/agents") {
    // `null` is a server too old to know the route.
    return __SCN.agents === null ? __answer({ detail: "Not Found" }, 404)
                                 : __answer({ agents: __SCN.agents }, 200);
  }
  if (u.pathname === "/sessions/recent") return __answer([], 200);
  if (u.pathname.startsWith("/watch/")) return new Promise(() => {});  // a live watch: no frames
  if ((opts.method || "GET") === "POST") {
    __posted.push({ path: u.pathname, body: opts.body ? JSON.parse(opts.body) : null });
  }
  return __answer({ queued: true }, 200);
};
function __look() {
  const offer = document.getElementById("start-agents");
  return {
    offered: !offer.hidden,
    offering: document.getElementById("empty").classList.contains("offering"),
    buttons: document.getElementById("start-agent-buttons").children.map((b) => b.textContent),
    line: document.getElementById("start-agent-line").textContent,
  };
}
__later(() => {
  const before = __look();
  if (__SCN.click !== null) {
    document.getElementById("start-agent-buttons").children[__SCN.click].onclick();
  }
  __later(() => {
    const errors = __created.filter((n) => n.parent !== null && n.classes.has("errcard"))
      .map((n) => n.textContent);
    process.stdout.write(JSON.stringify({ seen: before, posted: __posted, errors }));
    process.exit(0);
  }, 50);
}, 100);
"""


def _page(tmp_path: Path, agents: list[dict[str, str]] | None, click: int | None = None) -> Any:
    """Run the page on a new chat against a server answering ``agents`` (None: no such route)."""
    assert _NODE is not None
    script = tmp_path / f"page-{time.monotonic_ns()}.js"
    script.write_text(_DOM + _SERVER + _page_script(), encoding="utf-8")
    scenario = {"agents": agents, "click": click}
    proc = subprocess.run(
        [_NODE, str(script), json.dumps(scenario)], capture_output=True, timeout=60, check=False
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    shown = json.loads(proc.stdout.decode("utf-8"))
    assert shown["errors"] == [], shown["errors"]
    return shown


def _client(workspace: Path) -> TestClient:
    settings = Settings(
        default_model="scripted/test", context_window=8192, workspace_root=workspace
    )
    store = SessionStore(base_dir=workspace / ".sessions")
    return TestClient(create_app(settings=settings, store=store))


_KEEPER = {"name": "keeper", "start": "/start keeper --mode assistant"}
_SCOUT = {"name": "scout", "start": "/start scout --mode assistant"}


# ── the route ────────────────────────────────────────────────────────────────


def test_the_server_names_each_agent_and_the_line_that_starts_it(tmp_path: Path) -> None:
    client = _client(mind(tmp_path, "scout", "keeper"))
    assert client.get("/workspace/agents").json() == {"agents": [_KEEPER, _SCOUT]}


def test_an_agent_made_after_the_server_started_is_offered_at_the_next_ask(tmp_path: Path) -> None:
    client = _client(mind(tmp_path))
    assert client.get("/workspace/agents").json() == {"agents": []}
    (tmp_path / "agents" / "scout").mkdir(parents=True)
    (tmp_path / "agents" / "scout" / "self.md").write_text("# scout\n", encoding="utf-8")
    assert client.get("/workspace/agents").json() == {"agents": [_SCOUT]}


def test_a_workspace_without_the_framework_names_none(tmp_path: Path) -> None:
    (tmp_path / "agents" / "scout").mkdir(parents=True)
    (tmp_path / "agents" / "scout" / "self.md").write_text("# scout\n", encoding="utf-8")
    assert _client(tmp_path).get("/workspace/agents").json() == {"agents": []}


# ── the page ─────────────────────────────────────────────────────────────────

needs_node = pytest.mark.skipif(_NODE is None, reason="needs node on PATH (CI runners carry it)")


@needs_node
def test_a_new_chat_offers_each_agent_in_place_of_its_suggestions(tmp_path: Path) -> None:
    shown = _page(tmp_path, [_KEEPER, _SCOUT])
    assert shown["seen"] == {
        "offered": True,
        "offering": True,  # the class that hides the line and the suggestions
        "buttons": ["Start keeper in assistant mode", "Start scout in assistant mode"],
        "line": "/start keeper --mode assistant",
    }
    assert shown["posted"] == []  # an offer, never a send of its own


@needs_node
def test_a_button_sends_its_start_line_as_the_chat_s_first_message(tmp_path: Path) -> None:
    shown = _page(tmp_path, [_KEEPER, _SCOUT], click=1)
    assert shown["posted"] == [
        {
            "path": "/say",
            "body": {"text": "/start scout --mode assistant", "session_id": "abcdef0123456789"},
        }
    ]


@needs_node
@pytest.mark.parametrize("agents", [[], None], ids=["no-agents", "no-such-route"])
def test_without_agents_the_start_screen_is_as_it_was(
    tmp_path: Path, agents: list[dict[str, str]] | None
) -> None:
    shown = _page(tmp_path, agents)
    assert shown["seen"] == {"offered": False, "offering": False, "buttons": [], "line": ""}
    assert shown["posted"] == []
