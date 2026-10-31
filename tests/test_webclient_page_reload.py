"""An open web page reloads itself into the page the server serves (ADR-0345).

An open page runs the script it was served with for as long as it stays open. A server that
restarted into a newer build (ADR-0333) left every open page on its old script, and a fix to the
page reached nobody until they reloaded it by hand. The server now reads its page once, as it
starts, writes the page's name (the first 12 hex digits of its sha256) into the page it serves at
``/``, and sends that name with every watch response (``X-Page-Digest``). A page whose name is not
the server's reloads, once, and keeps what is typed in the composer.

The page tests run the page's OWN script under node against a REAL loopback server, held open as
test_webclient_bus_epoch.py holds it. A server on another build is a server started on another
copy of the page's file. Not a browser; see test_webclient_turn_clock.py for why.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.test_server_consumer import _LiveServer
from tests.test_webclient_bus_epoch import _OpenPage
from tests.test_webclient_reload_history import _app
from tests.test_webclient_turn_clock import _INDEX, _NODE, _page_script
from zakcode.server import app as server_app

_needs_node = pytest.mark.skipif(_NODE is None, reason="needs node on PATH (CI runners carry it)")

#: What another build adds to the page, so that its page has another name.
_ANOTHER_BUILD = b"<!-- another build -->\n"
#: The name of a page from a build no server here serves.
_ELSEWHERE = "0123456789ab"

#: Runs after the page's DOM and before its script, as test_webclient_bus_epoch.py's tail does, and
#: first sets up what the page finds as it opens (`__START`): its address, its session storage, and
#: what is in its composer. Commands arrive on stdin, one per line, and each is answered with one
#: JSON line: BASE <url> sends every later request to another server, OFFLINE aborts the page's
#: watch connection and refuses every later one, ONLINE lifts that, TYPE <text> fills the composer,
#: SNAP reports what the page asked for, the reloads it made, its address, its session storage,
#: its composer, its connection label and the note it shows.
_TAIL = """
const __START = __START_JSON__;
let __base = process.argv[2];
location.href = __START.href;
for (const [key, value] of Object.entries(__START.storage)) sessionStorage.items.set(key, value);
sessionStorage.broken = __START.broken;
document.getElementById("input").value = __START.typed;
document.getElementById("page-note").hidden = true;  // as the page's markup has it
const __realFetch = globalThis.fetch;
const __trace = [];
let __offline = false;
let __watch = null;
globalThis.fetch = (url, opts) => {
  const full = new URL(url, __base);
  __trace.push(full.pathname + full.search);
  if (!full.pathname.startsWith("/watch/")) return __realFetch(full, opts);
  if (__offline) return Promise.reject(new TypeError("network is down"));
  __watch = new AbortController();
  return __realFetch(full, { ...opts, signal: __watch.signal });
};
const __say = (obj) => process.stdout.write(JSON.stringify(obj) + "\\n");
let __pending = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => {
  __pending += chunk;
  let end;
  while ((end = __pending.indexOf("\\n")) >= 0) {
    const line = __pending.slice(0, end).trim();
    __pending = __pending.slice(end + 1);
    const [cmd, ...rest] = line.split(" ");
    if (cmd === "BASE") {
      __base = rest.join(" ");
    } else if (cmd === "OFFLINE") {
      __offline = true;
      if (__watch) __watch.abort();
    } else if (cmd === "ONLINE") {
      __offline = false;
    } else if (cmd === "TYPE") {
      document.getElementById("input").value = rest.join(" ");
    }
    if (cmd !== "SNAP") { __say({ ok: true }); continue; }
    const note = document.getElementById("page-note");
    __say({
      trace: __trace,
      replaced: location.replaced,
      href: location.href,
      storage: Object.fromEntries(sessionStorage.items),
      typed: document.getElementById("input").value,
      conn: document.getElementById("conn-label").textContent,
      note: note.hidden ? null : note.textContent,
      errs: __created
        .filter((n) => n.parent !== null && n.classes.has("errcard"))
        .map((n) => n.textContent),
    });
  }
});
process.stdin.on("end", () => process.exit(0));
"""


def _tail(
    href: str = "http://page.test/",
    storage: dict[str, str] | None = None,
    broken: bool = False,
    typed: str = "",
) -> str:
    """The tail, with what the page finds as it opens."""
    start = {"href": href, "storage": storage or {}, "broken": broken, "typed": typed}
    return _TAIL.replace("__START_JSON__", json.dumps(start))


def _name_of(page_file: Path) -> str:
    """The name a server gives the page it starts on."""
    return hashlib.sha256(page_file.read_bytes()).hexdigest()[:12]


def _named(name: str) -> str:
    """The shipped page's script as a server on a build whose page is named `name` serves it."""
    return _page_script().replace(server_app._PAGE_DIGEST_SLOT.decode(), name, 1)


def _served_script(base: str) -> str:
    """The page's script as the server at `base` serves it, its name written in."""
    resp = httpx.get(base + "/", timeout=5)
    resp.raise_for_status()
    scripts = re.findall(r"<script>\n(.*?)</script>", resp.text.replace("\r\n", "\n"), re.S)
    assert len(scripts) == 1, "the page carries exactly one inline script"
    script: str = scripts[0]
    return script


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


def _pages(tmp_path: Path, name: str, extra: bytes = b"") -> Path:
    """A folder holding the shipped page, or another build's page when `extra` is given."""
    folder = tmp_path / name
    folder.mkdir()
    (folder / "index.html").write_bytes(_INDEX.read_bytes() + extra)
    return folder


def _serving(monkeypatch: pytest.MonkeyPatch, workspace: Path, folder: Path) -> Any:
    """An app on `workspace` that serves the page in `folder`, as a server on that build does."""
    monkeypatch.setattr(server_app, "_STATIC_DIR", folder)
    return _app(workspace)


def _connected(page: _OpenPage) -> dict[str, Any]:
    """Snapshot once a watch response has reached the page. It looks at the response's page name
    as it marks itself connected, in the same step, so a reload it was to make is made by then."""
    return page._wait_for(lambda snap: snap["conn"] == "connected", 15.0)


def _page_header(base: str, session_id: str) -> str | None:
    """The page name a watch response sends, read from its headers alone."""
    timeout = httpx.Timeout(10.0, read=5.0)
    with httpx.stream("GET", f"{base}/watch/{session_id}", timeout=timeout) as resp:
        assert resp.status_code == 200
        return resp.headers.get("x-page-digest")


@_needs_node
def test_a_page_left_open_while_the_server_restarts_on_another_build_reloads_into_its_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The page stays while its server serves it, reloads once the server started in that one's
    place serves another page, and comes back from the reload with what was typed."""
    workspace = _workspace(tmp_path)
    old_pages, new_pages = _pages(tmp_path, "old"), _pages(tmp_path, "new", _ANOTHER_BUILD)
    old = _LiveServer(_serving(monkeypatch, workspace, old_pages))
    new = _LiveServer(_serving(monkeypatch, workspace, new_pages))
    with old as base, _OpenPage(base, tmp_path, _tail(), page=_served_script(base)) as page:
        assert _connected(page)["replaced"] == []  # the page its server serves: it stays
        page.ask("TYPE half a thought")
        # The server stops and one on another build starts at the same address, the page's watch
        # cut while it does (here by the page, which the stop would otherwise wait on).
        page.ask("OFFLINE")
        old.__exit__(None, None, None)
        with new as new_base:
            page.ask("BASE " + new_base)
            page.ask("ONLINE")
            left = page._wait_for(lambda snap: bool(snap["replaced"]), 15.0)
            page.ask("OFFLINE")  # it is leaving: its watch goes with it
            # The reload: the new server's page, opened at the address the old page went to,
            # with the session storage it left behind.
            asked = left["replaced"][0]
            back_tail = _tail(href=asked, storage=left["storage"])
            with _OpenPage(new_base, tmp_path, back_tail, page=_served_script(new_base)) as back:
                came = _connected(back)
    assert left["replaced"] == [f"http://page.test/?page={_name_of(new_pages / 'index.html')}"]
    assert left["storage"] == {"zakcode-draft": "half a thought"}
    assert came["replaced"] == []  # it is the server's page now: no second reload
    assert came["typed"] == "half a thought"
    assert came["href"] == "http://page.test/"  # ?page= has left the address
    assert came["storage"] == {}
    assert came["note"] is None
    assert left["errs"] == [] and came["errs"] == []


@_needs_node
def test_a_reload_that_brings_back_the_same_old_page_says_so_and_stays(tmp_path: Path) -> None:
    """Something between the browser and the server answers with a saved copy of the page, so
    reloading brings the old page back: it says so, and does not reload again."""
    shipped = _name_of(_INDEX)
    came_from = f"http://page.test/?page={shipped}"
    with (
        _LiveServer(_app(_workspace(tmp_path))) as base,
        _OpenPage(base, tmp_path, _tail(href=came_from), page=_named(_ELSEWHERE)) as page,
    ):
        snap = _connected(page)
    assert snap["replaced"] == []
    assert snap["note"] is not None and "saved copy" in snap["note"]
    assert snap["href"] == "http://page.test/"


@_needs_node
@pytest.mark.parametrize("typed", ["half a thought", ""])
def test_a_page_that_cannot_keep_what_is_typed_says_a_newer_one_is_there_and_stays(
    tmp_path: Path, typed: str
) -> None:
    """Where the browser lets the page keep nothing, a reload would lose what is typed: the page
    says a newer one is there and stays. With nothing typed there is nothing to lose: it reloads."""
    with (
        _LiveServer(_app(_workspace(tmp_path))) as base,
        _OpenPage(base, tmp_path, _tail(broken=True, typed=typed), page=_named(_ELSEWHERE)) as page,
    ):
        snap = _connected(page)
    if typed:
        assert snap["replaced"] == []
        assert snap["note"] is not None and "copy it first" in snap["note"]
        assert snap["typed"] == typed
    else:
        assert snap["replaced"] == [f"http://page.test/?page={_name_of(_INDEX)}"]


@_needs_node
def test_a_page_with_no_name_never_reloads(tmp_path: Path) -> None:
    """A page that did not come from the server's "/" (a copy kept somewhere else) still holds the
    slot for its name: it cannot tell which page it is, so it never reloads itself."""
    with (
        _LiveServer(_app(_workspace(tmp_path))) as base,
        _OpenPage(base, tmp_path, _tail()) as page,  # the shipped file as it is, slot and all
    ):
        snap = _connected(page)
    assert snap["replaced"] == []
    assert snap["note"] is None


@_needs_node
def test_a_server_that_names_no_page_never_reloads_a_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A server that sends no page name (an older one, or one in a checkout without the page)
    never differs from the page, whatever its name."""
    empty = tmp_path / "no-page"
    empty.mkdir()
    with (
        _LiveServer(_serving(monkeypatch, _workspace(tmp_path), empty)) as base,
        _OpenPage(base, tmp_path, _tail(), page=_named(_ELSEWHERE)) as page,
    ):
        snap = _connected(page)
    assert snap["replaced"] == []
    assert snap["note"] is None


def test_the_page_carries_its_name_and_every_watch_response_sends_it(tmp_path: Path) -> None:
    shipped = _INDEX.read_bytes()
    name = _name_of(_INDEX)
    assert shipped.count(server_app._PAGE_DIGEST_SLOT) == 1  # one slot, where the page reads it
    with _LiveServer(_app(_workspace(tmp_path))) as base:
        page = httpx.get(base + "/", timeout=5)
        session_id = httpx.post(base + "/sessions", timeout=5).json()["id"]
        header = _page_header(base, session_id)
    assert page.content == shipped.replace(server_app._PAGE_DIGEST_SLOT, name.encode(), 1)
    assert page.headers["cache-control"] == "no-cache"
    assert header == name


def test_a_server_serves_the_page_it_started_with_whatever_lands_on_disk_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An update puts another build's page on disk under a running server. The server goes on
    serving its own, the page its routes were written for, until it restarts into that build."""
    folder = _pages(tmp_path, "pages")
    started_with = _name_of(folder / "index.html")
    app = _serving(monkeypatch, _workspace(tmp_path), folder)
    (folder / "index.html").write_bytes(_INDEX.read_bytes() + _ANOTHER_BUILD)  # an update lands
    with _LiveServer(app) as base:
        page = httpx.get(base + "/", timeout=5)
        session_id = httpx.post(base + "/sessions", timeout=5).json()["id"]
        header = _page_header(base, session_id)
    slot = server_app._PAGE_DIGEST_SLOT
    assert page.content == _INDEX.read_bytes().replace(slot, started_with.encode(), 1)
    assert header == started_with
