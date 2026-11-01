"""The web page's brand comes from the deployment's settings, written in as it is served (ADR-0350).

A deployment that wanted its own name on the page had to edit the page in the install, and
``zakcode update`` replaces the install, so the next update put "Zak Code" back. The name, the new
chat's line and its suggestions are settings now (``ZAKCODE_BRAND_*``), which live with the
deployment, and the server writes them into the page's slots, escaped, as it writes in the page's
name (ADR-0345). Hermetic: TestClient, no network.
"""

from __future__ import annotations

import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.test_webclient_turn_clock import _page_script
from zakcode.config import Settings, load_settings
from zakcode.server import app as server_app
from zakcode.server.app import create_app
from zakcode.session.store import SessionStore

#: The page as it shipped before it had a brand, in every place it names itself.
_OWN = {
    "title": "Zak Code",
    "header": "Zak Code",
    "new_chat": "Zak Code",
    "line": "Ask about this workspace — Zak can read, edit, and run things.",
    "suggestions": [
        "explain this codebase",
        "find a bug and fix it",
        "run the tests and summarize",
    ],
    "placeholder": "Message Zak Code…",
    "root_class": "",
}
_BRAND_ENV = ("ZAKCODE_BRAND_NAME", "ZAKCODE_BRAND_TAGLINE", "ZAKCODE_BRAND_SUGGESTIONS")


class _Shown(HTMLParser):
    """Where the page names itself, as a browser reads it: entities decoded, markup not text."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: dict[str, object] = {"suggestions": []}
        self._into: str | None = None
        self._in_brand = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        classes = (a.get("class") or "").split()
        if tag == "html":
            self.seen["root_class"] = a.get("class") or ""
        elif tag == "title":
            self._into = "title"
        elif tag == "textarea" and a.get("id") == "input":
            self.seen["placeholder"] = a.get("placeholder") or ""
        elif tag == "div" and "brand" in classes:
            self._in_brand = True
        elif tag == "span" and self._in_brand and not classes:
            self._into = "header"
        elif tag == "div" and classes == ["t"]:
            self._into = "new_chat"
        elif tag == "div" and classes == ["h"]:
            self._into = "line"
        elif tag == "button" and "pill" in classes:
            self._into = "pill"
            self.seen["suggestions"].append("")  # type: ignore[attr-defined]

    def handle_endtag(self, tag: str) -> None:
        if tag == "div":
            self._in_brand = False
        self._into = None

    def handle_data(self, data: str) -> None:
        if self._into == "pill":
            self.seen["suggestions"][-1] += data  # type: ignore[index]
        elif self._into is not None:
            self.seen[self._into] = str(self.seen.get(self._into, "")) + data


def _shown(page: bytes) -> dict[str, object]:
    parser = _Shown()
    parser.feed(page.decode("utf-8"))
    return parser.seen


def _client(tmp_path: Path, **brand: object) -> TestClient:
    settings = Settings(  # type: ignore[arg-type]
        default_model="scripted/test", context_window=8192, workspace_root=tmp_path, **brand
    )
    app = create_app(settings=settings, store=SessionStore(base_dir=tmp_path / "sessions"))
    return TestClient(app)


def _page_name(page: bytes) -> str:
    found = re.search(rb'const PAGE_DIGEST = "([0-9a-f]{12})";', page)
    assert found, "the served page carries its name"
    return found.group(1).decode()


def test_with_no_brand_set_the_page_reads_as_it_always_has(tmp_path: Path) -> None:
    page = _client(tmp_path).get("/").content
    assert _shown(page) == _OWN
    assert b"__ZAKCODE_BRAND_" not in page


def test_a_deployment_s_brand_is_written_in_wherever_the_page_names_itself(tmp_path: Path) -> None:
    """Every value is text: markup in a setting is escaped, never read as markup."""
    name = 'Acme <Assist> & "Co"'
    page = (
        _client(
            tmp_path,
            brand_name=name,
            brand_tagline="Ask Acme about your <files>.",
            brand_suggestions=["summarize <b>this</b> folder", "draft a reply"],
        )
        .get("/")
        .content
    )
    assert _shown(page) == {
        "title": name,
        "header": name,
        "new_chat": name,
        "line": "Ask Acme about your <files>.",
        "suggestions": ["summarize <b>this</b> folder", "draft a reply"],
        "placeholder": f"Message {name}…",
        "root_class": "own-brand",
    }
    assert b"<Assist>" not in page and b"<files>" not in page and b"<b>this" not in page
    assert b"__ZAKCODE_BRAND_" not in page


def test_a_name_set_alone_is_the_name_the_new_chat_s_line_gives(tmp_path: Path) -> None:
    shown = _shown(_client(tmp_path, brand_name="Acme").get("/").content)
    assert shown["line"] == "Ask about this workspace — Acme can read, edit, and run things."
    assert shown["suggestions"] == _OWN["suggestions"]


def test_a_brand_naming_the_page_s_own_name_is_the_page_s_own(tmp_path: Path) -> None:
    """Setting the default name, or an empty one, is not a brand: the page keeps its own face."""
    for name in ("Zak Code", ""):
        assert _shown(_client(tmp_path, brand_name=name).get("/").content) == _OWN


def test_an_empty_list_of_suggestions_shows_none(tmp_path: Path) -> None:
    assert _shown(_client(tmp_path, brand_suggestions=[]).get("/").content)["suggestions"] == []


def test_the_page_s_name_covers_its_brand_so_a_new_brand_reloads_open_pages(
    tmp_path: Path,
) -> None:
    """A page's name is that of the page as branded (ADR-0345): a server restarted on a new brand
    gives its page another name, and an open page reloads into it, as after an update."""
    own = _client(tmp_path).get("/").content
    acme = _client(tmp_path, brand_name="Acme").get("/").content
    slot = server_app._PAGE_DIGEST_SLOT
    for page in (own, acme):
        name = _page_name(page)
        unnamed = page.replace(f'"{name}";'.encode(), b'"' + slot + b'";', 1)
        assert hashlib.sha256(unnamed).hexdigest()[:12] == name
    assert _page_name(own) != _page_name(acme)


def test_the_app_path_serves_the_branded_page_not_its_slots(tmp_path: Path) -> None:
    client = _client(tmp_path, brand_name="Acme")
    page = client.get("/").content
    for path in ("/app/", "/app/index.html"):
        resp = client.get(path)
        assert resp.status_code == 200
        assert resp.content == page


def test_the_brand_is_read_from_the_deployment_s_own_settings_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The brand lives in the config home's .env, outside the install, so an update that replaces
    the install leaves it where it was. The page in the install carries slots, not a brand."""
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text(
        "ZAKCODE_BRAND_NAME=Acme Assistant\n"
        "ZAKCODE_BRAND_TAGLINE=Ask Acme anything.\n"
        f"ZAKCODE_BRAND_SUGGESTIONS='{json.dumps(['summarize my notes'])}'\n",
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("ZAKCODE_HOME", str(home))
    for name in _BRAND_ENV:  # recorded, so the test takes back what the .env load exports
        monkeypatch.setenv(name, "recorded")
        monkeypatch.delenv(name)
    settings = load_settings(
        default_model="scripted/test", context_window=8192, workspace_root=workspace
    )
    app = create_app(settings=settings, store=SessionStore(base_dir=tmp_path / "sessions"))
    shown = _shown(TestClient(app).get("/").content)
    assert shown["title"] == "Acme Assistant"
    assert shown["line"] == "Ask Acme anything."
    assert shown["suggestions"] == ["summarize my notes"]
    installed = (server_app._STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "Acme" not in installed and "<title>__ZAKCODE_BRAND_NAME__</title>" in installed


def test_the_page_s_script_names_no_brand_of_its_own() -> None:
    """The composer's placeholder comes back from the page's markup, where the server wrote the
    brand, so the script holds no name an update could put back."""
    script = _page_script()
    assert "Zak Code" not in script and "Zak can" not in script
