"""Tests for the betas litellm is made to forward (``zakcode.providers.beta_headers``, ADR-0341).

litellm passes every ``anthropic-beta`` value through an allowlist and drops one it does not
name. A process that cannot fetch the remote allowlist reads the copy bundled in the installed
litellm, which predates ``thinking-binding-controls-2026-08-01``, so the Claude 5 request carried
``thinking.block_binding`` without the beta that permits it and Anthropic answered 400, "Extra
inputs are not permitted" (measured on a served deployment whose fetch answered 403). The
ADR-0321 tests assert that ``_build_kwargs`` carries the beta, one layer above the filter, and
they stayed green while the beta never reached the wire.

The unit tests hand the helper a fake manager. The subprocess tests run a fresh interpreter on
litellm's BUNDLED list (``LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS=True``, with which litellm fetches
nothing) and read the request at the HTTP client, the one Anthropic call answered from a canned
body. Offline and deterministic.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from zakcode.providers.beta_headers import allow_provider_betas

_ROOT = Path(__file__).resolve().parents[1]
_BETA = "thinking-binding-controls-2026-08-01"


def _bundled() -> dict[str, Any]:
    """A list shaped like litellm's bundled one: a mapping per provider, no binding beta."""
    return {
        "description": "Mapping of Anthropic beta headers for each provider.",
        "anthropic": {"effort-2025-11-24": "effort-2025-11-24", "bash_20241022": None},
        "bedrock": {"effort-2025-11-24": "effort-2025-11-24"},
    }


class _Manager:
    """The members of ``litellm.anthropic_beta_headers_manager`` the helper touches."""

    def __init__(self, lists: list[Any], cached: Any = None) -> None:
        self._lists = list(lists)
        self.loads = 0
        self._BETA_HEADERS_CONFIG = cached

    def get_beta_headers_config(self, url: str) -> Any:
        self.loads += 1
        return self._lists[min(self.loads, len(self._lists)) - 1]


class TestAllowProviderBetas:
    def test_a_list_loaded_later_names_the_beta(self) -> None:
        manager = _Manager([_bundled()])

        assert allow_provider_betas(manager, "anthropic", [_BETA]) is True

        config = manager.get_beta_headers_config("https://example.invalid/list.json")
        assert config["anthropic"][_BETA] == _BETA
        assert config["anthropic"]["bash_20241022"] is None  # what the list said is kept
        assert _BETA not in config["bedrock"]  # another provider is not touched

    def test_every_load_names_the_beta(self) -> None:
        # First use, then a reload: each hands back a fresh list.
        manager = _Manager([_bundled(), _bundled()])
        allow_provider_betas(manager, "anthropic", [_BETA])

        assert manager.get_beta_headers_config("u")["anthropic"][_BETA] == _BETA
        assert manager.get_beta_headers_config("u")["anthropic"][_BETA] == _BETA

    def test_a_list_already_loaded_gets_the_beta_at_once(self) -> None:
        loaded = _bundled()
        manager = _Manager([], cached=loaded)

        allow_provider_betas(manager, "anthropic", [_BETA])

        assert loaded["anthropic"][_BETA] == _BETA

    @pytest.mark.parametrize("held", [_BETA, "renamed-beta", None])
    def test_a_list_that_holds_the_key_is_left_as_it_is(self, held: str | None) -> None:
        # Named, renamed, or named as unsupported: the list's own word stands.
        config = _bundled()
        config["anthropic"][_BETA] = held
        manager = _Manager([config])

        allow_provider_betas(manager, "anthropic", [_BETA])

        assert manager.get_beta_headers_config("u")["anthropic"][_BETA] == held

    def test_a_second_call_wraps_nothing_more(self) -> None:
        manager = _Manager([_bundled()])
        assert allow_provider_betas(manager, "anthropic", [_BETA])
        wrapper = manager.get_beta_headers_config
        assert allow_provider_betas(manager, "anthropic", [_BETA])

        assert manager.get_beta_headers_config is wrapper
        assert not hasattr(wrapper.__wrapped__, "__wrapped__")  # one layer over litellm's loader
        manager.get_beta_headers_config("u")
        assert manager.loads == 1

    def test_calls_with_other_names_compose(self) -> None:
        manager = _Manager([_bundled()])
        allow_provider_betas(manager, "anthropic", [_BETA])
        allow_provider_betas(manager, "bedrock", ["another-beta-2026-01-01"])

        config = manager.get_beta_headers_config("u")
        assert config["anthropic"][_BETA] == _BETA
        assert config["bedrock"]["another-beta-2026-01-01"] == "another-beta-2026-01-01"

    @pytest.mark.parametrize("loaded", ["not a dict", {"anthropic": "not a mapping"}, {}])
    def test_a_list_of_another_shape_comes_back_untouched(self, loaded: Any) -> None:
        manager = _Manager([loaded])
        allow_provider_betas(manager, "anthropic", [_BETA])

        assert manager.get_beta_headers_config("u") == loaded

    def test_a_litellm_without_the_loader_is_left_alone(self) -> None:
        assert allow_provider_betas(object(), "anthropic", [_BETA]) is False

    def test_the_loaders_own_error_is_not_swallowed(self) -> None:
        class _Raising(_Manager):
            def get_beta_headers_config(self, url: str) -> Any:
                raise RuntimeError("fetch failed")

        manager = _Raising([])
        allow_provider_betas(manager, "anthropic", [_BETA])

        with pytest.raises(RuntimeError, match="fetch failed"):
            manager.get_beta_headers_config("u")


# -- the request as litellm sends it -----------------------------------------------------------

#: Builds the Claude 5 request with the provider's own ``_build_kwargs`` and sends it through
#: litellm with the HTTP client replaced: the one call to Anthropic is answered from a canned body,
#: any other host fails the script, and what litellm put on the wire is printed as one JSON line.
_WIRE_PROBE = """
import json
import httpx
import litellm
from litellm import anthropic_beta_headers_manager as manager

litellm_loader = manager.get_beta_headers_config  # litellm's own, before zakcode wraps it

from zakcode.providers.litellm_provider import LiteLLMProvider

if __UNSHIM__:
    # The process as it was before ADR-0341: litellm's own loader and no list loaded yet.
    manager.get_beta_headers_config = litellm_loader
    manager._BETA_HEADERS_CONFIG = None

seen = {}

def fake_send(self, request, *args, **kwargs):
    assert request.url.host == "api.anthropic.com", str(request.url)
    seen["anthropic_beta"] = request.headers.get("anthropic-beta")
    seen["thinking"] = json.loads(request.content).get("thinking")
    return httpx.Response(200, request=request, json={
        "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
        "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
        "stop_sequence": None, "usage": {"input_tokens": 3, "output_tokens": 1},
    })

httpx.Client.send = fake_send
kwargs = LiteLLMProvider(model="anthropic/claude-sonnet-5-5")._build_kwargs(
    [{"role": "user", "content": "hi"}], None
)
litellm.completion(api_key="sk-ant-test", **kwargs)
print(json.dumps(seen))
"""


#: Reads the allowlist litellm bundles, the one a process that cannot fetch the remote copy uses.
_BUNDLED_PROBE = """
import json
from litellm.anthropic_beta_headers_manager import GetAnthropicBetaHeadersConfig

anthropic = GetAnthropicBetaHeadersConfig.load_local_beta_headers_config().get("anthropic", {})
print(json.dumps({"named": __BETA__ in anthropic, "entries": len(anthropic)}))
"""


def _run_json(script: str) -> Any:
    """Run ``script`` in a fresh interpreter on litellm's bundled lists; return its JSON line.

    ``LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS=True`` is what a process that cannot fetch litellm's
    allowlist falls back to, and ``LITELLM_LOCAL_MODEL_COST_MAP=True`` the same for the price map.
    The script's stdout must be that one JSON line and nothing else.
    """
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(_ROOT / "src"),
        PYTHONDONTWRITEBYTECODE="1",
        LITELLM_LOG="ERROR",
        LITELLM_LOCAL_MODEL_COST_MAP="True",
        LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS="True",
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=100,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == 1, f"stdout is not one JSON line (import-time output?):\n{result.stdout}"
    return json.loads(lines[0])


def _wire(*, unshim: bool) -> Any:
    return _run_json(_WIRE_PROBE.replace("__UNSHIM__", str(unshim)))


class TestTheBetaReachesAnthropic:
    def test_litellms_bundled_list_still_lacks_the_beta(self) -> None:
        """The premise of ADR-0341. When it fails, the litellm in use names the beta itself, and
        ``allow_provider_betas`` and the ADR can be removed."""
        bundled = _run_json(_BUNDLED_PROBE.replace("__BETA__", repr(_BETA)))

        assert bundled["entries"] > 0, "the bundled list was not read; 'lacks' would mean nothing"
        assert bundled["named"] is False, (
            f"litellm's bundled list now names {_BETA}: allow_provider_betas is redundant"
        )

    def test_on_litellms_bundled_list_the_request_carries_the_beta(self) -> None:
        wire = _wire(unshim=False)

        assert wire["anthropic_beta"] == _BETA
        assert wire["thinking"]["block_binding"] == {"prefix_mismatch_behavior": "drop_block"}

    def test_without_the_helper_litellm_drops_the_beta_and_keeps_the_field(self) -> None:
        """The control: the same request on the same list, with litellm's own loader.

        It is why the test above can fail, and it does not depend on the helper being installed.
        """
        wire = _wire(unshim=True)

        assert wire["anthropic_beta"] is None
        assert "block_binding" in wire["thinking"]
