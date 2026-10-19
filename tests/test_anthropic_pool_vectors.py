"""ADR-0323: the Python pool passes the policy's shared test vectors.

The credit pool is one policy with two implementations, and the vectors are what keep them one:
each is a start state plus a list of assignments and responses, with the decision and the account
state the policy requires after each step. This file runs all of them against
:class:`zakcode.providers.anthropic_pool.Pool`, reading the vectors and the registry the way the
other runtime's runner does.

The vectors and the registry here are verbatim copies. ``fixtures/anthropic_pool/SOURCE.json``
pins their git blob ids, so a hand edit of either copy fails here; and with
``ZAKCODE_POOL_SOURCE_CHECKOUT`` pointing at a checkout of the source repository the same ids are
compared with the originals, which is the check that the copies have not fallen behind.

A runner that cannot fail proves nothing, so the control feeds it a vector whose expectation is
wrong in one field and requires the mismatch to be reported.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from zakcode.providers.anthropic_pool import (
    DEFAULT_REGISTRY_PATH,
    Pool,
    apply_registry_overrides,
    load_registry,
    parse_iso,
)

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURES = Path(__file__).parent / "fixtures" / "anthropic_pool"
_VECTORS_PATH = _FIXTURES / "policy-vectors.json"
_SOURCE = json.loads((_FIXTURES / "SOURCE.json").read_text(encoding="utf-8"))
_VECTORS: list[dict[str, Any]] = json.loads(_VECTORS_PATH.read_text(encoding="utf-8"))


def _blob_id(data: bytes) -> str:
    """The git blob id of ``data``, line endings normalised so a Windows checkout agrees."""
    data = data.replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _run(vector: dict[str, Any]) -> list[str]:
    """Run one vector; the mismatches it finds, empty when the policy does what it requires."""
    registry = apply_registry_overrides(load_registry(), vector.get("registry_overrides"))
    base_now = parse_iso(vector["now"])
    assert base_now is not None
    pool = Pool(registry, vector["env_present"], vector.get("state", {}), base_now)
    errors: list[str] = []
    for index, op in enumerate(vector["ops"]):
        now = parse_iso(op["now"]) if "now" in op else base_now
        assert now is not None
        where = f"op[{index}] {op['op']} {op.get('session', '')}"
        if op["op"] == "assign":
            decision = pool.assign(op["session"], op["use"], now)
        elif op["op"] == "response":
            decision = pool.handle_response(
                op["session"],
                op["status"],
                now,
                error_type=op.get("error_type"),
                message=op.get("message"),
                retry_after=op.get("retry_after"),
                stop_reason=op.get("stop_reason"),
            )
        else:
            errors.append(f"{where}: unknown op")
            continue
        got = decision.as_dict()
        expect = op.get("expect", {})
        for name in ("member", "retry", "moved", "alert", "retry_after"):
            if name in expect and got.get(name) != expect[name]:
                errors.append(f"{where}: {name} expected {expect[name]!r}, got {got.get(name)!r}")
        for member, fields in expect.get("state_after", {}).items():
            state = pool.get_member_state(member)
            for name, want in fields.items():
                if state.get(name) != want:
                    errors.append(
                        f"{where}: state_after {member}.{name} expected {want!r}, "
                        f"got {state.get(name)!r}"
                    )
    return errors


@pytest.mark.parametrize("vector", _VECTORS, ids=[v["name"] for v in _VECTORS])
def test_vector(vector: dict[str, Any]) -> None:
    assert _run(vector) == []


def test_every_vector_is_run() -> None:
    """The count is the file's, not a number remembered here: a vector added upstream is run
    by the parametrisation above with no edit, and this pins that none is silently dropped."""
    assert len(_VECTORS) == _SOURCE["vector_count"] == 22
    assert len({v["name"] for v in _VECTORS}) == len(_VECTORS)


def test_the_runner_reports_a_wrong_expectation() -> None:
    """Control: the same vector with one expected field flipped must come back with a mismatch."""
    vector = next(v for v in _VECTORS if v["name"] == "credit_402_exhausts_moves_retries")
    assert _run(vector) == []
    wrong = json.loads(json.dumps(vector))
    wrong["ops"][0]["expect"]["moved"] = not wrong["ops"][0]["expect"]["moved"]
    mismatches = _run(wrong)
    assert len(mismatches) == 1
    assert "moved expected" in mismatches[0]


def test_the_copies_are_the_pinned_ones() -> None:
    for relative, pinned in _SOURCE["files"].items():
        assert _blob_id((_ROOT / relative).read_bytes()) == pinned["git_blob"], (
            f"{relative} differs from the copy SOURCE.json pins: do not edit it by hand; "
            "refresh it from the source repository and update SOURCE.json with it"
        )
    assert (_ROOT / "src/zakcode/providers/anthropic_credit_registry.json") == DEFAULT_REGISTRY_PATH


@pytest.mark.skipif(
    not os.environ.get("ZAKCODE_POOL_SOURCE_CHECKOUT"),
    reason="set ZAKCODE_POOL_SOURCE_CHECKOUT to a checkout of the source repository",
)
def test_the_copies_have_not_fallen_behind_the_source() -> None:
    root = Path(os.environ["ZAKCODE_POOL_SOURCE_CHECKOUT"]) / _SOURCE["source_repository_path"]
    for pinned in _SOURCE["files"].values():
        original = root / pinned["source_file"]
        assert _blob_id(original.read_bytes()) == pinned["git_blob"], (
            f"{pinned['source_file']} at the source has changed since this copy was taken"
        )
