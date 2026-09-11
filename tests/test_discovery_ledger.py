"""The discovery ledger: the mind accumulates what the vessel's projection cannot hold.

``discoveryPerception`` is rebuilt from scratch every perception tick and bounded at the
character's bubble, so an entity that is explored and then walked away from VANISHES from the
next envelope. These tests pin the properties that follow from that, because each one is a
place a future reader could "simplify" the ledger back into a mirror of the slice and silently
re-lock everything the character has found:

- membership is sticky — absence, and an explicit ``discovered: false``, are both readings of
  the current frame, never retractions,
- the unlock is an EVENT, reported on the envelope that caused it and not again,
- merely-seen entities are not recorded, because the next envelope re-supplies them,
- every failure is open: a corrupt ledger, an unwritable one, and a garbage row each cost the
  ledger something and cost the perception nothing.

Hermetic: tmp_path workspaces, no network, no server.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from zakcode.session.discovery_ledger import (
    DISCOVERY_LEDGER_VERSION,
    discovery_path,
    fold_observation,
    read_ledger,
    write_ledger,
)


def _envelope(slice_: dict[str, Any] | None, *, at: str = "2026-09-11T21:00:00Z") -> dict[str, Any]:
    observation: dict[str, Any] = {"spatial": {"nearby": ["a wall"]}}
    if slice_ is not None:
        observation["discoveryPerception"] = slice_
    return {
        "envelopeVersion": 1,
        "externalClientRef": "vessel-1",
        "observedAt": at,
        "observation": observation,
        "droppedSlices": [],
        "frame": "FRAMED-AS-DATA\n\n",
    }


def _row(*, touch: int, status: str = "Within Touching Reach") -> dict[str, Any]:
    """The producer's own row shape: it derives `discovered` as touchCount > 0."""
    return {"touchCount": touch, "distanceStatus": status, "discovered": touch > 0}


def test_an_unlock_is_recorded_and_reported(tmp_path: Path) -> None:
    path = discovery_path(tmp_path)

    newly = fold_observation(path, _envelope({"fountain": _row(touch=1)}))

    assert newly == ["fountain"]
    assert list(read_ledger(path)) == ["fountain"]
    assert read_ledger(path)["fountain"]["discoveredAt"] == "2026-09-11T21:00:00Z"


def test_a_seen_but_untouched_entity_is_not_recorded(tmp_path: Path) -> None:
    """The frontier is in the next envelope anyway — only the unlock outlives the frame."""
    path = discovery_path(tmp_path)

    newly = fold_observation(path, _envelope({"statue": _row(touch=0, status="Within Sight")}))

    assert newly == []
    assert read_ledger(path) == {}
    assert path.exists() is False, "an all-untouched envelope wrote a ledger with nothing in it"


def test_the_unlock_is_reported_once_not_every_round(tmp_path: Path) -> None:
    path = discovery_path(tmp_path)
    assert fold_observation(path, _envelope({"fountain": _row(touch=1)})) == ["fountain"]

    again = fold_observation(
        path, _envelope({"fountain": _row(touch=2)}, at="2026-09-11T21:00:05Z")
    )

    assert again == [], "a standing unlock was re-announced as if it were new"
    assert read_ledger(path)["fountain"]["touchCount"] == 2, "latest-wins on the count"


def test_walking_away_does_not_re_lock(tmp_path: Path) -> None:
    """THE POINT OF THE MODULE: the vessel prunes on departure, the mind must not."""
    path = discovery_path(tmp_path)
    fold_observation(path, _envelope({"fountain": _row(touch=1), "gate": _row(touch=3)}))

    # The character walks off; the next projection is bounded to what is still in the bubble.
    fold_observation(path, _envelope({"gate": _row(touch=3)}, at="2026-09-11T21:01:00Z"))

    assert set(read_ledger(path)) == {"fountain", "gate"}, "an absent entity was re-locked"


def test_a_reported_false_is_a_reset_not_a_retraction(tmp_path: Path) -> None:
    """A vessel-side counter reset must stay VISIBLE without un-discovering anything."""
    path = discovery_path(tmp_path)
    fold_observation(path, _envelope({"fountain": _row(touch=4)}))

    fold_observation(
        path,
        _envelope({"fountain": {"touchCount": 0, "discovered": False}}, at="2026-09-11T22:00:00Z"),
    )

    entry = read_ledger(path)["fountain"]
    assert "fountain" in read_ledger(path), "a false reading retracted an unlock"
    assert entry["touchCount"] == 0, "the reset was hidden instead of reported"
    assert entry["discoveredAt"] == "2026-09-11T21:00:00Z", "the unlock stamp was overwritten"


def test_the_explicit_flag_outranks_the_counter(tmp_path: Path) -> None:
    """The producer derives the flag from the count today; if that changes, the flag decides."""
    path = discovery_path(tmp_path)

    newly = fold_observation(
        path,
        _envelope(
            {
                "sigil": {"touchCount": 0, "discovered": True},
                "rubble": {"touchCount": 7, "discovered": False},
            }
        ),
    )

    assert newly == ["sigil"], "the consumer re-derived a rule the vessel owns"
    assert "rubble" not in read_ledger(path)


def test_the_counter_is_the_fallback_when_no_flag_is_sent(tmp_path: Path) -> None:
    path = discovery_path(tmp_path)

    newly = fold_observation(
        path, _envelope({"torch": {"touchCount": 2}, "moss": {"touchCount": 0}})
    )

    assert newly == ["torch"]
    assert "moss" not in read_ledger(path)


def test_an_envelope_without_the_slice_is_inert(tmp_path: Path) -> None:
    """This consumer ships BEFORE the producer is merged — it must sit quiet, not error."""
    path = discovery_path(tmp_path)

    assert fold_observation(path, _envelope(None)) == []
    assert fold_observation(path, None) == []
    assert fold_observation(path, {"observation": "not-a-map"}) == []
    assert path.exists() is False


def test_a_malformed_row_costs_only_itself(tmp_path: Path) -> None:
    path = discovery_path(tmp_path)

    newly = fold_observation(
        path,
        _envelope({"fountain": _row(touch=1), "broken": "not-an-object", "gate": _row(touch=2)}),
    )

    assert newly == ["fountain", "gate"], "one bad row aborted the good rows beside it"
    assert "broken" not in read_ledger(path)


def test_a_non_integer_count_reads_as_zero_not_as_a_crash(tmp_path: Path) -> None:
    path = discovery_path(tmp_path)

    newly = fold_observation(path, _envelope({"idol": {"touchCount": "lots", "discovered": True}}))

    assert newly == ["idol"]
    assert read_ledger(path)["idol"]["touchCount"] == 0


def test_a_corrupt_ledger_rebuilds_instead_of_wedging(tmp_path: Path) -> None:
    path = discovery_path(tmp_path)
    path.write_text("{not json at all", encoding="utf-8")

    newly = fold_observation(path, _envelope({"fountain": _row(touch=1)}))

    assert newly == ["fountain"]
    assert read_ledger(path) == {
        "fountain": {
            "discoveredAt": "2026-09-11T21:00:00Z",
            "touchCount": 1,
            "lastObservedAt": "2026-09-11T21:00:00Z",
        }
    }


def test_a_ledger_from_a_version_we_do_not_speak_is_not_guessed_at(tmp_path: Path) -> None:
    path = discovery_path(tmp_path)
    path.write_text(
        json.dumps({"ledgerVersion": DISCOVERY_LEDGER_VERSION + 1, "discovered": {"x": {}}}),
        encoding="utf-8",
    )

    assert read_ledger(path) == {}


def test_an_unwritable_ledger_still_reports_the_unlock(tmp_path: Path) -> None:
    """Fail-open: the caller must still be able to say what was found, ledger or no ledger."""
    blocked = tmp_path / "not-a-dir"
    blocked.write_text("a file where a directory would have to be", encoding="utf-8")
    path = discovery_path(blocked)

    newly = fold_observation(path, _envelope({"fountain": _row(touch=1)}))

    assert newly == ["fountain"]
    assert write_ledger(path, {"fountain": {}}) is False


def test_no_temp_file_is_left_behind(tmp_path: Path) -> None:
    """The write is atomic; a stray .tmp would read as workspace litter to every other tool."""
    path = discovery_path(tmp_path)
    fold_observation(path, _envelope({"fountain": _row(touch=1)}))

    assert [p.name for p in tmp_path.iterdir()] == [".discovery"]


def test_an_unchanged_round_does_not_rewrite_the_file(tmp_path: Path) -> None:
    """The perception path fires every iteration; an idle round should cost no write."""
    path = discovery_path(tmp_path)
    envelope = _envelope({"fountain": _row(touch=1)})
    fold_observation(path, envelope)
    before = path.read_bytes()
    stamped = path.stat().st_mtime_ns

    assert fold_observation(path, _envelope({"fountain": _row(touch=1)})) == []

    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns == stamped, "an identical envelope rewrote the ledger"
