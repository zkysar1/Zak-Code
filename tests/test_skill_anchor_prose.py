"""A description is prose: one shared word is a topic, two are a request (ADR-0040).

Field transcript 2026-08-27: "go actually try to fetch some of those, there is no way it is
this big already" seeded ``/research`` as a plan step — anchored on nothing but the word
"fetch" in the skill's description — and the turn that followed never touched a tool.
"""

from __future__ import annotations

from zakcode.providers.routing import implied_skill_anchored

_RESEARCH = ("research", "Fetch and summarise web sources on a topic into a knowledge note")
_NOTIFY = ("notify-user", "Send the operator a message about an event")


def test_one_description_word_is_a_topic_not_a_request() -> None:
    assert not implied_skill_anchored(
        "go actually try to fetch some of those, there is no way it is this big already",
        *_RESEARCH,
    )


def test_the_name_still_anchors_on_one_stem() -> None:
    assert implied_skill_anchored("research the vendor's rate limits", *_RESEARCH)
    assert implied_skill_anchored("run the notifier", *_NOTIFY)  # notif ~ notify


def test_two_description_words_anchor() -> None:
    assert implied_skill_anchored("send the operator a heads-up", *_NOTIFY)  # send + operator
    assert implied_skill_anchored("fetch and summarise these sources", *_RESEARCH)


def test_stopwords_never_anchor() -> None:
    assert not implied_skill_anchored("use the tool on another one of these", *_NOTIFY)


# ── ADR-0109: a NAME made of everyday words needs a reference shape ────────────────────

# The Mind's control command, description as shipped (abridged to its first sentences).
_START = (
    "start",
    "Creates or resumes an agent in reader (read-only), assistant (user-directed), or "
    "autonomous mode (perpetual loop). USER-ONLY — Claude must NEVER invoke /start. Fires "
    "only when the user types /start {agent-name} [--mode {mode}].",
)
_TEST = ("test", "Run the project's test suite and report the failures")


def test_the_incident_string_does_not_anchor_a_skill_named_start() -> None:
    """2026-09-05: 'lets start from scratch' seeded `run /start`; the plan gate then made the
    model run the Mind's start-an-agent command."""
    assert not implied_skill_anchored("ok, clear that plan, and lets start from scratch", *_START)
    assert not implied_skill_anchored("lets test this quickly before we move on", *_TEST)


def test_a_generic_name_anchors_when_referenced_as_a_skill() -> None:
    assert implied_skill_anchored("run /start alpha", *_START)  # slash token
    assert implied_skill_anchored("use the start skill for alpha", *_START)  # "<name> skill"
    assert implied_skill_anchored("run the test command", *_TEST)  # invocation verb
    assert implied_skill_anchored("please invoke test", *_TEST)


def test_a_generic_name_still_anchors_on_two_description_words() -> None:
    assert implied_skill_anchored("run the suite and report failures", *_TEST)  # suite + report


def test_a_distinctive_name_still_anchors_on_its_stem() -> None:
    assert implied_skill_anchored("research the vendor's rate limits", *_RESEARCH)
    assert implied_skill_anchored(
        "add an aspiration for the report",
        "create-aspiration",
        "Create a new aspiration in the world queue",
    )  # `create` is generic, `aspiration` is not — the distinctive stem carries it


def test_generic_is_judged_on_whole_name_words_not_stems() -> None:
    from zakcode.providers.routing import _name_is_generic

    assert (
        _name_is_generic("start") and _name_is_generic("reset") and _name_is_generic("test-report")
    )
    assert not _name_is_generic("research")  # shares the stem `rese` with `reset`; not generic
    assert not _name_is_generic("create-aspiration") and not _name_is_generic("forge-skill")
    assert not _name_is_generic("")


def test_reference_shapes_span_multi_word_names() -> None:
    from zakcode.providers.routing import _references_skill

    assert _references_skill("run forge-skill on it", "forge-skill")
    assert _references_skill("use the forge skill", "forge-skill")
    assert _references_skill("try /forge_skill", "forge-skill")
    assert not _references_skill("we should forge ahead with the skill", "forge-skill")
    assert not _references_skill("a/start/b is a path", "start")  # not a slash token
