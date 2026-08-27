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
