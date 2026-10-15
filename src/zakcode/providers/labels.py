"""Display-only names for provider tokens (ADR-0314).

A model id is ``<provider>/<model>``, and the provider token is litellm's protocol, not the
vendor: ``openai/route-a`` through an OpenAI-compatible gateway sends nothing to OpenAI. The
``provider_labels`` setting maps a token to the name a person should read, for example
``{"openai": "Gateway"}``. Everything here builds text for a person and nothing else: the model
string on the wire, the ``source`` key in config, session, trace and transcript JSON, the log file
and every structured field keep the token as configured. With no labels every function returns
its input, so the default output is unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping


def provider_label(token: str, labels: Mapping[str, str] | None) -> str:
    """The name to show for a provider token: its label when one is set, else the token."""
    if not labels:
        return token
    return labels.get(token, token)


def display_model_id(model_id: str, labels: Mapping[str, str] | None) -> str:
    """``model_id`` with its LEADING provider token relabelled, for a person to read.

    Only the token before the first ``/`` is looked up, and only as a whole token:
    ``openai/route-a`` reads ``Gateway/route-a`` under ``{"openai": "Gateway"}``, while
    ``bedrock/converse/openai.route-c`` and ``openai-compatible/x`` are left as they are.
    """
    if not labels or "/" not in model_id:
        return model_id
    token, rest = model_id.split("/", 1)
    label = labels.get(token)
    return model_id if label is None else f"{label}/{rest}"


class ModelRef:
    """A model id handed to a log call, so the transcript can show it labelled (ADR-0314).

    The record's own text, which the log file keeps, renders it as the id: ``str()`` and
    ``repr()`` are the string's. The transcript handler reads :attr:`model_id` and shows
    :func:`display_model_id` of it instead.
    """

    __slots__ = ("model_id",)

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def __str__(self) -> str:
        return self.model_id

    def __repr__(self) -> str:
        return repr(self.model_id)


__all__ = ["ModelRef", "display_model_id", "provider_label"]
