"""Conversation state and persistence.

Holds the message history and usage accounting for a session and defines the on-disk
session format (resume/export). The concrete types live in
:mod:`zakcode.session.store`; this package re-exports them for convenience.
"""

from zakcode.session.store import Session, SessionStore

__all__ = ["Session", "SessionStore"]
