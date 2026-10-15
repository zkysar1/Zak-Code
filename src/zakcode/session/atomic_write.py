"""Publish a text file in one atomic step, safely from any thread.

``os.replace`` makes the final rename atomic, but only for a temp file that is the caller's
own. A temp name built from the PID is shared by every thread of the server, and the sync
routes run on its worker threads, so two overlapping calls wrote the SAME temp path and the
second ``os.replace`` found nothing left to rename and failed (ADR-0318). The name here
carries a fresh random suffix instead.
"""

from __future__ import annotations

import contextlib
import os
import uuid
from pathlib import Path

_TMP_SUFFIX = ".tmp"


def write_text_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` so a reader sees the old file or the whole new one.

    The temp file is unique to this call and sits beside ``path``, so ``os.replace`` is a
    rename on one filesystem. If the write or the rename fails, the temp file is removed, the
    exception propagates, and ``path`` is left as it was.
    """
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}{_TMP_SUFFIX}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        # Best-effort cleanup of the partial temp file; swallow cleanup errors so the
        # original failure is the one that propagates.
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise
