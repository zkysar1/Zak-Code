"""Publish a text file in one atomic step, safely from any thread.

``os.replace`` makes the final rename atomic, but only for a temp file that is the caller's
own. A temp name built from the PID is shared by every thread of the server, and the sync
routes run on its worker threads, so two overlapping calls wrote the SAME temp path and the
second ``os.replace`` found nothing left to rename and failed (ADR-0318). The name here
carries a fresh random suffix instead.

Windows adds a second way for overlapping writers to fail. Observed on CI's Windows cell:
eight renames onto one target at once were refused with ``PermissionError`` (access denied)
for six of the eight, and ``status_file._atomic_write`` records the same refusal for a target
a reader has open. The likely cause is a handle held on the target for a moment (inferred, not
isolated), so on Windows the rename is retried and the writers take turns. POSIX renames are
serialised by the kernel, and a ``PermissionError`` there is a real refusal, so it is raised
at once.

``replace_retrying`` is that rename on its own, for a writer that makes its temp file itself:
``SessionStore.save`` syncs its temp to disk before the rename (ADR-0326).
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
import uuid
from pathlib import Path

_TMP_SUFFIX = ".tmp"

#: Retry a refused rename. True only on Windows, where the refusal can be momentary (see above).
_RETRY_RENAME_REFUSAL = sys.platform == "win32"
#: The retry is bounded: 50 attempts, 10 ms apart, about half a second before the refusal is raised.
_RENAME_ATTEMPTS = 50
_RENAME_PAUSE_S = 0.01


def write_text_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` so a reader sees the old file or the whole new one.

    The temp file is unique to this call and sits beside ``path``, so ``os.replace`` is a
    rename on one filesystem. If the write or the rename fails, the temp file is removed, the
    exception propagates, and ``path`` is left as it was.
    """
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}{_TMP_SUFFIX}")
    try:
        tmp.write_text(text, encoding="utf-8")
        replace_retrying(tmp, path)
    except BaseException:
        # Best-effort cleanup of the partial temp file; swallow cleanup errors so the
        # original failure is the one that propagates.
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise


def replace_retrying(tmp: Path, path: Path) -> None:
    """``os.replace`` ``tmp`` onto ``path``. On Windows a refused rename is retried until it
    clears or the bound is spent, then the refusal is raised; elsewhere it is raised at once.
    ``tmp`` stays the caller's: if this raises, removing it is the caller's job.
    """
    for attempt in range(_RENAME_ATTEMPTS):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if not _RETRY_RENAME_REFUSAL or attempt == _RENAME_ATTEMPTS - 1:
                raise
            time.sleep(_RENAME_PAUSE_S)
