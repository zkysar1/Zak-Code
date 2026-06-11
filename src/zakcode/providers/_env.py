"""Pre-litellm environment defaults (imported FIRST by the providers package).

litellm reads ``LITELLM_LOG`` while it is being imported, and configures its own
"LiteLLM" logger then — so a default set any later (a logger ``setLevel``, a
``load_settings()``-time export) cannot quiet its import-time WARNING chatter
(missing optional backends like botocore). Every route to litellm runs the
``zakcode.providers`` package ``__init__`` first, and that ``__init__`` imports
this module before anything that imports litellm — making this the one place the
default is guaranteed to land in time. ``setdefault`` so an operator's explicit
``LITELLM_LOG`` always wins. (D20 rider; the CLI keeps a harmless backstop.)
"""

from __future__ import annotations

import os

os.environ.setdefault("LITELLM_LOG", "ERROR")
