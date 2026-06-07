"""Sandbox / isolation primitives.

Filesystem confinement is provided elsewhere (the workspace path guard in
``tools/builtins/_safety.py``). This package adds the **network-egress** half: a
domain-allowlisting forward proxy that constrains the agent's subprocess egress on a
**best-effort, cooperating-client** basis (a process that ignores the proxy env can still egress
directly — it is not a kernel firewall; see :mod:`zakcode.sandbox.egress`).
"""

from __future__ import annotations

from zakcode.sandbox.egress import EgressProxy

__all__ = ["EgressProxy"]
