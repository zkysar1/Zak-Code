"""``secret_names`` — list the names of configured secrets. Names only, never values.

The discovery half of the named-secrets feature (see ``_secrets.py``): the model asks
what is available, gets back NAMES plus the placeholder syntax, and then references a
secret as ``{{secret:NAME}}`` in ``web_fetch`` — where the value is substituted into
the outbound request outside the model's context.

Registered unconditionally (matching the always-registered web tools): with no secrets
configured it reports that cleanly, which is a better model experience than the tool
sometimes not existing.
"""

from __future__ import annotations

from zakcode.config import PermissionTier
from zakcode.tools.base import (
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from zakcode.tools.builtins._secrets import SecretsProvider


class SecretNamesTool(Tool):
    """List available secret NAMES for ``{{secret:NAME}}`` substitution."""

    spec = ToolSpec(
        name="secret_names",
        description=(
            "List the names of secrets saved for this session (API keys etc.). Values are "
            "never shown; use a name as {{secret:NAME}} in web_fetch's url or headers and "
            "the real value is substituted outside your context."
        ),
        parameters={"type": "object", "properties": {}},
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    def __init__(self, secrets: SecretsProvider | None = None) -> None:
        self._secrets = secrets or SecretsProvider(None)

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        names = self._secrets.names()
        if not names:
            return ToolResult.ok(
                "No secrets are configured for this session.",
                data={"names": []},
            )
        listing = "\n".join(f"- {name}" for name in names)
        return ToolResult.ok(
            f"Available secrets (reference as {{{{secret:NAME}}}} in web_fetch):\n{listing}",
            data={"names": names},
            hint='e.g. web_fetch with headers {"Authorization": "Bearer {{secret:'
            + names[0]
            + '}}"}',
        )
