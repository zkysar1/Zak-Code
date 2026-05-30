"""Provider abstraction: the vendor-agnostic seam between the agent loop and any LLM
backend.

Only modules in this package may import ``litellm`` or a vendor SDK (see
``docs/GUARDRAILS.md``). The agent loop depends solely on the base contracts re-exported
here.
"""

from zakcode.providers.base import (
    AuthError,
    Capabilities,
    ContextWindowExceeded,
    LLMResult,
    Provider,
    ProviderError,
    RateLimited,
    RequestFailed,
    ToolCall,
)
from zakcode.providers.litellm_provider import LiteLLMProvider
from zakcode.providers.registry import get_capabilities

__all__ = [
    "AuthError",
    "Capabilities",
    "ContextWindowExceeded",
    "LLMResult",
    "LiteLLMProvider",
    "Provider",
    "ProviderError",
    "RateLimited",
    "RequestFailed",
    "ToolCall",
    "get_capabilities",
]
