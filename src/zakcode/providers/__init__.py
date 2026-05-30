"""Vendor-agnostic LLM provider layer.

All model access in Zak Code flows through this package, built on top of ``litellm`` so
the same agent runs on ~100 providers (first-class, tested: Ollama and OpenAI). The agent
loop must never embed a provider-specific request shape — switching providers is a
configuration change only.

:mod:`zakcode.providers.base` defines the :class:`Provider` ABC and the canonical
request/response value objects. The concrete litellm-backed provider lands next.
See ``docs/ARCHITECTURE.md``.
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

__all__ = [
    "AuthError",
    "Capabilities",
    "ContextWindowExceeded",
    "LLMResult",
    "Provider",
    "ProviderError",
    "RateLimited",
    "RequestFailed",
    "ToolCall",
]
