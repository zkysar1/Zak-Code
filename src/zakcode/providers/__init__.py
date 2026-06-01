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
from zakcode.providers.text_tools import (
    TextToolCallingProvider,
    parse_text_tool_calls,
    render_tool_protocol,
    textify_messages,
)

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
    "TextToolCallingProvider",
    "ToolCall",
    "get_capabilities",
    "parse_text_tool_calls",
    "render_tool_protocol",
    "textify_messages",
]
