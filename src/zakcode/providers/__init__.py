"""Provider abstraction: the vendor-agnostic seam between the agent loop and any LLM
backend.

Only modules in this package may import ``litellm`` or a vendor SDK (see
``docs/GUARDRAILS.md``). The agent loop depends solely on the base contracts re-exported
here.
"""

from zakcode.providers import _env  # noqa: F401  — pre-litellm env defaults; keep first
from zakcode.providers.base import (
    AuthError,
    Capabilities,
    ContextWindowExceeded,
    LLMResult,
    ModelOutputRejected,
    Provider,
    ProviderError,
    RateLimited,
    RequestFailed,
    ToolCall,
    UnknownContextWindow,
    WindowResolution,
)
from zakcode.providers.bitnet import BitNetProvider
from zakcode.providers.claude_code import ClaudeCodeProvider, CompletionBridge
from zakcode.providers.litellm_provider import LiteLLMProvider
from zakcode.providers.registry import get_capabilities
from zakcode.providers.resolve import (
    AvailabilityResolver,
    ModelResolutionError,
    ModelResolver,
    ResolvedModel,
)
from zakcode.providers.structured import (
    StructuredResult,
    StructuredValidationError,
    coerce_structured,
    complete_structured,
    extract_json,
    make_response_format,
    schema_error,
)
from zakcode.providers.text_tools import (
    TextToolCallingProvider,
    parse_text_tool_calls,
    render_tool_protocol,
    textify_messages,
)

__all__ = [
    "AuthError",
    "AvailabilityResolver",
    "BitNetProvider",
    "Capabilities",
    "ClaudeCodeProvider",
    "CompletionBridge",
    "ContextWindowExceeded",
    "LLMResult",
    "LiteLLMProvider",
    "ModelOutputRejected",
    "ModelResolutionError",
    "ModelResolver",
    "Provider",
    "ProviderError",
    "RateLimited",
    "RequestFailed",
    "ResolvedModel",
    "UnknownContextWindow",
    "WindowResolution",
    "StructuredResult",
    "StructuredValidationError",
    "TextToolCallingProvider",
    "ToolCall",
    "coerce_structured",
    "complete_structured",
    "extract_json",
    "get_capabilities",
    "make_response_format",
    "parse_text_tool_calls",
    "render_tool_protocol",
    "schema_error",
    "textify_messages",
]
