"""Vendor-agnostic LLM provider contract for the ZDS agent family.

Depends only on pydantic. No litellm, no vendor SDK, no network calls.
Any Python 3.11+ environment can implement Provider.
"""

from zds_llm_provider.messages import (
    ContentBlock,
    Message,
    Role,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from zds_llm_provider.text_tools import (
    TOOL_CALLING_MODES,
    TextToolCallingProvider,
    parse_text_tool_calls,
    render_tool_protocol,
    textify_messages,
)
from zds_llm_provider.types import (
    AuthError,
    Capabilities,
    ContextWindowExceeded,
    LLMResult,
    Provider,
    ProviderError,
    ProviderStreamEvent,
    RateLimited,
    RequestFailed,
    StreamDone,
    StreamTextDelta,
    StreamToolCallDelta,
    StreamUsage,
    ToolCall,
)
from zds_llm_provider.usage import Usage, UsageTracker

__all__ = [
    # messages
    "ContentBlock",
    "Message",
    "Role",
    "TextBlock",
    "ThinkingBlock",
    "ToolResultBlock",
    "ToolUseBlock",
    # types (provider ABC + value objects + errors + streaming)
    "AuthError",
    "Capabilities",
    "ContextWindowExceeded",
    "LLMResult",
    "Provider",
    "ProviderError",
    "ProviderStreamEvent",
    "RateLimited",
    "RequestFailed",
    "StreamDone",
    "StreamTextDelta",
    "StreamToolCallDelta",
    "StreamUsage",
    "ToolCall",
    # usage
    "Usage",
    "UsageTracker",
    # text tools
    "TOOL_CALLING_MODES",
    "TextToolCallingProvider",
    "parse_text_tool_calls",
    "render_tool_protocol",
    "textify_messages",
]
