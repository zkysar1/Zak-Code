"""Backward-compatible re-export from the extracted zds-llm-provider package.

The canonical provider contract (Provider ABC, LLMResult, ToolCall, Capabilities,
streaming events, error taxonomy) now lives in ``zds_llm_provider.types`` (the
vendor-agnostic, pydantic-only package extracted in M-7). This module remains so
existing ``from zakcode.providers.base import ...`` call sites keep working unchanged.
"""

from zds_llm_provider.types import *  # noqa: F401,F403
from zds_llm_provider.types import __all__  # noqa: F401
