"""Backward-compatible re-export from the extracted zds-llm-provider package.

The canonical text-mode tool-calling layer (TextToolCallingProvider, parse/render
helpers) now lives in ``zds_llm_provider.text_tools`` (the vendor-agnostic,
pydantic-only package extracted in M-7). This module remains so existing
``from zakcode.providers.text_tools import ...`` call sites keep working unchanged.
"""
from zds_llm_provider.text_tools import *  # noqa: F401,F403
from zds_llm_provider.text_tools import __all__  # noqa: F401
