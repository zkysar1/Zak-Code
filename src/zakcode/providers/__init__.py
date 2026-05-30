"""Vendor-agnostic LLM provider layer.

All model access in Zak Code flows through this package, built on top of ``litellm`` so
the same agent runs on ~100 providers (first-class, tested: Ollama and OpenAI). The agent
loop must never embed a provider-specific request shape — switching providers is a
configuration change only.

Status: scaffolded. The provider interface (sync + async completion, streaming, and
cross-provider tool-call normalization) is implemented in milestone M0.
See ``docs/ARCHITECTURE.md``.
"""
