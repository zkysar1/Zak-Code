# Live-Model Shakedown

Validating Zak Code's provider/agent loop against a **real** local model (not just the
hermetic scripted-provider tests), per the M0 → live-shakedown → M1 plan.

> **Status (2026-05-30): blocked by host CPU; provider wiring verified, real inference pending.**
> A genuine attempt to run a local model on this machine hit a hardware wall (details below).
> The code path is proven as far as the environment allows; a clean run is a follow-up on
> AVX2-capable hardware (or once a CPU-matched llama.cpp binary builds).

## Goal

Serve a local GGUF behind an OpenAI-compatible HTTP API and drive it through Zak Code's
`LiteLLMProvider` → `AgentLoop`, exercising (a) plain chat and (b) tool-calling
(`write_file` → `read_file`). This is the first end-to-end run against weights, and the
forcing function for making **any** OpenAI-compatible local server first-class.

## What shipped from this effort (code, already merged)

Even before a model could run, the shakedown drove a real, tested improvement:

- **`Settings.api_base` / `Settings.api_key`** (env `ZAKCODE_API_BASE` / `ZAKCODE_API_KEY`),
  threaded through `LiteLLMProvider` so Zak Code can target **any** OpenAI-compatible server
  (llama.cpp / llama-cpp-python / vLLM / LM Studio) by config alone — previously only Ollama
  had a configurable base URL. Covered by `tests/test_endpoint_config.py` (env → Settings →
  provider → the kwargs handed to litellm). Commits `344d445` + fix `a17ec72`.

## Environment

| Thing | Value |
| --- | --- |
| CPU | **AMD FX-8320E** (Piledriver / Bulldozer family, 2012) |
| CPU SIMD | SSE4.2, **AVX (1.0)**, FMA3/FMA4 — **no AVX2, no F16C, no AVX-512** |
| OS | Windows 10 |
| Models dir | `C:\ZakNoCloud\GitHub\Models` (GGUFs incl. Qwen2.5-Coder-14B-Q5_K_M, Qwen_Qwen3.5-9B-Q6_K, Qwen3.5-0.8B-Q4_0) |
| Server venv | `C:\ZakNoCloud\_llama_server\.venv` (isolated; **not** the repo venv) |
| Compilers | mingw-w64 `gcc`/`g++` + `ninja` + `cmake` on PATH; **no MSVC** |

## The blocker

`pip install "llama-cpp-python[server]"` from the prebuilt CPU wheel index
(`https://abetlen.github.io/llama-cpp-python/whl/cpu`) **installs cleanly** (v0.3.23,
`import llama_cpp.server` works). But starting the server crashes the instant it initializes
the native backend:

```
File ".../llama_cpp/llama.py", line 208, in __init__
    llama_cpp.llama_backend_init()
OSError: [WinError -1073741795] Windows Error 0xc000001d
```

`0xc000001d` is **`STATUS_ILLEGAL_INSTRUCTION`**. The prebuilt wheels are compiled with an
**AVX2** baseline; the FX-8320E predates AVX2, so the first AVX2 instruction in
`ggml`/`llama` faults. This is a CPU/packaging mismatch — **not a Zak Code defect** (Zak
Code never even gets a connection to make a request).

## Remediation attempts

1. **Prebuilt CPU wheel (py3.11 and py3.12)** — installs, but `llama_backend_init()` →
   illegal instruction (above). ❌
2. **Source rebuild with AVX2 disabled** — rebuild `llama-cpp-python` from source against
   this exact CPU using the available mingw toolchain:
   ```
   set CMAKE_ARGS=-DGGML_AVX=ON -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_AVX512=OFF
   set CMAKE_GENERATOR=Ninja
   set CC=gcc & set CXX=g++ & set FORCE_CMAKE=1
   uv pip install --no-binary llama-cpp-python --force-reinstall --no-cache-dir llama-cpp-python
   ```
   _(Result recorded once the build completes. mingw↔CPython ABI mixing on Windows is the risk.)_

## Recommended paths to a real run (any one suffices)

- **Rebuild for this CPU** (attempt #2 above) — keeps everything local. Best if the mingw
  build links.
- **Ollama** — not installed here; its bundled llama.cpp ships non-AVX2 fallback builds and
  would likely run on this CPU. `ollama serve` + `ZAKCODE_DEFAULT_MODEL=ollama_chat/<model>`.
- **Different hardware** — any AVX2-capable machine runs the prebuilt wheel as-is; point
  `ZAKCODE_API_BASE` at it.
- **A hosted OpenAI-compatible endpoint** — set `ZAKCODE_DEFAULT_MODEL=openai/<model>` +
  `OPENAI_API_KEY`; the same `LiteLLMProvider` path is exercised.

## How to run the shakedown once a server is up

```powershell
# 1. serve (from the isolated venv), e.g. the small model to prove plumbing:
C:\ZakNoCloud\_llama_server\.venv\Scripts\python.exe -m llama_cpp.server `
  --model "C:\ZakNoCloud\GitHub\Models\Qwen3.5-0.8B-Q4_0.gguf" `
  --chat_format chatml-function-calling --host 127.0.0.1 --port 8000 --n_ctx 8192

# 2. point Zak Code at it and run the two-test harness (gitignored scratch _shakedown.py):
$env:OPENAI_API_KEY = "sk-local-noop"
uv run python _shakedown.py qwen3.5-0.8b http://127.0.0.1:8000/v1
# -> writes _shakedown_out.txt (plain-chat result + write_file→read_file tool result)
```

Findings (tool-call quirks, content-then-tool_calls handling, etc.) will be appended here
after the first successful run.
