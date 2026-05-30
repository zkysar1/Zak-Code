# zak/llm/provider.py
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator, Optional
import litellm

litellm.drop_params = True          # silently drop params a provider doesn't support
litellm.set_verbose = False
# litellm.add_function_to_prompt = True  # see Gotchas (tool fallback)

@dataclass
class ModelConfig:
    model: str                       # e.g. "openai/gpt-4o" or "ollama_chat/llama3.1"
    api_base: Optional[str] = None   # local Ollama: "http://localhost:11434"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout: float = 600.0
    num_retries: int = 2
    extra: dict = field(default_factory=dict)

@dataclass
class LLMResult:
    text: str
    tool_calls: list[dict]           # normalized: [{"id","name","arguments"(dict)}]
    finish_reason: str
    usage: dict                      # {"prompt_tokens","completion_tokens","total_tokens"}
    cost_usd: float
    raw: Any

class LLMProvider:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    def _params(self, messages, tools, tool_choice, stream, **kw):
        p = dict(
            model=self.cfg.model, messages=messages,
            temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens,
            timeout=self.cfg.timeout, num_retries=self.cfg.num_retries,
            api_base=self.cfg.api_base, api_key=self.cfg.api_key, stream=stream,
        )
        if tools:       p["tools"] = tools
        if tool_choice: p["tool_choice"] = tool_choice
        if stream:      p["stream_options"] = {"include_usage": True}  # usage in last chunk
        return {**p, **self.cfg.extra, **kw}

    def complete(self, messages, tools=None, tool_choice="auto", **kw) -> LLMResult:
        r = litellm.completion(**self._params(messages, tools, tool_choice, False, **kw))
        return _to_result(r)

    async def acomplete(self, messages, tools=None, tool_choice="auto", **kw) -> LLMResult:
        r = await litellm.acompletion(**self._params(messages, tools, tool_choice, False, **kw))
        return _to_result(r)

    def stream(self, messages, tools=None, tool_choice="auto", **kw) -> Iterator:
        return litellm.completion(**self._params(messages, tools, tool_choice, True, **kw))

    async def astream(self, messages, tools=None, tool_choice="auto", **kw) -> AsyncIterator:
        return await litellm.acompletion(**self._params(messages, tools, tool_choice, True, **kw))
```

Surface to expose to the rest of the app: `complete` / `acomplete` (non-streaming), `stream` / `astream` (returns the chunk iterator), plus normalization helpers `_to_result` and the streaming accumulator (below). `acompletion` is a true coroutine; its streaming object supports `async for`.

### 2. Model-string + config patterns (Ollama & OpenAI)

```python
# OpenAI (first-class). Prefix optional but use it explicitly for clarity.
ModelConfig(model="openai/gpt-4o", api_key=os.environ["OPENAI_API_KEY"])
# OpenAI-compatible endpoint (incl. Ollama's /v1 shim): keep "openai/" prefix + api_base
ModelConfig(model="openai/llama3.1", api_base="http://localhost:11434/v1", api_key="ollama")

# Local Ollama (native path). Prefer ollama_chat/ (hits /api/chat) over ollama/ (/api/generate).
ModelConfig(model="ollama_chat/llama3.1", api_base="http://localhost:11434")
```

- Model string is `provider/model`. Routing is decided by the prefix: `openai/…`, `ollama_chat/…`, `ollama/…`, `anthropic/…`, `vertex_ai/…`, etc. New providers later = just a new prefix, no code change.
- `api_base` can be passed per-call (preferred for Zak Code) or via env (`OPENAI_BASE_URL`, `OLLAMA_API_BASE`). Default Ollama host is `http://localhost:11434`. Note a known bug where some configs ignore `api_base` and require `OLLAMA_API_BASE` env var — set both to be safe for local.
- Ollama JSON mode: `format="json"` (legacy) or `response_format={...schema...}` for structured output.

### 3. Tool-call normalization (request + response)

Request `tools` is OpenAI JSON-schema format for **all** providers — LiteLLM translates it:

```python
tools = [{
  "type": "function",
  "function": {
    "name": "read_file",
    "description": "Read a file from disk",
    "parameters": {"type":"object","properties":{"path":{"type":"string"}},"required":["path"]},
  },
}]
```

Response normalization (provider-agnostic) — arguments arrive as a JSON **string**, so parse:

```python
import json
def _to_result(r) -> LLMResult:
    msg = r.choices[0].message
    calls = []
    for tc in (getattr(msg, "tool_calls", None) or []):
        try:    args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError: args = {"_raw": tc.function.arguments}
        calls.append({"id": tc.id, "name": tc.function.name, "arguments": args})
    u = r.usage or {}
    return LLMResult(
        text=msg.content or "",
        tool_calls=calls,
        finish_reason=r.choices[0].finish_reason,
        usage={"prompt_tokens":getattr(u,"prompt_tokens",0),
               "completion_tokens":getattr(u,"completion_tokens",0),
               "total_tokens":getattr(u,"total_tokens",0)},
        cost_usd=(r._hidden_params or {}).get("response_cost") or 0.0,
        raw=r,
    )
```

Feed a tool result back as a `role:"tool"` message keyed by `tool_call_id`:

```python
messages.append({"role":"assistant","content":res.text or None,
                 "tool_calls":res.raw.choices[0].message.tool_calls})
messages.append({"role":"tool","tool_call_id":call["id"],
                 "name":call["name"],"content":json.dumps(tool_output)})
```

### 4. Streaming (sync + async) and reconstruction

Each chunk exposes `chunk.choices[0].delta.content` (text) and `delta.tool_calls` (tool deltas). Tool-call args stream **incrementally**: only the first delta carries `id`/`name`; subsequent deltas carry partial `function.arguments` and rely on `index` to know which call they extend. Accumulate by index:

```python
async def consume_stream(stream):
    text, tool_acc, usage = [], {}, None
    async for chunk in stream:
        d = chunk.choices[0].delta
        if getattr(d, "content", None):
            text.append(d.content)
        for tc in (getattr(d, "tool_calls", None) or []):
            slot = tool_acc.setdefault(tc.index, {"id":None,"name":None,"args":""})
            if tc.id:                 slot["id"] = tc.id
            if tc.function and tc.function.name: slot["name"] = tc.function.name
            if tc.function and tc.function.arguments: slot["args"] += tc.function.arguments
        if getattr(chunk, "usage", None):   # final chunk when stream_options.include_usage
            usage = chunk.usage
    return "".join(text), tool_acc, usage
```

Alternatively collect chunks and call `litellm.stream_chunk_builder(chunks, messages=messages)` to rebuild a full `ModelResponse` — but it has known gaps (content-then-tool_calls, o1, some providers drop tool args), so the manual index-based accumulator above is more robust for an agent loop. Pass `stream_options={"include_usage": True}` to get a final usage-only chunk (empty `choices`).

### 5. Token & cost accounting

```python
from litellm import token_counter, completion_cost, cost_per_token, get_max_tokens

token_counter(model="openai/gpt-4o", messages=messages)        # pre-flight count (tiktoken fallback)
get_max_tokens("openai/gpt-4o")                                 # context budget guard
res._hidden_params["response_cost"]                            # per-call USD (preferred)
completion_cost(completion_response=res)                        # same, recompute from response
cost_per_token(model="openai/gpt-4o", prompt_tokens=5, completion_tokens=10)  # (in_usd, out_usd)
```

`usage` (prompt/completion/total tokens) is on `response.usage`. For **streaming**, cost/usage are absent unless you set `include_usage` (or run `completion_cost` on the rebuilt response). Local Ollama models usually have no entry in the cost map → cost is `0.0` (fine; treat local as free, or `litellm.register_model()` with custom pricing). Track per-call cost centrally in the provider wrapper and accumulate a session total.

### 6. Error handling & retries

All exceptions subclass the matching `openai.*` types, so existing OpenAI handling works. Catch `litellm.*` (or `openai.*`):

| litellm | inherits | code | retry? |
|---|---|---|---|
| `AuthenticationError` | openai.AuthenticationError | 401 | no |
| `PermissionDeniedError` | openai.PermissionDeniedError | 403 | no |
| `BadRequestError` / `ContextWindowExceededError` / `ContentPolicyViolationError` | openai.BadRequestError | 400 | no |
| `NotFoundError` | openai.NotFoundError | 404 | no |
| `Timeout` | openai.APITimeoutError | 408 | yes |
| `RateLimitError` | openai.RateLimitError | 429 | yes |
| `APIConnectionError` / `APIError` / `InternalServerError` | openai.* | 500 | yes |
| `ServiceUnavailableError` | openai.APIStatusError | 503 | yes |

Every exception carries `.status_code`, `.message`, `.llm_provider`. Built-in retry via `num_retries=` (exponential backoff on 408/429/5xx). Cross-provider failover via `fallbacks=[{"primary":["backup"]}]` per call, or use `litellm.Router` for a model_list with fallbacks/load-balancing (`router.acompletion(...)`).

```python
import litellm, openai
try:
    res = litellm.completion(model=cfg.model, messages=msgs,
                             num_retries=2,
                             fallbacks=[{cfg.model: ["openai/gpt-4o-mini"]}])
except litellm.ContextWindowExceededError:
    ...  # trim history / summarize, don't retry
except (openai.RateLimitError, openai.APITimeoutError) as e:
    if litellm._should_retry(e.status_code): ...   # already retried; escalate/queue
except openai.AuthenticationError:
    raise  # fix creds, no retry
```

### 7. Gotchas (esp. local Ollama tool calling)

- **Not all Ollama models support tools.** Native tool calling needs LiteLLM ≥ 1.41.27 *and* a tool-capable model (e.g. `llama3.1`, `qwen3`, `mistral`). For others LiteLLM falls back to JSON-mode prompt injection, which is flaky.
- **`ollama_chat/` + tools has known bugs:** mixed `content`+`tool_calls` assistant turns can error (`cannot unmarshal array … of type string`), and tool results sometimes come back as raw JSON in `content` instead of `tool_calls`. **Recommended workaround for local tool-calling: use the OpenAI-compatible path** — `model="openai/<ollama-model>"` with `api_base="http://localhost:11434/v1"`. This routes through Ollama's OpenAI shim and yields correct `tool_calls`.
- **Capability detection / fallback strategy:** gate behavior with `litellm.supports_function_calling(model=...)` and `litellm.get_supported_openai_params(model=...)`. If a model lacks native tools, either (a) set `litellm.add_function_to_prompt = True` to inline tool schemas into the prompt and parse JSON from `content` yourself, or (b) route that request to OpenAI as a fallback. Build this branch into `LLMProvider` so callers stay agnostic.
- **`drop_params=True`** prevents hard failures when a local model rejects OpenAI-only params (`parallel_tool_calls`, `seed`, `logit_bias`, etc.); use `allowed_openai_params`/`additional_drop_params` for fine control.
- **Streaming usage/cost** is missing unless `stream_options={"include_usage": True}`; local models report `cost=0`.
- **`stream_chunk_builder`** drops tool-call data in several known cases — prefer the manual index accumulator for the agent loop.
- **`api_base` config bug:** for local Ollama set the `OLLAMA_API_BASE` env var in addition to passing `api_base`, since some paths ignore the param.

Sources: [docs.litellm.ai/docs](https://docs.litellm.ai/docs/), [input](https://docs.litellm.ai/docs/completion/input), [stream](https://docs.litellm.ai/docs/completion/stream), [function_call](https://docs.litellm.ai/docs/completion/function_call), [ollama](https://docs.litellm.ai/docs/providers/ollama), [openai](https://docs.litellm.ai/docs/providers/openai), [token_usage](https://docs.litellm.ai/docs/completion/token_usage), [exception_mapping](https://docs.litellm.ai/docs/exception_mapping), [routing](https://docs.litellm.ai/docs/routing), [drop_params](https://docs.litellm.ai/docs/completion/drop_params).
