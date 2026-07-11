# agent-trace

**SDK-agnostic OpenTelemetry tracing for AI agent reasoning chains.**

Captures the full agent reasoning chain — `agent → step → tool → LLM call` — as distributed traces compatible with Jaeger, Datadog, and Honeycomb. Works with any SDK (Anthropic, OpenAI, Groq, Ollama, …) via a single import, with no per-SDK packages and no framework lock-in.

## The problem

Debugging a multi-step agent in production is hard. When a tool call 20 steps in causes a failure at step 38, flat logs don't tell you which reasoning path led there, how many tokens were spent, or whether a retry loop silently inflated your costs.

## The approach

Instead of wrapping each SDK individually (the OpenLLMetry approach), `agent-trace` intercepts at the **`httpx` transport layer** — the HTTP client used by the OpenAI, Anthropic, and most other Python SDKs. One monkey-patch catches all providers. `contextvars.ContextVar` propagates the current span context through `asyncio` call chains so every auto-detected LLM call is parented to the right reasoning step automatically.

Streaming responses (SSE) are fully supported — the interceptor wraps the byte stream, accumulates chunks, and emits the span with token counts and completion text only after the stream closes. **Note:** OpenAI only includes usage (`gen_ai.usage.*`) in a streamed response if the caller sets `stream_options={"include_usage": True}` on the request — that's an OpenAI API behavior, not a limitation of the interceptor; without it, the token counts simply aren't in the stream to parse.

```
agent ReActAgent
  ├── step think          ← gen_ai.operation.name=agent_step
  │     └── gen_ai anthropic claude-haiku-4-5  ← auto-detected, no SDK wrapper
  ├── tool search         ← gen_ai.operation.name=execute_tool
  └── step think [retry]  ← agent_trace.retry.attempt=1
        └── gen_ai anthropic claude-haiku-4-5
```

## Quickstart

```bash
pip install -e ".[dev]"
```

### See traces in the console (zero setup)

```python
import agent_trace
agent_trace.init(exporter="console")
```

### See traces in Jaeger (no Docker)

```powershell
# Downloads and starts a single Jaeger binary
.\scripts\start_jaeger.ps1
```

Then configure your keys and run an example:

```bash
cp sample.env .env   # add your API key(s)
python examples/react_agent.py      # Anthropic
python examples/openai_agent.py     # OpenAI
python examples/groq_agent.py       # Groq (free tier)
python examples/streaming_model.py --provider anthropic  # streaming, no wrappers needed
python examples/langchain_react_agent.py   # LangChain adapter
python examples/langgraph_react_agent.py   # LangGraph adapter
```

Open [http://localhost:16686](http://localhost:16686) and select the service (e.g. `groq-agent-demo`).

### Swap backends — one line in .env

```bash
# Jaeger (default)
AGENT_TRACE_OTLP_ENDPOINT=http://localhost:4318/v1/traces

# Grafana Tempo
AGENT_TRACE_OTLP_ENDPOINT=http://<tempo-host>:4318/v1/traces
```

The trace structure is identical across all providers. The tracer has no idea which SDK was used.

## Framework integrations (optional)

The core tracer remains framework-agnostic. If you use LangChain or LangGraph, you can opt into adapters that map framework lifecycle events to `agent` / `step` / `tool` spans while the interceptor continues to own LLM `chat` spans.

Install extras:

```bash
pip install -e ".[langchain]"
pip install -e ".[langgraph]"
```

### LangChain

`trace_runnable` wraps any LangChain `Runnable` and injects `AgentTraceCallbackHandler` automatically:

```python
from agent_trace.adapters.langchain import trace_runnable, AgentTraceCallbackHandler

traced = trace_runnable(my_runnable, agent_name="MyAgent")
result = traced.invoke({"question": "..."})        # sync
result = await traced.ainvoke({"question": "..."}) # async
```

Use `AgentTraceCallbackHandler` directly when you need more control:

```python
handler = AgentTraceCallbackHandler(
    agent_name="MyAgent",
    step_on_agent_action=True,   # emit a step span on each AgentAction
    record_tool_output=False,    # set True to attach tool output to span
)
chain.invoke(input, config={"callbacks": [handler]})
```

### LangGraph

`graph_config` injects the callback handler into the graph's run config — this alone is the recommended usage, no wrapper needed:

```python
from agent_trace.adapters.langgraph import graph_config

result = app.invoke(state, config=graph_config(agent_name="MyGraph"))
result = await app.ainvoke(state, config=graph_config(agent_name="MyGraph"))  # async, same hierarchy
```

`trace_graph` is an optional sync/async context manager for grouping several `invoke`/`ainvoke` calls under one shared agent span:

```python
from agent_trace.adapters.langgraph import trace_graph, graph_config

with trace_graph("MyGraph"):
    result = app.invoke(state, config=graph_config(agent_name="MyGraph"))
```

Read the full integration guides:

- [LangChain integration](docs/integrations/langchain.md)
- [LangGraph integration](docs/integrations/langgraph.md)

## API

```python
import agent_trace

# All parameters are optional — defaults are read from environment variables:
#   AGENT_TRACE_EXPORTER         (default: "otlp")
#   AGENT_TRACE_OTLP_ENDPOINT    (default: "http://localhost:4318/v1/traces")
agent_trace.init(
    service_name="my-agent",    # shown in Jaeger/Tempo service list
    exporter="otlp",            # "otlp" | "console"
    otlp_endpoint="http://localhost:4318/v1/traces",
)

async with agent_trace.agent("MyAgent"):
    async with agent_trace.step("plan"):
        response = await client.messages.create(...)   # auto-traced

    async with agent_trace.tool("search_web", input=query):
        result = await web_search(query)
```

To remove the httpx monkey-patch (e.g. in tests):

```python
from agent_trace import interceptor
interceptor.uninstall()
```

Custom base URLs, proxies, or self-hosted gateways can opt into tracing without a code change via `AGENT_TRACE_EXTRA_HOSTS` (comma-separated hostnames, merged into the built-in host list on every request check):

```bash
export AGENT_TRACE_EXTRA_HOSTS="my-llm-gateway.internal,another-proxy.example.com"
```

## Summary exporter

Spans are great for deep inspection in Jaeger/Tempo, but sometimes you just want a results file. Attach `RunSummaryProcessor` via `init(summary=...)` to get a per-run aggregate (agent name, wall duration, LLM call count, total input/output tokens, models used, tool call count, error count, per-step retry counts) alongside the normal span export:

```python
import agent_trace

agent_trace.init(exporter="console", summary="bench/summary.json")  # or summary=True for in-memory only

async with agent_trace.agent("MyAgent"):
    ...

# Query in-process at any time:
print(agent_trace.get_summary())
# [{"agent_name": "MyAgent", "duration_seconds": 1.42, "llm_call_count": 3,
#   "input_tokens": 512, "output_tokens": 128, "models": ["gpt-4o-mini"],
#   "tool_call_count": 1, "error_count": 0, "retry_counts": {}}]
```

`summary="bench/summary.json"` also writes the JSON above to disk when the provider shuts down (the existing `atexit`-registered `TracerProvider.shutdown()` path — no extra wiring needed). Pass a `.md` path instead (e.g. `summary="bench/summary.md"`) to additionally get a human-readable markdown table as a sibling file, alongside a `.json` with the same basename.

## Threads and executors

`agent_trace` tracks the active span with a `contextvars.ContextVar`, which propagates automatically across `asyncio` `await` boundaries but **not** across real OS threads — work submitted to a `ThreadPoolExecutor` or dispatched via `loop.run_in_executor` starts with no ambient context, so an LLM call made there becomes an orphaned root trace instead of nesting under the step/tool that submitted it.

Wrap the callable with `agent_trace.bind_context` at submission time to fix this — it captures the calling context and replays it inside the thread:

```python
from concurrent.futures import ThreadPoolExecutor
import agent_trace

async with agent_trace.agent("MyAgent"):
    async with agent_trace.step("plan"):
        with ThreadPoolExecutor() as pool:
            future = pool.submit(agent_trace.bind_context(call_llm), prompt)
            result = await asyncio.wrap_future(future)
```

This is a best-effort helper for code you control at the submission point — it can't retroactively fix a raw `threading.Thread` target that was already created elsewhere. As a secondary fallback, `agent_trace` spans also nest under any *other* library's OTel span that's current in ambient OTel context (e.g. if that library propagates across threads itself), so tracing degrades gracefully rather than crashing when full propagation isn't possible.

## Span attributes (GenAI semantic conventions)

| Attribute | Value / source |
|---|---|
| `gen_ai.operation.name` | `invoke_agent` / `agent_step` / `execute_tool` / `chat` |
| `gen_ai.system` | interceptor — provider name from hostname |
| `gen_ai.request.model` | interceptor — from request body |
| `gen_ai.usage.input_tokens` | interceptor — from response body |
| `gen_ai.usage.output_tokens` | interceptor — from response body |
| `gen_ai.response.finish_reasons` | interceptor — from response body |
| `gen_ai.prompt_preview` | interceptor — first 500 chars of prompt |
| `gen_ai.completion_preview` | interceptor — first 500 chars of completion |
| `gen_ai.agent.name` | `agent_trace.agent()` |
| `agent_trace.step.name` | `agent_trace.step()` |
| `agent_trace.retry.attempt` | `agent_trace.step()` — auto-incremented per name |
| `gen_ai.tool.name` | `agent_trace.tool()` |
| `gen_ai.tool.call.arguments` | `agent_trace.tool(input=...)` |

## How it compares

| | agent-trace | OpenLLMetry | Langfuse | Helicone |
|---|---|---|---|---|
| Instrumentation layer | `httpx` transport | SDK monkey-patch | SDK wrapper | Reverse proxy |
| SDK-agnostic | ✅ one import | ❌ one package per SDK | ❌ explicit integration | ✅ |
| Agent span hierarchy | ✅ | Minimal | ✅ | ❌ |
| Self-hostable backend | ✅ (Jaeger, Grafana Tempo) | ✅ | ✅ | ❌ |
| OTel native | ✅ | ✅ | Partial | ❌ |
| Framework required | None | None | None | None |

## Supported providers

Any SDK that uses `httpx` under the hood is automatically instrumented. Tested providers:

| Provider | SDK | Model example |
|---|---|---|
| Anthropic | `anthropic` | `claude-haiku-4-5-20251001` |
| OpenAI | `openai` | `gpt-4o-mini` |
| Groq | `groq` | `llama-3.1-8b-instant` |
| Google Gemini | `google-generativeai` | `gemini-1.5-flash` |
| Ollama | direct HTTP (`/api/chat`, `/api/generate`, and the OpenAI-compatible `/v1/chat/completions`) | `llama3` |
| OpenRouter | direct HTTP / OpenAI-compatible SDKs | any OpenRouter-hosted model |
| Azure OpenAI | `openai` (Azure config) | your deployment name |

Other providers with OpenAI-compatible APIs (Together AI, Mistral, Cohere) are detected automatically — no code changes needed. Anything else reachable at a custom host can be added via `AGENT_TRACE_EXTRA_HOSTS` (see [API](#api)).

## Requirements

- Python 3.10+
- `opentelemetry-sdk >= 1.24.0`
- `opentelemetry-exporter-otlp-proto-http >= 1.24.0`
- `opentelemetry-semantic-conventions >= 0.45b0`
- `httpx >= 0.25.0` (transitive dependency of the OpenAI, Anthropic, and Groq SDKs)

All installed automatically via `pip install -e ".[dev]"`.
