# agent-trace

**SDK-agnostic OpenTelemetry tracing for AI agent reasoning chains.**

Captures the full agent reasoning chain — `agent → step → tool → LLM call` — as distributed traces compatible with Jaeger, Datadog, and Honeycomb. Works with any SDK (Anthropic, OpenAI, Groq, Ollama, …) via a single import, with no per-SDK packages and no framework lock-in.

## The problem

Debugging a multi-step agent in production is hard. When a tool call 20 steps in causes a failure at step 38, flat logs don't tell you which reasoning path led there, how many tokens were spent, or whether a retry loop silently inflated your costs.

## The approach

Instead of wrapping each SDK individually (the OpenLLMetry approach), `agent-trace` intercepts at the **`httpx` transport layer** — the HTTP client used by the OpenAI, Anthropic, and most other Python SDKs. One monkey-patch catches all providers. `contextvars.ContextVar` propagates the current span context through `asyncio` call chains so every auto-detected LLM call is parented to the right reasoning step automatically.

Streaming responses (SSE) are fully supported — the interceptor wraps the byte stream, accumulates chunks, and emits the span with token counts and completion text only after the stream closes.

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

`trace_graph` is a sync/async context manager for the outer agent span. `graph_config` injects the callback handler into the graph's run config:

```python
from agent_trace.adapters.langgraph import trace_graph, graph_config

with trace_graph("MyGraph"):
    result = app.invoke(state, config=graph_config(agent_name="MyGraph"))

# async variant
async with trace_graph("MyGraph"):
    result = await app.ainvoke(state, config=graph_config(agent_name="MyGraph"))
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
| Ollama | direct HTTP | `llama3` |

Other providers with OpenAI-compatible APIs (Together AI, Mistral, Cohere) are detected automatically — no code changes needed.

## Requirements

- Python 3.10+
- `opentelemetry-sdk >= 1.24.0`
- `opentelemetry-exporter-otlp-proto-http >= 1.24.0`
- `opentelemetry-semantic-conventions >= 0.45b0`
- `httpx >= 0.25.0` (transitive dependency of the OpenAI, Anthropic, and Groq SDKs)

All installed automatically via `pip install -e ".[dev]"`.
