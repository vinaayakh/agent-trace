# How agent-trace works

## The problem with SDK wrapping

The obvious approach to tracing LLM calls is to wrap the SDK client:

```python
# What OpenLLMetry does — a separate package per provider
from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
AnthropicInstrumentor().instrument()
```

This works, but it has a structural problem: you need one instrumentation package for every SDK (Anthropic, OpenAI, Bedrock, Ollama, …), each one patching specific internal methods that break whenever the SDK ships a new version.

agent-trace takes a different path.

---

## Transport-level interception via httpx

The OpenAI Python SDK and the Anthropic Python SDK both use [`httpx`](https://www.python-httpx.org/) as their HTTP client. So does nearly every other major Python LLM SDK. Instead of patching SDK methods, agent-trace patches the single lowest-level method that all of them share:

```
your code
  └── anthropic.AsyncAnthropic().messages.create()
        └── httpx.AsyncClient.send()   ← patched here
              └── TCP socket
```

`agent_trace/interceptor.py` replaces `httpx.Client.send` and `httpx.AsyncClient.send` with wrapper functions that:

1. Check whether the destination host + URL path match a known LLM endpoint (e.g. `api.anthropic.com/v1/messages`)
2. If yes: create an OTel span, call the original `send`, parse the response body for tokens/finish-reason, record them as span attributes, close the span
3. If no: pass through transparently

```python
# interceptor.py (simplified)
_original_async_send = httpx.AsyncClient.send

async def _patched_async_send(self, request, **kwargs):
    if not _is_llm_request(request):
        return await _original_async_send(self, request, **kwargs)

    span = _make_span(request)          # creates gen_ai.* attributes from request body
    with span:
        response = await _original_async_send(self, request, **kwargs)
        span.set_attributes(_parse_response(response))   # tokens, finish_reason
        return response

httpx.AsyncClient.send = _patched_async_send
```

This runs on `import agent_trace`. No further configuration needed.

---

## Context propagation via `contextvars`

The interceptor knows how to capture a single LLM call. But to produce a *hierarchy* (`agent → step → llm_call`), it needs to know which agent step caused the call. This is the job of `contextvars.ContextVar`.

Python's `ContextVar` is an async-safe slot: each `asyncio` task gets its own copy, and child tasks inherit the parent's value. agent-trace stores the current `TraceContext` (which holds the active OTel `Span`) in a `ContextVar`:

```python
# context.py
_current: ContextVar[Optional[TraceContext]] = ContextVar("agent_trace_ctx", default=None)
```

When you do `async with agent_trace.step("plan"):`, the library:
1. Reads the current context (the enclosing `agent` span)
2. Creates a new child `step` span with that as its parent
3. Sets `_current` to a new `TraceContext` holding the new span
4. Saves a reset token so the previous value is restored on exit

```python
# __init__.py (simplified)
@asynccontextmanager
async def step(name):
    span = start_span(f"step {name}", attributes=step_attrs(name), ...)
    token = _current.set(TraceContext(span=span, ...))
    with use_span(span, end_on_exit=True):
        yield
    _current.reset(token)          # restores parent context
```

When the interceptor fires, it calls `current_otel_context()`, which reads `_current` and returns the current span as an OTel parent context. The auto-detected LLM span is therefore automatically nested under whatever `agent/step/tool` block is active.

---

## Data flow for a single agent run

```
agent_trace.agent("ReActAgent")
│  creates span: name="agent ReActAgent"
│  sets _current = TraceContext(span=agent_span)
│
├── agent_trace.step("think")
│   │  creates span: name="step think", parent=agent_span
│   │  sets _current = TraceContext(span=step_span)
│   │
│   └── client.messages.create(...)        ← user code, any SDK
│       │
│       └── httpx.AsyncClient.send()       ← interceptor fires
│           │  reads _current → parent = step_span
│           │  creates span: name="gen_ai anthropic claude-haiku-4-5"
│           │  records: gen_ai.usage.input_tokens, output_tokens, finish_reason
│           └── [response returned to user code]
│
├── agent_trace.tool("search", input=query)
│   │  creates span: name="tool search", parent=agent_span
│   └── [synchronous tool code — no LLM call here]
│
└── agent_trace.step("think")              ← same step name, second time
    │  retry_attempt auto-incremented → agent_trace.retry.attempt=1
    └── httpx.AsyncClient.send()
```

---

## OTel span attributes

All attributes follow the [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/).

| Span | Attribute | Value |
|---|---|---|
| `agent *` | `gen_ai.operation.name` | `invoke_agent` |
| | `gen_ai.agent.name` | your agent name |
| `step *` | `gen_ai.operation.name` | `agent_step` |
| | `agent_trace.step.name` | your step name |
| | `agent_trace.retry.attempt` | 0 (first), 1, 2… |
| `tool *` | `gen_ai.operation.name` | `execute_tool` |
| | `gen_ai.tool.name` | your tool name |
| | `gen_ai.tool.call.arguments` | stringified input |
| `gen_ai *` (auto) | `gen_ai.system` | `anthropic` / `openai` / … |
| | `gen_ai.request.model` | e.g. `claude-haiku-4-5-20251001` |
| | `gen_ai.usage.input_tokens` | integer |
| | `gen_ai.usage.output_tokens` | integer |
| | `gen_ai.response.finish_reasons` | `end_turn` / `stop` / … |

---

## Module map

```
agent_trace/
  __init__.py       Public API — init(), agent(), step(), tool()
  _runtime.py       Shared span lifecycle helpers (used by API and adapters)
  context.py        ContextVar[TraceContext] — ambient span carrier
  interceptor.py    httpx monkey-patch — LLM auto-detection and span creation
  spans.py          OTel span factory + GenAI semconv attribute builders
  exporter.py       TracerProvider setup — OTLP or console
  adapters/         Optional framework adapters (LangChain, LangGraph)
```
