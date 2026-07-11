# Testing guide

## Running the tests

```powershell
cd N:\Github\agent-trace
pip install -e ".[dev]"
pytest
```

All tests run without any API keys or a running backend — they use in-process OTel spans and mock httpx responses.

Run with verbose output to see each test name:

```powershell
pytest -v
```

---

## Test structure

```
tests/
  test_interceptor.py           httpx monkey-patch: LLM calls traced (OpenAI, Anthropic, Gemini,
                                 Ollama native, Azure, OpenRouter, AGENT_TRACE_EXTRA_HOSTS),
                                 non-LLM calls pass through, streaming SSE parsing
  test_context.py               ContextVar propagation across asyncio tasks
  test_spans.py                 GenAI semconv attribute builders
  test_api.py                   Public API: agent/step/tool context managers, retry detection
  test_runtime_hardening.py     Explicit span parenting, OTel-current-span fallback,
                                 tolerant exit_frame across contextvars contexts
  test_threading.py             agent_trace.bind_context for ThreadPoolExecutor/run_in_executor
  test_summary.py               RunSummaryProcessor aggregation + JSON/markdown file output
  adapters/
    test_stack.py                Run-id keyed span-frame stack (RunStack)
    test_langchain_handler.py    AgentTraceCallbackHandler span hierarchy
    test_langgraph.py            trace_graph()/graph_config() basics
    test_realworld_execution.py  Real langgraph graphs: sync/async/combined/concurrent execution
```

---

## How the tests avoid real API calls

Tests use a custom `httpx.Transport` that returns canned responses instead of making network requests. This lets us exercise the full interception path — request body parsing, response body parsing, span attribute extraction — without hitting any real API.

```python
# Pattern used across tests
import httpx
from unittest.mock import MagicMock

def fake_response(status=200, body=None):
    return httpx.Response(status, json=body or {})

# Mount it on a client
transport = httpx.MockTransport(handler=lambda r: fake_response(...))
client = httpx.Client(transport=transport)
```

The interceptor fires on `client.send()` regardless of the transport — so mock transports exercise the real code path.

---

## Checking what spans were emitted

Tests use an in-memory `SpanExporter` that collects all finished spans:

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

exporter = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(provider)

# ... run code ...

spans = exporter.get_finished_spans()
assert any(s.name == "gen_ai anthropic claude-haiku" for s in spans)
```

---

## Manual smoke test (end-to-end with a real API key)

After running the unit tests, do a quick manual check with a real key to confirm nothing is broken end-to-end:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
python examples/react_agent.py
```

Expected: the agent prints a final answer and span JSON is written to stdout (with `exporter="console"`).

To test the Jaeger path, start Jaeger first:

```powershell
.\scripts\start_jaeger.ps1
# (in a second terminal)
python examples/react_agent.py    # after changing exporter="otlp" in the example
```

Then verify the trace appears in the Jaeger UI at [http://localhost:16686](http://localhost:16686).
