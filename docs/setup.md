# Setup guide

## Prerequisites

- Python 3.10 or later
- pip
- An Anthropic or OpenAI API key (only needed to run the live examples)

Check your Python version:
```powershell
python --version   # must be 3.10+
```

---

## Install

Clone the repo and install in editable mode with the development extras:

```powershell
cd N:\Github\agent-trace
pip install -e ".[dev]"
```

The `[dev]` extras install:
- `anthropic` and `openai` SDKs (needed by the examples)
- `pytest` and `pytest-asyncio` (needed by the tests)

Core library dependencies (`opentelemetry-sdk`, `httpx`, etc.) are installed automatically.

Verify the install:
```powershell
python -c "import agent_trace; print('OK')"
```

---

## Quickstart — console exporter (no backend needed)

The console exporter prints spans to stdout. It is the fastest way to see something working.

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
python examples/react_agent.py
```

You will see output like this interleaved with the agent's answer:

```
{
    "name": "gen_ai anthropic claude-haiku-4-5-20251001",
    "context": { "trace_id": "0x...", "span_id": "0x..." },
    "parent_id": "0x...",
    "attributes": {
        "gen_ai.system": "anthropic",
        "gen_ai.request.model": "claude-haiku-4-5-20251001",
        "gen_ai.usage.input_tokens": 312,
        "gen_ai.usage.output_tokens": 87,
        "gen_ai.response.finish_reasons": "end_turn"
    },
    ...
}
```

To use the OpenAI example instead:

```powershell
$env:OPENAI_API_KEY = "sk-..."
python examples/openai_agent.py
```

The trace structure is identical — same span names, same attributes. The tracer has no knowledge of which SDK was used.

---

## Quickstart — Jaeger UI (no Docker)

Jaeger provides a visual trace timeline. agent-trace ships a PowerShell script that downloads and runs the Jaeger all-in-one binary for Windows with no Docker required.

**Step 1 — Start Jaeger**

```powershell
.\scripts\start_jaeger.ps1
```

On first run this downloads `jaeger-all-in-one-windows-amd64.exe` (~80 MB) into the `scripts/` directory. Subsequent runs start immediately from the cached binary.

Jaeger is ready when you see:
```
{"level":"info","msg":"Starting GRPC server","port":14250}
{"level":"info","msg":"Starting HTTP server","port":16686}
```

**Step 2 — Run an example with OTLP exporter**

Edit either example to change the exporter from `"console"` to `"otlp"`:

```python
# examples/react_agent.py  (bottom of file)
agent_trace.init(exporter="otlp")   # was: exporter="console"
```

Then run:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
python examples/react_agent.py
```

**Step 3 — Open the Jaeger UI**

Navigate to [http://localhost:16686](http://localhost:16686).

1. Select **Service → agent-trace**
2. Click **Find Traces**
3. Click any trace to expand the span tree

You should see a tree like:
```
agent ReActAgent           450ms
  step think               210ms
    gen_ai anthropic ...   200ms
  tool search               5ms
  step think [retry=1]     235ms
    gen_ai anthropic ...   225ms
```

---

## Pointing at a different OTLP backend

Any OTLP/HTTP-compatible backend works. Pass the endpoint to `init()`:

```python
# Honeycomb
agent_trace.init(
    exporter="otlp",
    otlp_endpoint="https://api.honeycomb.io/v1/traces",
)

# Grafana Cloud
agent_trace.init(
    exporter="otlp",
    otlp_endpoint="https://otlp-gateway-prod-us-east-0.grafana.net/otlp/v1/traces",
)
```

Add provider-specific auth headers by setting the `OTEL_EXPORTER_OTLP_HEADERS` environment variable:

```powershell
$env:OTEL_EXPORTER_OTLP_HEADERS = "x-honeycomb-team=YOUR_API_KEY"
```
