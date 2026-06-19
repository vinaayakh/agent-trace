# LangChain integration

`agent-trace` can instrument LangChain runs through a callback handler while still using transport-level interception for model calls.

## Install

```bash
pip install -e ".[langchain]"
```

## Usage

```python
import agent_trace
from agent_trace.adapters.langchain import AgentTraceCallbackHandler, trace_runnable

agent_trace.init(service_name="langchain-demo", exporter="console")

handler = AgentTraceCallbackHandler(agent_name="LangChainAgent")
traced = trace_runnable(runnable, handler=handler)
result = traced.invoke({"input": "hello"})
```

## Span mapping

- Root chain run -> `agent <name>`
- Nested chain runs -> `step <name>`
- Tool callbacks -> `tool <name>`
- LLM calls -> `gen_ai ...` spans from the `httpx` interceptor (no duplicate callback LLM spans)
