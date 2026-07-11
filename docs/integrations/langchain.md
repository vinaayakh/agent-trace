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

- Root chain run (no active agent-trace context yet) -> `agent <name>`
- Root chain run when an agent-trace context is *already* active (e.g. this handler is attached inside an `agent_trace.agent()` block) -> `step <name>` instead, so you never get two agent spans for one logical run
- Nested chain runs -> `step <name>`
- Tool callbacks -> `tool <name>`
- LLM calls -> `gen_ai ...` spans from the `httpx` interceptor (no duplicate callback LLM spans)

## Async correctness

`AgentTraceCallbackHandler` sets `run_inline = True` so LangChain dispatches it on the calling task instead of via an executor thread — required for correct span nesting under `ainvoke`. Parenting is also resolved explicitly from the callback run-id stack (not only from ambient `contextvars` state), so nesting stays correct even if a run's start and end callbacks happen to fire on different `asyncio` Tasks internally.
