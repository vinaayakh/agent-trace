# LangGraph integration

LangGraph support is built on top of the LangChain callback adapter.

## Install

```bash
pip install -e ".[langgraph]"
```

## Usage

`graph_config()` alone is the recommended usage — no separate wrapper needed:

```python
import agent_trace
from agent_trace.adapters.langgraph import graph_config

agent_trace.init(service_name="langgraph-demo", exporter="console")

result = graph.invoke(initial_state, config=graph_config(agent_name="LangGraphAgent"))
```

The attached callback handler opens the `agent <name>` span itself on the graph's
root callback. Both `invoke` and `ainvoke` produce the same span hierarchy.

### Grouping multiple invokes under one agent span

`trace_graph()` is optional, for when you want several `invoke`/`ainvoke` calls
(e.g. a multi-turn loop) to share a single agent span instead of getting one
agent span per call:

```python
from agent_trace.adapters.langgraph import trace_graph, graph_config

with trace_graph("LangGraphAgent"):
    result = graph.invoke(initial_state, config=graph_config(agent_name="LangGraphAgent"))
    result = graph.invoke(next_state, config=graph_config(agent_name="LangGraphAgent"))
```

When an agent-trace context is already active (as `trace_graph` establishes),
the handler detects it and enters a `step` for the graph root instead of a
second `agent` span, so combining both never produces duplicate agent spans.

## Span mapping

- Graph invocation boundary -> `agent <name>` (or `step <name>` if nested under an existing `trace_graph`/`agent_trace.agent()` context)
- Node/chain callbacks -> `step <name>`
- Tool callbacks -> `tool <name>`
- LLM calls -> `gen_ai ...` spans from `httpx` interception

## Threads and async execution

- `AgentTraceCallbackHandler` sets `run_inline = True` so LangChain dispatches
  it on the calling task instead of via an executor thread, which keeps
  ContextVar-based nesting correct under `ainvoke`.
- Span parenting is additionally resolved explicitly from the callback
  run-id stack (not only from ambient context), so nesting stays correct
  even when LangGraph schedules a node's callbacks on a different `asyncio`
  Task than the one that started it.
- For code that crosses real OS threads (`ThreadPoolExecutor`, etc.), see
  [Threads and executors](../../README.md#threads-and-executors) in the README.
