# LangGraph integration

LangGraph support is built on top of the LangChain callback adapter plus a graph-level context manager.

## Install

```bash
pip install -e ".[langgraph]"
```

## Usage

```python
import agent_trace
from agent_trace.adapters.langgraph import trace_graph, graph_config

agent_trace.init(service_name="langgraph-demo", exporter="console")

with trace_graph("LangGraphAgent"):
    result = graph.invoke(initial_state, config=graph_config(agent_name="LangGraphAgent"))
```

## Span mapping

- Graph invocation boundary -> `agent <name>`
- Node/chain callbacks -> `step <name>`
- Tool callbacks -> `tool <name>`
- LLM calls -> `gen_ai ...` spans from `httpx` interception
