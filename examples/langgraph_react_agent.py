"""Minimal LangGraph integration using graph_config.

graph_config() alone is the recommended usage: it attaches a callback handler
that opens the agent span itself on the first (parent_run_id=None) callback,
so no separate trace_graph() wrapper is needed. trace_graph() remains
available for grouping multiple invokes under one shared agent span.

Run:
    pip install -e ".[langgraph]"
    python examples/langgraph_react_agent.py
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

import agent_trace
from agent_trace.adapters.langgraph import graph_config


class GraphState(TypedDict):
    question: str
    answer: str


def think_node(state: GraphState) -> GraphState:
    question = state.get("question", "")
    return {"question": question, "answer": f"Echo: {question}"}


def main() -> None:
    agent_trace.init(service_name="langgraph-adapter-demo", exporter="console")

    graph = StateGraph(GraphState)
    graph.add_node("think", think_node)
    graph.set_entry_point("think")
    graph.add_edge("think", END)
    app = graph.compile()

    result = app.invoke(
        {"question": "How does the graph adapter work?", "answer": ""},
        config=graph_config(agent_name="LangGraphAgent"),
    )

    print(result["answer"])


if __name__ == "__main__":
    main()
