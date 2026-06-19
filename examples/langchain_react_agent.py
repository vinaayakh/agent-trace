"""Minimal LangChain integration using AgentTraceCallbackHandler.

Run:
    pip install -e ".[langchain]"
    python examples/langchain_react_agent.py
"""
from __future__ import annotations

from langchain_core.runnables import RunnableLambda

import agent_trace
from agent_trace.adapters.langchain import trace_runnable


def think_step(payload: dict) -> dict:
    question = payload.get("question", "")
    return {"answer": f"Echo: {question}"}


def main() -> None:
    agent_trace.init(service_name="langchain-adapter-demo", exporter="console")

    runnable = RunnableLambda(think_step)
    traced = trace_runnable(runnable, agent_name="LangChainAgent")

    result = traced.invoke({"question": "What is agent-trace?"})
    print(result["answer"])


if __name__ == "__main__":
    main()
