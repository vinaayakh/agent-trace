"""Optional framework adapters for LangChain and LangGraph."""

from __future__ import annotations

__all__ = [
    "AgentTraceCallbackHandler",
    "trace_runnable",
    "trace_graph",
]


def __getattr__(name: str):
    if name in {"AgentTraceCallbackHandler", "trace_runnable"}:
        from .langchain import AgentTraceCallbackHandler, trace_runnable

        exports = {
            "AgentTraceCallbackHandler": AgentTraceCallbackHandler,
            "trace_runnable": trace_runnable,
        }
        return exports[name]

    if name == "trace_graph":
        from .langgraph import trace_graph

        return trace_graph

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
