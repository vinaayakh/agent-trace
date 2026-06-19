"""LangGraph adapter helpers for agent_trace."""
from __future__ import annotations

from typing import Any, Optional

from agent_trace._runtime import SpanFrame, enter_agent, exit_frame

from .langchain import AgentTraceCallbackHandler, _merge_callbacks


class _GraphTraceContext:
    def __init__(self, name: str) -> None:
        self._name = name
        self._frame: Optional[SpanFrame] = None

    def __enter__(self) -> "_GraphTraceContext":
        self._frame = enter_agent(self._name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._frame is None:
            return
        exit_frame(self._frame, exc_type, exc_val, exc_tb)
        self._frame = None

    async def __aenter__(self) -> "_GraphTraceContext":
        self._frame = enter_agent(self._name)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._frame is None:
            return
        exit_frame(self._frame, exc_type, exc_val, exc_tb)
        self._frame = None


def trace_graph(name: str = "LangGraphAgent") -> _GraphTraceContext:
    """Create a sync/async context manager for a LangGraph run."""
    return _GraphTraceContext(name)


def graph_config(
    config: Optional[dict[str, Any]] = None,
    *,
    agent_name: str = "LangGraphAgent",
    handler: Optional[AgentTraceCallbackHandler] = None,
) -> dict[str, Any]:
    """Return config with AgentTraceCallbackHandler attached."""
    resolved_handler = handler or AgentTraceCallbackHandler(agent_name=agent_name)
    return _merge_callbacks(config, resolved_handler)
