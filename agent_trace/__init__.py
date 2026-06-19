"""agent-trace: SDK-agnostic OpenTelemetry tracing for AI agent reasoning chains.

Usage:
    import agent_trace

    agent_trace.init()  # configure OTel exporter once at startup

    async with agent_trace.agent("MyAgent"):
        async with agent_trace.step("plan"):
            result = await llm_client.messages.create(...)   # auto-traced
        async with agent_trace.tool("search_web", input=query):
            data = await search(query)
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from ._runtime import enter_agent, enter_step, enter_tool, exit_frame
from .exporter import configure

# Activating the interceptor is a side-effect of importing it.
from . import interceptor as _interceptor  # noqa: F401


def init(
    service_name: str = "agent-trace",
    exporter: str | None = None,
    otlp_endpoint: str | None = None,
) -> None:
    """Configure the OTel tracer provider.

    Call once at application startup before running any agents.

    Args:
        service_name: Appears as the service name in your trace backend.
        exporter: "otlp" (Jaeger/Tempo/any OTLP backend) or "console" (stdout).
                  Defaults to AGENT_TRACE_EXPORTER env var, then "otlp".
        otlp_endpoint: OTLP/HTTP endpoint.
                       Defaults to AGENT_TRACE_OTLP_ENDPOINT env var, then
                       http://localhost:4318/v1/traces.
    """
    if exporter is None:
        exporter = os.environ.get("AGENT_TRACE_EXPORTER", "otlp")
    if otlp_endpoint is None:
        otlp_endpoint = os.environ.get(
            "AGENT_TRACE_OTLP_ENDPOINT", "http://localhost:4318/v1/traces"
        )
    configure(service_name=service_name, exporter=exporter, otlp_endpoint=otlp_endpoint)


# ── Context managers ───────────────────────────────────────────────────────────

@asynccontextmanager
async def agent(name: str) -> AsyncGenerator[None, None]:
    """Top-level span for a complete agent run."""
    frame = enter_agent(name)
    try:
        yield
    except Exception as exc:
        exit_frame(frame, type(exc), exc, exc.__traceback__)
        raise
    else:
        exit_frame(frame)


@asynccontextmanager
async def step(name: str) -> AsyncGenerator[None, None]:
    """A reasoning step within an agent run. Automatically tracks retry attempts."""
    frame = enter_step(name)
    try:
        yield
    except Exception as exc:
        exit_frame(frame, type(exc), exc, exc.__traceback__)
        raise
    else:
        exit_frame(frame)


@asynccontextmanager
async def tool(name: str, input: Any = None) -> AsyncGenerator[None, None]:
    """A tool call within a step."""
    frame = enter_tool(name, input_value=input)
    try:
        yield
    except Exception as exc:
        exit_frame(frame, type(exc), exc, exc.__traceback__)
        raise
    else:
        exit_frame(frame)


# Expose for convenience
__all__ = ["init", "agent", "step", "tool"]
