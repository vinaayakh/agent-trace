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

import asyncio
import os
from contextlib import asynccontextmanager, contextmanager
from contextvars import copy_context
from typing import Any, AsyncGenerator, Generator, Optional

from opentelemetry.trace import StatusCode, use_span

from .context import TraceContext, _current, get_context
from .exporter import configure
from .spans import agent_attrs, llm_request_attrs, start_span, step_attrs, tool_attrs

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
    span = start_span(f"agent {name}", attributes=agent_attrs(name))
    ctx = TraceContext(span=span)
    token = _current.set(ctx)
    with use_span(span, end_on_exit=True, record_exception=True):
        try:
            yield
        finally:
            _current.reset(token)


@asynccontextmanager
async def step(name: str) -> AsyncGenerator[None, None]:
    """A reasoning step within an agent run. Automatically tracks retry attempts."""
    ctx = get_context()
    retry_attempt = 0
    if ctx is not None:
        count = ctx.step_counts.get(name, 0)
        ctx.step_counts[name] = count + 1
        retry_attempt = count  # 0 = first attempt, 1+ = retries

    span = start_span(f"step {name}", attributes=step_attrs(name, retry_attempt))
    token = _current.set(TraceContext(span=span, step_counts=ctx.step_counts if ctx else {}))
    with use_span(span, end_on_exit=True, record_exception=True):
        try:
            yield
        finally:
            _current.reset(token)


@asynccontextmanager
async def tool(name: str, input: Any = None) -> AsyncGenerator[None, None]:
    """A tool call within a step."""
    span = start_span(f"tool {name}", attributes=tool_attrs(name, input))
    ctx = get_context()
    token = _current.set(TraceContext(span=span, step_counts=ctx.step_counts if ctx else {}))
    with use_span(span, end_on_exit=True, record_exception=True):
        try:
            yield
        finally:
            _current.reset(token)


# Expose for convenience
__all__ = ["init", "agent", "step", "tool"]
