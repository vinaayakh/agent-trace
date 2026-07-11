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

import contextvars
import functools
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable, TypeVar, Union

from ._runtime import enter_agent, enter_step, enter_tool, exit_frame
from .exporter import configure, get_summary_processor

# Activating the interceptor is a side-effect of importing it.
from . import interceptor as _interceptor  # noqa: F401


def init(
    service_name: str = "agent-trace",
    exporter: str | None = None,
    otlp_endpoint: str | None = None,
    summary: Union[bool, str, None] = None,
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
        summary: Attach a `RunSummaryProcessor` alongside the span exporter.
                 `True` keeps aggregated per-run summaries in memory only
                 (query with `agent_trace.get_summary()`). A path string
                 (e.g. "bench/summary.json") additionally writes the summary
                 to disk when the provider shuts down — pass a ".md" path to
                 also get a markdown table alongside the JSON.
    """
    if exporter is None:
        exporter = os.environ.get("AGENT_TRACE_EXPORTER", "otlp")
    if otlp_endpoint is None:
        otlp_endpoint = os.environ.get(
            "AGENT_TRACE_OTLP_ENDPOINT", "http://localhost:4318/v1/traces"
        )
    configure(
        service_name=service_name,
        exporter=exporter,
        otlp_endpoint=otlp_endpoint,
        summary=summary,
    )


def get_summary() -> list[dict[str, Any]]:
    """Return aggregated per-run summaries collected so far.

    Requires `agent_trace.init(summary=True)` (or a path) to have been called.
    Returns an empty list if no summary processor is attached.
    """
    processor = get_summary_processor()
    if processor is None:
        return []
    return processor.get_summary()


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


# ── Thread/executor helper ───────────────────────────────────────────────────

_T = TypeVar("_T")


def bind_context(fn: Callable[..., _T]) -> Callable[..., _T]:
    """Wrap `fn` so it runs inside the calling code's current context.

    ContextVars (what agent_trace uses to track the active span) don't cross
    real OS thread boundaries — code submitted to a `ThreadPoolExecutor` or
    dispatched via `loop.run_in_executor` runs in a fresh thread with no
    ambient agent_trace context, so any LLM call made there becomes an
    orphaned root trace instead of nesting under the step/tool that
    submitted it. Wrap the callable at submission time to fix that:

        with ThreadPoolExecutor() as pool:
            future = pool.submit(agent_trace.bind_context(call_llm), prompt)

    This can't make raw `threading.Thread` targets or pre-existing callables
    retroactively inherit context — it only helps for work wrapped before
    being handed off.
    """
    ctx = contextvars.copy_context()

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> _T:
        return ctx.run(fn, *args, **kwargs)

    return wrapper


# Expose for convenience
__all__ = ["init", "agent", "step", "tool", "bind_context", "get_summary"]
