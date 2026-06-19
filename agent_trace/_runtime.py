"""Internal span lifecycle helpers shared by APIs and adapters."""
from __future__ import annotations

from contextvars import Token
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Optional

from opentelemetry.trace import use_span

from .context import TraceContext, _current, get_context
from .spans import agent_attrs, start_span, step_attrs, tool_attrs


@dataclass
class SpanFrame:
    """Active span frame with context reset token and OTel scope."""

    token: Token
    scope: Any


def enter_agent(name: str) -> SpanFrame:
    span = start_span(f"agent {name}", attributes=agent_attrs(name))
    token = _current.set(TraceContext(span=span))
    scope = use_span(span, end_on_exit=True, record_exception=True)
    scope.__enter__()
    return SpanFrame(token=token, scope=scope)


def enter_step(name: str) -> SpanFrame:
    ctx = get_context()
    retry_attempt = 0
    if ctx is not None:
        count = ctx.step_counts.get(name, 0)
        ctx.step_counts[name] = count + 1
        retry_attempt = count

    span = start_span(f"step {name}", attributes=step_attrs(name, retry_attempt))
    token = _current.set(TraceContext(span=span, step_counts=ctx.step_counts if ctx else {}))
    scope = use_span(span, end_on_exit=True, record_exception=True)
    scope.__enter__()
    return SpanFrame(token=token, scope=scope)


def enter_tool(name: str, input_value: Any = None) -> SpanFrame:
    span = start_span(f"tool {name}", attributes=tool_attrs(name, input_value))
    ctx = get_context()
    token = _current.set(TraceContext(span=span, step_counts=ctx.step_counts if ctx else {}))
    scope = use_span(span, end_on_exit=True, record_exception=True)
    scope.__enter__()
    return SpanFrame(token=token, scope=scope)


def exit_frame(
    frame: SpanFrame,
    exc_type: Optional[type[BaseException]] = None,
    exc_value: Optional[BaseException] = None,
    exc_tb: Optional[TracebackType] = None,
) -> None:
    try:
        frame.scope.__exit__(exc_type, exc_value, exc_tb)
    finally:
        _current.reset(frame.token)
