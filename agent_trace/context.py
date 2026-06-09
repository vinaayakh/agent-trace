"""Ambient trace context carried across asyncio boundaries via ContextVar."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional

from opentelemetry import trace
from opentelemetry.trace import Span


@dataclass
class TraceContext:
    span: Span
    # Step name → how many times it has been entered in this agent run (for retry detection)
    step_counts: dict[str, int] = field(default_factory=dict)


_current: ContextVar[Optional[TraceContext]] = ContextVar("agent_trace_ctx", default=None)


def get_context() -> Optional[TraceContext]:
    return _current.get()


def set_context(ctx: Optional[TraceContext]):
    _current.set(ctx)


def get_current_span() -> Optional[Span]:
    ctx = _current.get()
    return ctx.span if ctx else None


def current_otel_context():
    """Return an OTel Context containing the current span, for use as a parent."""
    span = get_current_span()
    if span is None:
        return None
    return trace.set_span_in_context(span)
