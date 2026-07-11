"""Internal span lifecycle helpers shared by APIs and adapters."""
from __future__ import annotations

from contextvars import Token
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Optional

from opentelemetry.trace import Span, StatusCode

from .context import TraceContext, _current, get_context
from .spans import agent_attrs, start_span, step_attrs, tool_attrs


@dataclass
class SpanFrame:
    """Active span frame with context reset token and the underlying span.

    Deliberately does not use `opentelemetry.trace.use_span` to manage the
    span lifecycle: that helper attaches/detaches a token in OTel's own
    ambient Context, which — like our `_current` ContextVar — does not
    survive being set in one asyncio Task and torn down in another (e.g.
    LangGraph schedules each node, and the callbacks around it, as separate
    Tasks). We already thread parenting explicitly (see `parent=` in
    `start_span`), so we don't need OTel's ambient context for nesting and
    can end spans/record exceptions directly instead.
    """

    token: Token
    span: Span


def enter_agent(name: str, parent: Optional[Span] = None) -> SpanFrame:
    span = start_span(f"agent {name}", attributes=agent_attrs(name), parent=parent)
    token = _current.set(TraceContext(span=span))
    return SpanFrame(token=token, span=span)


def enter_step(name: str, parent: Optional[Span] = None) -> SpanFrame:
    ctx = get_context()
    retry_attempt = 0
    if ctx is not None:
        count = ctx.step_counts.get(name, 0)
        ctx.step_counts[name] = count + 1
        retry_attempt = count

    span = start_span(f"step {name}", attributes=step_attrs(name, retry_attempt), parent=parent)
    token = _current.set(TraceContext(span=span, step_counts=ctx.step_counts if ctx else {}))
    return SpanFrame(token=token, span=span)


def enter_tool(name: str, input_value: Any = None, parent: Optional[Span] = None) -> SpanFrame:
    span = start_span(f"tool {name}", attributes=tool_attrs(name, input_value), parent=parent)
    ctx = get_context()
    token = _current.set(TraceContext(span=span, step_counts=ctx.step_counts if ctx else {}))
    return SpanFrame(token=token, span=span)


def exit_frame(
    frame: SpanFrame,
    exc_type: Optional[type[BaseException]] = None,
    exc_value: Optional[BaseException] = None,
    exc_tb: Optional[TracebackType] = None,
) -> None:
    try:
        if exc_value is not None:
            frame.span.record_exception(exc_value)
            frame.span.set_status(StatusCode.ERROR, str(exc_value))
        frame.span.end()
    finally:
        try:
            _current.reset(frame.token)
        except ValueError:
            # Cross-context reset: the callback fired from a different
            # contextvars context (e.g. an executor thread or a LangGraph
            # node Task). The span is already ended above; nothing else to do.
            pass
