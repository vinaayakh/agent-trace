"""Tests for ContextVar trace context propagation."""
import asyncio
import pytest
from agent_trace.context import (
    TraceContext,
    _current,
    get_context,
    get_current_span,
)
from unittest.mock import MagicMock


def make_span(name="test"):
    span = MagicMock()
    span.name = name
    return span


def test_get_context_default_is_none():
    _current.set(None)
    assert get_context() is None
    assert get_current_span() is None


def test_set_and_get_context():
    span = make_span("root")
    ctx = TraceContext(span=span)
    token = _current.set(ctx)
    try:
        assert get_current_span() is span
        assert get_context() is ctx
    finally:
        _current.reset(token)


def test_context_restored_after_reset():
    outer_span = make_span("outer")
    outer_ctx = TraceContext(span=outer_span)
    token = _current.set(outer_ctx)

    inner_span = make_span("inner")
    inner_ctx = TraceContext(span=inner_span)
    inner_token = _current.set(inner_ctx)
    assert get_current_span() is inner_span

    _current.reset(inner_token)
    assert get_current_span() is outer_span

    _current.reset(token)
    assert get_current_span() is None


@pytest.mark.asyncio
async def test_context_propagates_across_await():
    span = make_span("async-root")
    ctx = TraceContext(span=span)
    token = _current.set(ctx)

    async def nested():
        # ContextVar value is visible in child coroutines
        return get_current_span()

    result = await nested()
    assert result is span
    _current.reset(token)


@pytest.mark.asyncio
async def test_concurrent_tasks_have_independent_contexts():
    results = {}

    async def task(name, span):
        token = _current.set(TraceContext(span=span))
        await asyncio.sleep(0)  # yield to event loop
        results[name] = get_current_span()
        _current.reset(token)

    span_a = make_span("a")
    span_b = make_span("b")

    await asyncio.gather(task("a", span_a), task("b", span_b))

    assert results["a"] is span_a
    assert results["b"] is span_b


def test_step_counts_track_retries():
    span = make_span()
    ctx = TraceContext(span=span)
    assert ctx.step_counts == {}

    ctx.step_counts["plan"] = 1
    assert ctx.step_counts["plan"] == 1
