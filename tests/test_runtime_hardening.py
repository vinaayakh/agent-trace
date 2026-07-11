"""Phase 1 gate: explicit parenting, OTel-current-span fallback, tolerant exit_frame."""
import contextvars

from agent_trace._runtime import enter_agent, enter_step, exit_frame
from agent_trace.context import _current
from agent_trace.spans import start_span
from tests.conftest import get_span


def test_explicit_parent_wins_over_ambient_context(span_exporter):
    outer_frame = enter_agent("Outer")

    # Start a standalone span to use as an explicit parent, distinct from ambient.
    detached_parent_span = start_span("agent Detached")

    step_frame = enter_step("plan", parent=detached_parent_span)
    exit_frame(step_frame)
    detached_parent_span.end()
    exit_frame(outer_frame)

    detached_span = get_span(span_exporter, "agent Detached")
    step_span = get_span(span_exporter, "step plan")
    outer_span = get_span(span_exporter, "agent Outer")

    assert step_span.parent.span_id == detached_span.context.span_id
    assert step_span.parent.span_id != outer_span.context.span_id


def test_otel_current_span_fallback_when_ambient_empty(span_exporter):
    from opentelemetry.trace import use_span

    foreign_span = start_span("foreign instrumentation span")
    with use_span(foreign_span, end_on_exit=True):
        # No agent_trace ambient context is active here, but an OTel span is
        # current — agent-trace should nest under it.
        frame = enter_step("nested-under-foreign")
        exit_frame(frame)

    foreign = get_span(span_exporter, "foreign instrumentation span")
    nested = get_span(span_exporter, "step nested-under-foreign")
    assert nested.parent.span_id == foreign.context.span_id


def test_exit_frame_tolerant_of_cross_context_reset(span_exporter):
    frame = enter_agent("CrossContext")

    def reset_in_other_context():
        # Runs in a copied context — resetting frame.token here raises ValueError
        # in vanilla contextvars usage. exit_frame must swallow it.
        exit_frame(frame)

    ctx = contextvars.copy_context()
    ctx.run(reset_in_other_context)

    span = get_span(span_exporter, "agent CrossContext")
    assert span is not None
    assert span.end_time is not None

    # The copy's reset attempt can't touch *this* (the real/root) context's
    # `_current` mapping — clean it up directly so this test, which
    # deliberately provokes the cross-context failure, doesn't leak ambient
    # state into whatever test runs next in the same process.
    _current.reset(frame.token)
