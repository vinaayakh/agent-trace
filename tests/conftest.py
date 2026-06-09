"""Shared test fixtures: in-memory OTel exporter and span helpers."""
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import agent_trace.exporter as _exporter


@pytest.fixture
def span_exporter():
    """Fresh in-memory exporter injected into agent_trace for each test.

    Bypasses trace.set_tracer_provider() (which OTel only honours once)
    by writing directly to the module-level _provider slot in exporter.py.
    """
    mem_exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(mem_exporter))

    previous = _exporter.get_provider()
    _exporter.set_provider(provider)

    yield mem_exporter

    _exporter.set_provider(previous)
    mem_exporter.clear()


def get_span(exporter: InMemorySpanExporter, name: str):
    """Return the first finished span whose name equals `name`, or None."""
    return next((s for s in exporter.get_finished_spans() if s.name == name), None)


def span_attr(exporter: InMemorySpanExporter, span_name: str, attr: str):
    """Return a span attribute value, or None if span/attr not found."""
    span = get_span(exporter, span_name)
    if span is None:
        return None
    return span.attributes.get(attr)
