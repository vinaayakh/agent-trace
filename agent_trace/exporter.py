"""OTel tracer provider setup — OTLP (Jaeger) or console."""
from __future__ import annotations

from typing import Optional, Union

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from .summary import RunSummaryProcessor

# Module-level provider reference — allows tests to inject their own provider
# without fighting OTel's "set_tracer_provider can only be called once" guard.
_provider: TracerProvider | None = None

# The summary processor attached by the most recent configure() call, if any —
# lets agent_trace.get_summary() reach it without threading it through init().
_summary_processor: Optional[RunSummaryProcessor] = None


def get_provider() -> TracerProvider | None:
    return _provider


def set_provider(provider: TracerProvider) -> None:
    global _provider
    _provider = provider


def get_summary_processor() -> Optional[RunSummaryProcessor]:
    return _summary_processor


def set_summary_processor(processor: Optional[RunSummaryProcessor]) -> None:
    global _summary_processor
    _summary_processor = processor


def configure(
    service_name: str = "agent-trace",
    exporter: str = "otlp",
    otlp_endpoint: str = "http://localhost:4318/v1/traces",
    summary: Union[bool, str, None] = None,
) -> None:
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if exporter == "console":
        span_exporter = ConsoleSpanExporter()
    else:
        # Lazy import so otlp dep is only required when actually used
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        span_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)

    provider.add_span_processor(BatchSpanProcessor(span_exporter))

    if summary:
        path = summary if isinstance(summary, str) else None
        processor = RunSummaryProcessor(path=path)
        provider.add_span_processor(processor)
        set_summary_processor(processor)
    else:
        set_summary_processor(None)

    set_provider(provider)
    trace.set_tracer_provider(provider)
