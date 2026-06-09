"""OTel tracer provider setup — OTLP (Jaeger) or console."""
from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

# Module-level provider reference — allows tests to inject their own provider
# without fighting OTel's "set_tracer_provider can only be called once" guard.
_provider: TracerProvider | None = None


def get_provider() -> TracerProvider | None:
    return _provider


def set_provider(provider: TracerProvider) -> None:
    global _provider
    _provider = provider


def configure(
    service_name: str = "agent-trace",
    exporter: str = "otlp",
    otlp_endpoint: str = "http://localhost:4318/v1/traces",
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
    set_provider(provider)
    trace.set_tracer_provider(provider)
