"""OTel span helpers following the GenAI semantic conventions.

Attribute names: https://opentelemetry.io/docs/specs/semconv/gen-ai/
"""
from __future__ import annotations

from typing import Any, Optional

from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind

from .context import current_otel_context

_tracer_name = "agent_trace"


def get_tracer() -> trace.Tracer:
    # Use the explicitly configured provider when available (supports test injection).
    # Falls back to the OTel global so that bare `import agent_trace` still works
    # without calling init() first.
    from .exporter import get_provider
    provider = get_provider()
    if provider is not None:
        return provider.get_tracer(_tracer_name)
    return trace.get_tracer(_tracer_name)


def start_span(
    name: str,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[dict[str, Any]] = None,
    parent: Optional[Span] = None,
) -> Span:
    tracer = get_tracer()
    if parent is not None:
        parent_ctx = trace.set_span_in_context(parent)
    else:
        parent_ctx = current_otel_context()
        if parent_ctx is None:
            current_span = trace.get_current_span()
            if current_span is not None and current_span.is_recording():
                parent_ctx = trace.set_span_in_context(current_span)
    kwargs: dict[str, Any] = {"kind": kind}
    if attributes:
        kwargs["attributes"] = attributes
    if parent_ctx is not None:
        kwargs["context"] = parent_ctx
    return tracer.start_span(name, **kwargs)


# --- GenAI semconv attribute builders ---

def agent_attrs(name: str) -> dict[str, Any]:
    return {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.agent.name": name,
    }


def step_attrs(name: str, retry_attempt: int = 0) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "gen_ai.operation.name": "agent_step",
        "agent_trace.step.name": name,
    }
    if retry_attempt > 0:
        attrs["agent_trace.retry.attempt"] = retry_attempt
    return attrs


def tool_attrs(name: str, input_value: Any = None) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": name,
    }
    if input_value is not None:
        attrs["gen_ai.tool.call.arguments"] = str(input_value)
    return attrs


def llm_request_attrs(provider: str, model: str, prompt_preview: str = "") -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "gen_ai.operation.name": "chat",
        "gen_ai.system": provider,
        "gen_ai.request.model": model,
    }
    if prompt_preview:
        attrs["gen_ai.prompt_preview"] = prompt_preview[:500]
    return attrs


def llm_response_attrs(
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    finish_reason: Optional[str],
    completion_preview: str = "",
) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    if input_tokens is not None:
        attrs["gen_ai.usage.input_tokens"] = input_tokens
    if output_tokens is not None:
        attrs["gen_ai.usage.output_tokens"] = output_tokens
    if finish_reason is not None:
        attrs["gen_ai.response.finish_reasons"] = finish_reason
    if completion_preview:
        attrs["gen_ai.completion_preview"] = completion_preview[:500]
    return attrs
