"""Aggregated per-run summary exporter.

Spans go to Jaeger/console for deep inspection, but there was no path from
traces to a results file — `RunSummaryProcessor` closes that gap by grouping
finished spans by trace_id and deriving per-run stats (token totals, call
counts, error counts) without needing a query against a tracing backend.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.trace import StatusCode


class RunSummaryProcessor(SpanProcessor):
    """SpanProcessor that aggregates finished spans into a per-run summary.

    Attach via `agent_trace.init(summary=True)` (in-memory only, query with
    `agent_trace.get_summary()`) or `agent_trace.init(summary="out.json")`
    to also write the summary to disk when the provider shuts down.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = path
        self._traces: dict[int, list[ReadableSpan]] = defaultdict(list)

    def on_start(self, span: ReadableSpan, parent_context: Any = None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        self._traces[span.context.trace_id].append(span)

    def shutdown(self) -> None:
        if self._path:
            self.write(self._path)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def get_summary(self) -> list[dict[str, Any]]:
        return [_summarize_trace(trace_id, spans) for trace_id, spans in self._traces.items()]

    def write(self, path: str) -> None:
        """Write the summary as JSON; also write a markdown table if `path` ends in `.md`."""
        summary = self.get_summary()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        if target.suffix == ".md":
            target.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            target.write_text(_render_markdown(summary), encoding="utf-8")
        else:
            target.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _summarize_trace(trace_id: int, spans: list[ReadableSpan]) -> dict[str, Any]:
    agent_name = "unknown"
    start_times: list[int] = []
    end_times: list[int] = []
    llm_calls = 0
    input_tokens = 0
    output_tokens = 0
    models: set[str] = set()
    tool_calls = 0
    error_count = 0
    retry_counts: dict[str, int] = {}

    for span in spans:
        attrs = span.attributes or {}
        if span.start_time is not None:
            start_times.append(span.start_time)
        if span.end_time is not None:
            end_times.append(span.end_time)
        if span.status is not None and span.status.status_code == StatusCode.ERROR:
            error_count += 1

        op = attrs.get("gen_ai.operation.name")
        if op == "invoke_agent":
            agent_name = attrs.get("gen_ai.agent.name", agent_name)
        elif op == "agent_step":
            step_name = attrs.get("agent_trace.step.name")
            retry_attempt = attrs.get("agent_trace.retry.attempt", 0)
            if step_name is not None:
                retry_counts[step_name] = max(retry_counts.get(step_name, 0), retry_attempt)
        elif op == "execute_tool":
            tool_calls += 1
        elif op == "chat":
            llm_calls += 1
            input_tokens += attrs.get("gen_ai.usage.input_tokens", 0) or 0
            output_tokens += attrs.get("gen_ai.usage.output_tokens", 0) or 0
            model = attrs.get("gen_ai.request.model")
            if model:
                models.add(model)

    duration_seconds = None
    if start_times and end_times:
        duration_seconds = (max(end_times) - min(start_times)) / 1e9

    return {
        "trace_id": format(trace_id, "032x"),
        "agent_name": agent_name,
        "duration_seconds": duration_seconds,
        "llm_call_count": llm_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "models": sorted(models),
        "tool_call_count": tool_calls,
        "error_count": error_count,
        "retry_counts": retry_counts,
    }


def _render_markdown(summary: list[dict[str, Any]]) -> str:
    header = "| Agent | Duration (s) | LLM Calls | Input Tokens | Output Tokens | Models | Tool Calls | Errors |"
    sep = "|---|---|---|---|---|---|---|---|"
    rows = [header, sep]
    for run in summary:
        duration = f"{run['duration_seconds']:.3f}" if run["duration_seconds"] is not None else "-"
        models = ", ".join(run["models"]) or "-"
        rows.append(
            f"| {run['agent_name']} | {duration} | {run['llm_call_count']} | "
            f"{run['input_tokens']} | {run['output_tokens']} | {models} | "
            f"{run['tool_call_count']} | {run['error_count']} |"
        )
    return "\n".join(rows) + "\n"
