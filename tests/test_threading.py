"""Phase 3 gate: thread-crossing story via agent_trace.bind_context."""
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

import agent_trace
from tests.conftest import get_span

_ANTHROPIC_BODY = {
    "model": "claude-haiku-4-5-20251001",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 5},
    "content": [{"type": "text", "text": "done"}],
}


def _call_llm() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ANTHROPIC_BODY)

    client = httpx.Client(base_url="https://api.anthropic.com", transport=httpx.MockTransport(handler))
    body = json.dumps({"model": "claude-haiku-4-5-20251001", "messages": [{"role": "user", "content": "hi"}]})
    client.post("/v1/messages", content=body.encode())


@pytest.mark.asyncio
async def test_thread_executor_without_bind_context_orphans_trace(span_exporter):
    """Documents current (pre-helper) behavior: a raw threadpool submission
    loses ambient context, so the LLM span becomes an unparented root trace."""
    async with agent_trace.agent("A"):
        async with agent_trace.step("think"):
            with ThreadPoolExecutor() as pool:
                future = pool.submit(_call_llm)
                await asyncio.wrap_future(future)

    finished = span_exporter.get_finished_spans()
    step_span = get_span(span_exporter, "step think")
    llm_span = next(s for s in finished if "anthropic" in s.name)

    assert llm_span.parent is None
    assert llm_span.context.trace_id != step_span.context.trace_id


@pytest.mark.asyncio
async def test_thread_executor_with_bind_context_nests_correctly(span_exporter):
    async with agent_trace.agent("A"):
        async with agent_trace.step("think"):
            with ThreadPoolExecutor() as pool:
                future = pool.submit(agent_trace.bind_context(_call_llm))
                await asyncio.wrap_future(future)

    finished = span_exporter.get_finished_spans()
    step_span = get_span(span_exporter, "step think")
    llm_span = next(s for s in finished if "anthropic" in s.name)

    assert llm_span.parent is not None
    assert llm_span.parent.span_id == step_span.context.span_id
    assert llm_span.context.trace_id == step_span.context.trace_id
