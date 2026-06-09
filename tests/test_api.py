"""Tests for the public API — agent/step/tool context managers."""
import json
import pytest
import httpx

import agent_trace
from agent_trace.context import get_current_span, _current
from tests.conftest import get_span, span_attr


def _make_async_client(response_body: dict, base_url: str) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response_body)
    return httpx.AsyncClient(base_url=base_url, transport=httpx.MockTransport(handler))


_ANTHROPIC_BODY = {
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 5},
    "content": [{"type": "text", "text": "done"}],
}


# ── agent() ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_agent_creates_span(span_exporter):
    async with agent_trace.agent("TestAgent"):
        pass

    span = get_span(span_exporter, "agent TestAgent")
    assert span is not None
    assert span.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert span.attributes["gen_ai.agent.name"] == "TestAgent"


@pytest.mark.asyncio
async def test_agent_context_cleared_after_exit(span_exporter):
    async with agent_trace.agent("TestAgent"):
        assert get_current_span() is not None

    assert get_current_span() is None


# ── step() ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_step_creates_child_of_agent(span_exporter):
    async with agent_trace.agent("A"):
        async with agent_trace.step("plan"):
            pass

    agent_span = get_span(span_exporter, "agent A")
    step_span = get_span(span_exporter, "step plan")
    assert step_span is not None
    assert step_span.parent.span_id == agent_span.context.span_id


@pytest.mark.asyncio
async def test_step_retry_detection(span_exporter):
    async with agent_trace.agent("A"):
        async with agent_trace.step("think"):
            pass
        async with agent_trace.step("think"):   # second time → retry
            pass
        async with agent_trace.step("think"):   # third time → retry attempt 2
            pass

    finished = span_exporter.get_finished_spans()
    step_spans = [s for s in finished if s.name == "step think"]
    assert len(step_spans) == 3

    retry_attempts = [s.attributes.get("agent_trace.retry.attempt", 0) for s in step_spans]
    assert sorted(retry_attempts) == [0, 1, 2]


@pytest.mark.asyncio
async def test_step_first_attempt_has_no_retry_attr(span_exporter):
    async with agent_trace.agent("A"):
        async with agent_trace.step("plan"):
            pass

    step_span = get_span(span_exporter, "step plan")
    assert "agent_trace.retry.attempt" not in step_span.attributes


# ── tool() ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tool_creates_span(span_exporter):
    async with agent_trace.agent("A"):
        async with agent_trace.step("act"):
            async with agent_trace.tool("search_web", input="Python history"):
                pass

    tool_span = get_span(span_exporter, "tool search_web")
    assert tool_span is not None
    assert tool_span.attributes["gen_ai.tool.name"] == "search_web"
    assert tool_span.attributes["gen_ai.tool.call.arguments"] == "Python history"


# ── full trace hierarchy ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_llm_call_nested_under_step(span_exporter):
    """LLM span auto-detected by interceptor should be a child of the active step."""
    client = _make_async_client(
        {"model": "claude-haiku-4-5-20251001", **_ANTHROPIC_BODY},
        "https://api.anthropic.com",
    )
    body = json.dumps({"model": "claude-haiku-4-5-20251001", "messages": [{"role": "user", "content": "hi"}]})

    async with agent_trace.agent("FullAgent"):
        async with agent_trace.step("think"):
            await client.post("/v1/messages", content=body.encode())

    finished = span_exporter.get_finished_spans()
    names = {s.name for s in finished}
    assert "agent FullAgent" in names
    assert "step think" in names
    assert any("anthropic" in n for n in names)

    step_span = get_span(span_exporter, "step think")
    llm_span = next(s for s in finished if "anthropic" in s.name)
    assert llm_span.parent.span_id == step_span.context.span_id
