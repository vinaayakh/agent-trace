"""Tests for the httpx transport-level interceptor.

All tests use httpx mock transports — no real network calls are made.
"""
import json
import pytest
import httpx
from opentelemetry import trace

import agent_trace  # activates the interceptor
from agent_trace import interceptor
from tests.conftest import get_span, span_attr


# ── Mock response helpers ─────────────────────────────────────────────────────

def _anthropic_response(model="claude-haiku-4-5-20251001", input_tokens=50, output_tokens=20):
    return {
        "id": "msg_123",
        "model": model,
        "stop_reason": "end_turn",
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        "content": [{"type": "text", "text": "The answer is 42."}],
    }


def _openai_response(model="gpt-4o-mini", prompt_tokens=50, completion_tokens=20):
    return {
        "id": "chatcmpl-123",
        "model": model,
        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "The answer is 42."}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def _make_client(response_body: dict, base_url: str) -> httpx.Client:
    """Sync httpx client backed by a mock transport."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response_body)

    transport = httpx.MockTransport(handler)
    return httpx.Client(base_url=base_url, transport=transport)


def _make_async_client(response_body: dict, base_url: str) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response_body)

    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(base_url=base_url, transport=transport)


# ── Interceptor is installed ──────────────────────────────────────────────────

def test_interceptor_is_installed():
    assert interceptor._installed is True
    assert httpx.Client.send is interceptor._patched_send
    assert httpx.AsyncClient.send is interceptor._patched_async_send


# ── Sync interception ─────────────────────────────────────────────────────────

def test_sync_anthropic_call_creates_span(span_exporter):
    client = _make_client(_anthropic_response(), "https://api.anthropic.com")
    body = json.dumps({"model": "claude-haiku-4-5-20251001", "messages": [{"role": "user", "content": "Hi"}]})
    client.post("/v1/messages", content=body.encode())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert "anthropic" in spans[0].name


def test_sync_anthropic_span_has_token_attrs(span_exporter):
    client = _make_client(_anthropic_response(input_tokens=75, output_tokens=30), "https://api.anthropic.com")
    body = json.dumps({"model": "claude-haiku-4-5-20251001", "messages": []})
    client.post("/v1/messages", content=body.encode())

    assert span_attr(span_exporter, "gen_ai anthropic claude-haiku-4-5-20251001", "gen_ai.usage.input_tokens") == 75
    assert span_attr(span_exporter, "gen_ai anthropic claude-haiku-4-5-20251001", "gen_ai.usage.output_tokens") == 30
    assert span_attr(span_exporter, "gen_ai anthropic claude-haiku-4-5-20251001", "gen_ai.response.finish_reasons") == "end_turn"


def test_sync_openai_call_creates_span(span_exporter):
    client = _make_client(_openai_response(), "https://api.openai.com")
    body = json.dumps({"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]})
    client.post("/v1/chat/completions", content=body.encode())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert "openai" in spans[0].name
    assert span_attr(span_exporter, spans[0].name, "gen_ai.system") == "openai"


def test_non_llm_call_not_intercepted(span_exporter):
    def handler(request):
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(base_url="https://example.com", transport=transport)
    client.get("/api/data")

    assert len(span_exporter.get_finished_spans()) == 0


# ── Async interception ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_async_anthropic_call_creates_span(span_exporter):
    client = _make_async_client(_anthropic_response(), "https://api.anthropic.com")
    body = json.dumps({"model": "claude-haiku-4-5-20251001", "messages": [{"role": "user", "content": "Hi"}]})
    await client.post("/v1/messages", content=body.encode())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert "anthropic" in spans[0].name


@pytest.mark.asyncio
async def test_async_openai_call_creates_span(span_exporter):
    client = _make_async_client(_openai_response(), "https://api.openai.com")
    body = json.dumps({"model": "gpt-4o-mini", "messages": []})
    await client.post("/v1/chat/completions", content=body.encode())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert "openai" in spans[0].name


@pytest.mark.asyncio
async def test_async_non_llm_call_not_intercepted(span_exporter):
    def handler(request):
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(base_url="https://example.com", transport=transport) as client:
        await client.get("/health")

    assert len(span_exporter.get_finished_spans()) == 0


# ── Error handling ────────────────────────────────────────────────────────────

def test_http_error_sets_span_error_status(span_exporter):
    def handler(request):
        return httpx.Response(429, json={"error": "rate_limited"})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(base_url="https://api.anthropic.com", transport=transport)
    body = json.dumps({"model": "claude-haiku-4-5-20251001", "messages": []})
    client.post("/v1/messages", content=body.encode())

    from opentelemetry.trace import StatusCode
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR
