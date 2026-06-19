"""Tests for the httpx transport-level interceptor.

All tests use httpx mock transports — no real network calls are made.
"""
import json
import pytest
import httpx
from opentelemetry import trace

import agent_trace  # activates the interceptor
from agent_trace import interceptor
from agent_trace.interceptor import _is_llm_request, _is_streaming_response, _parse_sse_chunks
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


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://api.openai.com/v1/chat/completions", True),
        ("https://api.anthropic.com/v1/messages", True),
        ("https://localhost:11434/v1/chat/completions", True),
        ("https://127.0.0.1:11434/v1/generate", True),
        ("https://notlocalhost.com/v1/messages", False),
        ("https://evil-api.openai.com.attacker.net/v1/chat/completions", False),
        ("https://api.openai.com/proxy/v1/chat/completions", False),
        ("https://example.com/health", False),
    ],
)
def test_is_llm_request_matching_rules(url: str, expected: bool):
    request = httpx.Request("POST", url)
    assert _is_llm_request(request) is expected


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


# ── Streaming support ─────────────────────────────────────────────────────────

def _openai_sse_bytes() -> bytes:
    chunks = [
        b'data: {"choices":[{"delta":{"content":"Hello "},"finish_reason":null}],"usage":null}\n\n',
        b'data: {"choices":[{"delta":{"content":"world"},"finish_reason":null}],"usage":null}\n\n',
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}\n\n',
        b'data: [DONE]\n\n',
    ]
    return b"".join(chunks)


def _anthropic_sse_bytes() -> bytes:
    chunks = [
        b'data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}\n\n',
        b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello "}}\n\n',
        b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"world"}}\n\n',
        b'data: {"type":"message_delta","usage":{"output_tokens":5},"delta":{"stop_reason":"end_turn"}}\n\n',
        b'data: {"type":"message_stop"}\n\n',
    ]
    return b"".join(chunks)


class _FakeSyncSseStream(httpx.SyncByteStream):
    def __init__(self, data: bytes) -> None:
        self._data = data

    def __iter__(self):
        yield self._data

    def close(self) -> None:
        pass


class _FakeAsyncSseStream(httpx.AsyncByteStream):
    def __init__(self, data: bytes) -> None:
        self._data = data

    async def __aiter__(self):
        yield self._data

    async def aclose(self) -> None:
        pass


def _make_streaming_client(data: bytes, base_url: str) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_FakeSyncSseStream(data),
        )
    return httpx.Client(base_url=base_url, transport=httpx.MockTransport(handler))


def _make_async_streaming_client(data: bytes, base_url: str) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_FakeAsyncSseStream(data),
        )
    return httpx.AsyncClient(base_url=base_url, transport=httpx.MockTransport(handler))


def test_is_streaming_response_detects_event_stream():
    streaming = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        stream=_FakeSyncSseStream(b"data: {}\n\n"),
    )
    assert _is_streaming_response(streaming) is True

    buffered = httpx.Response(200, json={"ok": True})
    assert _is_streaming_response(buffered) is False


def test_parse_sse_chunks_openai():
    in_tok, out_tok, finish, preview = _parse_sse_chunks(_openai_sse_bytes(), "openai")
    assert in_tok == 10
    assert out_tok == 5
    assert finish == "stop"
    assert preview == "Hello world"


def test_parse_sse_chunks_anthropic():
    in_tok, out_tok, finish, preview = _parse_sse_chunks(_anthropic_sse_bytes(), "anthropic")
    assert in_tok == 10
    assert out_tok == 5
    assert finish == "end_turn"
    assert preview == "Hello world"


def test_sync_streaming_span_ends_after_stream_not_at_send(span_exporter):
    client = _make_streaming_client(_openai_sse_bytes(), "https://api.openai.com")
    request = httpx.Request(
        "POST",
        "https://api.openai.com/v1/chat/completions",
        content=json.dumps({"model": "gpt-4o-mini", "messages": [], "stream": True}).encode(),
    )

    response = client.send(request, stream=True)
    assert len(span_exporter.get_finished_spans()) == 0, "span must not end at send()"

    list(response.iter_lines())
    assert len(span_exporter.get_finished_spans()) == 1, "span must end after stream exhausted"


def test_sync_streaming_span_has_token_attrs(span_exporter):
    client = _make_streaming_client(_openai_sse_bytes(), "https://api.openai.com")
    request = httpx.Request(
        "POST",
        "https://api.openai.com/v1/chat/completions",
        content=json.dumps({"model": "gpt-4o-mini", "messages": [], "stream": True}).encode(),
    )
    response = client.send(request, stream=True)
    list(response.iter_lines())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs.get("gen_ai.usage.input_tokens") == 10
    assert attrs.get("gen_ai.usage.output_tokens") == 5
    assert attrs.get("gen_ai.response.finish_reasons") == "stop"
    assert attrs.get("gen_ai.completion_preview") == "Hello world"


def test_sync_streaming_span_ends_on_response_close_without_read(span_exporter):
    client = _make_streaming_client(_openai_sse_bytes(), "https://api.openai.com")
    request = httpx.Request(
        "POST",
        "https://api.openai.com/v1/chat/completions",
        content=json.dumps({"model": "gpt-4o-mini", "messages": [], "stream": True}).encode(),
    )
    response = client.send(request, stream=True)
    assert len(span_exporter.get_finished_spans()) == 0

    response.close()
    assert len(span_exporter.get_finished_spans()) == 1


def test_sync_streaming_finalize_is_idempotent(span_exporter):
    client = _make_streaming_client(_openai_sse_bytes(), "https://api.openai.com")
    request = httpx.Request(
        "POST",
        "https://api.openai.com/v1/chat/completions",
        content=json.dumps({"model": "gpt-4o-mini", "messages": [], "stream": True}).encode(),
    )
    response = client.send(request, stream=True)
    list(response.iter_lines())
    response.close()

    assert len(span_exporter.get_finished_spans()) == 1


@pytest.mark.asyncio
async def test_async_streaming_span_attrs(span_exporter):
    client = _make_async_streaming_client(_anthropic_sse_bytes(), "https://api.anthropic.com")
    request = httpx.Request(
        "POST",
        "https://api.anthropic.com/v1/messages",
        content=json.dumps({"model": "claude-haiku-4-5-20251001", "messages": [], "stream": True}).encode(),
    )

    response = await client.send(request, stream=True)
    assert len(span_exporter.get_finished_spans()) == 0, "span must not end at send()"

    async for _ in response.aiter_lines():
        pass
    assert len(span_exporter.get_finished_spans()) == 1

    spans = span_exporter.get_finished_spans()
    attrs = spans[0].attributes
    assert attrs.get("gen_ai.usage.input_tokens") == 10
    assert attrs.get("gen_ai.usage.output_tokens") == 5
    assert attrs.get("gen_ai.response.finish_reasons") == "end_turn"
    assert attrs.get("gen_ai.completion_preview") == "Hello world"


@pytest.mark.asyncio
async def test_async_streaming_span_ends_on_response_close_without_read(span_exporter):
    client = _make_async_streaming_client(_anthropic_sse_bytes(), "https://api.anthropic.com")
    request = httpx.Request(
        "POST",
        "https://api.anthropic.com/v1/messages",
        content=json.dumps({"model": "claude-haiku-4-5-20251001", "messages": [], "stream": True}).encode(),
    )

    response = await client.send(request, stream=True)
    assert len(span_exporter.get_finished_spans()) == 0

    await response.aclose()
    assert len(span_exporter.get_finished_spans()) == 1
