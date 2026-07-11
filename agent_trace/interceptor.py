"""Transport-level httpx interceptor.

Monkey-patches httpx.Client.send and httpx.AsyncClient.send so that any
outbound request to a known LLM endpoint is automatically wrapped in an
OTel span — regardless of which SDK made the call.

Activation: importing this module (or calling install()) is all that's needed.
"""
from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, Iterator, Optional

import httpx
from opentelemetry.trace import StatusCode

from .spans import (
    start_span,
    llm_request_attrs,
    llm_response_attrs,
)

# Hosts that indicate an LLM API call
_LLM_HOSTS: set[str] = {
    "api.openai.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
    "api.cohere.com",
    "api.mistral.ai",
    "api.together.xyz",
    "api.groq.com",
    "openrouter.ai",
    "openai.azure.com",  # suffix-matches <resource>.openai.azure.com
    "localhost",        # Ollama and local proxies
    "127.0.0.1",
}

# URL path fragments that confirm this is a chat/completion endpoint
_LLM_PATHS: tuple[str, ...] = (
    "/v1/messages",          # Anthropic
    "/v1/chat/completions",  # OpenAI, Groq, Together, Ollama
    "/v1/completions",
    "/v1beta/models",        # Gemini
    "/v1/generate",          # Ollama (legacy)
    "/api/chat",             # Ollama native
    "/api/generate",         # Ollama native
    "/api/v1/chat/completions",  # OpenRouter
    "/api/v1/completions",       # OpenRouter
)

_installed = False


def _configured_hosts() -> set[str]:
    """`_LLM_HOSTS` plus any hosts from AGENT_TRACE_EXTRA_HOSTS (comma-separated).

    Read fresh on every call (not cached) so custom base URLs/proxies can be
    opted into tracing via env var without a code change or re-import.
    """
    extra = os.environ.get("AGENT_TRACE_EXTRA_HOSTS", "")
    extra_hosts = {h.strip() for h in extra.split(",") if h.strip()}
    return _LLM_HOSTS | extra_hosts


def _is_azure_chat_path(path: str) -> bool:
    """Azure OpenAI: /openai/deployments/<deployment>/chat/completions"""
    return "/openai/deployments/" in path and path.rstrip("/").endswith("/chat/completions")


def _host_matches(host: str, allowed: str) -> bool:
    normalized_host = host.lower().rstrip(".")
    normalized_allowed = allowed.lower()
    return (
        normalized_host == normalized_allowed
        or normalized_host.endswith("." + normalized_allowed)
    )


def _is_streaming_response(response: httpx.Response) -> bool:
    ct = response.headers.get("content-type", "")
    return "text/event-stream" in ct and not response.is_stream_consumed


def _parse_sse_chunks(
    data: bytes, provider: str
) -> tuple[Optional[int], Optional[int], Optional[str], str]:
    """Parse accumulated SSE bytes into (input_tokens, output_tokens, finish_reason, completion_preview)."""
    input_tokens = output_tokens = finish_reason = None
    content_parts: list[str] = []
    try:
        text = data.decode("utf-8", errors="replace")
        for frame in text.split("\n\n"):
            for line in frame.splitlines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    continue
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if provider == "anthropic":
                    t = obj.get("type")
                    if t == "message_start":
                        usage = obj.get("message", {}).get("usage", {})
                        input_tokens = usage.get("input_tokens")
                    elif t == "content_block_delta":
                        delta = obj.get("delta", {})
                        if delta.get("type") == "text_delta":
                            content_parts.append(delta.get("text", ""))
                    elif t == "message_delta":
                        usage = obj.get("usage", {})
                        output_tokens = usage.get("output_tokens")
                        finish_reason = obj.get("delta", {}).get("stop_reason")
                elif provider == "google":
                    usage_metadata = obj.get("usageMetadata")
                    if usage_metadata:
                        input_tokens = usage_metadata.get("promptTokenCount")
                        output_tokens = usage_metadata.get("candidatesTokenCount")
                    candidates = obj.get("candidates") or []
                    if candidates:
                        candidate = candidates[0]
                        parts = candidate.get("content", {}).get("parts", [])
                        for part in parts:
                            if isinstance(part, dict) and part.get("text"):
                                content_parts.append(part["text"])
                        fr = candidate.get("finishReason")
                        if fr:
                            finish_reason = fr
                else:
                    usage = obj.get("usage") or {}
                    if usage:
                        input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
                        output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
                    choices = obj.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta", {})
                        if delta.get("content"):
                            content_parts.append(delta["content"])
                        fr = choices[0].get("finish_reason")
                        if fr:
                            finish_reason = fr
    except Exception:
        pass
    return input_tokens, output_tokens, finish_reason, "".join(content_parts)[:500]


class _SyncSseWrapper(httpx.SyncByteStream):
    def __init__(self, original: httpx.SyncByteStream, span: Any, provider: str) -> None:
        self._original = original
        self._span = span
        self._provider = provider
        self._accumulator: bytearray = bytearray()
        self._finalized = False

    def __iter__(self) -> Iterator[bytes]:
        exc_to_raise: Optional[BaseException] = None
        try:
            for chunk in self._original:
                self._accumulator.extend(chunk)
                yield chunk
        except Exception as exc:
            exc_to_raise = exc
            self._span.set_status(StatusCode.ERROR, str(exc))
            self._span.record_exception(exc)
        finally:
            self._finalize()
        if exc_to_raise is not None:
            raise exc_to_raise

    def close(self) -> None:
        self._finalize()
        self._original.close()

    def _finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        in_tok, out_tok, finish, preview = _parse_sse_chunks(bytes(self._accumulator), self._provider)
        self._span.set_attributes(llm_response_attrs(in_tok, out_tok, finish, preview))
        self._span.end()


class _AsyncSseWrapper(httpx.AsyncByteStream):
    def __init__(self, original: httpx.AsyncByteStream, span: Any, provider: str) -> None:
        self._original = original
        self._span = span
        self._provider = provider
        self._accumulator: bytearray = bytearray()
        self._finalized = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        exc_to_raise: Optional[BaseException] = None
        try:
            async for chunk in self._original:
                self._accumulator.extend(chunk)
                yield chunk
        except Exception as exc:
            exc_to_raise = exc
            self._span.set_status(StatusCode.ERROR, str(exc))
            self._span.record_exception(exc)
        finally:
            self._finalize()
        if exc_to_raise is not None:
            raise exc_to_raise

    async def aclose(self) -> None:
        self._finalize()
        await self._original.aclose()

    def _finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        in_tok, out_tok, finish, preview = _parse_sse_chunks(bytes(self._accumulator), self._provider)
        self._span.set_attributes(llm_response_attrs(in_tok, out_tok, finish, preview))
        self._span.end()


def _is_llm_request(request: httpx.Request) -> bool:
    host = request.url.host or ""
    path = request.url.path
    if not any(_host_matches(host, allowed) for allowed in _configured_hosts()):
        return False
    return any(path.startswith(p) for p in _LLM_PATHS) or _is_azure_chat_path(path)


def _detect_provider(request: httpx.Request) -> str:
    host = request.url.host or ""
    if _host_matches(host, "api.anthropic.com"):
        return "anthropic"
    if _host_matches(host, "api.openai.com"):
        return "openai"
    if _host_matches(host, "generativelanguage.googleapis.com"):
        return "google"
    if _host_matches(host, "api.cohere.com"):
        return "cohere"
    if _host_matches(host, "api.mistral.ai"):
        return "mistral"
    if _host_matches(host, "api.groq.com"):
        return "groq"
    if _host_matches(host, "api.together.xyz"):
        return "together"
    if _host_matches(host, "openrouter.ai"):
        return "openrouter"
    if _host_matches(host, "openai.azure.com"):
        return "azure"
    if host in {"localhost", "127.0.0.1"}:
        return "ollama"
    return "unknown"


def _parse_request(request: httpx.Request) -> tuple[str, str]:
    """Return (model, prompt_preview)."""
    model = "unknown"
    prompt_preview = ""
    try:
        body = json.loads(request.content)
        model = body.get("model", "unknown")
        messages = body.get("messages") or body.get("prompt") or []
        if isinstance(messages, list) and messages:
            last = messages[-1]
            content = last.get("content", "")
            if isinstance(content, list):
                # Anthropic multi-part content
                parts = [b.get("text", "") for b in content if isinstance(b, dict)]
                content = " ".join(parts)
            prompt_preview = str(content)
        elif isinstance(messages, str):
            prompt_preview = messages
    except Exception:
        pass
    return model, prompt_preview


def _parse_response(response: httpx.Response) -> tuple[Optional[int], Optional[int], Optional[str], str]:
    """Return (input_tokens, output_tokens, finish_reason, completion_preview)."""
    input_tokens = output_tokens = finish_reason = None
    completion_preview = ""
    try:
        body = json.loads(response.content)
        # OpenAI-style
        usage = body.get("usage", {})
        input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
        output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")

        choices = body.get("choices")
        if choices and isinstance(choices, list):
            finish_reason = choices[0].get("finish_reason")
            msg = choices[0].get("message", {})
            completion_preview = msg.get("content", "")

        # Anthropic-style
        if not finish_reason:
            finish_reason = body.get("stop_reason")
        if not completion_preview:
            content = body.get("content", [])
            if isinstance(content, list) and content:
                completion_preview = content[0].get("text", "")

        # Gemini-style
        usage_metadata = body.get("usageMetadata")
        if usage_metadata:
            if input_tokens is None:
                input_tokens = usage_metadata.get("promptTokenCount")
            if output_tokens is None:
                output_tokens = usage_metadata.get("candidatesTokenCount")
        candidates = body.get("candidates")
        if candidates and isinstance(candidates, list):
            candidate = candidates[0]
            if not finish_reason:
                finish_reason = candidate.get("finishReason")
            if not completion_preview:
                parts = candidate.get("content", {}).get("parts", [])
                completion_preview = "".join(
                    p.get("text", "") for p in parts if isinstance(p, dict)
                )
    except Exception:
        pass
    return input_tokens, output_tokens, finish_reason, completion_preview


def _make_span(request: httpx.Request):
    provider = _detect_provider(request)
    model, prompt_preview = _parse_request(request)
    attrs = llm_request_attrs(provider, model, prompt_preview)
    span_name = f"gen_ai {provider} {model}"
    return start_span(span_name, attributes=attrs)


# ── Sync patch ────────────────────────────────────────────────────────────────

_original_send: Any = None
_original_async_send: Any = None


def _patched_send(self: httpx.Client, request: httpx.Request, **kwargs):
    if not _is_llm_request(request):
        return _original_send(self, request, **kwargs)

    span = _make_span(request)
    try:
        response = _original_send(self, request, **kwargs)
    except Exception as exc:
        span.set_status(StatusCode.ERROR, str(exc))
        span.record_exception(exc)
        span.end()
        raise

    if _is_streaming_response(response):
        response.stream = _SyncSseWrapper(response.stream, span, _detect_provider(request))
        return response

    try:
        in_tok, out_tok, finish, preview = _parse_response(response)
        span.set_attributes(llm_response_attrs(in_tok, out_tok, finish, preview))
        if response.is_error:
            span.set_status(StatusCode.ERROR, f"HTTP {response.status_code}")
        return response
    except Exception as exc:
        span.set_status(StatusCode.ERROR, str(exc))
        span.record_exception(exc)
        raise
    finally:
        span.end()


async def _patched_async_send(self: httpx.AsyncClient, request: httpx.Request, **kwargs):
    if not _is_llm_request(request):
        return await _original_async_send(self, request, **kwargs)

    span = _make_span(request)
    try:
        response = await _original_async_send(self, request, **kwargs)
    except Exception as exc:
        span.set_status(StatusCode.ERROR, str(exc))
        span.record_exception(exc)
        span.end()
        raise

    if _is_streaming_response(response):
        response.stream = _AsyncSseWrapper(response.stream, span, _detect_provider(request))
        return response

    try:
        in_tok, out_tok, finish, preview = _parse_response(response)
        span.set_attributes(llm_response_attrs(in_tok, out_tok, finish, preview))
        if response.is_error:
            span.set_status(StatusCode.ERROR, f"HTTP {response.status_code}")
        return response
    except Exception as exc:
        span.set_status(StatusCode.ERROR, str(exc))
        span.record_exception(exc)
        raise
    finally:
        span.end()


def install() -> None:
    """Monkey-patch httpx at the transport level. Idempotent."""
    global _installed, _original_send, _original_async_send
    if _installed:
        return
    _original_send = httpx.Client.send
    _original_async_send = httpx.AsyncClient.send
    httpx.Client.send = _patched_send
    httpx.AsyncClient.send = _patched_async_send
    _installed = True


def uninstall() -> None:
    """Restore original httpx send methods."""
    global _installed
    if not _installed:
        return
    httpx.Client.send = _original_send
    httpx.AsyncClient.send = _original_async_send
    _installed = False


# Auto-install on import
install()
