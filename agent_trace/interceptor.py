"""Transport-level httpx interceptor.

Monkey-patches httpx.Client.send and httpx.AsyncClient.send so that any
outbound request to a known LLM endpoint is automatically wrapped in an
OTel span — regardless of which SDK made the call.

Activation: importing this module (or calling install()) is all that's needed.
"""
from __future__ import annotations

import json
import time
from typing import Any, Optional

import httpx
from opentelemetry.trace import StatusCode

from .context import get_context, set_context, TraceContext
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
    "localhost",        # Ollama and local proxies
    "127.0.0.1",
}

# URL path fragments that confirm this is a chat/completion endpoint
_LLM_PATHS: tuple[str, ...] = (
    "/v1/messages",          # Anthropic
    "/v1/chat/completions",  # OpenAI, Groq, Together, Ollama
    "/v1/completions",
    "/v1beta/models",        # Gemini
    "/v1/generate",          # Ollama
)

_installed = False


def _is_llm_request(request: httpx.Request) -> bool:
    host = request.url.host
    path = request.url.path
    return (
        any(h in host for h in _LLM_HOSTS)
        and any(path.startswith(p) or p in path for p in _LLM_PATHS)
    )


def _detect_provider(request: httpx.Request) -> str:
    host = request.url.host
    if "anthropic" in host:
        return "anthropic"
    if "openai" in host:
        return "openai"
    if "googleapis" in host:
        return "google"
    if "cohere" in host:
        return "cohere"
    if "mistral" in host:
        return "mistral"
    if "groq" in host:
        return "groq"
    if "together" in host:
        return "together"
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
    with span:
        try:
            response = _original_send(self, request, **kwargs)
            in_tok, out_tok, finish, preview = _parse_response(response)
            span.set_attributes(llm_response_attrs(in_tok, out_tok, finish, preview))
            if response.is_error:
                span.set_status(StatusCode.ERROR, f"HTTP {response.status_code}")
            return response
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise


async def _patched_async_send(self: httpx.AsyncClient, request: httpx.Request, **kwargs):
    if not _is_llm_request(request):
        return await _original_async_send(self, request, **kwargs)

    span = _make_span(request)
    with span:
        try:
            response = await _original_async_send(self, request, **kwargs)
            in_tok, out_tok, finish, preview = _parse_response(response)
            span.set_attributes(llm_response_attrs(in_tok, out_tok, finish, preview))
            if response.is_error:
                span.set_status(StatusCode.ERROR, f"HTTP {response.status_code}")
            return response
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise


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
