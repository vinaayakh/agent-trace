"""Phase 5 gate: RunSummaryProcessor aggregation and JSON/markdown output."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from opentelemetry.sdk.trace import TracerProvider

import agent_trace
import agent_trace.exporter as _exporter
from agent_trace.summary import RunSummaryProcessor

_OPENAI_BODY = {
    "model": "gpt-4o-mini",
    "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 4},
}


def _make_client() -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_OPENAI_BODY)

    return httpx.Client(base_url="https://api.openai.com", transport=httpx.MockTransport(handler))


@pytest.fixture
def summary_processor():
    """Fresh RunSummaryProcessor on its own TracerProvider, wired the same
    way agent_trace.init(summary=...) would wire it, without needing a real
    exporter or the batching/background-thread overhead of one."""
    processor = RunSummaryProcessor()
    provider = TracerProvider()
    provider.add_span_processor(processor)

    previous_provider = _exporter.get_provider()
    previous_summary = _exporter.get_summary_processor()
    _exporter.set_provider(provider)
    _exporter.set_summary_processor(processor)

    yield processor

    _exporter.set_provider(previous_provider)
    _exporter.set_summary_processor(previous_summary)


# ── aggregation ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_summary_aggregates_nested_run(summary_processor):
    client = _make_client()
    body = json.dumps({"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]})

    async with agent_trace.agent("Agent1"):
        async with agent_trace.step("think"):
            client.post("/v1/chat/completions", content=body.encode())
        async with agent_trace.step("think"):  # same name again -> retry
            client.post("/v1/chat/completions", content=body.encode())
        async with agent_trace.tool("search", input="q"):
            pass

    summaries = agent_trace.get_summary()
    assert len(summaries) == 1
    summary = summaries[0]

    assert summary["agent_name"] == "Agent1"
    assert summary["llm_call_count"] == 2
    assert summary["input_tokens"] == 20
    assert summary["output_tokens"] == 8
    assert summary["models"] == ["gpt-4o-mini"]
    assert summary["tool_call_count"] == 1
    assert summary["error_count"] == 0
    assert summary["retry_counts"] == {"think": 1}
    assert summary["duration_seconds"] is not None
    assert summary["duration_seconds"] >= 0


@pytest.mark.asyncio
async def test_summary_counts_errors(summary_processor):
    with pytest.raises(RuntimeError):
        async with agent_trace.agent("Agent2"):
            async with agent_trace.step("think"):
                raise RuntimeError("boom")

    [summary] = agent_trace.get_summary()
    assert summary["agent_name"] == "Agent2"
    assert summary["error_count"] == 2  # step span + agent span both get status ERROR


@pytest.mark.asyncio
async def test_summary_two_distinct_agent_runs_stay_separate(summary_processor):
    async with agent_trace.agent("First"):
        pass
    async with agent_trace.agent("Second"):
        async with agent_trace.tool("search", input="q"):
            pass

    summaries = {s["agent_name"]: s for s in agent_trace.get_summary()}
    assert set(summaries) == {"First", "Second"}
    assert summaries["First"]["tool_call_count"] == 0
    assert summaries["Second"]["tool_call_count"] == 1


def test_get_summary_empty_without_processor():
    assert agent_trace.get_summary() == []


# ── JSON / markdown file output ────────────────────────────────────────────────

def _run_trivial_agent(name: str) -> None:
    async def run():
        async with agent_trace.agent(name):
            pass

    asyncio.run(run())


def test_write_json(tmp_path, summary_processor):
    _run_trivial_agent("FileAgent")

    path = tmp_path / "summary.json"
    summary_processor.write(str(path))

    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["agent_name"] == "FileAgent"


def test_write_markdown_also_writes_json_sibling(tmp_path, summary_processor):
    _run_trivial_agent("MdAgent")

    path = tmp_path / "summary.md"
    summary_processor.write(str(path))

    md_content = path.read_text(encoding="utf-8")
    assert "MdAgent" in md_content
    assert "| Agent | Duration (s) |" in md_content

    json_path = tmp_path / "summary.json"
    assert json_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data[0]["agent_name"] == "MdAgent"


def test_shutdown_writes_summary_when_path_given(tmp_path):
    path = tmp_path / "on_shutdown.json"
    processor = RunSummaryProcessor(path=str(path))
    provider = TracerProvider()
    provider.add_span_processor(processor)

    previous_provider = _exporter.get_provider()
    previous_summary = _exporter.get_summary_processor()
    _exporter.set_provider(provider)
    _exporter.set_summary_processor(processor)
    try:
        _run_trivial_agent("ShutdownAgent")
        provider.shutdown()

        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data[0]["agent_name"] == "ShutdownAgent"
    finally:
        _exporter.set_provider(previous_provider)
        _exporter.set_summary_processor(previous_summary)


def test_shutdown_does_not_write_without_path(summary_processor):
    _run_trivial_agent("NoWriteAgent")
    # Should not raise even though no path was configured.
    summary_processor.shutdown()


# ── configure() wiring ──────────────────────────────────────────────────────────

def test_configure_with_summary_true_attaches_in_memory_processor():
    previous_provider = _exporter.get_provider()
    previous_summary = _exporter.get_summary_processor()
    try:
        _exporter.configure(exporter="console", summary=True)
        processor = _exporter.get_summary_processor()
        assert processor is not None
        assert processor._path is None
    finally:
        _exporter.set_provider(previous_provider)
        _exporter.set_summary_processor(previous_summary)


def test_configure_with_summary_path_sets_path():
    previous_provider = _exporter.get_provider()
    previous_summary = _exporter.get_summary_processor()
    try:
        _exporter.configure(exporter="console", summary="out/summary.json")
        processor = _exporter.get_summary_processor()
        assert processor is not None
        assert processor._path == "out/summary.json"
    finally:
        _exporter.set_provider(previous_provider)
        _exporter.set_summary_processor(previous_summary)


def test_configure_without_summary_leaves_no_processor():
    previous_provider = _exporter.get_provider()
    previous_summary = _exporter.get_summary_processor()
    try:
        _exporter.configure(exporter="console")
        assert _exporter.get_summary_processor() is None
    finally:
        _exporter.set_provider(previous_provider)
        _exporter.set_summary_processor(previous_summary)
