"""Phase 2 gate: real langgraph graphs exercising sync/async/concurrent execution.

Ports the repro scenarios from the code review: duplicate agent spans under
graph_config() + trace_graph(), broken nesting under ainvoke, and cross-run
contamination under asyncio.gather.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import TypedDict

import httpx
import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

from langgraph.graph import END, StateGraph

from agent_trace.adapters.langgraph import graph_config, trace_graph
from tests.conftest import get_span


class GraphState(TypedDict):
    question: str
    answer: str


def _llm_response_body() -> dict:
    return {
        "id": "chatcmpl-1",
        "model": "gpt-4o-mini",
        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
    }


def _think_node(state: GraphState) -> GraphState:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_llm_response_body())

    client = httpx.Client(base_url="https://api.openai.com", transport=httpx.MockTransport(handler))
    body = json.dumps({"model": "gpt-4o-mini", "messages": [{"role": "user", "content": state["question"]}]})
    client.post("/v1/chat/completions", content=body.encode())
    return {"question": state["question"], "answer": f"Echo: {state['question']}"}


async def _think_node_async(state: GraphState) -> GraphState:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_llm_response_body())

    async with httpx.AsyncClient(base_url="https://api.openai.com", transport=httpx.MockTransport(handler)) as client:
        body = json.dumps({"model": "gpt-4o-mini", "messages": [{"role": "user", "content": state["question"]}]})
        await client.post("/v1/chat/completions", content=body.encode())
    # Give the event loop a chance to schedule other tasks, exercising
    # cross-task context isolation.
    await asyncio.sleep(0)
    return {"question": state["question"], "answer": f"Echo: {state['question']}"}


def _build_graph(node_fn):
    graph = StateGraph(GraphState)
    graph.add_node("think", node_fn)
    graph.set_entry_point("think")
    graph.add_edge("think", END)
    return graph.compile()


# ── sync invoke ────────────────────────────────────────────────────────────────

def test_sync_invoke_single_agent_span_with_full_hierarchy(span_exporter):
    app = _build_graph(_think_node)
    app.invoke({"question": "hi", "answer": ""}, config=graph_config(agent_name="SyncAgent"))

    finished = span_exporter.get_finished_spans()
    agent_spans = [s for s in finished if s.name == "agent SyncAgent"]
    assert len(agent_spans) == 1

    step_span = get_span(span_exporter, "step think")
    assert step_span is not None
    assert step_span.parent.span_id == agent_spans[0].context.span_id

    llm_span = next(s for s in finished if "openai" in s.name)
    assert llm_span.parent.span_id == step_span.context.span_id


# ── async ainvoke ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_async_ainvoke_same_hierarchy_no_callback_errors(span_exporter, caplog):
    app = _build_graph(_think_node_async)
    with caplog.at_level(logging.WARNING):
        await app.ainvoke({"question": "hi", "answer": ""}, config=graph_config(agent_name="AsyncAgent"))

    assert "Failed to detach context" not in caplog.text
    assert "Token was created in a different Context" not in caplog.text

    finished = span_exporter.get_finished_spans()
    agent_spans = [s for s in finished if s.name == "agent AsyncAgent"]
    assert len(agent_spans) == 1

    step_span = get_span(span_exporter, "step think")
    assert step_span is not None
    assert step_span.parent.span_id == agent_spans[0].context.span_id

    llm_span = next(s for s in finished if "openai" in s.name)
    assert llm_span.parent.span_id == step_span.context.span_id


# ── trace_graph + graph_config combined ───────────────────────────────────────

def test_trace_graph_and_graph_config_combined_single_agent_span(span_exporter):
    app = _build_graph(_think_node)
    with trace_graph("CombinedAgent"):
        app.invoke({"question": "hi", "answer": ""}, config=graph_config(agent_name="CombinedAgent"))

    finished = span_exporter.get_finished_spans()
    agent_spans = [s for s in finished if s.name == "agent CombinedAgent"]
    assert len(agent_spans) == 1  # not two — this was the duplicate-span bug

    step_spans = [s for s in finished if s.name.startswith("step ")]
    # Graph root becomes a step under the agent span.
    root_step = next(s for s in step_spans if s.parent and s.parent.span_id == agent_spans[0].context.span_id)
    assert root_step is not None

    node_step = get_span(span_exporter, "step think")
    assert node_step.parent.span_id == root_step.context.span_id

    llm_span = next(s for s in finished if "openai" in s.name)
    assert llm_span.parent.span_id == node_step.context.span_id


# ── concurrent asyncio.gather ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_concurrent_gather_two_distinct_traces_no_cross_contamination(span_exporter):
    app = _build_graph(_think_node_async)

    async def run(agent_name: str, question: str):
        await app.ainvoke({"question": question, "answer": ""}, config=graph_config(agent_name=agent_name))

    await asyncio.gather(run("AgentOne", "question one"), run("AgentTwo", "question two"))

    finished = span_exporter.get_finished_spans()
    agent_one = next(s for s in finished if s.name == "agent AgentOne")
    agent_two = next(s for s in finished if s.name == "agent AgentTwo")
    assert agent_one.context.trace_id != agent_two.context.trace_id

    def descendants_of(agent_span):
        return [s for s in finished if s.context.trace_id == agent_span.context.trace_id]

    one_spans = descendants_of(agent_one)
    two_spans = descendants_of(agent_two)
    # No span from one run's trace leaks into the other's.
    assert not (set(s.context.span_id for s in one_spans) & set(s.context.span_id for s in two_spans))
    assert len(one_spans) >= 3  # agent + step + llm
    assert len(two_spans) >= 3
