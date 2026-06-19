import pytest

from tests.conftest import get_span

pytest.importorskip("langchain_core")

from agent_trace.adapters.langgraph import graph_config, trace_graph


def test_trace_graph_context_creates_agent_span(span_exporter):
    with trace_graph("LangGraphAgent"):
        pass

    agent_span = get_span(span_exporter, "agent LangGraphAgent")
    assert agent_span is not None


def test_graph_config_adds_callback():
    cfg = graph_config({"callbacks": ["existing"]}, agent_name="LangGraphAgent")
    assert len(cfg["callbacks"]) == 2
