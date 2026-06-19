from uuid import uuid4

import pytest

from tests.conftest import get_span

langchain_core = pytest.importorskip("langchain_core")

from agent_trace.adapters.langchain import AgentTraceCallbackHandler, trace_runnable


def test_handler_creates_agent_step_tool_hierarchy(span_exporter):
    handler = AgentTraceCallbackHandler(agent_name="LangChainAgent")

    root_run = uuid4()
    step_run = uuid4()
    tool_run = uuid4()

    handler.on_chain_start({"name": "AgentExecutor"}, {}, run_id=root_run, parent_run_id=None)
    handler.on_chain_start({"name": "think"}, {}, run_id=step_run, parent_run_id=root_run)
    handler.on_tool_start({"name": "search"}, "python", run_id=tool_run, parent_run_id=step_run)
    handler.on_tool_end("ok", run_id=tool_run)
    handler.on_chain_end({}, run_id=step_run)
    handler.on_chain_end({}, run_id=root_run)

    agent_span = get_span(span_exporter, "agent LangChainAgent")
    step_span = get_span(span_exporter, "step think")
    tool_span = get_span(span_exporter, "tool search")

    assert agent_span is not None
    assert step_span is not None
    assert tool_span is not None
    assert step_span.parent.span_id == agent_span.context.span_id
    assert tool_span.parent.span_id == step_span.context.span_id


def test_handler_closes_span_on_chain_error(span_exporter):
    handler = AgentTraceCallbackHandler(agent_name="LangChainAgent")
    root_run = uuid4()

    handler.on_chain_start({"name": "AgentExecutor"}, {}, run_id=root_run, parent_run_id=None)
    handler.on_chain_error(RuntimeError("boom"), run_id=root_run)

    assert get_span(span_exporter, "agent LangChainAgent") is not None


def test_trace_runnable_appends_callback():
    class _FakeRunnable:
        def __init__(self):
            self.last_config = None

        def invoke(self, input, config=None, **kwargs):
            self.last_config = config
            return {"ok": True}

    runnable = _FakeRunnable()
    traced = trace_runnable(runnable, agent_name="LangChainAgent")
    result = traced.invoke({"question": "hi"}, config={"callbacks": ["existing"]})

    assert result == {"ok": True}
    assert len(runnable.last_config["callbacks"]) == 2
