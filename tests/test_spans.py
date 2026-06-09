"""Tests for the GenAI semconv attribute builders in spans.py."""
from agent_trace.spans import (
    agent_attrs,
    llm_request_attrs,
    llm_response_attrs,
    step_attrs,
    tool_attrs,
)


def test_agent_attrs():
    attrs = agent_attrs("MyAgent")
    assert attrs["gen_ai.operation.name"] == "invoke_agent"
    assert attrs["gen_ai.agent.name"] == "MyAgent"


def test_step_attrs_first_attempt():
    attrs = step_attrs("plan", retry_attempt=0)
    assert attrs["gen_ai.operation.name"] == "agent_step"
    assert attrs["agent_trace.step.name"] == "plan"
    assert "agent_trace.retry.attempt" not in attrs


def test_step_attrs_retry():
    attrs = step_attrs("plan", retry_attempt=2)
    assert attrs["agent_trace.retry.attempt"] == 2


def test_tool_attrs_with_input():
    attrs = tool_attrs("search_web", input_value="Python history")
    assert attrs["gen_ai.operation.name"] == "execute_tool"
    assert attrs["gen_ai.tool.name"] == "search_web"
    assert attrs["gen_ai.tool.call.arguments"] == "Python history"


def test_tool_attrs_without_input():
    attrs = tool_attrs("get_time")
    assert "gen_ai.tool.call.arguments" not in attrs


def test_llm_request_attrs():
    attrs = llm_request_attrs("anthropic", "claude-haiku-4-5-20251001", "Hello")
    assert attrs["gen_ai.system"] == "anthropic"
    assert attrs["gen_ai.request.model"] == "claude-haiku-4-5-20251001"
    assert attrs["gen_ai.prompt_preview"] == "Hello"


def test_llm_request_attrs_truncates_long_prompt():
    long_prompt = "x" * 1000
    attrs = llm_request_attrs("openai", "gpt-4o", long_prompt)
    assert len(attrs["gen_ai.prompt_preview"]) == 500


def test_llm_response_attrs():
    attrs = llm_response_attrs(100, 50, "end_turn", "The answer is 42")
    assert attrs["gen_ai.usage.input_tokens"] == 100
    assert attrs["gen_ai.usage.output_tokens"] == 50
    assert attrs["gen_ai.response.finish_reasons"] == "end_turn"
    assert attrs["gen_ai.completion_preview"] == "The answer is 42"


def test_llm_response_attrs_partial():
    # Should not include keys for None values
    attrs = llm_response_attrs(None, None, None)
    assert "gen_ai.usage.input_tokens" not in attrs
    assert "gen_ai.usage.output_tokens" not in attrs
    assert "gen_ai.response.finish_reasons" not in attrs
