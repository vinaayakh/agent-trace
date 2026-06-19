"""LangChain callback adapter for agent_trace spans."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

from agent_trace._runtime import enter_agent, enter_step, enter_tool, exit_frame

from ._stack import RunStack

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
except ImportError as exc:  # pragma: no cover - exercised in import tests
    raise ImportError(
        "LangChain adapter requires optional dependency 'langchain-core'. "
        "Install with: pip install agent-trace[langchain]"
    ) from exc


def _name_from_serialized(serialized: Optional[dict[str, Any]], fallback: str) -> str:
    if not isinstance(serialized, dict):
        return fallback
    if serialized.get("name"):
        return str(serialized["name"])
    if isinstance(serialized.get("id"), list) and serialized["id"]:
        return str(serialized["id"][-1])
    return fallback


def _merge_callbacks(config: Optional[dict[str, Any]], handler: Any) -> dict[str, Any]:
    cfg = dict(config) if config else {}
    callbacks = cfg.get("callbacks")
    if callbacks is None:
        cfg["callbacks"] = [handler]
        return cfg
    if isinstance(callbacks, list):
        cfg["callbacks"] = [*callbacks, handler]
        return cfg
    if isinstance(callbacks, tuple):
        cfg["callbacks"] = [*callbacks, handler]
        return cfg
    cfg["callbacks"] = [callbacks, handler]
    return cfg


class AgentTraceCallbackHandler(BaseCallbackHandler):
    """Maps LangChain callbacks to agent/step/tool spans."""

    ignore_llm = True

    def __init__(
        self,
        agent_name: str = "LangChainAgent",
        *,
        step_on_agent_action: bool = True,
        record_tool_output: bool = False,
    ) -> None:
        super().__init__()
        self.agent_name = agent_name
        self.step_on_agent_action = step_on_agent_action
        self.record_tool_output = record_tool_output
        self._stack = RunStack()
        self._agent_action_frames: dict[str, list[Any]] = defaultdict(list)

    def _exit_run(self, run_id: Any, error: Optional[BaseException] = None) -> None:
        run_frame = self._stack.pop(run_id)
        if run_frame is None:
            return
        if error is not None:
            exit_frame(run_frame.frame, type(error), error, error.__traceback__)
            return
        exit_frame(run_frame.frame)

    def _close_action_frames(self, run_id: Any, error: Optional[BaseException] = None) -> None:
        key = str(run_id)
        frames = self._agent_action_frames.pop(key, [])
        for frame in reversed(frames):
            if error is not None:
                exit_frame(frame, type(error), error, error.__traceback__)
            else:
                exit_frame(frame)

    def on_chain_start(
        self,
        serialized: Optional[dict[str, Any]],
        inputs: dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        if parent_run_id is None:
            frame = enter_agent(self.agent_name)
            self._stack.push(run_id, frame, kind="agent", parent_run_id=parent_run_id)
            return
        step_name = "think"
        if isinstance(metadata, dict):
            step_name = str(metadata.get("langgraph_node") or metadata.get("step_name") or step_name)
        if step_name == "think":
            step_name = _name_from_serialized(serialized, "think")
        frame = enter_step(step_name)
        self._stack.push(run_id, frame, kind="step", parent_run_id=parent_run_id)

    def on_chain_end(self, outputs: dict[str, Any], *, run_id: Any, **kwargs: Any) -> Any:
        self._close_action_frames(run_id)
        self._exit_run(run_id)

    def on_chain_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> Any:
        self._close_action_frames(run_id, error=error)
        self._exit_run(run_id, error=error)

    def on_tool_start(
        self,
        serialized: Optional[dict[str, Any]],
        input_str: str,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> Any:
        tool_name = _name_from_serialized(serialized, "tool")
        frame = enter_tool(tool_name, input_value=input_str)
        self._stack.push(run_id, frame, kind="tool", parent_run_id=parent_run_id)

    def on_tool_end(self, output: Any, *, run_id: Any, **kwargs: Any) -> Any:
        self._exit_run(run_id)

    def on_tool_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> Any:
        self._exit_run(run_id, error=error)

    def on_agent_action(self, action: Any, *, run_id: Any, **kwargs: Any) -> Any:
        if not self.step_on_agent_action:
            return
        frame = enter_step("act")
        self._agent_action_frames[str(run_id)].append(frame)

    def on_agent_finish(self, finish: Any, *, run_id: Any, **kwargs: Any) -> Any:
        self._close_action_frames(run_id)


class _TracedRunnable:
    def __init__(self, runnable: Any, handler: AgentTraceCallbackHandler) -> None:
        self._runnable = runnable
        self._handler = handler

    def invoke(self, input: Any, config: Optional[dict[str, Any]] = None, **kwargs: Any) -> Any:
        cfg = _merge_callbacks(config, self._handler)
        return self._runnable.invoke(input, config=cfg, **kwargs)

    async def ainvoke(self, input: Any, config: Optional[dict[str, Any]] = None, **kwargs: Any) -> Any:
        cfg = _merge_callbacks(config, self._handler)
        return await self._runnable.ainvoke(input, config=cfg, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runnable, name)


def trace_runnable(
    runnable: Any,
    *,
    agent_name: str = "LangChainAgent",
    handler: Optional[AgentTraceCallbackHandler] = None,
) -> Any:
    """Attach an AgentTraceCallbackHandler to a runnable."""
    resolved_handler = handler or AgentTraceCallbackHandler(agent_name=agent_name)
    return _TracedRunnable(runnable, resolved_handler)
