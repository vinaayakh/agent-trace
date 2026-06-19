from agent_trace.adapters._stack import RunStack


def test_run_stack_push_pop():
    stack = RunStack()
    frame = object()

    stack.push("run-1", frame=frame, kind="step", parent_run_id="parent")
    stored = stack.get("run-1")

    assert stored is not None
    assert stored.frame is frame
    assert stored.kind == "step"
    assert stored.parent_run_id == "parent"

    popped = stack.pop("run-1")
    assert popped is not None
    assert popped.frame is frame
    assert stack.get("run-1") is None
