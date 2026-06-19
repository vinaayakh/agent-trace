"""Run-id keyed frame tracking for callback-based integrations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from agent_trace._runtime import SpanFrame


@dataclass
class RunFrame:
    frame: SpanFrame
    kind: str
    parent_run_id: Optional[str] = None


class RunStack:
    """Tracks active span frames by callback run id."""

    def __init__(self) -> None:
        self._frames: dict[str, RunFrame] = {}

    def push(
        self,
        run_id: Any,
        frame: SpanFrame,
        kind: str,
        parent_run_id: Any = None,
    ) -> None:
        key = str(run_id)
        parent_key = str(parent_run_id) if parent_run_id is not None else None
        self._frames[key] = RunFrame(frame=frame, kind=kind, parent_run_id=parent_key)

    def pop(self, run_id: Any) -> Optional[RunFrame]:
        return self._frames.pop(str(run_id), None)

    def get(self, run_id: Any) -> Optional[RunFrame]:
        return self._frames.get(str(run_id))

    def clear(self) -> None:
        self._frames.clear()
