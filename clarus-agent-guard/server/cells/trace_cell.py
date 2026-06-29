"""TraceCell — terminal cell. Commits the finished DAGlet to the store so
it becomes part of memory M and is queryable by motif. It writes nothing
itself beyond persistence; the actual building happens in the scheduler.
"""

from __future__ import annotations

from typing import Any

from . import Cell, Decision
from ..trace.schema import EdgeKind


class TraceCell(Cell):
    name = "trace"

    def fire(self, event: str, field: dict[str, Any]) -> Decision | None:
        return Decision(self.name, "", EdgeKind.CONTROL, "committed DAGlet")
