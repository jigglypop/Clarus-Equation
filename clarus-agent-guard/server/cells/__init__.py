"""Cell substrate.

A Cell is a small, always-available unit that, given an event + field,
decides whether it fires and what edge it emits. Cells are deliberately
thin in this MVP — most return a rule-based decision and never call an
LLM. The point of the runtime is that *small cells judge first*; the LLM
is one optional cell among many.

Each cell exposes `fire(event, field) -> Decision | None`. Returning None
means "I don't apply to this event"; the scheduler skips me.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dfield
from typing import Any

from ..trace.schema import EdgeKind


@dataclass
class Decision:
    cell: str
    next: str                      # logical target node id this cell points to
    kind: EdgeKind
    reason: str
    note: str = ""
    data: dict[str, Any] = dfield(default_factory=dict)


class Cell:
    name: str = "cell"

    def fire(self, event: str, field: dict[str, Any]) -> Decision | None:
        raise NotImplementedError
