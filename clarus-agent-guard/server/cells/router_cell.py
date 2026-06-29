"""RouterCell — reads the field set by salience and picks the path:
answer / memory / search / verify. This is the cell the route-accuracy
benchmark scores.
"""

from __future__ import annotations

from typing import Any

from . import Cell, Decision
from ..trace.schema import EdgeKind


class RouterCell(Cell):
    name = "router"

    def fire(self, event: str, field: dict[str, Any]) -> Decision | None:
        # priority: external action must be verified before anything else.
        if field.get("risk"):
            return Decision(self.name, "policy", EdgeKind.EXTERNAL_ACTION,
                            "external action detected -> route to verify")
        if field.get("needs_memory"):
            return Decision(self.name, "memory", EdgeKind.MEMORY_READ,
                            "past context required -> route to memory")
        if field.get("needs_search"):
            return Decision(self.name, "search", EdgeKind.RETRIEVAL,
                            "fresh info required -> route to search")
        return Decision(self.name, "answer", EdgeKind.CONTROL,
                        "self-contained -> answer directly (no LLM tool calls)")
