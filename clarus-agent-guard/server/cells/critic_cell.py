"""CriticCell — runs after the path resolves. Scores the DAGlet so the
store can tell, next time the same motif appears, whether it tends to
succeed. This score is what closes the replay loop.
"""

from __future__ import annotations

from typing import Any

from . import Cell, Decision
from ..trace.schema import EdgeKind


class CriticCell(Cell):
    name = "critic"

    def score(self, result: str, field: dict[str, Any]) -> float:
        # crude rubric: blocking an unapproved action is a win; answering
        # a memory/search need without resolving it is penalised.
        if result == "draft" and field.get("risk"):
            return 0.95
        if result == "answer" and field.get("needs_memory") \
                and not field.get("recalled"):
            return 0.3
        return 0.8

    def fire(self, event: str, field: dict[str, Any]) -> Decision | None:
        return Decision(self.name, "trace", EdgeKind.CRITIC,
                        "scored execution outcome")
