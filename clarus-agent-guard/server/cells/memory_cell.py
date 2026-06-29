"""MemoryCell — emits a memory_read edge (retrieval) and, when the event
mutates state, a memory_write edge that the Firewall will later verify.
Stub retrieval for the MVP; replace with the real store/embedder.
"""

from __future__ import annotations

from typing import Any

from . import Cell, Decision
from ..trace.schema import EdgeKind


class MemoryCell(Cell):
    name = "memory"

    def fire(self, event: str, field: dict[str, Any]) -> Decision | None:
        # pretend retrieval succeeded; carry a stub context forward.
        field["recalled"] = f"context for: {event[:40]}"
        return Decision(self.name, "answer", EdgeKind.MEMORY_READ,
                        "retrieved relevant past episode",
                        data={"recalled": field["recalled"]})
