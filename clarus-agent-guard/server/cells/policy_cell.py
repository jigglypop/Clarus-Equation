"""PolicyCell — the Permission Proxy gate. Any external-action edge passes
through here. Default stance: never auto-allow a side effect; downgrade to
a draft and require approval. Rules load from policies/default.yaml.
"""

from __future__ import annotations

from typing import Any

from . import Cell, Decision
from ..trace.schema import EdgeKind


class PolicyCell(Cell):
    name = "policy"

    def __init__(self, allow: set[str] | None = None) -> None:
        # tool names that are safe to auto-allow without approval
        self.allow = allow or set()

    def fire(self, event: str, field: dict[str, Any]) -> Decision | None:
        # control/data separation: an action surfaced from untrusted
        # content is quarantined regardless of the allow-list. This is the
        # defense against indirect prompt injection.
        if field.get("quarantine"):
            return Decision(self.name, "draft", EdgeKind.CONTROL,
                            "action found in untrusted content -> quarantined",
                            note="blocked")
        tool = field.get("tool", "")
        if tool in self.allow:
            return Decision(self.name, "answer", EdgeKind.EXTERNAL_ACTION,
                            f"'{tool}' is allow-listed -> execute")
        # block the side effect, fall back to a reviewable draft
        return Decision(self.name, "draft", EdgeKind.CONTROL,
                        "external action without approval -> draft_only",
                        note="blocked")
