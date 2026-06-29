"""PolicyCell — the Permission Proxy gate. Any external-action edge passes
through here. Default stance: never auto-allow a side effect; downgrade to
a draft and require approval. Rules load from policies/default.yaml.
"""

from __future__ import annotations

from typing import Any

from . import Cell, Decision
from ..trace.schema import EdgeKind
from ..executor import DEFAULT


class PolicyCell(Cell):
    name = "policy"

    def __init__(self, executor=DEFAULT) -> None:
        self.ex = executor

    def fire(self, event: str, field: dict[str, Any]) -> Decision | None:
        # control/data separation: an action surfaced from untrusted
        # content is quarantined regardless of anything else (indirect PI).
        if field.get("quarantine"):
            return Decision(self.name, "draft", EdgeKind.CONTROL,
                            "action found in untrusted content -> quarantined",
                            note="blocked")

        tool = field.get("tool", "")
        spec = self.ex.tool(tool)
        granted = field.get("granted", set())

        # read-only / unknown-but-non-side-effecting: run through the gate.
        if spec is not None and not spec.side_effecting:
            return Decision(self.name, "answer", EdgeKind.EXTERNAL_ACTION,
                            f"'{tool}' is read-only -> execute via chokepoint")

        # side-effecting tool: capability decides REFUSE vs needs-approval.
        if spec is not None and spec.side_effecting:
            if spec.required_cap not in granted:
                # I1: no capability from the user turn -> structural refusal.
                return Decision(self.name, "draft", EdgeKind.CONTROL,
                                f"'{tool}' refused: no capability "
                                f"({spec.required_cap.value}) from user turn",
                                note="blocked")
            # authorised, but a side effect still needs human approval.
            return Decision(self.name, "draft", EdgeKind.CONTROL,
                            f"'{tool}' authorised -> draft, awaiting approval",
                            note="blocked")

        # no declared tool but text looked risky: default-deny to draft.
        return Decision(self.name, "draft", EdgeKind.CONTROL,
                        "risky action without a declared tool -> draft_only",
                        note="blocked")
