"""Graph invariant: no labelled external action may bypass the PolicyCell.

This is a *structural* guarantee, distinct from detection. Detection (did
we correctly recognise an action?) is the rule cells' job and is fallible.
This invariant says: GIVEN an edge is labelled `external_action`, the
DAGlet must contain a `policy` node — i.e. the action provably passed
through the gate. If a future cell ever wires an action straight to
execution, this fires at commit time instead of leaking silently.

Because the scheduler always routes risk -> policy, normal DAGlets satisfy
this by construction; the checker exists to keep it true as cells change.
"""

from __future__ import annotations

from .schema import DAGlet, EdgeKind


def audit(d: DAGlet) -> list[str]:
    violations: list[str] = []
    has_policy = any(n.id == "policy" for n in d.nodes)
    executed = d.result not in {"draft"}

    for e in d.edges_of(EdgeKind.EXTERNAL_ACTION):
        # an external_action edge that reaches execution without a policy
        # node anywhere in the graph is a gate bypass.
        if e.dst != "policy" and not has_policy:
            violations.append(
                f"external_action {e.src}->{e.dst} bypassed PolicyCell")
        # an external_action that executed must have been gated.
        if executed and not has_policy:
            violations.append(
                f"external_action {e.src}->{e.dst} executed without gate")
    # de-dup while preserving order
    return list(dict.fromkeys(violations))
