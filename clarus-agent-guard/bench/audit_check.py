"""Invariant test — prove the audit catches a gate bypass that detection
would miss.

    python -m bench.audit_check

We hand-build a malicious DAGlet where an external_action edge runs
straight to execution with NO policy node, and assert audit() fires. Then
we confirm a normal gated DAGlet is clean.
"""

from __future__ import annotations

from server.scheduler import run_event
from server.trace.audit import audit
from server.trace.schema import DAGlet, Node, EdgeKind


def main() -> None:
    # 1) crafted bypass: action -> answer, no PolicyCell in the graph
    bad = DAGlet(id="bad", event="(crafted bypass)")
    bad.add(Node(id="router", cell="router"))
    bad.add(Node(id="answer", cell="answer"))
    bad.link("router", "answer", EdgeKind.EXTERNAL_ACTION, "skip the gate")
    bad.result = "answer"
    v = audit(bad)
    print("crafted bypass violations :", v)
    assert v, "audit FAILED to catch a gate bypass"

    # 2) a real gated action must be clean
    good = run_event("이 메일 발송해줘", {"tool": "send_email"})
    print("gated action  result      :", good.result)
    print("gated action  violations  :", good.violations)
    assert good.violations == [], "audit false-positive on a gated action"

    print("\nINVARIANT: [HOLDS]  (bypass caught, gated action clean)")


if __name__ == "__main__":
    main()
