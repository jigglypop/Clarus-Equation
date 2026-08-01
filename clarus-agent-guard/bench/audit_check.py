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
from server.trace.store import STORE, AuditError


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

    # 1b) FAIL-CLOSED: committing that bypass must be REFUSED, not recorded.
    bad.violations = v
    n_before = len(STORE.all())
    log_before = len(STORE.audit_log())
    refused = False
    try:
        STORE.commit(bad)  # enforce defaults to True
    except AuditError as e:
        refused = True
        assert e.violations == v
    print("fail-closed commit refused:", refused)
    assert refused, "fail-open: a breaching DAGlet was committed"
    # no partial state: it never entered the store, but it IS in the log.
    assert STORE.get("bad") is None, "partial state: bypass leaked into store"
    assert len(STORE.all()) == n_before, "store size changed on a refused commit"
    assert len(STORE.audit_log()) == log_before + 1, "breach not audit-logged"
    print("post-refusal store size    :", len(STORE.all()), "(unchanged)")

    # 1c) explicit diagnostic opt-out still logs but is allowed through.
    diag = DAGlet(id="diag", event="(diagnostic opt-out)")
    diag.add(Node(id="router", cell="router"))
    diag.add(Node(id="answer", cell="answer"))
    diag.link("router", "answer", EdgeKind.EXTERNAL_ACTION, "skip the gate")
    diag.result = "answer"
    diag.violations = audit(diag)
    STORE.commit(diag, enforce=False)
    assert STORE.get("diag") is not None, "enforce=False should permit commit"
    print("enforce=False opt-out       : committed + logged")

    # 2) a real gated action must be clean (no false-block)
    good = run_event("이 메일 발송해줘", {"tool": "send_email"})
    print("gated action  result      :", good.result)
    print("gated action  violations  :", good.violations)
    assert good.violations == [], "audit false-positive on a gated action"

    print("\nINVARIANT: [HOLDS]  (bypass refused fail-closed, gated action clean)")


if __name__ == "__main__":
    main()
