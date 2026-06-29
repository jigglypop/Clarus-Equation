"""Scheduler — turns an event into a DAGlet by walking the cell substrate.

This is the heart of the runtime. It does NOT execute a fixed workflow
graph; it starts at salience and follows whichever cell each decision
points to, recording every fired cell as a node and every decision as a
causal edge. The same substrate yields different DAGlets for different
events — that is the whole point.

The walk is intrinsically acyclic here (each cell points forward), so one
event = one time-expanded causal DAG.
"""

from __future__ import annotations

import itertools
from typing import Any

from .cells.salience_cell import SalienceCell
from .cells.router_cell import RouterCell
from .cells.policy_cell import PolicyCell
from .cells.memory_cell import MemoryCell
from .cells.critic_cell import CriticCell
from .cells.trace_cell import TraceCell
from .trace.audit import audit
from .trace.schema import DAGlet, Node, NodeStatus
from .trace.store import STORE

_ids = itertools.count(1)

# the substrate: logical target -> cell instance
_CELLS = {
    "salience": SalienceCell(),
    "router": RouterCell(),
    "policy": PolicyCell(),
    "memory": MemoryCell(),
    "critic": CriticCell(),
    "trace": TraceCell(),
}
_CRITIC = _CELLS["critic"]

# routes that are terminal "leaf" outcomes, not cells to fire
_TERMINALS = {"answer", "search", "draft"}


def run_event(event: str, field: dict[str, Any] | None = None) -> DAGlet:
    field = field or {}
    daglet = DAGlet(id=f"d_{next(_ids):04d}", event=event)

    # check memory M first: has this kind of event burned us before?
    reuse_hint = None

    cursor = "salience"
    daglet.add(Node(id="salience", cell="salience"))
    guard = 0
    while cursor and cursor not in _TERMINALS and guard < 32:
        guard += 1
        cell = _CELLS.get(cursor)
        if cell is None:
            break
        decision = cell.fire(event, field)
        if decision is None:
            break
        nxt = decision.next
        # record the edge with its causal reason
        daglet.link(cursor, nxt or "trace", decision.kind, decision.reason)
        if decision.note == "blocked":
            for n in daglet.nodes:
                if n.id == cursor:
                    n.status = NodeStatus.BLOCKED
        if nxt and nxt not in {n.id for n in daglet.nodes}:
            daglet.add(Node(id=nxt, cell=nxt if nxt in _CELLS else nxt,
                            data=decision.data))
        cursor = nxt

    daglet.result = cursor or "answer"

    # critic scores the resolved path, then we consult/record memory.
    daglet.score = _CRITIC.score(daglet.result, field)
    prior = STORE.similar(daglet.motif())
    if prior:
        reuse_hint = {"seen": len(prior), "best_score": prior[0].score}
        daglet.nodes[0].data["reuse"] = reuse_hint

    # structural invariant: no labelled action bypassed the gate.
    daglet.violations = audit(daglet)

    STORE.commit(daglet)
    return daglet
