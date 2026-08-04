"""Diffusion orchestration scheduler (docs/7_AGI/20_DiffusionOrchestration.md).

A *candidate* alternative to the sequential cell walk in `scheduler.py`. It
computes an orchestration prior pi over the cell substrate by diffusing a
field on the substrate graph G:

    d phi / d tau = -Delta_G phi + s - lambda phi   ->   phi*,  pi = softmax(phi*/T)

STATUS (honest): this is `Open`/`Hypothesis`. The diffusion does NOT yet
*replace* routing (rules 1-2 in the spec are unproven). It overlays pi as an
orchestration prior while the action->policy execution path is still produced
by the same safe cell walk, so the structural gate invariant
(`trace/audit.py`) holds by construction. No performance claim is made here;
this is the scaffolding that makes the A/B measurable.

Pure stdlib on purpose (mirrors scheduler.py): no numpy, |V| is tiny.
"""

from __future__ import annotations

import itertools
import math
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

_CELLS = {
    "salience": SalienceCell(),
    "router": RouterCell(),
    "policy": PolicyCell(),
    "memory": MemoryCell(),
    "critic": CriticCell(),
    "trace": TraceCell(),
}
_CRITIC = _CELLS["critic"]
_TERMINALS = {"answer", "search", "draft"}

# substrate graph G: undirected "possible connections" between cells. This is
# the cell topology the sequential walk can traverse (salience seeds, router
# fans out to policy/memory, every path ends through critic -> trace).
SUBSTRATE_ADJ: dict[str, list[str]] = {
    "salience": ["router"],
    "router": ["salience", "policy", "memory", "search", "answer", "critic"],
    "policy": ["router", "critic", "trace"],
    "memory": ["router", "critic"],
    "search": ["router", "critic"],
    "answer": ["router"],
    "critic": ["router", "policy", "memory", "search", "trace"],
    "trace": ["policy", "critic"],
}

# diffusion route-target node -> the bench route label it stands for.
_ROUTE_NODES = {"policy": "verify", "memory": "memory",
                "search": "search", "answer": "answer"}


def diffuse(
    source: dict[str, float],
    *,
    lam: float = 0.5,
    steps: int = 256,
    dt: float = 0.1,
    adj: dict[str, list[str]] | None = None,
) -> dict[str, float]:
    """Integrate phi_tau = -Delta_G phi + s - lambda phi to steady state.

    Delta_G is the normalized symmetric graph Laplacian; its action is
        (Delta_G phi)_v = phi_v - (1/sqrt(d_v)) * sum_{u~v} phi_u / sqrt(d_u).
    Explicit Euler is fine here: G is tiny and lambda>0 guarantees a unique,
    bounded steady state (spec section 2). Everything is dimensionless.
    """
    adj = adj or SUBSTRATE_ADJ
    nodes = list(adj)
    deg = {v: max(len(adj[v]), 1) for v in nodes}
    phi = {v: 0.0 for v in nodes}
    s = {v: float(source.get(v, 0.0)) for v in nodes}
    for _ in range(max(1, steps)):
        lap = {}
        for v in nodes:
            neigh = sum(phi[u] / math.sqrt(deg[u]) for u in adj[v])
            lap[v] = phi[v] - neigh / math.sqrt(deg[v])
        phi = {v: phi[v] + dt * (-lap[v] + s[v] - lam * phi[v]) for v in nodes}
    return phi


def readout(phi: dict[str, float], *, temperature: float = 0.5) -> dict[str, float]:
    """pi_v = softmax(phi_v / T): normalized activation priority (spec sec 3)."""
    t = max(temperature, 1e-6)
    mx = max(phi.values()) if phi else 0.0
    exps = {v: math.exp((val - mx) / t) for v, val in phi.items()}
    z = sum(exps.values()) or 1.0
    return {v: e / z for v, e in exps.items()}


def event_source(field: dict[str, Any]) -> dict[str, float]:
    """Event-conditioned source s (spec sec 1): inject mass at the cells whose
    salience flags fired, so phi* concentrates on the event-relevant cells.
    Without this, s is the same for every event and pi carries no event
    information (the diffusion would be a constant prior)."""
    s = {"salience": 1.0}
    if field.get("risk"):
        s["policy"] = 1.5  # an action draws mass toward the gate
    if field.get("needs_memory"):
        s["memory"] = 1.0
    if field.get("needs_search"):
        s["search"] = 1.0
    if not (field.get("risk") or field.get("needs_memory")
            or field.get("needs_search")):
        s["answer"] = 1.0
    return s


def orchestration_prior(
    event: str,
    field: dict[str, Any],
    *,
    lam: float = 0.5,
    temperature: float = 0.5,
) -> dict[str, float]:
    """Compute the event-conditioned orchestration prior pi."""
    phi = diffuse(event_source(field), lam=lam)
    return readout(phi, temperature=temperature)


def diffusion_route(pi: dict[str, float], field: dict[str, Any]) -> str:
    """Pick a bench route from the diffusion prior pi.

    Safety floor (NOT learned, NOT diffused): a flagged action always routes
    to verify. Diffusion only decides among the non-risk routes. This keeps
    the action->policy invariant independent of the diffusion numbers.
    """
    if field.get("risk"):
        return "verify"
    ranked = sorted(_ROUTE_NODES, key=lambda c: pi.get(c, 0.0), reverse=True)
    for cell in ranked:
        if cell == "policy":
            continue  # only the risk floor may select verify
        return _ROUTE_NODES[cell]
    return "answer"


def run_event_diffusion(event: str, field: dict[str, Any] | None = None,
                        enforce: bool = True) -> DAGlet:
    """Build a DAGlet using diffusion as an orchestration prior.

    Safety: the execution path is still produced by the same forward cell
    decisions as `scheduler.run_event`, so any `external_action` edge still
    flows through PolicyCell and `audit()` holds by construction. Diffusion
    only annotates pi and records the priority order; it never wires an action
    straight to execution.
    """
    field = field or {}
    daglet = DAGlet(id=f"x_{next(_ids):04d}", event=event)

    # Pre-pass salience once so the field (risk/needs_*) exists before we
    # compute the orchestration prior. This mirrors what the walk would do.
    pre = _CELLS["salience"].fire(event, field)

    pi = orchestration_prior(event, field)
    order = sorted(pi, key=pi.get, reverse=True)

    cursor = "salience"
    salience_node = daglet.add(Node(id="salience", cell="salience"))
    salience_node.data["pi"] = round(pi.get("salience", 0.0), 6)
    salience_node.data["diffusion_order"] = order
    guard = 0
    while cursor and cursor not in _TERMINALS and guard < 32:
        guard += 1
        cell = _CELLS.get(cursor)
        if cell is None:
            break
        decision = pre if (cursor == "salience" and pre is not None) else cell.fire(event, field)
        pre = None
        if decision is None:
            break
        nxt = decision.next
        daglet.link(cursor, nxt or "trace", decision.kind, decision.reason)
        if decision.note == "blocked":
            for n in daglet.nodes:
                if n.id == cursor:
                    n.status = NodeStatus.BLOCKED
        if nxt and nxt not in {n.id for n in daglet.nodes}:
            node = daglet.add(Node(id=nxt, cell=nxt if nxt in _CELLS else nxt,
                                   data=decision.data))
            node.data["pi"] = round(pi.get(nxt, 0.0), 6)
        cursor = nxt

    daglet.result = cursor or "answer"
    daglet.score = _CRITIC.score(daglet.result, field)

    # Same structural invariant as the walk scheduler — must stay 0.
    daglet.violations = audit(daglet)
    # fail-closed, same policy as the walk scheduler.
    STORE.commit(daglet, enforce=enforce)
    return daglet
