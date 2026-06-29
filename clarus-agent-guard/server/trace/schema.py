"""DAGlet schema — the core data model of Clarus Agent Guard.

A DAGlet is a *time-expanded causal DAG*: one episode of an agent's
thinking/acting, recorded as the cells that actually fired and the causal
edges between them. The recurrent cell substrate is cyclic; a single run
through it is acyclic. Every product (Trace Guard, Permission Proxy,
Memory Firewall, Metric RAG) is an operation over this object.

Pure stdlib on purpose — this module must never import a web framework or
an LLM client, so the schema stays portable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class EdgeKind(str, Enum):
    """Why one cell handed control to the next. Guards key off these."""

    CONTROL = "control"            # plain sequencing
    MEMORY_READ = "memory_read"    # retrieved past context
    MEMORY_WRITE = "memory_write"  # proposes a long-term memory mutation -> Firewall
    EXTERNAL_ACTION = "external_action"  # tool/side-effect -> Permission Proxy gate
    RETRIEVAL = "retrieval"        # search/RAG -> Metric router
    CRITIC = "critic"              # evaluation feedback


class NodeStatus(str, Enum):
    OK = "ok"
    BLOCKED = "blocked"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Node:
    id: str
    cell: str                       # which cell produced this node
    status: NodeStatus = NodeStatus.OK
    note: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    src: str
    dst: str
    kind: EdgeKind = EdgeKind.CONTROL
    reason: str = ""                # human-readable causal reason


@dataclass
class DAGlet:
    """One recorded execution graph for a single event."""

    id: str
    event: str                       # the triggering input, summarised
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    result: str = ""                 # terminal route, e.g. "draft_only"
    score: float = 0.0               # critic score in [0, 1]
    signature: str = ""              # motif key for similarity lookup
    violations: list[str] = field(default_factory=list)  # invariant breaches

    # --- construction helpers -------------------------------------------
    def add(self, node: Node) -> Node:
        self.nodes.append(node)
        return node

    def link(self, src: str, dst: str, kind: EdgeKind = EdgeKind.CONTROL,
             reason: str = "") -> Edge:
        e = Edge(src=src, dst=dst, kind=kind, reason=reason)
        self.edges.append(e)
        return e

    # --- queries used by guards -----------------------------------------
    def edges_of(self, kind: EdgeKind) -> list[Edge]:
        return [e for e in self.edges if e.kind == kind]

    def failed_nodes(self) -> list[Node]:
        return [n for n in self.nodes
                if n.status in (NodeStatus.FAILED, NodeStatus.BLOCKED)]

    def motif(self) -> str:
        """Structural fingerprint: the cell path, ignoring payloads.

        Two events that flow through the same cells in the same shape share
        a motif, which is how `store.similar()` finds reusable DAGlets.
        """
        return " -> ".join(n.cell for n in self.nodes)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # asdict turns enums into Enum members under some configs; normalise.
        for n in d["nodes"]:
            n["status"] = str(n["status"].value if isinstance(
                n["status"], NodeStatus) else n["status"])
        for e in d["edges"]:
            e["kind"] = str(e["kind"].value if isinstance(
                e["kind"], EdgeKind) else e["kind"])
        return d
