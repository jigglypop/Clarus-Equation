"""HTTP surface. FastAPI is optional — the runtime works headless via
`scheduler.run_event`. These routes just expose it.

    POST /event              run an event -> DAGlet
    GET  /daglet/{id}        fetch a recorded DAGlet
    GET  /daglet/similar     past DAGlets sharing a motif
    GET  /trace              list everything recorded
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from .memory_firewall import STORE as MEM, MemoryUpdate
from .replay import replay
from .scheduler import run_event
from .trace.store import STORE, AuditError

router = APIRouter()


class EventIn(BaseModel):
    event: str
    tool: str | None = None  # if the agent is proposing a specific tool call


@router.post("/event")
def post_event(body: EventIn):
    field = {"tool": body.tool} if body.tool else {}
    try:
        d = run_event(body.event, field)
    except AuditError as e:
        # fail-closed: the DAGlet breached a structural invariant and was
        # refused before any store mutation. Report an explicit denial.
        return {"refused": True, "daglet_id": e.daglet_id,
                "violations": e.violations}
    return d.to_dict()


@router.get("/daglet/{daglet_id}")
def get_daglet(daglet_id: str):
    d = STORE.get(daglet_id)
    return d.to_dict() if d else {"error": "not found"}


@router.get("/daglet/similar")
def get_similar(motif: str):
    return [d.to_dict() for d in STORE.similar(motif)]


@router.get("/trace")
def get_trace():
    return [{"id": d.id, "event": d.event, "result": d.result,
             "score": d.score, "motif": d.motif(),
             "violations": d.violations} for d in STORE.all()]


@router.get("/audit")
def get_audit():
    """Every gate-invariant breach the store has refused (fail-closed) or
    logged under an explicit enforce=False opt-out. Committed DAGlets are
    breach-free by construction, so this reads the audit log, not the store.
    """
    log = STORE.audit_log()
    return {"breaches": len(log), "daglets": log}


@router.post("/replay")
def post_replay():
    """Idle consolidation: learn action verbs from past blocked actions."""
    return replay(STORE)


class MemoryIn(BaseModel):
    key: str
    value: str
    provenance: str = "user"
    raw_episode_id: str | None = None


@router.post("/memory/propose")
def post_memory(body: MemoryIn):
    """Propose a long-term memory write; the firewall verifies it first."""
    v = MEM.commit(MemoryUpdate(body.key, body.value, body.provenance,
                                body.raw_episode_id))
    return {"action": v.action, "ok": v.ok, "reasons": v.reasons}
