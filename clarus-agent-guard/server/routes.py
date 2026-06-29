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

from .replay import replay
from .scheduler import run_event
from .trace.store import STORE

router = APIRouter()


class EventIn(BaseModel):
    event: str
    tool: str | None = None  # if the agent is proposing a specific tool call


@router.post("/event")
def post_event(body: EventIn):
    field = {"tool": body.tool} if body.tool else {}
    d = run_event(body.event, field)
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
    """DAGlets that breached the gate invariant. Should always be empty."""
    bad = [d for d in STORE.all() if d.violations]
    return {"breaches": len(bad),
            "daglets": [{"id": d.id, "event": d.event,
                         "violations": d.violations} for d in bad]}


@router.post("/replay")
def post_replay():
    """Idle consolidation: learn action verbs from past blocked actions."""
    return replay(STORE)
