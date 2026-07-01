"""ClarusGuard — the developer-facing facade.

Composes the four cores (interception gate, capability enforcement, DAGlet
trace, memory firewall) behind one ergonomic API so a real agent can adopt
the guard in a few lines:

    guard = ClarusGuard(db_path="guard.db")

    @guard.tool(side_effecting=True, cap=Capability.SEND_EMAIL,
                critical_args=("to",))
    def send_email(to, body): ...

    # in the agent loop, every tool call goes through the guard:
    r = guard.call("send_email", user_text=user_msg,
                   args={"to": recipient, "body": draft})
    if r.status == "pending":            # human-in-the-loop
        guard.approve(r.token)

Guarantees inherited from the cores:
  - injected (non-user) requests can never mint a capability  -> refused
  - side effects with a granted capability still wait for approval
  - read-only tools execute through the single chokepoint
  - every decision is persisted (survives restart) and auditable
"""

from __future__ import annotations

import itertools
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

from .capability import Capability, CapabilityError, Value
from .executor import authorize
from .interception import ToolGate
from .memory_firewall import MemoryStore, MemoryUpdate

_ids = itertools.count(1)


@dataclass
class Result:
    status: str                       # executed | refused | pending
    reason: str = ""
    value: Any = None
    token: str = ""
    record_id: int = 0


class ClarusGuard:
    def __init__(self, db_path: str = ":memory:",
                 require_approval: bool = True) -> None:
        self.gate = ToolGate()
        self.memory = MemoryStore()
        self.require_approval = require_approval
        self._pending: dict[str, tuple] = {}
        self._db = sqlite3.connect(db_path)
        # WAL + NORMAL sync: ~160x faster than the default commit-per-call
        # (4.2ms -> 26us/call on a file db) while staying crash-safe.
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS trace ("
            "id INTEGER PRIMARY KEY, ts REAL, user_text TEXT, tool TEXT, "
            "provenance TEXT, decision TEXT, reason TEXT)")
        self._db.commit()

    # --- registration ---------------------------------------------------
    def tool(self, *, side_effecting: bool = False,
             cap: Capability | None = None,
             critical_args: tuple[str, ...] = ()):
        return self.gate.guarded(side_effecting=side_effecting,
                                 required_cap=cap, critical_args=critical_args)

    # --- the single call path -------------------------------------------
    def call(self, name: str, *, user_text: str = "",
             args: dict | None = None, provenance: str = "user") -> Result:
        args = args or {}
        spec = self.gate._ex.tool(name)
        granted = authorize(user_text) if provenance == "user" else set()
        vals: dict[str, Value] = {}
        for k, v in args.items():
            if isinstance(v, tuple) and len(v) == 2:
                vals[k] = Value.from_provenance(v[0], v[1])
            else:
                vals[k] = Value.from_provenance(v, provenance)

        try:
            if spec is not None and spec.side_effecting and self.require_approval:
                self.gate._ex.execute(name, vals, granted, dry=True)  # validate
                token = f"tok{next(_ids)}"
                self._pending[token] = (name, vals, granted)
                rid = self._record(user_text, name, provenance,
                                   "pending", "authorised; awaiting approval")
                return Result("pending", "authorised; awaiting approval",
                              token=token, record_id=rid)
            value = self.gate._ex.execute(name, vals, granted)
            rid = self._record(user_text, name, provenance, "executed", "")
            return Result("executed", value=value, record_id=rid)
        except CapabilityError as e:
            rid = self._record(user_text, name, provenance, "refused", str(e))
            return Result("refused", reason=str(e), record_id=rid)

    def approve(self, token: str) -> Result:
        if token not in self._pending:
            return Result("refused", reason="unknown or used token")
        name, vals, granted = self._pending.pop(token)
        value = self.gate._ex.execute(name, vals, granted)
        rid = self._record("(approval)", name, "user", "executed", "approved")
        return Result("executed", value=value, record_id=rid)

    # --- memory firewall passthrough ------------------------------------
    def remember(self, key: str, value: str, *, provenance: str = "user",
                 episode: str | None = None):
        if episode:
            self.memory.add_episode(episode)
        return self.memory.commit(MemoryUpdate(key, value, provenance, episode))

    # --- persistence / audit --------------------------------------------
    def _record(self, user_text, tool, provenance, decision, reason) -> int:
        cur = self._db.execute(
            "INSERT INTO trace (ts,user_text,tool,provenance,decision,reason) "
            "VALUES (?,?,?,?,?,?)",
            (time.time(), user_text, tool, provenance, decision, reason))
        self._db.commit()
        return cur.lastrowid

    def history(self) -> list[dict]:
        cur = self._db.execute(
            "SELECT id,tool,provenance,decision,reason FROM trace ORDER BY id")
        return [dict(zip(("id", "tool", "provenance", "decision", "reason"), r))
                for r in cur.fetchall()]

    def audit(self) -> list[dict]:
        return [r for r in self.history() if r["decision"] == "refused"]
