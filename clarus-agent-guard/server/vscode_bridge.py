"""JSON-line stdio bridge so an editor plugin can drive ClarusGuard tools
without an HTTP server. Spawn once, keep stdin/stdout open, one request per
line in each direction.

Request  -> {"tool": "write_file", "args": {...}, "user_text": "...", "provenance": "user"}
Approve  -> {"approve": "tok3"}
Response <- {"status": "executed"|"refused"|"pending", "reason": "", "value": ..., "token": ""}

The tool bodies here are the REAL side effects (file write, shell command) —
this is the only place in the extension stack that touches disk/process, and
every call to it goes through ClarusGuard first.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .capability import Capability
from .sdk import ClarusGuard

guard = ClarusGuard(db_path="guard.db")


@guard.tool(side_effecting=False)
def read_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


@guard.tool(side_effecting=False)
def list_dir(path: str) -> list[str]:
    return sorted(p.name for p in Path(path).iterdir())


@guard.tool(side_effecting=True, cap=Capability.WRITE_FILE, critical_args=("path",))
def write_file(path: str, content: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} bytes to {path}"

@guard.tool(side_effecting=True, cap=Capability.RUN_COMMAND, critical_args=("command",))
def run_command(command: str, cwd: str | None = None) -> str:
    r = subprocess.run(command, shell=True, capture_output=True, text=True,
                       timeout=120, cwd=cwd or None)
    return (r.stdout + r.stderr)[-8000:]


def _handle(req: dict) -> dict:
    if "approve" in req:
        r = guard.approve(req["approve"])
    else:
        r = guard.call(req["tool"], user_text=req.get("user_text", ""),
                       args=req.get("args", {}),
                       provenance=req.get("provenance", "user"))
    return {"status": r.status, "reason": r.reason, "value": r.value,
            "token": r.token}


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            out = _handle(json.loads(line))
        except Exception as e:  # noqa: BLE001 - bridge must never crash on bad input
            out = {"status": "refused", "reason": f"bridge error: {e}",
                   "value": None, "token": ""}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
