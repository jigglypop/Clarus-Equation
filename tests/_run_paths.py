"""Resolve CE run directories that may have been archived by run.sh gc."""
from pathlib import Path

_WS = Path(__file__).resolve().parents[1] / "_workspace" / "ce"


def run_dir(name: str) -> Path:
    live = _WS / name
    return live if live.exists() else _WS / "_archive" / name
