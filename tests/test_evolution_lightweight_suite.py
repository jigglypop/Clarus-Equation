"""Regression test for the lightweight evolution closure suite."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_evolution_lightweight_closure_suite_passes() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "examples" / "physics" / "evolution" / "evolution_lightweight_closure_suite.py"

    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    summary = json.loads(completed.stdout)

    assert summary["passed"] is True
    assert summary["passed_count"] == summary["total"] == 17
