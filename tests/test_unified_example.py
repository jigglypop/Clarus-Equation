from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_unified_reality_stone_clarus_example_runs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "reality_stone" / "examples" / "unified_clarus_demo.py"

    completed = subprocess.run(
        [sys.executable, "-B", str(script)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    summary = json.loads(completed.stdout)

    assert summary["reality_stone"]["version"] == "0.2.10"
    assert summary["clarus_runtime"]["mode"] == "WAKE"
    assert summary["metric_attention"]["finite"] is True
    assert summary["geodesic"]["point"] == [0.5, 0.5]
