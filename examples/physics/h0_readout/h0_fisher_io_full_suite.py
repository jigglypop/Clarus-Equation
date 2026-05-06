"""Run the full H0 Fisher/covariance IO smoke suite."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PHYSICS = Path(__file__).resolve().parent


COMMANDS = [
    ["h0_fisher_manifest_validate_gate.py"],
    ["h0_fisher_manifest_negative_gate.py"],
    ["h0_fisher_io_validate_gate.py"],
    ["h0_fisher_io_negative_gate.py"],
    ["h0_fisher_io_regression_gate.py"],
    ["h0_fisher_io_batch_gate.py"],
    ["h0_tdcosmo_factor_role_gate.py"],
    ["h0_tdcosmo_notebook_factor_extract_gate.py"],
    ["h0_tdcosmo_role_transition_gate.py"],
    ["h0_tdcosmo_role_ablation_gate.py"],
]


def main() -> int:
    print("# H0 Fisher IO Full Suite")
    print()
    for command in COMMANDS:
        script = PHYSICS / command[0]
        print(f"## {script.name}")
        print()
        completed = subprocess.run(
            [sys.executable, str(script), *command[1:]],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.stdout:
            print(completed.stdout.strip())
            print()
        if completed.stderr:
            print("stderr:")
            print(completed.stderr.strip())
            print()
        if completed.returncode != 0:
            print(f"FAILED: {script.name} exited with {completed.returncode}")
            return completed.returncode
    print("Verdict: full Fisher/covariance IO suite passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
