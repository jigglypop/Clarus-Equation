"""Run the single locked ACBSM development-only promise score."""

import json
from pathlib import Path

from reality_stone.clarus.integrated_latent_state_bridge import run_development


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    report = run_development(
        ROOT / "experiments/preregistration/sparse_causal_bridge_v8.json",
        lock_path=ROOT / "_workspace/ce/agi-acbsm-development-20260811/implementation-lock.json",
        output_path=ROOT / "artifacts/agi/acbsm_core_development_v1.json",
    )
    print(json.dumps(report["promise_score"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
