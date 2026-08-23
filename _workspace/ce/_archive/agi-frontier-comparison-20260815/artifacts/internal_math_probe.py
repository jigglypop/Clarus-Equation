"""Small, read-only probes for the CE-AGI frontier comparison.

RBE is intentionally not imported or inspected.  This script checks only
current non-RBE public objects and committed/result artifacts.
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path

import numpy as np

from reality_stone.clarus.agent import ConsciousnessMonitor
from reality_stone.clarus.episodic_memory_benchmark import evaluate_episodic_memory
from reality_stone.clarus.learnable_small_gain_local_cloud import (
    LearnableSmallGainConfig,
)
from reality_stone.clarus.local_cloud_kernel import LocalCloudTransitionKernel
from reality_stone.clarus.universe_life_kernel import (
    HybridState,
    internal_kernel,
    registered_host_pair,
)


ROOT = Path(__file__).resolve().parents[4]


def determinant_vs_spectral_norm() -> dict[str, object]:
    """Counterexample to |det T|^2 <= 1 iff sigma_max(T) <= 1."""

    matrix = np.diag([2.0, 0.4])
    return {
        "matrix": matrix.tolist(),
        "determinant_squared": float(np.linalg.det(matrix) ** 2),
        "spectral_norm": float(np.linalg.norm(matrix, ord=2)),
        "determinant_condition": bool(np.linalg.det(matrix) ** 2 <= 1.0),
        "spectral_condition": bool(np.linalg.norm(matrix, ord=2) <= 1.0),
    }


def lbonorm_convergence_condition() -> dict[str, object]:
    """Counterexample to the stated eta < 1/lambda_max(V^T V) condition.

    In one dimension V=[2], Delta=I-V^T V=-3.  The documented update
    h' = h-eta*Delta*h has factor 1.6 for eta=0.2, although 0.2 < 1/4.
    """

    v = 2.0
    eigenvalue_vtv = v * v
    eta = 0.2
    update_factor = 1.0 - eta * (1.0 - eigenvalue_vtv)
    return {
        "V": v,
        "lambda_max_VtV": eigenvalue_vtv,
        "eta": eta,
        "declared_condition_holds": eta < 1.0 / eigenvalue_vtv,
        "update_factor": update_factor,
        "is_contractive": abs(update_factor) < 1.0,
    }


def finite_host_boundaries() -> dict[str, object]:
    host = registered_host_pair()[0]
    sigma_zero = internal_kernel(replace(host, sigma=0))
    sigma_one = internal_kernel(replace(host, sigma=1))
    invalid_cube = HybridState(Fraction(2), Fraction(0), Fraction(0))
    return {
        "sigma_zero_and_one_sensor_equal": sigma_zero.sensor == sigma_one.sensor,
        "sigma_zero_and_one_action_equal": sigma_zero.action == sigma_one.action,
        "sigma_is_only_preserved_metadata": sigma_zero.sigma == 0 and sigma_one.sigma == 1,
        "direct_out_of_cube_constructor_accepted": invalid_cube.mass == 2,
    }


def internal_mechanism_snapshot() -> dict[str, object]:
    v10 = LocalCloudTransitionKernel().certificate
    v12 = LearnableSmallGainConfig().certificate()
    monitor = ConsciousnessMonitor()
    return {
        "v10_certificate": {
            "spectral_radius": v10.spectral_radius,
            "q": v10.contraction_factor,
            "certified": v10.certified,
        },
        "v12_config_certificate": {
            "spectral_radius": v12["spectral_radius"],
            "q": v12["q"],
            "certified": v12["certified"],
        },
        "metacognition_step_from_1": monitor.metacognition_step(1.0),
        "metacognition_is_prescribed_scalar_multiplication": True,
    }


def v13_results() -> dict[str, object]:
    result: dict[str, object] = {}
    for path in sorted((ROOT / "artifacts" / "agi").glob("local_cloud_v13*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        result[path.name] = {
            "overall": payload.get("overall"),
            "gates": payload.get("gates"),
            "seed_count": payload.get("seed_count"),
            "variant": payload.get("variant"),
            "split": payload.get("split"),
        }
    return result


def main() -> None:
    episodic = evaluate_episodic_memory()
    payload = {
        "determinant_vs_spectral_norm": determinant_vs_spectral_norm(),
        "lbonorm_convergence_condition": lbonorm_convergence_condition(),
        "finite_host_boundaries": finite_host_boundaries(),
        "mechanisms": internal_mechanism_snapshot(),
        "episodic_memory": {
            "hard_gate": episodic["hard_gate"],
            "grade": episodic["grade"],
            "claim_limit": episodic["claim_limit"],
            "candidate_means": episodic["means"]["candidate"],
            "lcb_candidate_composite_minus": episodic[
                "lcb_candidate_composite_minus"
            ],
        },
        "v13": v13_results(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
