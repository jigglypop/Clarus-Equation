"""Identifiability checks for the brain inverse maps.

Inverse formulas are only meaningful when the experiment design has enough
independent variation. This script checks rank and conditioning for the current
pilot designs:

1. Task sensitivity beta_r.
2. Background-reserve sensitivity chi_r.
3. Homeostatic sensitivity d_r.
4. Graph temporal-mode inverse for rho_B and gamma.

The important result is not just pass/fail. It also shows when a design can
identify only a reduced set of axes and needs extra interventions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


TASK_LOADS = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
REST_BURDENS = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)

HOMEOSTATIC_AXES = ["sleep", "arousal", "autonomic", "endocrine", "immune", "metabolic"]
CURRENT_HOMEOSTATIC_DESIGN = {
    "normal": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    "sleep_deprivation": [0.34, 0.16, 0.08, 0.06, 0.03, 0.10],
    "high_arousal_task": [0.04, 0.38, 0.14, 0.10, 0.02, 0.08],
    "metabolic_stress": [0.08, 0.10, 0.12, 0.08, 0.06, 0.36],
    "recovery_sleep": [-0.18, -0.06, -0.04, -0.03, 0.00, -0.05],
}
EXTENDED_HOMEOSTATIC_DESIGN = {
    **CURRENT_HOMEOSTATIC_DESIGN,
    "autonomic_challenge": [0.02, 0.12, 0.42, 0.08, 0.03, 0.05],
    "endocrine_stress": [0.06, 0.10, 0.08, 0.44, 0.08, 0.06],
    "immune_challenge": [0.04, 0.06, 0.06, 0.10, 0.40, 0.08],
}

GRAPH_LAMBDAS = np.asarray(
    [0.0, 0.3435455646714175, 0.9936188679366691, 1.4854728726731559, 1.9944215472200446],
    dtype=np.float64,
)


def linear_design(values: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones_like(values), values])


def positive_matrix(design: dict[str, list[float]]) -> tuple[list[str], np.ndarray]:
    names = list(design)
    matrix = np.asarray([np.maximum(design[name], 0.0) for name in names], dtype=np.float64)
    # The all-zero normal row does not harm rank, but it can make condition
    # numbers look less interpretable for the nonzero intervention subspace.
    nonzero = matrix[np.linalg.norm(matrix, axis=1) > 0.0]
    return names, nonzero


def matrix_report(matrix: np.ndarray, expected_columns: int) -> dict[str, object]:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.linalg.matrix_rank(matrix))
    positive_singular = singular_values[singular_values > 1e-12]
    condition = float(positive_singular[0] / positive_singular[-1]) if len(positive_singular) else float("inf")
    return {
        "shape": list(matrix.shape),
        "rank": rank,
        "expected_columns": expected_columns,
        "full_column_rank": rank == expected_columns,
        "singular_values": [float(value) for value in singular_values],
        "condition_number": condition,
    }


def graph_temporal_design(lambdas: np.ndarray) -> np.ndarray:
    # mu_k = rho_B - gamma lambda_k. Columns identify rho_B and gamma.
    nonzero = lambdas[lambdas > 1e-12]
    return np.column_stack([np.ones_like(nonzero), -nonzero])


def main() -> None:
    task = matrix_report(linear_design(TASK_LOADS), expected_columns=2)
    rest = matrix_report(linear_design(REST_BURDENS), expected_columns=2)

    _, current_homeostatic = positive_matrix(CURRENT_HOMEOSTATIC_DESIGN)
    _, extended_homeostatic = positive_matrix(EXTENDED_HOMEOSTATIC_DESIGN)
    current_homeostasis = matrix_report(current_homeostatic, expected_columns=len(HOMEOSTATIC_AXES))
    extended_homeostasis = matrix_report(extended_homeostatic, expected_columns=len(HOMEOSTATIC_AXES))

    graph_modes = matrix_report(graph_temporal_design(GRAPH_LAMBDAS), expected_columns=2)

    out = {
        "task_beta_design": task,
        "background_chi_design": rest,
        "current_homeostatic_design": current_homeostasis,
        "extended_homeostatic_design": extended_homeostasis,
        "graph_temporal_design": graph_modes,
        "interpretation": {
            "task_beta": "three graded loads identify intercept and beta_r",
            "background_chi": "three burden levels identify intercept and chi_r",
            "current_homeostasis": "current sleep/arousal/metabolic pilot is rank-deficient for all six q axes",
            "extended_homeostasis": "adding autonomic, endocrine, and immune challenges gives full column rank",
            "graph_temporal": "two or more nonzero graph modes identify rho_B and gamma",
        },
        "required_next_interventions": [
            "autonomic challenge for q_aut",
            "endocrine stress contrast for q_endo",
            "immune challenge for q_immune",
        ],
    }

    out_path = Path(__file__).with_name("brain_identifiability_gate_results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("Brain identifiability gate")
    print(f"  task beta rank: {task['rank']}/2, condition={task['condition_number']:.4f}")
    print(f"  background chi rank: {rest['rank']}/2, condition={rest['condition_number']:.4f}")
    print(
        "  current homeostasis rank: "
        f"{current_homeostasis['rank']}/6, condition={current_homeostasis['condition_number']:.4f}"
    )
    print(
        "  extended homeostasis rank: "
        f"{extended_homeostasis['rank']}/6, condition={extended_homeostasis['condition_number']:.4f}"
    )
    print(f"  graph temporal rank: {graph_modes['rank']}/2, condition={graph_modes['condition_number']:.4f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
