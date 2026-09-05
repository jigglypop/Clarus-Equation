"""동일한 25-cell 양의 측도에서 국소 접힘과 상위 접힘의 조건부 합성을 검사한다.

Post-refutation diagnostic, not a preregistered prediction or a repaired F-01.
The global Wick measure is supplied once; no independent coarse measure is reset.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

from examples.physics.gravity.gaussian_fold import log_density_at_zero, split_log_density, stationary_scale

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "F-01" / "predict_fold_budget.py"
spec = importlib.util.spec_from_file_location("fold_budget_reference", SOURCE)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def row_basis(rows):
    _, singular, vt = np.linalg.svd(rows, full_matrices=False)
    rank = int(np.sum(singular > 1e-9 * singular[0]))
    return vt[:rank].T


def run():
    points = reference.points_from_squared(np.full(10, 2.0))
    top = reference.refine([tuple(range(5))], points)
    cells = reference.refine(top, points)
    kappas = reference.equal_split_kappas(cells, tuple(range(5)), np.full(10, math.pi))
    precision = np.zeros((250, 250))
    for i, (cell, kappa) in enumerate(zip(cells, kappas)):
        hessian = reference.simplex_hessian(reference.cell_lengths(cell, points), kappa)
        values, vectors = np.linalg.eigh(hessian)
        precision[i*10:(i+1)*10, i*10:(i+1)*10] = (vectors * np.abs(values)) @ vectors.T
    rows = np.vstack([reference.gluing_rows(cells), reference.gauge_directions(cells, points, list(range(5, 11))).T])
    global_basis = row_basis(rows)
    local_rows = []
    for i in range(5):
        subcells = cells[i*5:(i+1)*5]
        constraints = np.vstack([reference.gluing_rows(subcells), reference.gauge_directions(subcells, points, [6+i]).T])
        embedded = np.zeros((len(constraints), 250))
        embedded[:, i*50:(i+1)*50] = constraints
        local_rows.append(embedded)
    local = row_basis(np.vstack(local_rows))
    remainder = global_basis - local @ (local.T @ global_basis)
    outer = row_basis(remainder.T)
    # row_basis expects row vectors, so this yields the orthogonal outer directions.
    joined = np.column_stack([local, outer])
    covariance = joined.T @ np.linalg.solve(precision, joined)
    symmetry_residual = float(np.linalg.norm(covariance-covariance.T)/np.linalg.norm(covariance))
    if symmetry_residual > 1e-10:
        raise RuntimeError("projected covariance lost symmetry")
    covariance = (covariance+covariance.T)/2
    direct = log_density_at_zero(covariance)
    first, conditional = split_log_density(covariance, local.shape[1])
    independent_second = log_density_at_zero(covariance[local.shape[1]:, local.shape[1]:])
    global_covariance = global_basis.T @ np.linalg.solve(precision, global_basis)
    global_covariance = (global_covariance+global_covariance.T)/2
    global_log = log_density_at_zero(global_covariance)
    spectrum = np.linalg.eigvalsh(global_covariance)
    action = 10 * math.sqrt(3) / 4 * (math.pi-math.acos(0.25))
    result = {
        "scope": "post-refutation diagnostic; supplied global positive Wick measure, dimensionless coordinates, kappa=1",
        "python": sys.executable, "python_version": sys.version.split()[0], "numpy": np.__version__,
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "ranks": {"global": global_basis.shape[1], "local": local.shape[1], "outer": outer.shape[1]},
        "subspace_residual": float(np.linalg.norm(local-global_basis @ (global_basis.T @ local))),
        "symmetry_residual": symmetry_residual,
        "orthogonality_residual": float(np.linalg.norm(joined.T @ joined-np.eye(joined.shape[1]))),
        "global_basis_log_density": global_log,
        "direct_log_density": direct, "local_log_density": first,
        "conditional_outer_log_density": conditional,
        "reset_outer_log_density": independent_second,
        "chain_residual": direct-first-conditional,
        "basis_residual": direct-global_log,
        "correlation_correction": conditional-independent_second,
        "d_W": global_log+0.5*len(global_covariance)*math.log(2*math.pi),
        "soft_fold_stationary_t": {str(eps): stationary_scale(spectrum, action, eps) for eps in (0.1, 1.0, 10.0)},
        "soft_fold_action_assumption": "positive boundary cost s=10 sqrt(3)/4 (pi-acos(1/4)); kappa=t/(8*pi)",
        "hard_density_stationary_t_upper_bound": 4*math.pi*len(spectrum)/action,
        "physical_measure_uniqueness_proved": False,
        "common_metric_attractor_proved": False,
    }
    if result["ranks"] != {"global": 234, "local": 195, "outer": 39}:
        raise RuntimeError(f"unexpected ranks: {result['ranks']}")
    if max(abs(result[k]) for k in ("subspace_residual", "orthogonality_residual", "chain_residual", "basis_residual")) > 1e-7:
        raise RuntimeError("conditional composition check failed")
    return result


if __name__ == "__main__":
    result = run()
    (HERE / "conditional_composition.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
