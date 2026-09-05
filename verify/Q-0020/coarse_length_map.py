"""상위 변 길이 사상과 두 게이지 절단면의 측도를 비교하는 사후 진단.

This constructs geometry maps, not a new physical measure or a preregistered prediction.
"""

from __future__ import annotations

import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

from conditional_composition import reference as r, row_basis

HERE = Path(__file__).resolve().parent


def kernel(rows):
    _, singular, vt = np.linalg.svd(rows, full_matrices=True)
    rank = int(np.sum(singular > 1e-9 * singular[0]))
    return vt[rank:].T


def edge_readout(cells, edges):
    readout = np.zeros((len(edges), 10*len(cells)))
    for i, edge in enumerate(edges):
        wanted = set(edge)
        occurrences = [10*a+j for a, cell in enumerate(cells)
                       for j, current in enumerate(combinations(cell, 2)) if set(current) == wanted]
        if not occurrences:
            raise ValueError(f"edge {edge} has no fine representative")
        readout[i, occurrences] = 1/len(occurrences)
    return readout


def covariance_on_slice(basis, precision):
    return basis @ np.linalg.solve(basis.T @ precision @ basis, basis.T)


def relative_difference(left, right):
    return float(np.linalg.norm(left-right)/np.linalg.norm(right))


def marginalized_covariance(basis, gauge, precision):
    """게이지로 명명한 방향을 영으로 고정하지 않고 적분한 좌표 공분산."""
    a = basis.T @ precision @ basis
    b = basis.T @ precision @ gauge
    d = gauge.T @ precision @ gauge
    schur = a-b @ np.linalg.solve(d,b.T)
    return basis @ np.linalg.solve(schur,basis.T)


def run():
    points = r.points_from_squared(np.full(10, 2.0))
    top = r.refine([tuple(range(5))], points)
    cells = r.refine(top, points)
    local_rows = []
    e = np.zeros((50, 250))
    for i, parent in enumerate(top):
        children = cells[5*i:5*i+5]
        rows = np.vstack([r.gluing_rows(children), r.gauge_directions(children, points, [6+i]).T])
        embedded = np.zeros((len(rows), 250))
        embedded[:, 50*i:50*i+50] = rows
        local_rows.append(embedded)
        e[10*i:10*i+10,50*i:50*i+50] = edge_readout(children, list(combinations(parent, 2)))
    local_rows = np.vstack(local_rows)
    local_kernel = kernel(local_rows)
    top_glue = r.gluing_rows(top) @ e
    prior_slice = kernel(np.vstack([local_rows, top_glue]))
    fine_glue = r.gluing_rows(cells)
    sub_gauge = r.gauge_directions(cells, points, list(range(6, 11)))
    direct_prior = kernel(np.vstack([fine_glue, sub_gauge.T]))
    top_gauge = r.gauge_directions(cells, points, [5])
    adjusted_gauge = top_gauge-sub_gauge @ np.linalg.solve(sub_gauge.T @ sub_gauge, sub_gauge.T @ top_gauge)
    f_top = r.gauge_directions(top, points, [5]).T @ e
    global_slice = kernel(np.vstack([fine_glue, sub_gauge.T, top_gauge.T]))
    # This is an explicit change of gauge representative, not a change of geometry.
    shift = adjusted_gauge @ np.linalg.solve(f_top @ adjusted_gauge, f_top)
    transported = global_slice-shift @ global_slice
    hierarchical = kernel(np.vstack([local_rows, top_glue, f_top]))
    kappas = r.equal_split_kappas(cells, tuple(range(5)), np.full(10, np.pi))
    precision = np.zeros((250, 250))
    for i, (cell, kappa) in enumerate(zip(cells, kappas)):
        h = r.simplex_hessian(r.cell_lengths(cell, points), kappa)
        w, v = np.linalg.eigh(h)
        precision[10*i:10*i+10,10*i:10*i+10] = (v*np.abs(w)) @ v.T
    outer = edge_readout(cells, list(combinations(range(5), 2)))
    c_global = outer @ covariance_on_slice(global_slice, precision) @ outer.T
    c_hierarchical = outer @ covariance_on_slice(hierarchical, precision) @ outer.T
    c_transported = outer @ covariance_on_slice(transported, precision) @ outer.T
    c_marginal = outer @ covariance_on_slice(prior_slice, precision) @ outer.T
    c_global_corrected = outer @ marginalized_covariance(global_slice,adjusted_gauge,precision) @ outer.T
    c_hier_corrected = outer @ marginalized_covariance(hierarchical,adjusted_gauge,precision) @ outer.T
    # Integrate all 24 internal displacement directions; do not retain the local slices.
    full_glued = kernel(fine_glue)
    c_fine_all = outer @ covariance_on_slice(full_glued,precision) @ outer.T
    top_precision = np.zeros((50,50))
    top_kappas = r.equal_split_kappas(top,tuple(range(5)),np.full(10,np.pi))
    for i,(cell,kappa) in enumerate(zip(top,top_kappas)):
        h = r.simplex_hessian(r.cell_lengths(cell,points),kappa)
        w,v = np.linalg.eigh(h)
        top_precision[10*i:10*i+10,10*i:10*i+10] = (v*np.abs(w)) @ v.T
    top_boundary = edge_readout(top,list(combinations(range(5),2)))
    top_glued = kernel(r.gluing_rows(top))
    c_top_all = top_boundary @ covariance_on_slice(top_glued,top_precision) @ top_boundary.T
    w,v = np.linalg.eigh((c_top_all+c_top_all.T)/2)
    inverse_root = (v*w**-0.5) @ v.T
    covariance_ratios = np.linalg.eigvalsh(inverse_root @ c_fine_all @ inverse_root)
    result = {
        "scope": "post-refutation diagnostic; regular Euclidean 25 cells; supplied positive Wick precision; kappa=1",
        "python": sys.version.split()[0], "numpy": np.__version__,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(Path(r.__file__).read_bytes()).hexdigest(),
        "local_dimension": int(local_kernel.shape[1]),
        "parent_length_rank": int(np.linalg.matrix_rank(e @ local_kernel, tol=1e-9)),
        "parent_invisible_dimension": int(local_kernel.shape[1]-np.linalg.matrix_rank(e @ local_kernel, tol=1e-9)),
        "before_top_gauge_dimension": int(prior_slice.shape[1]),
        "global_dimension": int(global_slice.shape[1]),
        "hierarchical_dimension": int(hierarchical.shape[1]),
        "gluing_projector_residual": float(np.linalg.norm(prior_slice @ prior_slice.T-direct_prior @ direct_prior.T)),
        "adjusted_gauge_constraint_residual": float(np.linalg.norm(np.vstack([local_rows,top_glue]) @ adjusted_gauge)),
        "gauge_fixing_matrix_rank": int(np.linalg.matrix_rank(f_top @ adjusted_gauge)),
        "gauge_transport_constraint_residual": float(np.linalg.norm(np.vstack([local_rows,top_glue,f_top]) @ transported)),
        "boundary_gauge_shift_residual": float(np.linalg.norm(outer @ adjusted_gauge)),
        "transported_covariance_residual": relative_difference(c_transported,c_hierarchical),
        "boundary_covariance_slice_relative_difference": relative_difference(c_hierarchical,c_global),
        "global_vs_gauge_marginal_relative_difference": relative_difference(c_global,c_marginal),
        "hierarchical_vs_gauge_marginal_relative_difference": relative_difference(c_hierarchical,c_marginal),
        "global_integrated_gauge_residual": relative_difference(c_global_corrected,c_marginal),
        "hierarchical_integrated_gauge_residual": relative_difference(c_hier_corrected,c_marginal),
        "positive_gauge_cost_eigenvalues": np.linalg.eigvalsh(adjusted_gauge.T @ precision @ adjusted_gauge).tolist(),
        "all_gauge_integrated_comparison": {
            "fine_glued_dimension":int(full_glued.shape[1]),
            "top_glued_dimension":int(top_glued.shape[1]),
            "boundary_covariance_relative_difference":relative_difference(c_fine_all,c_top_all),
            "fine_over_top_covariance_eigenvalues":covariance_ratios.tolist(),
            "normalization":"same global dimensionless length coordinates and kappa=1; all non-boundary variables integrated",
            "scope":"post hoc comparison of two supplied Wick model levels, not a universal no-go",
        },
        "physical_measure_derived": False,
        "common_metric_attractor_proved": False,
    }
    expected = {"local_dimension":55,"parent_length_rank":50,"parent_invisible_dimension":5,
                "before_top_gauge_dimension":20,"global_dimension":16,"hierarchical_dimension":16,
                "gauge_fixing_matrix_rank":4}
    if any(result[k]!=v for k,v in expected.items()):
        raise RuntimeError(f"unexpected geometry dimensions: {result}")
    for name in ("gluing_projector_residual","adjusted_gauge_constraint_residual",
                 "gauge_transport_constraint_residual","boundary_gauge_shift_residual","transported_covariance_residual",
                 "global_integrated_gauge_residual","hierarchical_integrated_gauge_residual"):
        if result[name]>1e-7:
            raise RuntimeError(f"geometry identity failed: {name}={result[name]}")
    return result


if __name__ == "__main__":
    result=run()
    (HERE/"coarse_length_map.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(result,indent=2,allow_nan=False))
