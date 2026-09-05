"""Boundary dependence of the supplied fixed-boundary Regge one-loop measure.

Lengths b are dimensionless lengths in a fixed reference unit. beta is held fixed.
Every b defines its own flat barycentric interior branch before taking derivatives.
Phi_D(b)=-log Z_internal,D(b) and Gamma_D=beta*S_coarse+Phi_D.
The log determinant uses the supplied Euclidean orthonormal global-length quotient.
No vertex-coordinate gauge-volume factor or physical measure is derived here.

The local cell Hessian is analytic; boundary derivatives of its quotient determinant
use central differences with Richardson extrapolation. Repeated step sizes are an
empirical convergence check, not a rigorous error bound. Moving gauge subspaces are
included by reconstructing the quotient at every boundary.

At the regular point, permutation symmetry and degree-zero homogeneity imply
grad Phi=0 and H_Phi=0*P1+mu4*P4+mu5*P5. The code checks this prediction against
independent directional differences; it does not identify common-metric selection
with selection of one regular shape. Length gluing is already imposed.

A shape curvature of Gamma below uses the AFFINE constraint sum(b)=constant.
Fixed volume or fixed norm are different curved constraints because grad S is radial
and nonzero. Positive curvature on that affine slice is neither a normalized global
boundary measure nor a controlled nonperturbative stability result. In particular,
loop/classical competition alone cannot establish validity of the loop expansion.

References: arXiv:1110.6866 (flat Regge moves), arXiv:1404.5288 (one-loop measure).
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np

import fixed_boundary_composition as fixed


HERE = Path(__file__).resolve().parent
REGULAR = np.full(10, math.sqrt(2.))
EDGES = fixed.EDGES


def shape_projectors():
    incidence = np.array([[float(v in edge) for v in range(5)] for edge in EDGES])
    p1 = np.ones((10, 10))/10
    p4 = incidence @ np.linalg.solve(incidence.T @ incidence, incidence.T)-p1
    p5 = np.eye(10)-p1-p4
    u4 = (incidence[:, 0]-incidence[:, 1])/math.sqrt(6)
    u5 = np.zeros(10)
    for edge, value in [((0, 1), 1), ((0, 2), -1), ((1, 3), -1), ((2, 3), 1)]:
        u5[EDGES.index(edge)] = value/2
    return (p1, p4, p5), (u4, u5)


def centroid_one_loop(lengths, beta=1.):
    """Closed one-insertion determinant and analytic boundary derivatives.

    For the positive Regge sign, arXiv:1110.6866 equations 5.29-5.33 give
    H_ij=25*T*r_i*r_j/(48*V) at a barycenter, T=sum_i r_i^2.
    Since T=sum_edges b_e^2/5, lambda=(sum_edges b_e^2)^2/(48*V).
    Four vertex translations are omitted using the supplied orthonormal quotient.
    The returned derivatives are of Phi, with beta held fixed.
    """
    b = np.asarray(lengths, dtype=float)
    if b.shape != (10,) or not np.isfinite(b).all() or np.any(b <= 0):
        raise ValueError("Boundary lengths must be ten finite positive numbers")
    beta = float(beta)
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    pairs = np.zeros((10, 5, 5))
    for k, (u, v) in enumerate(EDGES):
        pairs[k, u, v] = pairs[k, v, u] = 1.
    coefficients = (pairs[:, 0, 1:, None]+pairs[:, 0, None, 1:]-pairs[:, 1:, 1:])/2
    gram = np.einsum("k,kij->ij", b*b, coefficients)
    try:
        np.linalg.cholesky(gram)
    except np.linalg.LinAlgError as error:
        raise ValueError("Boundary Gram matrix must be positive definite") from error
    log_volume = np.linalg.slogdet(gram)[1]/2-math.log(24)
    total = float(b @ b)
    log_curvature = 2*math.log(total)-math.log(48)-log_volume
    derivative = 2*b[:, None, None]*coefficients
    product = np.einsum("ij,kjl->kil", np.linalg.solve(gram, np.eye(4)), derivative)
    trace = np.trace(product, axis1=1, axis2=2)
    gradient = 2*b/total-trace/4
    hessian = (2*np.eye(10)/total-4*np.outer(b, b)/total**2
               -np.diag(trace/b)/4+np.einsum("aij,bji->ab", product, product)/4)
    return {
        "curvature": math.exp(log_curvature),
        "phi": (log_curvature-math.log(2*math.pi/beta))/2,
        "gradient": gradient, "hessian": hessian,
    }


def evaluate(lengths, depth=2, beta=1.):
    """Evaluate the whole flat branch and both coordinate forms of its integral."""
    b = np.asarray(lengths, dtype=float)
    if b.shape != (10,) or not np.isfinite(b).all() or np.any(b <= 0):
        raise ValueError("Boundary lengths must be ten finite positive numbers")
    if isinstance(depth, bool) or not isinstance(depth, int) or depth not in (1, 2):
        raise ValueError("Depth must be one or two")
    beta = float(beta)
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    r = fixed.gluing.reference()
    try:
        points = r.points_from_squared(b*b)
    except np.linalg.LinAlgError as error:
        raise ValueError("Boundary Gram matrix must be positive definite") from error
    cells = [tuple(range(5))]
    insertion_boundaries = []
    for _ in range(depth):
        insertion_boundaries.extend(r.cell_lengths(c, points) for c in cells)
        cells = r.refine(cells, points)
    edges = sorted({tuple(sorted(e)) for c in cells for e in combinations(c, 2)})
    index = {e: i for i, e in enumerate(edges)}
    boundary = [i for i, e in enumerate(edges) if max(e) < 5]
    internal = [i for i, e in enumerate(edges) if max(e) >= 5]
    vertices = list(range(5, max(points)+1))
    kappas = r.equal_split_kappas(cells, tuple(range(5)), np.full(10, math.pi))
    h = np.zeros((len(edges), len(edges)))
    gradient = np.zeros(len(edges))
    action = 0.
    for cell, kappa in zip(cells, kappas):
        value, local_gradient, local_hessian, _ = fixed.simplex_derivatives(
            r.cell_lengths(cell, points), kappa
        )
        local = [index[tuple(sorted(e))] for e in combinations(cell, 2)]
        h[np.ix_(local, local)] += local_hessian
        gradient[local] += local_gradient
        action += value
    gauge = np.zeros((len(internal), 4*len(vertices)))
    for row, j in enumerate(internal):
        u, v = edges[j]
        direction = (points[u]-points[v])/np.linalg.norm(points[u]-points[v])
        for vertex, sign in ((u, 1.), (v, -1.)):
            if vertex >= 5:
                gauge[row, 4*(vertex-5):4*(vertex-4)] = sign*direction
    z = fixed.complement(gauge, 4*len(vertices))
    hi, hb = h[np.ix_(internal, internal)], h[np.ix_(boundary, internal)]
    qz = z.T @ hi @ z
    eigenvalues = np.linalg.eigvalsh(qz)
    if eigenvalues[0] <= 0:
        raise ArithmeticError("The internal quotient must be positive")
    # Disjoint hierarchical radial slices; they need not be globally gauge-orthogonal.
    t = np.zeros((len(internal), len(vertices)))
    for vertex in vertices:
        rows = [i for i, j in enumerate(internal) if max(edges[j]) == vertex]
        columns = list(range(4*(vertex-5), 4*(vertex-4)))
        t[rows, vertex-5] = fixed.complement(gauge[np.ix_(rows, columns)], 4)[:, 0]
    qt = t.T @ hi @ t
    if depth == 2:
        child = qt[1:, 1:]
        np.linalg.cholesky(child)
        parent = qt[0, 0]-qt[0, 1:] @ np.linalg.solve(child, qt[1:, 0])
        child_logdet = np.linalg.slogdet(child)[1]
    else:
        parent, child_logdet = qt[0, 0], 0.
    if parent <= 0:
        raise ArithmeticError("The parent Schur curvature must be positive")
    sign, log_j = np.linalg.slogdet(z.T @ t)
    if sign == 0:
        raise ArithmeticError("The hierarchical slice must be transverse")
    rank = len(qz)
    base = rank*math.log(2*math.pi/beta)/2
    phi = np.linalg.slogdet(qz)[1]/2-base
    phi_raw = (child_logdet+math.log(parent))/2-base
    coarse = fixed.simplex_derivatives(b, np.full(10, math.pi))
    mixing = hb @ z
    effective = h[np.ix_(boundary, boundary)]-mixing @ np.linalg.solve(qz, mixing.T)
    return {
        "phi": float(phi), "phi_hierarchical_raw": float(phi_raw),
        "minus_log_jacobian": float(-log_j), "phi_sequential": float(phi_raw-log_j),
        "gamma": float(beta*coarse[0]+phi), "classical_action": coarse[0],
        "rank": rank, "smallest_internal_curvature": float(eigenvalues[0]),
        "centroid_formula_phi": float(sum(
            centroid_one_loop(local, beta)["phi"] for local in insertion_boundaries)-log_j),
        "classical_action_residual": float(abs(action-coarse[0])),
        "boundary_schur_residual": float(np.linalg.norm(effective-coarse[2])),
        "gauge_hessian_residual": float(np.linalg.norm(hi @ gauge)),
        "internal_gradient_residual": float(np.linalg.norm(gradient[internal])),
    }


@lru_cache(maxsize=512)
def _potential(lengths, depth):
    row = evaluate(lengths, depth)
    return np.array([row["phi"], row["phi_sequential"],
                     row["phi_hierarchical_raw"], row["minus_log_jacobian"]])


def directional_derivatives(lengths, direction, depth=2, step=.004):
    """Return Richardson gradient, curvature, and the half-step correction."""
    b, u = np.asarray(lengths, dtype=float), np.asarray(direction, dtype=float)
    if u.shape != (10,) or not np.isfinite(u).all():
        raise ValueError("Direction must contain ten finite components")
    step = float(step)
    if not math.isfinite(step) or step <= 0:
        raise ValueError("The finite-difference step must be finite and positive")
    if b.shape != (10,) or not np.isfinite(b).all() or np.any(b <= 0):
        raise ValueError("Boundary lengths must be ten finite positive numbers")
    center = _potential(tuple(b), depth)
    results = []
    for h in (step, step/2):
        plus = _potential(tuple(b+h*u), depth)
        minus = _potential(tuple(b-h*u), depth)
        results.append(((plus-minus)/(2*h), (plus-2*center+minus)/h**2))
    gradient = (4*results[1][0]-results[0][0])/3
    curvature = (4*results[1][1]-results[0][1])/3
    return gradient, curvature, curvature-results[1][1]


def regular_shape_case(depth=2, step=.004):
    projectors, directions = shape_projectors()
    derivatives = [directional_derivatives(REGULAR, u, depth, step) for u in directions]
    curvatures = np.array([x[1] for x in derivatives])
    phi_hessian = sum(curvatures[i, 0]*projectors[i+1] for i in range(2))
    coarse = fixed.simplex_derivatives(REGULAR, np.full(10, math.pi))
    classical = [float(u @ coarse[2] @ u) for u in directions]
    # This is a local threshold on an affine mean-length slice, not a physical prediction.
    thresholds = [float(curvatures[i, 0]/(-classical[i])) for i in range(2)]
    return {
        "depth": depth, "boundary_difference_step": step,
        "regular": evaluate(REGULAR, depth),
        "phi_shape_curvatures_multiplicity_4_5": curvatures[:, 0].tolist(),
        "sequential_curvature_residual": float(np.max(abs(curvatures[:, 1]-curvatures[:, 0]))),
        "omitted_jacobian_curvatures": curvatures[:, 3].tolist(),
        "richardson_correction": max(float(np.max(abs(x[2]))) for x in derivatives),
        "shape_gradient_residual": max(float(np.max(abs(x[0]))) for x in derivatives),
        "classical_affine_shape_curvatures": classical,
        "gamma_affine_shape_curvatures_beta_1": (np.array(classical)+curvatures[:, 0]).tolist(),
        "formal_beta_zero_curvature_thresholds": thresholds,
        "phi_hessian": phi_hessian.tolist(),
        "exact_one_insertion_hessian_residual": (
            float(np.linalg.norm(phi_hessian-centroid_one_loop(REGULAR)["hessian"]))
            if depth == 1 else None),
    }


def run():
    cases = [regular_shape_case(depth, step) for depth in (1, 2) for step in (.004, .002)]
    rng = np.random.default_rng(20260905)
    checks = []
    for depth in (1, 2):
        case = cases[2*(depth-1)+1]
        hessian = np.array(case["phi_hessian"])
        direction = rng.normal(size=10)
        direction -= direction.mean()
        direction /= np.linalg.norm(direction)
        _, measured, _ = directional_derivatives(REGULAR, direction, depth, .002)
        radial = directional_derivatives(REGULAR, REGULAR/np.linalg.norm(REGULAR), depth, .002)
        checks.append({
            "depth": depth,
            "random_shape_curvature": float(measured[0]),
            "s5_prediction_residual": float(abs(measured[0]-direction @ hessian @ direction)),
            "radial_gradient": float(radial[0][0]), "radial_curvature": float(radial[1][0]),
            "scale_residual": abs(evaluate(2*REGULAR, depth)["phi"]-case["regular"]["phi"]),
        })
    shear_rows = []
    r = fixed.gluing.reference()
    for shear in (0., .1, .2):
        transform = np.eye(4)
        transform[0, 1] = shear
        points = {v: transform @ p for v, p in r.points_from_squared(REGULAR**2).items()}
        b = r.cell_lengths(tuple(range(5)), points)
        first, second = evaluate(b, 1), evaluate(b, 2)
        shear_rows.append({
            "volume_preserving_shear": shear,
            "phi_depth_1": first["phi"], "phi_depth_2": second["phi"],
            "refinement_phi_increment": second["phi"]-first["phi"],
            "classical_action": first["classical_action"],
        })
    increment = (np.array(cases[3]["phi_shape_curvatures_multiplicity_4_5"])
                 -np.array(cases[1]["phi_shape_curvatures_multiplicity_4_5"]))
    classical = np.array(cases[3]["classical_affine_shape_curvatures"])
    fitted_delta_beta = float(increment @ classical/(classical @ classical))
    files = ["boundary_one_loop_action.py", "fixed_boundary_composition.py",
             "length_gluing_bath.py", "local_refinement_bath.py", "continuum_bath.py",
             "F-01/predict_fold_budget.py", "F-01/regge_one_to_five_boundary_hessian.py",
             "F-01/regge_one_to_five_refinement.py"]
    return {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "source_sha256": {p: hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in files},
        "cases": cases, "independent_direction_checks": checks, "volume_preserving_controls": shear_rows,
        "refinement_increment_shape_curvatures": increment.tolist(),
        "single_beta_fit_on_affine_shape_slice": {
            "least_squares_delta_beta": fitted_delta_beta,
            "unabsorbed_curvature_norm": float(np.linalg.norm(increment-fitted_delta_beta*classical)),
            "global_scale_identity_already_excludes_nonzero_delta_beta": True,
        },
        "scope": {
            "flat_interior_recomputed_for_each_boundary": True,
            "orthonormal_quotient_measure_supplied": True, "length_gluing_imposed": True,
            "fixed_dimensionless_reference_and_beta": True,
            "full_nonlinear_internal_integral_computed": False,
            "boundary_derivative_error_rigorously_bounded": False,
            "controlled_loop_expansion_proven_at_formal_threshold": False,
            "global_boundary_measure_normalized": False,
            "microscopic_measure_derived": False, "dynamic_common_metric_selection_proven": False,
            "continuum_einstein_limit_derived": False,
        },
    }


if __name__ == "__main__":
    result = run()
    (HERE/"boundary_one_loop_action.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
