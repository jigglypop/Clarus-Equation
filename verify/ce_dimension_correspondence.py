"""Exact relation/space correspondence tools; CE-DIM-F1, kinematics only."""
import json

import sympy as s

from ce_state_dimensions import (
    analyse_state, density_matrix, exact_matrix, hermitian_coordinates,
    is_zero, traceless_basis,
)


def exact_rectangle(value, name):
    matrix = s.Matrix(value)
    if not matrix.rows or not matrix.cols:
        raise ValueError(f"{name} must be nonempty")
    if matrix.free_symbols or any(v.has(s.Float) or v.is_finite is not True for v in matrix):
        raise ValueError(f"{name} requires finite exact numeric entries")
    return matrix


def frame_geometry(frame, derivatives):
    """Metric and Berry curvature at one witnessed Stiefel-frame jet, ell=1.

    A valid first jet always has a local smooth Stiefel extension. This does
    not certify a global chart, constant rank neighbourhood or dynamics.
    """
    v = exact_rectangle(frame, "frame")
    if any(not is_zero(x) for x in v.H*v-s.eye(v.cols)):
        raise ValueError("frame columns must be orthonormal")
    ds = [exact_rectangle(a, "derivative") for a in derivatives]
    if any(a.shape != v.shape for a in ds):
        raise ValueError("frame derivative dimensions differ")
    if any(not is_zero(x) for a in ds for x in v.H*a+a.H*v):
        raise ValueError("derivatives must satisfy the Stiefel tangent constraint")
    q = s.eye(v.rows)-v*v.H
    horizontal = [q*a for a in ds]
    tensor = [[s.simplify(a.H*b) for b in horizontal] for a in horizontal]
    metric = s.Matrix(len(ds), len(ds), lambda i, j: s.re(s.trace(tensor[i][j])).simplify())
    curvature = {(i, j): s.simplify(s.I*(tensor[i][j]-tensor[j][i]))
                 for i in range(len(ds)) for j in range(i+1, len(ds))}
    return {"metric": metric, "metric_rank": metric.rank(), "curvature": curvature,
            "internal_frame_rank": v.cols,
            "max_real_grassmann_dimension": 2*v.cols*(v.rows-v.cols),
            "lorentzian_signature_from_this_metric": False}


def commutant_basis(dimension, operators, traceless=True):
    """Hermitian convention: Lie bracket i[A,B], real vector-space kernel."""
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError("dimension must be a positive integer")
    ops = [exact_matrix(a, "commutant operator") for a in operators]
    if any(a.rows != dimension for a in ops):
        raise ValueError("commutant dimensions differ")
    basis = traceless_basis(dimension)
    if not traceless:
        basis = [s.eye(dimension)]+basis
    if not basis or not ops:
        return basis
    rows = [s.Matrix.hstack(*(hermitian_coordinates(s.I*(a*b-b*a)) for b in basis))
            for a in ops]
    constraints = s.Matrix.vstack(*rows)
    return [sum((c*b for c, b in zip(vec, basis)), s.zeros(dimension))
            for vec in constraints.nullspace()]


def observation_flag(rho, operators):
    """Nested expectations on the SAME prepared state, not sequential collapse."""
    rho = density_matrix(rho)
    ops = [exact_matrix(a, "flag operator") for a in operators]
    if any(a.shape != rho.shape for a in ops):
        raise ValueError("flag dimensions differ")
    rows = []
    for k in range(len(ops)+1):
        result = analyse_state(rho, ops[:k])
        rows.append({"questions": k, "visible_rank": result["jacobian_rank"],
                     "local_hidden_dimension": result["certified_local_fibre_dimension"],
                     "certificate": result["dimension_certificate"],
                     "compatible_su_dimension": len(commutant_basis(rho.rows, ops[:k]))})
    return rows


def serial_geometry(result):
    return {**result, "metric": [[str(x) for x in row] for row in result["metric"].tolist()],
            "curvature": {f"{i},{j}": [[str(x) for x in row] for row in a.tolist()]
                          for (i, j), a in result["curvature"].items()}}


def demonstration():
    x = s.Matrix([[0, 1], [1, 0]])
    y = s.Matrix([[0, -s.I], [s.I, 0]])
    z = s.diag(1, -1)
    frame = s.eye(5)[:, :1]
    real = [s.eye(5)[:, i:i+1] for i in range(1, 5)]
    complex_tangent = [real[0], s.I*real[0], real[1], real[2]]
    return {
        "schema_version": 1,
        "qubit_expectation_flag": observation_flag(s.eye(2)/2, [z, z, x, y]),
        "same_origin_metric_zero_curvature": serial_geometry(frame_geometry(frame, real)),
        "same_origin_metric_nonzero_curvature": serial_geometry(frame_geometry(frame, complex_tangent)),
        "internal_rank_options": [{"m": m, "u_m_dimension": m*m, "su_m_dimension": m*m-1}
                                  for m in range(1, 5)],
        "spacetime_or_force_identity_inferred": False,
        "scientific_success": False, "full_joint_rmse": None,
    }


if __name__ == "__main__":
    print(json.dumps(demonstration(), ensure_ascii=False, indent=2))
