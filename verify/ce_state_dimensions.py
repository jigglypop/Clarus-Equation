"""Exact, finite quantum-state dimension tools for manuscript CE-DIM1.

Dimensions count real state parameters, not spacetime or fitted parameters.
Unknown signs, floating-point input and unwitnessed constraints are rejected.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path

import sympy as s


def is_zero(value):
    return s.simplify(value) == 0


def exact_matrix(value, name="matrix", hermitian=True):
    matrix = s.Matrix(value)
    if matrix.rows == 0 or matrix.rows != matrix.cols:
        raise ValueError(f"{name} must be nonempty and square")
    if matrix.free_symbols or any(v.has(s.Float) for v in matrix):
        raise ValueError(f"{name} requires exact numeric entries, without floats or free symbols")
    if any(v.is_finite is not True for v in matrix):
        raise ValueError(f"{name} must have finite entries")
    if hermitian and any(not is_zero(v) for v in matrix-matrix.H):
        raise ValueError(f"{name} must be Hermitian")
    return matrix


def density_matrix(value):
    rho = exact_matrix(value, "rho")
    if not is_zero(s.trace(rho)-1):
        raise ValueError("rho must have trace 1")
    if rho.is_positive_semidefinite is not True:
        raise ValueError("rho must be certifiably positive semidefinite")
    return rho


def hermitian_coordinates(matrix):
    """Real coordinates, without sqrt(2) normalization."""
    d = matrix.rows
    values = [s.re(matrix[i, i]).simplify() for i in range(d)]
    for i in range(d):
        for j in range(i+1, d):
            values.extend([s.re(matrix[i, j]).simplify(), s.im(matrix[i, j]).simplify()])
    return s.Matrix(values)


def traceless_basis(d):
    basis = []
    for i in range(d-1):
        b = s.zeros(d)
        b[i, i], b[d-1, d-1] = 1, -1
        basis.append(b)
    for i in range(d):
        for j in range(i+1, d):
            b = s.zeros(d)
            b[i, j] = b[j, i] = 1
            basis.append(b)
            b = s.zeros(d)
            b[i, j], b[j, i] = s.I, -s.I
            basis.append(b)
    return basis


def measurement_matrix(observables, basis):
    return s.Matrix(len(observables), len(basis),
                    lambda i, j: s.simplify(s.trace(observables[i]*basis[j])))


def rank_stratum_tangent(rho):
    """Tangent: trace X=0 and Q X Q=0, where Q projects onto ker(rho)."""
    rho = density_matrix(rho)
    d, rank = rho.rows, rho.rank()
    basis = traceless_basis(d)
    kernel = rho.nullspace()
    if not kernel:
        return basis
    n = s.Matrix.hstack(*kernel)
    q = n*(n.H*n).inv()*n.H
    constraints = s.Matrix.hstack(*(hermitian_coordinates(q*b*q) for b in basis))
    tangent = [
        sum((c*b for c, b in zip(vector, basis)), s.zeros(d))
        for vector in constraints.nullspace()
    ]
    if len(tangent) != 2*d*rank-rank*rank-1:
        raise ArithmeticError("rank-stratum tangent dimension disagrees with theorem D2")
    return tangent


def analyse_state(rho, observables, values=None):
    """A supplied state witnesses feasibility; singular fibres are not dimension-certified."""
    rho = density_matrix(rho)
    obs = [exact_matrix(a, f"observable[{i}]") for i, a in enumerate(observables)]
    if any(a.shape != rho.shape for a in obs):
        raise ValueError("observables and rho must have the same shape")
    actual = [s.simplify(s.trace(rho*a)) for a in obs]
    if values is not None:
        if len(values) != len(obs) or any(not is_zero(a-v) for a, v in zip(actual, values)):
            raise ValueError("rho does not witness the supplied expectation values")
    tangent = rank_stratum_tangent(rho)
    independent = measurement_matrix(obs, traceless_basis(rho.rows)).rank()
    jacobian = measurement_matrix(obs, tangent)
    rank = jacobian.rank()
    nullity = len(tangent)-rank
    if rank == independent:
        certificate = "regular_value_after_removing_affine_dependencies"
        dimension = nullity
    elif rank == len(tangent):
        certificate = "injective_differential_local_fibre_isolated"
        dimension = 0
    else:
        certificate = "critical_point_only_linearized_nullity_known"
        dimension = None
    return {
        "hilbert_complex_dimension": rho.rows,
        "state_rank": rho.rank(),
        "rank_stratum_real_dimension": len(tangent),
        "independent_affine_constraints": independent,
        "jacobian_rank": rank,
        "linearized_nullity": nullity,
        "certified_local_fibre_dimension": dimension,
        "dimension_certificate": certificate,
        "expectation_values": [str(v) for v in actual],
    }


def balanced_dimensions(operator):
    """Zero expectation level, including empty, semidefinite and singular cases."""
    a = exact_matrix(operator, "boundary_operator")
    d = a.rows
    spectrum = a.eigenvals()
    positive = negative = zero = 0
    for val, count in spectrum.items():
        if val.is_zero is True:
            zero += count
        elif val.is_positive is True:
            positive += count
        elif val.is_negative is True:
            negative += count
        else:
            raise ValueError("spectrum signs are not certified; no dimension inferred")
    common = {"hilbert_complex_dimension": d,
              "inertia": {"positive": positive, "negative": negative, "zero": zero}}
    if zero == d:
        return {**common, "kind": "whole_state_space", "mixed_dimension": d*d-1,
                "pure_dimension": 2*d-2, "separating_boundary": False}
    if positive and negative:
        paired = all(sum(count for other, count in spectrum.items() if is_zero(other+val)) == mult
                     for val, mult in spectrum.items())
        return {**common, "kind": "indefinite_zero_level",
                "mixed_dimension": d*d-2, "pure_regular_locus_dimension": 2*d-3,
                "pure_level_globally_regular": zero == 0,
                "pure_critical_locus_dimension": 2*zero-2 if zero else None,
                "exact_unitary_toggle_exists": paired, "separating_boundary": True}
    if zero:
        return {**common, "kind": "kernel_face", "mixed_dimension": zero*zero-1,
                "pure_dimension": 2*zero-2, "separating_boundary": False}
    return {**common, "kind": "empty", "mixed_dimension": None,
            "pure_dimension": None, "separating_boundary": False}


def fixed_state_dimensions(unitary):
    u = exact_matrix(unitary, "unitary", hermitian=False)
    if any(not is_zero(v) for v in u.H*u-s.eye(u.rows)):
        raise ValueError("unitary is not unitary")
    basis = traceless_basis(u.rows)
    constraints = (s.Matrix.hstack(*(hermitian_coordinates(u*b*u.H-b) for b in basis))
                   if basis else s.zeros(u.rows*u.rows, 0))
    dim = len(basis)-constraints.rank()
    multiplicities = list(u.eigenvals().values())
    spectral_dim = sum(m*m for m in multiplicities)-1
    if dim != spectral_dim:
        raise ArithmeticError("commutator and spectral dimensions disagree")
    return {"mixed_fixed_dimension": dim, "spectral_multiplicities": multiplicities,
            "pure_fixed_component_dimensions": [2*m-2 for m in multiplicities],
            "pure_fixed_dimension": max(2*m-2 for m in multiplicities)}


def natural(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def scalar_from_json(value):
    """Parse rational JSON entries without eval/sympify of external text."""
    if isinstance(value, bool):
        raise ValueError("Boolean is not an exact scalar")
    if isinstance(value, dict):
        if set(value) != {"re", "im"}:
            raise ValueError("complex entries require exactly re and im")
        re, im = scalar_from_json(value["re"]), scalar_from_json(value["im"])
        if re.is_real is not True or im.is_real is not True:
            raise ValueError("re and im must be rational real scalars")
        return re+s.I*im
    if not isinstance(value, (int, str)):
        raise ValueError("use integer or fraction string; JSON floats are rejected")
    try:
        f = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("invalid rational scalar") from exc
    return s.Rational(f.numerator, f.denominator)


def observation_fibre(d, p, visible_rank, visible_weight):
    """Fix the raw block R=P rho P (hence its trace), not just normalized R."""
    d, p, r = natural(d, "d", 1), natural(p, "p"), natural(visible_rank, "visible_rank")
    w = scalar_from_json(visible_weight)
    if w.is_real is not True or not 0 <= w <= 1 or p > d or r > p:
        raise ValueError("invalid split, visible rank, or visible weight")
    if (w == 0) != (r == 0):
        raise ValueError("a PSD visible block has zero trace iff its rank is zero")
    q = d-p
    if q == 0 and w != 1:
        raise ValueError("Q=0 requires visible weight 1")
    if w == 1:
        strata = [{"total_state_rank": r, "schur_rank": 0, "real_dimension": 0}]
    else:
        strata = [
            {"total_state_rank": r+t, "schur_rank": t,
             "real_dimension": 2*r*q+2*q*t-t*t-1}
            for t in range(0 if r else 1, q+1)
        ]
    pure = next((row["real_dimension"] for row in strata if row["total_state_rank"] == 1), None)
    return {
        "hilbert_complex_dimension": d, "visible_complex_dimension": p,
        "complement_complex_dimension": q, "visible_rank": r, "visible_weight": str(w),
        "raw_fibre_real_dimension": max(row["real_dimension"] for row in strata),
        "pure_fibre_real_dimension": pure, "rank_strata": strata,
        "only_normalized_visible_block_fixed_dimension": (2*r*q+q*q if q else 0) if r else None,
    }


def observable_history_basis(hamiltonian, observables):
    """Span of traceless observables under L(A)=i[H,A], closed exactly."""
    h = exact_matrix(hamiltonian, "hamiltonian")
    d = h.rows
    operators, columns = [], s.zeros(d*d, 0)

    def admit(a):
        nonlocal columns
        a = s.simplify(a-s.trace(a)*s.eye(d)/d)
        proposed = columns.row_join(hermitian_coordinates(a))
        if proposed.rank() > len(operators):
            operators.append(a)
            columns = proposed

    for i, a in enumerate(observables):
        a = exact_matrix(a, f"observable[{i}]")
        if a.shape != h.shape:
            raise ValueError("Hamiltonian and observable dimensions differ")
        admit(a)
    cursor = 0
    while cursor < len(operators):
        a = operators[cursor]
        admit(s.I*(h*a-a*h))
        cursor += 1
    return operators


def dimension_profile(d):
    d = natural(d, "d", 1)
    return {
        "hilbert_complex_dimension": d, "mixed_real_dimension": d*d-1,
        "pure_real_dimension": 2*d-2,
        "nontrivial_binary_question_exists": d >= 2,
        "uncertain_0_lt_p_lt_1": {
            "mixed_dimension": d*d-1, "pure_dimension": 2*d-2} if d >= 2 else None,
        "balanced_p_equals_half": {
            "mixed_dimension": d*d-2, "pure_dimension": 2*d-3} if d >= 2 else None,
        "condition": "nontrivial projective binary question; real dimensions; no spacetime inference",
    }


def matrix_from_json(value):
    return s.Matrix([[scalar_from_json(v) for v in row] for row in value])


def analyse_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("input must be a JSON object")
    if set(payload)-{"rho", "observables", "values", "hamiltonian", "unitary", "boundary_operator"}:
        raise ValueError("unknown input fields")
    rho = density_matrix(matrix_from_json(payload["rho"]))
    obs = [matrix_from_json(v) for v in payload.get("observables", [])]
    values = ([scalar_from_json(v) for v in payload["values"]] if "values" in payload else None)
    result = {"state_measurement": analyse_state(rho, obs, values)}
    if "boundary_operator" in payload:
        a = matrix_from_json(payload["boundary_operator"])
        if a.shape != rho.shape:
            raise ValueError("boundary_operator dimension differs from rho")
        result["balanced_boundary"] = balanced_dimensions(a)
    if "unitary" in payload:
        u = matrix_from_json(payload["unitary"])
        if u.shape != rho.shape:
            raise ValueError("unitary dimension differs from rho")
        result["fixed_states"] = fixed_state_dimensions(u)
    if "hamiltonian" in payload:
        h = matrix_from_json(payload["hamiltonian"])
        if h.shape != rho.shape:
            raise ValueError("hamiltonian dimension differs from rho")
        history = observable_history_basis(h, obs)
        result["history"] = {
            "observable_span_dimension": len(history),
            "full_rank_invisible_dimension": rho.rows*rho.rows-1-len(history),
            "at_supplied_state": analyse_state(rho, history),
            "assumption": "known closed unitary dynamics and repeat-prepared expectation histories",
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--levels", type=int)
    mode.add_argument("--input", type=Path, help="exact rational JSON state and observables")
    parser.add_argument("--visible", type=int)
    parser.add_argument("--visible-rank", type=int)
    parser.add_argument("--visible-weight", type=str)
    parser.add_argument("--output", type=Path, help="write JSON; otherwise stdout only")
    args = parser.parse_args()
    try:
        split = [args.visible, args.visible_rank, args.visible_weight]
        if args.input:
            if any(v is not None for v in split):
                raise ValueError("split flags apply only to --levels mode")
            result = analyse_payload(json.loads(args.input.read_text(encoding="utf-8")))
        else:
            d = args.levels if args.levels is not None else 2
            result = {"profile": dimension_profile(d)}
            if any(v is not None for v in split):
                if any(v is None for v in split):
                    raise ValueError("supply all three visible split flags")
                result["observation_fibre"] = observation_fibre(d, *split)
        result.update({"schema_version": 1, "dimension_type": "real_state_parameters",
                       "scientific_success": False, "full_joint_rmse": None,
                       "goedel_failure_region_defined": False})
        encoded = json.dumps(result, ensure_ascii=False, indent=2)+"\n"
        if args.output:
            args.output.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
