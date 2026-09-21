"""CE-RB5: events, relational clocks, projector geometry and Maxwell conditions.

Finite synthetic witnesses and symbolic checks; no emergent-spacetime completion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.linalg import eigh, expm

from verify_reverse import Evidence


def partial_b(matrix):
    return np.einsum("abcb->ac", matrix.reshape(2, 2, 2, 2))


def bures_speed_squared(rho, derivative):
    probs, vec = eigh(rho)
    transformed = vec.conj().T@derivative@vec
    denominator = probs[:, None]+probs[None, :]
    valid = denominator > 1e-13
    return float(.5*np.sum(np.abs(transformed[valid])**2/denominator[valid]))


def sld(rho, derivative):
    probs, vec = eigh(rho)
    transformed = vec.conj().T@derivative@vec
    denominator = probs[:, None]+probs[None, :]
    local = np.zeros_like(transformed, dtype=complex)
    valid = denominator > 1e-13
    local[valid] = 2*transformed[valid]/denominator[valid]
    return vec@local@vec.conj().T


def scalar_curvature(metric, coordinates):
    n = len(coordinates)
    inverse = metric.inv()
    connection = [[[sp.simplify(sum(inverse[a, d]*(sp.diff(metric[d, c], coordinates[b])
                     +sp.diff(metric[d, b], coordinates[c])-sp.diff(metric[b, c], coordinates[d]))/2
                     for d in range(n))) for c in range(n)] for b in range(n)] for a in range(n)]
    ricci = sp.zeros(n)
    for b in range(n):
        for d in range(n):
            ricci[b, d] = sp.simplify(sum(sp.diff(connection[a][d][b], coordinates[a])
                                -sp.diff(connection[a][a][b], coordinates[d])
                                +sum(connection[a][a][c]*connection[c][d][b]
                                     -connection[a][d][c]*connection[c][a][b] for c in range(n))
                                for a in range(n)))
    return sp.simplify(sum(inverse[b, d]*ricci[b, d] for b in range(n) for d in range(n)))


def projector_geometry(tangents):
    n = len(tangents)
    h = np.array([[np.vdot(a, b).real for b in tangents] for a in tangents])
    curvature = [[1j*(tangents[a].conj().T@tangents[b]-tangents[b].conj().T@tangents[a])
                  for b in range(n)] for a in range(n)]
    return h, curvature


def run():
    e = Evidence()
    values = {}
    rng = np.random.default_rng(20260922)
    eye2 = np.eye(2)
    pauli_x = np.array([[0., 1.], [1., 0.]])
    pauli_z = np.diag([1., -1.])

    # R51: independently formed event products and state errors.
    first = expm(-.17j*np.kron(pauli_x, pauli_x))
    last = expm(-.23j*np.kron(pauli_z, pauli_z))
    event_a = expm(-.31j*np.kron(pauli_z, eye2))
    random_state = rng.normal(size=4)+1j*rng.normal(size=4)
    random_state /= np.linalg.norm(random_state)
    rho = np.outer(random_state, random_state.conj())
    order_rows = []
    for perturbation in [0., .001, .1]:
        event_b = expm(-1j*(.21*np.kron(eye2, pauli_x)+perturbation*np.kron(pauli_x, pauli_z)))
        va, vb = last@event_b@event_a@first, last@event_a@event_b@first
        commutator = event_a@event_b-event_b@event_a
        delta = float(np.linalg.norm(commutator, 2))
        difference = va@rho@va.conj().T-vb@rho@vb.conj().T
        state_error = float(np.sum(np.abs(eigh(difference, eigvals_only=True))))
        e.close("R51", f"one_swap_operator_difference_{perturbation}", np.linalg.norm(va-vb, 2), delta, 1e-13)
        e.check("R51", f"one_swap_state_bound_{perturbation}", state_error <= 2*delta+1e-13,
                {"commutator_norm": delta, "state_error": state_error})
        order_rows.append({"perturbation": perturbation, "state_error": state_error, "bound": 2*delta})
    pure = np.array([1., 2j])/np.sqrt(5)
    state = np.outer(pure, pure.conj())
    xz, zx = pauli_x@pauli_z, pauli_z@pauli_x
    e.close("R51", "projective_commutation_same_state_map", xz@state@xz.conj().T, zx@state@zx.conj().T)
    e.close("R51", "projective_commutation_nonzero_operator_commutator", np.linalg.norm(xz-zx, 2), 2.)
    values["event_order_errors"] = order_rows

    # R52: variational SLD formula and partial-trace contraction.
    raw = rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4))
    rho = raw@raw.conj().T
    rho /= np.trace(rho)
    raw_h = rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4))
    h = (raw_h+raw_h.conj().T)/2
    drho = -1j*(h@rho-rho@h)
    vg2 = bures_speed_squared(rho, drho)
    ra, da = partial_b(rho), partial_b(drho)
    va2 = bures_speed_squared(ra, da)
    local_sld = sld(ra, da)
    global_sld = sld(rho, drho)
    fisher = float(np.trace(rho@global_sld@global_sld).real)
    e.close("R52", "SLD_spectral_normalization", fisher, 4*vg2)
    objective = float((2*np.trace(drho@global_sld)-np.trace(rho@global_sld@global_sld)).real)
    e.close("R52", "SLD_variational_maximizer", objective, fisher)
    lift = np.kron(local_sld, eye2)
    restricted = float((2*np.trace(drho@lift)-np.trace(rho@lift@lift)).real)
    e.close("R52", "partial_trace_restricted_variation", restricted, 4*va2)
    e.check("R52", "partial_trace_speed_contraction", va2 <= vg2,
            {"global_speed_squared": vg2, "local_speed_squared": va2})
    bell = np.array([1., 0., 0., 1.])/np.sqrt(2)
    rb = np.outer(bell, bell)
    hb = np.kron(pauli_z, eye2)/2
    db = -1j*(hb@rb-rb@hb)
    e.close("R52", "global_bell_clock", bures_speed_squared(rb, db), .25)
    e.close("R52", "stationary_local_bell_clock", bures_speed_squared(partial_b(rb), partial_b(db)), 0.)
    phase_cycle = expm(-2j*np.pi*hb)
    e.close("R52", "same_endpoint_after_nonzero_length", phase_cycle@rb@phase_cycle.conj().T, rb)
    shifted = hb+13*np.eye(4)
    e.close("R52", "energy_origin_does_not_change_clock", -1j*(shifted@rb-rb@shifted), db)
    values["bell_clock"] = {"joint_cycle_length": float(np.pi), "cycle_time": float(2*np.pi), "each_local_length": 0.}

    # R53: constant rank, projector metric and doubled-origin leaves.
    x, ell = sp.symbols("x ell", real=True, positive=True)
    vector = sp.Matrix([sp.cos(x), sp.sin(x)])
    projector = vector*vector.T
    dp = projector.diff(x)
    e.zero("R53", "projector_trace_metric", sp.trace(dp*dp)/2-1)
    e.zero("R53", "horizontal_velocity", (vector.T*vector.diff(x))[0])
    e.zero("R53", "constant_positive_rank", (vector.diff(x).T*vector.diff(x))[0]-1)
    # Each nonzero-x vertical fiber joins its upper and lower points without crossing the puncture.
    detours = []
    for epsilon in [.1, .01, .001]:
        points = np.array([[0., 1.], [epsilon, 1.], [epsilon, -1.], [0., -1.]])
        length = float(np.sum(np.abs(np.diff(points[:, 0]))))
        e.close("R53", f"null_leaf_detour_length_{epsilon}", length, 2*epsilon)
        detours.append({"epsilon": epsilon, "length_for_ell_one": length})
    values["doubled_origin_zero_distance_detours"] = detours

    # R54: finite graph realizations, curvature-area bound, common mass trace.
    tangents = [rng.normal(size=(4, 2))+1j*rng.normal(size=(4, 2)) for _ in range(3)]
    hmetric, curvature = projector_geometry(tangents)
    for a in range(3):
        for b in range(a+1, 3):
            actual = float(np.linalg.norm(curvature[a][b], "fro")**2)
            limit = float(4*(hmetric[a, a]*hmetric[b, b]-hmetric[a, b]**2))
            e.check("R54", f"curvature_area_bound_{a}{b}", actual <= limit+1e-12,
                    {"curvature_squared": actual, "area_bound": limit})
    he, hv = eigh(hmetric)
    whitening = (hv*(he**-.5))@hv.T
    orthogonal = [sum(whitening[a, b]*tangents[b] for b in range(3)) for a in range(3)]
    white_h, white_f = projector_geometry(orthogonal)
    e.close("R54", "same_metric_orthonormal_basis", white_h, np.eye(3))
    phi = sum(b.conj().T@b for b in orthogonal)
    phi_eigenvalues = eigh(phi, eigvals_only=True)
    e.close("R54", "common_geometric_mass_trace", np.trace(phi).real, 3.)
    e.check("R54", "mass_spectrum_trace_bounds", np.min(phi_eigenvalues) >= 0
            and np.max(phi_eigenvalues) <= 3 and 4.5 <= np.trace(phi@phi).real <= 9.,
            {"eigenvalues": phi_eigenvalues.tolist(), "trace_squared_operator": float(np.trace(phi@phi).real)})
    total_f = sum(np.linalg.norm(white_f[a][b], "fro")**2 for a in range(3) for b in range(3))
    e.check("R54", "contracted_curvature_bound", total_f <= 24., {"curvature_contraction": float(total_f), "bound": 24.})
    curved = [np.array([[1.], [0.]], complex), np.array([[1j], [0.]], complex)]
    flat = [np.array([[1.], [0.]], complex), np.array([[0.], [1.]], complex)]
    hc, fc = projector_geometry(curved)
    hf, ff = projector_geometry(flat)
    e.close("R54", "same_pointwise_metric", hc, hf)
    e.close("R54", "saturating_curvature", fc[0][1], [[-2.]])
    e.close("R54", "flat_curvature_counterexample", ff[0][1], [[0.]])

    # R55: real congruence preserves nonnegative quadratic form; lapse changes curvature.
    positive = rng.normal(size=(6, 4))
    gram = positive.T@positive
    change = rng.normal(size=(4, 4))
    pulled = change.T@gram@change
    e.check("R55", "positive_real_pullback", eigh(pulled, eigvals_only=True).min() >= -1e-12,
            {"eigenvalues": eigh(pulled, eigvals_only=True).tolist()})
    time, xc, yc, zc, kappa = sp.symbols("t x y z kappa", real=True)
    lapse = 1+kappa*xc**2
    lorentz = sp.diag(-lapse*lapse, 1, 1, 1)
    scalar_r = scalar_curvature(lorentz, [time, xc, yc, zc])
    e.zero("R55", "same_spatial_metric_different_spacetime_curvature", scalar_r+4*kappa/lapse)
    e.zero("R55", "minkowski_limit", scalar_r.subs(kappa, 0))

    # R56: explicit product frames and independent intrinsic-curvature calculation.
    def real_qubit(angle):
        return np.array([np.cos(angle/2), np.sin(angle/2)])
    def real_qubit_prime(angle):
        return np.array([-np.sin(angle/2), np.cos(angle/2)])/2
    def tensor(vectors):
        result = vectors[0]
        for v in vectors[1:]:
            result = np.kron(result, v)
        return result
    angles = [.4, .7, -.3]
    factors = [real_qubit(a) for a in angles]
    vflat = tensor(factors)
    bflat = []
    for a in range(3):
        changed = factors.copy()
        changed[a] = real_qubit_prime(angles[a])
        tangent = tensor(changed)
        bflat.append((tangent-vflat*np.vdot(vflat, tangent))[:, None])
    flat_metric, flat_f = projector_geometry(bflat)
    e.close("R56", "three_real_qubit_metric", flat_metric, np.eye(3)/4)
    e.close("R56", "three_real_qubit_curvature_zero", np.array(flat_f), np.zeros((3, 3, 1, 1)))
    mass = sum(np.linalg.inv(flat_metric)[a, b]*(bflat[a].conj().T@bflat[b])[0, 0]
               for a in range(3) for b in range(3))
    e.close("R56", "flat_space_nonzero_geometric_mass", mass, 3.)
    theta, phase, coordinate = sp.symbols("theta phase eta", real=True)
    scale = sp.symbols("ell", positive=True)
    flat_g = scale**2*sp.eye(3)/4
    sphere_g = scale**2*sp.diag(1, sp.sin(theta)**2, 1)/4
    e.zero("R56", "flat_product_scalar_curvature", scalar_curvature(flat_g, [theta, phase, coordinate]))
    e.zero("R56", "sphere_product_scalar_curvature", scalar_curvature(sphere_g, [theta, phase, coordinate])-8/scale**2)

    # R57: vacuum isotropic rank restriction, including a rank-four non-vacuum six-coordinate map.
    dimension_rows = []
    for n in [2, 3, 4]:
        symplectic = np.block([[np.zeros((n, n)), -np.eye(n)], [np.eye(n), np.zeros((n, n))]])
        jacobian = np.zeros((2*n, 4))
        jacobian[:min(n, 4), :min(n, 4)] = np.eye(min(n, 4))
        pulled_f = jacobian.T@symplectic@jacobian
        rank = int(np.linalg.matrix_rank(symplectic@jacobian))
        e.close("R57", f"canonical_vacuum_pullback_{n}", pulled_f, np.zeros((4, 4)))
        e.check("R57", f"vacuum_variational_rank_{n}", rank == min(n, 4),
                {"pairs": n, "variation_rank": rank, "required_rank": 4})
        dimension_rows.append({"pairs": n, "vacuum_rank": rank})
    n = 3
    symplectic = np.block([[np.zeros((n, n)), -np.eye(n)], [np.eye(n), np.zeros((n, n))]])
    jacobian = np.zeros((6, 4))
    jacobian[:3, :3] = np.eye(3)
    jacobian[3, 3] = 1.
    e.check("R57", "six_coordinate_full_rank_is_not_this_vacuum", np.linalg.matrix_rank(jacobian) == 4
            and np.linalg.norm(jacobian.T@symplectic@jacobian) > 0.,
            {"rank": int(np.linalg.matrix_rank(jacobian)), "F_squared_norm": float(np.linalg.norm(jacobian.T@symplectic@jacobian)**2)})
    values["canonical_vacuum_ranks"] = dimension_rows

    # R58: principal symbol requires zero pp metric, incompatible with canonical Berry curvature.
    momenta = sp.symbols("k0:4", real=True)
    coeffs = sp.symbols("g0:10", real=True)
    symmetric = sp.zeros(4)
    index = 0
    for a in range(4):
        for b in range(a, 4):
            symmetric[a, b] = symmetric[b, a] = coeffs[index]
            index += 1
    km = sp.Matrix(momenta)
    principal = sum(z*z for z in momenta)*(km.T*symmetric*km)[0]
    solutions = sp.solve(sp.Poly(principal, *momenta).coeffs(), coeffs, dict=True)
    e.check("R58", "gauge_principal_symbol_forces_zero_pp_metric", solutions == [{c: 0 for c in coeffs}],
            {"solution": {str(k): str(v) for k, v in solutions[0].items()}})
    for squeeze in [.25, 1., 4.]:
        bq, bp = np.sqrt(squeeze/2), 1j/np.sqrt(2*squeeze)
        curvature_pq = 1j*(np.conjugate(bp)*bq-np.conjugate(bq)*bp)
        e.close("R58", f"canonical_curvature_under_squeezing_{squeeze}", curvature_pq, 1.)
        e.close("R58", f"QGT_metric_product_lower_bound_{squeeze}", 4*abs(bp)**2*abs(bq)**2, 1.)
    radial = sp.symbols("r", positive=True)
    integral = 2*sp.pi**2*sp.integrate(radial**3*sp.exp(-2*radial**2)*(16*radial**4-16*radial**2+16), (radial, 0, sp.oo))
    e.zero("R58", "four_dimensional_gaussian_pure_gauge_cost", integral-6*sp.pi**2)
    numerical = 2*np.pi**2*quad(lambda r: r**3*np.exp(-2*r*r)*(16*r**4-16*r*r+16), 0., np.inf,
                                 epsabs=1e-11, epsrel=1e-12)[0]
    e.close("R58", "independent_pure_gauge_cost_integral", numerical, 6*np.pi**2, 1e-10)

    ids = sorted({row["claim"] for row in e.checks})
    assert ids == [f"R{i:02d}" for i in range(51, 59)]
    return {"schema": "CE-RB5-v1", "scope": "events, clocks, common geometry and fixed Maxwell candidates",
            "observational_validation": False, "full_CE_completion": False,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "helper_sha256": hashlib.sha256(Path(__file__).with_name("verify_reverse.py").read_bytes()).hexdigest(),
            "claim_ids": ids, "number_of_checks": len(e.checks), "all_passed": all(row["passed"] for row in e.checks),
            "checks": e.checks, "values": values}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results_geometry.json"))
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"PASS {result['number_of_checks']} checks; {len(result['claim_ids'])} claim groups")
