"""CE-RB3: conditional state-selection proofs and finite-record countermodels.

Synthetic verification only; no observed constants or continuum error certification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.linalg import eigh, expm
from scipy.special import eval_jacobi, roots_jacobi

from verify_reverse import Evidence


def cp2_matrices(size=32):
    x, quadrature = roots_jacobi(size+2, 0., 1.)
    y = (x+1.)/2.
    basis = np.array([np.sqrt(n+1.)*eval_jacobi(n, 0., 1., x) for n in range(size)])
    gram = (basis*(quadrature/2.))@basis.T
    ymatrix = (basis*(quadrature*y/2.))@basis.T
    free = np.diag(np.arange(size)*(np.arange(size)+2.))
    return free, ymatrix, gram


def ce_matrix(epsilon, nphi=8, nrad=5):
    # Form powers in a larger oscillator basis before projecting.
    nwide = nphi+4
    phi = np.diag(np.sqrt(np.arange(1, nwide)/2.), 1)
    phi += phi.T
    uwide = phi@phi/2.
    u = uwide[:nphi, :nphi]
    u2 = (uwide@uwide)[:nphi, :nphi]
    tphi = np.diag(np.arange(nphi)+.5)-u
    hphi = tphi+10.*(u2-u+.25*np.eye(nphi))

    def radial(dimension):
        n = np.arange(nrad)
        off = -.5*np.sqrt((n[:-1]+1.)*(n[:-1]+dimension/2.))
        op = np.diag(n+dimension/4.)+np.diag(off, 1)+np.diag(off, -1)
        kinetic = np.diag(2*n+dimension/2.)-op
        return op, kinetic

    ol, tl = radial(2)
    oh, th = radial(4)
    ir = np.eye(nrad)
    ll, hh = np.kron(ol, ir), np.kron(ir, oh)
    hr = np.kron(tl, ir)+np.kron(ir, th)+(.5-2*epsilon)*ll+(.5+epsilon)*hh
    h = np.kron(hphi, np.eye(nrad*nrad))+np.kron(np.eye(nphi), hr)+np.kron(u, ll+hh)
    d = np.kron(np.eye(nphi), -2*ll+hh)
    return h, d


def run():
    e = Evidence()
    values = {}
    rng = np.random.default_rng(20260921)

    # R41: passive rearrangement versus exactly stationary higher-energy state.
    energies = np.array([.1, 3.2, 8.4, 15.6])
    h = np.diag(energies)
    zeta = .25
    minimum = (1-zeta)*energies[0]+zeta*energies[1]
    gaps = []
    for _ in range(16):
        unitary, _ = np.linalg.qr(rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4)))
        rho = unitary@np.diag([1-zeta, zeta, 0., 0.])@unitary.conj().T
        gaps.append(float(np.trace(rho@h).real-minimum))
    e.check("R41", "rank_two_passive_lower_bound", min(gaps) >= -1e-12, {"minimum_margin": min(gaps)})
    rho02 = np.diag([1-zeta, 0., zeta, 0.])
    evolved = expm(-1j*h*1.7)@rho02@expm(1j*h*1.7)
    e.close("R41", "same_action_nonrelaxing_state", evolved, rho02)
    e.close("R41", "strict_energy_excess", np.trace(rho02@h)-minimum, zeta*(energies[2]-energies[1]))
    e.check("R41", "unitary_cannot_create_global_mixedness", np.trace(rho02@rho02) < 1.,
            {"target_purity": float(np.trace(rho02@rho02)), "pure_initial_purity": 1.})

    # R42: original regular CP2 operator and unavailable minima on a small-z range.
    y = sp.symbols("y")
    for n in range(1, 5):
        applied = -y*(1-y)*sp.diff(y**n, y, 2)-(2-3*y)*sp.diff(y**n, y)
        e.zero("R42", f"free_spectrum_leading_term_{n}", sp.expand(applied).coeff(y, n)-n*(n+2))
    free, ymatrix, gram = cp2_matrices()
    e.close("R42", "independent_jacobi_quadrature_orthonormality", gram, np.eye(len(free)), 2e-12)
    slopes = []
    zmax = np.sqrt(3*zeta)/2.
    for z in [zmax/10, zmax/2, zmax]:
        eta = z*z/2
        en, vectors = eigh(free+eta*ymatrix, subset_by_index=(0, 1))
        derivatives = np.diag(vectors.T@ymatrix@vectors)
        mixed_e = (1-zeta)*en[0]+zeta*en[1]
        mixed_d = (1-zeta)*derivatives[0]+zeta*derivatives[1]
        slope = (2*eta*mixed_d-mixed_e)/(z*z)
        upper = (z*z-3*zeta)/(z*z)
        slopes.append({"z": float(z), "slope": float(slope), "bound": float(upper)})
        e.check("R42", f"restricted_branch_decreasing_{z:.6f}", slope <= upper+1e-10 and upper < 0,
                slopes[-1])
    values["restricted_cp2_branch"] = slopes

    # R43: nonconstant moments and full KK periodic reindexing.
    q, radius, mass = sp.symbols("q R m2", positive=True)
    d = q*q/radius**2
    xs = [mass, mass+d, mass+d]
    e.zero("R43", "first_trace_derivative", sp.diff(sum(xs), q)-4*q/radius**2)
    e.zero("R43", "second_trace", sum(x*x for x in xs)-(3*mass**2+4*mass*d+2*d**2))
    mean = sum(xs)/3
    e.zero("R43", "centered_second_trace", sum((x-mean)**2 for x in xs)-2*d**2/3)
    e.zero("R43", "pointwise_theta_pi_embedding", mean-2*(d/3)-mass)
    qv = .31
    n = np.arange(-8, 9)
    e.close("R43", "full_plus_channel_reindexing", ((n-1)+(qv+1))**2, (n+qv)**2)
    e.close("R43", "full_minus_channel_reindexing", ((n+1)-(qv+1))**2, (n-qv)**2)
    e.check("R43", "fixed_zero_mode_not_periodic", qv*qv != (qv+1)**2,
            {"q_squared": qv*qv, "shifted_zero_mode": (qv+1)**2})
    step = 1e-6
    nearest = lambda v: min(v*v, (1-v)**2)
    left = (nearest(.5)-nearest(.5-step))/step
    right = (nearest(.5+step)-nearest(.5))/step
    e.check("R43", "lowest_mode_crossing", left > .99 and right < -.99,
            {"left_derivative": left, "right_derivative": right})

    # R44: stationary elimination and negative curvature after canonicalization.
    t, u, u0, k, loop = sp.symbols("t u u0 k loop", positive=True)
    along = k*(u*u*sp.exp(-5*t)-2*u*u0*sp.exp(-3*t)+u0*u0*sp.exp(-t))+loop*sp.exp(-6*t)
    loop_stationary = sp.solve(sp.diff(along, t).subs(t, 0), loop)[0]
    curvature = sp.diff(along, t, 2).subs(t, 0).subs(loop, loop_stationary)
    e.zero("R44", "same_y_stationary_curvature", curvature-k*(-5*u*u+18*u*u0-5*u0*u0))
    x = sp.symbols("x", real=True)
    poly = -5*x*x+18*x-5
    e.zero("R44", "strict_boundary_value", poly.subs(x, sp.Rational(1, 5))+sp.Rational(8, 5))
    e.zero("R44", "monotonicity_derivative", sp.diff(poly, x)-(18-10*x))
    # This matrix is a diagnostic of the Rayleigh principle, not a computed CE Hessian.
    v = np.array([1., -.2, 0.])
    vhat = v/np.linalg.norm(v)
    projector = np.outer(vhat, vhat)
    hessian = -3.25*projector+2.*(np.eye(3)-projector)
    gamma = np.array([[2., .2, .1], [.2, 1.2, -.1], [.1, -.1, .9]])
    physical = eigh(hessian, gamma, eigvals_only=True)
    rayleigh = float(v@hessian@v/(v@gamma@v))
    e.check("R44", "negative_generalized_rayleigh_direction", physical[0] <= rayleigh < 0.,
            {"generalized_eigenvalues": physical.tolist(), "rayleigh": rayleigh})

    # R45: gap, variance, trace-distance and the extra second-moment condition.
    en = np.array([1., 3., 6.])
    probs = np.array([.97, .02, .01])
    psi = np.sqrt(probs)
    average = float(en@probs)
    variance = float(((en-average)**2)@probs)
    delta, gap = average-en[0], en[1]-en[0]
    e.check("R45", "energy_gap_infidelity_bound", 1-probs[0] <= delta/gap,
            {"infidelity": 1-probs[0], "bound": delta/gap})
    rho_difference = np.outer(psi, psi)-np.diag([1., 0., 0.])
    trace_norm = float(np.sum(np.abs(np.linalg.eigvalsh(rho_difference))))
    e.close("R45", "pure_state_trace_distance", trace_norm, 2*np.sqrt(1-probs[0]))
    observable = np.array([[0., 1., .3], [1., 2., -.2], [.3, -.2, -.1]])
    mean_error = abs(float(psi@observable@psi)-observable[0, 0])
    e.check("R45", "bounded_observable_transfer", mean_error <= 2*np.linalg.norm(observable, 2)*np.sqrt(delta/gap),
            {"mean_error": mean_error, "energy_error": delta, "gap": gap})
    excited_lower = 2.8
    temple = variance/(excited_lower-average)
    e.check("R45", "variance_energy_certificate", delta <= temple,
            {"delta": delta, "certificate": temple, "excited_lower": excited_lower})
    e.check("R45", "variance_overlap_certificate", 1-probs[0] <= variance/(excited_lower-average)**2,
            {"infidelity": 1-probs[0], "bound": variance/(excited_lower-average)**2})
    lam, uv, uref = sp.symbols("lambda u u0", positive=True)
    e.zero("R45", "higgs_coercive_moment_bound", lam*(uv-uref)**2-(lam*uv**2/2-lam*uref**2)-lam*(uv-2*uref)**2/2)
    coordinate = np.array([.2, 1.7, 8.])
    msum = float(probs@coordinate**2+coordinate[0]**2)
    optimal_r = np.sqrt(msum/trace_norm)
    capped = np.minimum(coordinate, optimal_r)
    actual = abs(float(probs@coordinate-coordinate[0]))
    e.check("R45", "moment_tail_observable_bound", actual <= 2*np.sqrt(trace_norm*msum),
            {"actual": actual, "bound": 2*np.sqrt(trace_norm*msum), "cutoff": optimal_r,
             "capped_difference": float(probs@capped-capped[0])})
    gapless_h = np.diag([0., 1e-8])
    gapless_psi = np.array([0., 1.])
    gapless_en, gapless_vec = eigh(gapless_h)
    gapless_average = float(gapless_psi@gapless_h@gapless_psi)
    gapless_overlap = float(abs(gapless_vec[:, 0]@gapless_psi)**2)
    gapless_variance = float(np.linalg.norm((gapless_h-gapless_average*np.eye(2))@gapless_psi)**2)
    e.check("R45", "gapless_excited_countermodel", gapless_average-gapless_en[0] < 1e-6
            and gapless_overlap == 0. and gapless_variance == 0.,
            {"H": gapless_en.tolist(), "energy_error": gapless_average-gapless_en[0],
             "ground_overlap": gapless_overlap, "variance": gapless_variance})

    # R46: independently reconstruct from full and projected covariance.
    kinetic = np.array([[0., -.3, .2], [-.3, 4., 0.], [.2, 0., 7.]])
    basis_op = np.array([[0., 1., 2.], [1., 0., 0.], [2., 0., 0.]])
    target_c = .3
    h = kinetic+target_c*basis_op
    trial = np.array([1., 0., 0.])
    average = trial@h@trial
    residual = (h-average*np.eye(3))@trial
    w = (basis_op-(trial@basis_op@trial)*np.eye(3))@trial
    covariance = w@w
    b = w@(kinetic@trial)
    estimate = -b/covariance
    e.close("R46", "projected_eigenstate_hides_residual", residual[:2], [0., 0.])
    e.close("R46", "full_covariance_bias_identity", estimate-target_c, -(w@residual)/covariance)
    e.close("R46", "explicit_wrong_full_estimate", estimate, -.02)
    e.check("R46", "singular_value_bias_bound", abs(estimate-target_c) <= np.linalg.norm(residual)/np.sqrt(covariance),
            {"bias": estimate-target_c, "full_residual_norm": float(np.linalg.norm(residual)), "covariance": covariance})
    dm, db = .02, -.01
    noisy_estimate = -(b+db)/(covariance+dm)
    noisy_bound = (abs(w@residual)+abs(db)+abs(dm)*abs(target_c))/(covariance-abs(dm))
    e.check("R46", "noise_and_state_residual_bound", abs(noisy_estimate-target_c) <= noisy_bound,
            {"noisy_error": abs(noisy_estimate-target_c), "bound": noisy_bound})
    # A residual orthogonal to the dictionary does not bias its coefficient.
    unseen = np.array([0., -2., 1.])
    e.close("R46", "dictionary_orthogonal_residual", w@unseen, 0.)
    e.check("R46", "zero_bias_does_not_certify_state", np.linalg.norm(unseen) > 0., {"hidden_norm": float(np.linalg.norm(unseen))})

    # R47: smooth, positive exact ground states, same resolved moments, unstable V.
    angle = np.arange(4096)*2*np.pi/4096
    aq = .5
    band = 3
    functions = np.array([np.ones_like(angle)]+[f(j*angle) for j in range(1, band+1) for f in [np.cos, np.sin]])
    derivatives = np.array([np.zeros_like(angle)]+[f(j*angle)*scale for j in range(1, band+1)
                            for f, scale in [(np.sin, -j), (np.cos, j)]])
    uniform_moment = (functions@functions.T)/len(angle)
    uniform_m1 = .5*(derivatives@derivatives.T)/len(angle)
    instability = []
    for frequency in [8, 16, 32]:
        cosine, sine = np.cos(frequency*angle), np.sin(frequency*angle)
        density = (1+aq/frequency**2*cosine)/(2*np.pi)
        first = -aq/frequency*sine/(2*np.pi)
        second = -aq*cosine/(2*np.pi)
        wave = np.sqrt(density)
        wave_second = second/(2*wave)-first*first/(4*wave**3)
        potential = second/(4*density)-first*first/(8*density*density)
        e.close("R47", f"exact_ground_equation_{frequency}", -.5*wave_second+potential*wave, 0., 1e-14)
        weights = density*2*np.pi/len(angle)
        e.close("R47", f"same_static_resolved_moments_{frequency}", (functions*weights)@functions.T, uniform_moment, 3e-14)
        e.close("R47", f"same_first_energy_moments_{frequency}", .5*(derivatives*weights)@derivatives.T, uniform_m1, 3e-14)
        fisher = float(np.sum(first*first/density)*2*np.pi/len(angle))
        fisher_bound = aq*aq/(2*frequency**2*(1-aq/frequency**2))
        e.check("R47", f"small_fisher_large_potential_{frequency}", fisher <= fisher_bound+1e-14 and np.ptp(potential) > aq/2,
                {"fisher": fisher, "bound": fisher_bound, "potential_range": float(np.ptp(potential))})
        instability.append({"k": frequency, "density_C0": float(np.max(np.abs(density-1/(2*np.pi)))),
                            "density_derivative_C0": float(np.max(np.abs(first))), "V_C0": float(np.max(np.abs(potential)))})
    values["inverse_density_instability"] = instability

    # R48: analytic same-family kappa=0, and an interacting finite-basis diagnostic.
    eps, s0 = sp.symbols("epsilon s0", positive=True)
    ce_gaussian_energy = sp.sqrt(s0-2*eps)+2*sp.sqrt(s0+eps)
    e.zero("R48", "symmetric_zero_slope", sp.diff(ce_gaussian_energy, eps).subs(eps, 0))
    e.zero("R48", "strict_gaussian_curvature", sp.diff(ce_gaussian_energy, eps, 2)
           +(s0-2*eps)**(-sp.Rational(3, 2))+(s0+eps)**(-sp.Rational(3, 2))/2)
    h, derivative = ce_matrix(.15)
    en, vectors = eigh(h)
    ground = vectors[:, 0]
    couplings = vectors[:, 1:].T@derivative@ground
    analytic_second = float(-2*np.sum(couplings*couplings/(en[1:]-en[0])))
    ds = 1e-4
    lows = [eigh(ce_matrix(.15+shift)[0], eigvals_only=True, subset_by_index=(0, 0))[0] for shift in [-ds, 0., ds]]
    finite_second = (lows[0]-2*lows[1]+lows[2])/(ds*ds)
    slope = float(ground@derivative@ground)
    e.close("R48", "interacting_finite_basis_second_derivative", analytic_second, finite_second, 5e-5)
    e.check("R48", "interacting_negative_slope_and_curvature", slope < 0. and analytic_second < 0.,
            {"basis_dimension": len(h), "energy_upper_bound": float(en[0]), "slope": slope,
             "second_derivative": analytic_second, "continuum_certified": False})
    shifted_grounds, excitation_gaps = [], []
    for chosen_epsilon in [.1, .2]:
        candidate, _ = ce_matrix(chosen_epsilon)
        original = eigh(candidate, eigvals_only=True, subset_by_index=(0, 1))
        shifted = eigh(candidate-original[0]*np.eye(len(candidate)), eigvals_only=True, subset_by_index=(0, 1))
        shifted_grounds.append(float(shifted[0]))
        excitation_gaps.append(float(shifted[1]-shifted[0]))
    e.check("R48", "parameterwise_subtraction_erases_selection_only", max(map(abs, shifted_grounds)) < 1e-11
            and abs(excitation_gaps[0]-excitation_gaps[1]) > 1e-4,
            {"shifted_ground_energies": shifted_grounds, "different_excitation_gaps": excitation_gaps})

    ids = sorted({row["claim"] for row in e.checks})
    assert ids == [f"R{i:02d}" for i in range(41, 49)]
    return {"schema": "CE-RB3-v1", "scope": "source-specific state and inverse-record boundaries",
            "observational_validation": False, "full_CE_completion": False,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "helper_sha256": hashlib.sha256(Path(__file__).with_name("verify_reverse.py").read_bytes()).hexdigest(),
            "claim_ids": ids, "number_of_checks": len(e.checks), "all_passed": all(row["passed"] for row in e.checks),
            "checks": e.checks, "values": values}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results_states.json"))
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"PASS {result['number_of_checks']} checks; {len(result['claim_ids'])} claim groups")
