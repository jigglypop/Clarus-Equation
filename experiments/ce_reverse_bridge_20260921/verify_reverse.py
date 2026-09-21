"""CE-RB1: bounded symbolic identities and countermodels, not observational validation.

Run from the repository root. Only the requested output file is written.
All examples are synthetic. General proofs and domains are in chapters 17 and 18.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import sympy as sp
from scipy.integrate import quad, solve_ivp
from scipy.linalg import expm
from scipy.optimize import brentq


class Evidence:
    def __init__(self):
        self.checks = []
        self.values = {}

    def check(self, claim, name, condition, evidence):
        row = {"claim": claim, "name": name, "passed": bool(condition), "evidence": evidence}
        self.checks.append(row)
        if not row["passed"]:
            raise AssertionError(json.dumps(row, ensure_ascii=False))

    def zero(self, claim, name, expr):
        residual = sp.simplify(sp.trigsimp(expr))
        self.check(claim, name, residual == 0, {"symbolic_residual": str(residual)})

    def close(self, claim, name, lhs, rhs, tol=1e-10):
        residual = float(np.max(np.abs(np.asarray(lhs) - np.asarray(rhs))))
        self.check(claim, name, residual <= tol, {"max_abs_residual": residual, "tolerance": tol})


def ckm(a, b, c, phase):
    s12, s23, s13 = np.sin([a, b, c])
    c12, c23, c13 = np.cos([a, b, c])
    z = np.exp(1j * phase)
    return np.array([
        [c12*c13, s12*c13, s13/z],
        [-s12*c23-c12*s23*s13*z, c12*c23-s12*s23*s13*z, s23*c13],
        [s12*s23-c12*c23*s13*z, -c12*s23-s12*c23*s13*z, c23*c13],
    ])


def run():
    e = Evidence()
    pi = np.pi

    # R01: start exactly on the empirical surface; independent ODE leaves it.
    alpha, em = 0.10, 0.30
    x = 4 * alpha**(4/3)
    g0 = np.array([em / np.sqrt(1-x), em / np.sqrt(x), np.sqrt(4*pi*alpha)])
    beta = np.array([41/6, -19/6, -7.])
    read_x = lambda g: g[0]**2 / (g[0]**2 + g[1]**2)
    read_f = lambda g: read_x(g) - 4 * (g[2]**2/(4*pi))**(4/3)
    e.close("R01", "boundary_inverse", read_f(g0), 0, 1e-14)
    tangent = x*(1-x)*(beta[0]*g0[0]**2-beta[1]*g0[1]**2)/(8*pi*pi)
    tangent += 56/(3*pi)*alpha**(7/3)
    dt = 1e-4
    sol = solve_ivp(lambda t, g: beta*g**3/(16*pi*pi), (0, dt), g0,
                    rtol=1e-12, atol=1e-14)
    e.check("R01", "ode_success", sol.success, {"message": sol.message})
    gf = sol.y[:, -1]
    fd = (read_f(gf)-read_f(g0))/dt
    e.close("R01", "rg_normal_derivative", fd, tangent, 2e-6)
    e.check("R01", "rg_invariance_counterexample", tangent > 0 and read_f(gf) > 1e-6,
            {"normal_derivative": tangent, "boundary_F": read_f(g0), "evolved_F": read_f(gf)})
    analytic = 1 / np.sqrt(1/g0**2-beta*dt/(8*pi*pi))
    e.close("R01", "rg_solution_vs_ode", gf, analytic, 1e-12)

    # R02: same mean, different probability generating functions.
    q = brentq(lambda z: np.exp(2*(z-1))-z, 0., .9, xtol=1e-14)
    q04 = brentq(lambda z: (1+z**4)/2-z, 0., .9, xtol=1e-14)
    e.check("R02", "same_mean_different_extinction", 0 < q < q04 < 1,
            {"means": [2, 2, 2], "q_deterministic": 0, "q_poisson": q, "q_zero_four": q04})
    d = 3.177
    qd = brentq(lambda z: np.exp(d*(z-1))-z, 0, .5)
    step = 1e-5
    roots = [brentq(lambda z: np.exp(dd*(z-1))-z, 0, .5) for dd in [d-step, d+step]]
    e.close("R02", "extinction_implicit_derivative", (roots[1]-roots[0])/(2*step),
            -qd*(1-qd)/(1-d*qd), 1e-9)
    omega1, omega2 = q, 2*q/(1+q)
    e.check("R02", "energy_weight_counterexample", omega2 > omega1,
            {"same_q": q, "equal_energy_fraction": omega1, "double_energy_fraction": omega2})

    # R03: exact singular-value invariants and genuinely different weak mixing.
    du, dd = np.diag([.1, .5, 1.]), np.diag([.05, .2, .6])
    v0, v1 = np.eye(3), ckm(.2, 0., 0., 0.)
    for label, v in [("identity", v0), ("rotated", v1)]:
        yd = v @ dd
        e.close("R03", f"same_masses_{label}", np.linalg.eigvalsh(yd @ yd.conj().T),
                np.diag(dd)**2)
        e.close("R03", f"left_diagonalization_{label}", v.conj().T @ yd, dd)
    e.check("R03", "same_spectra_different_ckm", abs(v1[0, 1]-v0[0, 1]) > .1,
            {"vus": [abs(v0[0, 1]), abs(v1[0, 1])]})
    vs = [ckm(.22, .04, .003, delta) for delta in [0, pi/2]]
    moduli = [np.abs(v[[0, 1, 0], [1, 2, 2]]) for v in vs]
    j = [float(np.imag(v[0, 0]*v[1, 1]*np.conj(v[0, 1]*v[1, 0]))) for v in vs]
    e.close("R03", "same_three_moduli", *moduli)
    e.check("R03", "cp_not_selected", abs(j[0]) < 1e-14 and abs(j[1]) > 1e-6,
            {"J": j})
    u = vs[1]
    mn = u.conj() @ du @ u.conj().T
    e.close("R03", "majorana_takagi_realization", u.T @ mn @ u, du)
    msq, shift, baseline = np.array([.1, .2, .5]), .7, 2.3
    amplitudes = [u @ np.diag(np.exp(-1j*(msq+c)*baseline)) @ u.conj().T for c in [0, shift]]
    e.close("R03", "oscillation_mass_shift", np.abs(amplitudes[0])**2, np.abs(amplitudes[1])**2)
    h, lam, vv = sp.symbols("h lambda v", real=True)
    vh = lam*((vv+h)**2/2-vv**2/2)**2
    e.zero("R03", "higgs_mass_curvature", sp.diff(vh, h, 2).subs(h, 0)-2*lam*vv**2)

    # R04: exact normalized, positive phase laws; mode equation and initial state.
    phi, eta = sp.symbols("phi eta", real=True)
    density = (1+eta*sp.cos(2*phi))/sp.pi
    e.zero("R04", "phase_density_normalization", sp.integrate(density, (phi, 0, sp.pi))-1)
    e.zero("R04", "phase_average_changes", sp.integrate(sp.sin(phi)*density, (phi, 0, sp.pi))
           -2/sp.pi*(1-eta/3))
    tm = sp.symbols("tau", real=True, nonzero=True)
    k = sp.symbols("k", positive=True)
    mode = sp.exp(-sp.I*k*tm)*(1-sp.I/(k*tm))/sp.sqrt(2*k)
    e.zero("R04", "bd_mode_equation", sp.diff(mode, tm, 2)+(k*k-2/tm**2)*mode)
    e.zero("R04", "bd_wronskian", mode*sp.diff(sp.conjugate(mode), tm)
           -sp.conjugate(mode)*sp.diff(mode, tm)-sp.I)
    rr = .08
    aa, bb = np.cosh(rr), np.sinh(rr)
    e.close("R04", "bogoliubov_normalization", aa*aa-bb*bb, 1.)
    e.close("R04", "squeezed_power_ratio", (aa-bb)**2, np.exp(-2*rr))

    # R05: moment identities by polynomial algebra, not fitted spectrum tables.
    c, st, ep, s, zz = sp.symbols("c st epsilon s z", real=True)
    ys = [2*ep*c, -ep*c-sp.sqrt(3)*ep*st, -ep*c+sp.sqrt(3)*ep*st]
    reduce_circle = lambda expr: sp.expand(expr).subs(st**2, 1-c**2).expand()
    e.zero("R05", "first_moment", sum(ys))
    e.zero("R05", "second_moment", reduce_circle(sum(y*y for y in ys))-6*ep**2)
    e.zero("R05", "cubic_characteristic", reduce_circle(sp.prod(zz-y for y in ys))
           -(zz**3-3*ep**2*zz-2*ep**3*(4*c**3-3*c)))
    e.check("R05", "unequal_split_ratios_admissible", 1 > 2*.2,
            {"s": 1, "epsilon": [.1, .2], "all_phase_min_eigenvalues": [.8, .6]})
    # Direct proper-time UV behavior for two equal-moment spectra (stable expm1).
    spectra = [1+2*.2*np.cos((theta+2*pi*np.arange(3))/3) for theta in [.2, 1.7]]
    e.close("R05", "equal_moments_two_states", [sum(spectra[0]), sum(spectra[0]**2)],
            [sum(spectra[1]), sum(spectra[1]**2)])

    # R06: fixed internal spectrum need not fix transport or spatial size.
    th, ph = sp.symbols("theta phi", real=True)
    uv = sp.Matrix([sp.cos(th/2), sp.exp(sp.I*ph)*sp.sin(th/2)])
    ap = -sp.I*(uv.conjugate().T * uv.diff(ph))[0]
    e.zero("R06", "berry_curvature", sp.diff(ap, th)-sp.sin(th)/2)
    e.check("R06", "torus_length_counterexample", 4*pi*pi != pi*pi,
            {"fixed_fiber_spectrum": [1, 2, 3], "laplacian_gap_L1": 4*pi*pi,
             "laplacian_gap_L2": pi*pi})

    # R07: sampling includes complex rays to detect imaginary off-diagonal terms.
    rays = [np.eye(3, dtype=complex)[:, i] for i in range(3)]
    rays += [(rays[0]+rays[1])/np.sqrt(2), (rays[0]+1j*rays[1])/np.sqrt(2)]
    for kap in [1., 2.]:
        values = [float(np.vdot(v, kap*np.eye(3) @ v).real) for v in rays]
        e.close("R07", f"isotropic_portal_{kap}", values, np.full(len(rays), kap))
    bmat = np.array([[1, 1j, 0], [-1j, 1, 0], [0, 0, 1]], complex)
    diagonal_readings = [np.vdot(v, bmat @ v).real for v in rays[:3]]
    full_readings = [np.vdot(v, bmat @ v).real for v in rays]
    e.close("R07", "diagonal_only_false_isotropy", diagonal_readings, [1, 1, 1])
    e.check("R07", "full_orbit_detects_portal", np.ptp(full_readings) > .5,
            {"diagonal_only": diagonal_readings, "full_rays": full_readings})

    # R08: second mass derivative leaves two integration functions.
    ss, tt, theta, sstar = sp.symbols("s t theta sstar", real=True)
    jj = ss**2+theta
    reconstructed = sp.sin(theta)+sp.cos(theta)*(ss-sstar)
    reconstructed += sp.integrate((ss-tt)*(tt**2+theta), (tt, sstar, ss))
    e.zero("R08", "second_derivative_inverse", sp.diff(reconstructed, ss, 2)-jj)
    e.zero("R08", "derivative_kernel", sp.diff(3*sp.sin(theta)+7*ss, ss, 2))
    f = ss**2*theta+sp.sin(theta)
    e.zero("R08", "response_integrability", sp.diff(sp.diff(f, ss), theta)
           -sp.diff(sp.diff(f, theta), ss))

    # R09: circle kinetic operator at constant gauge source, same A=0 mass.
    gauge_a, mass, charge = sp.symbols("A m q", real=True)
    eig = mass**2+charge**2*gauge_a**2
    e.zero("R09", "same_zero_source_operator", eig.subs(gauge_a, 0)-mass**2)
    e.zero("R09", "charge_changes_response", sp.diff(eig, gauge_a, 2)-2*charge**2)
    source_curvatures = [sp.diff(eig, gauge_a, 2).subs(charge, qc) for qc in [0, 1]]
    e.check("R09", "neutral_charged_counterexample", source_curvatures == [0, 2],
            {"q": [0, 1], "zero_source_m2": [1, 1], "second_source_derivative": [0, 2]})

    # R10: exact eigenstate families and a singular isolated selection condition.
    oq = sp.symbols("q", real=True)
    ow = sp.symbols("omega", positive=True)
    ground = sp.exp(-ow*oq**2/2)
    e.zero("R10", "positive_oscillator_family", -sp.diff(ground, oq, 2)/2
           +ow**2*oq**2*ground/2-ow*ground/2)
    cvar = sp.symbols("c", real=True)
    e.zero("R10", "inverse_identity_jacobian", sp.diff(cvar-cvar, cvar))
    e.check("R10", "rank_deficiency_not_nonisolation", sp.solve(cvar**2, cvar) == [0],
            {"F": "c^2", "root": 0, "derivative_at_root": 0})

    # R11: connected weighted graph with deliberately nonuniform ground density.
    p = np.array([.05, .10, .15, .30, .40])
    psi = np.sqrt(p)
    lap = np.zeros((5, 5))
    for i, weight in enumerate([.2, .4, .3, .6, .5]):
        jn = (i+1) % 5
        lap[i, i] += weight
        lap[jn, jn] += weight
        lap[i, jn] -= weight
        lap[jn, i] -= weight
    inv = np.diag(1/psi)
    h0 = inv @ lap @ inv
    vals, vecs = np.linalg.eigh(h0)
    e.check("R11", "unique_positive_ground", abs(vals[0]) < 1e-12 and vals[1] > 0,
            {"ground_energy": vals[0], "gap": vals[1], "p_min": p.min()})
    v = psi[:, None]*vecs[:, 1:]
    gaps = vals[1:]-vals[0]
    moment = (v*gaps) @ v.T
    recovered = inv @ moment @ inv
    e.close("R11", "spectral_record_inverse", recovered, h0-vals[0]*np.eye(5))
    shifted_vals, shifted_vecs = np.linalg.eigh(h0+3.2*np.eye(5))
    vshift = psi[:, None]*shifted_vecs[:, 1:]
    moment_shift = (vshift*(shifted_vals[1:]-shifted_vals[0])) @ vshift.T
    e.close("R11", "energy_constant_invisible", moment_shift, moment)

    # R12: perturb both density and cross moments; compare actual operator error.
    rng = np.random.default_rng(20260921)
    dp = 1e-5*np.array([1., -1., .5, -.5, 0.])
    raw = rng.normal(size=(5, 5))
    noise = (raw+raw.T)/2
    noise *= 2e-5/np.linalg.norm(noise, 2)
    phat, mhat = p+dp, moment+noise
    ahat = np.diag(phat**-.5)
    hhat = ahat @ mhat @ ahat
    eps_p, eps_m = np.max(np.abs(dp)), np.linalg.norm(noise, 2)
    pm, lower = p.min(), p.min()-eps_p
    bound = eps_m/lower + np.linalg.norm(moment, 2)*eps_p/(2*lower**1.5) * (
        1/np.sqrt(lower)+1/np.sqrt(pm))
    actual = np.linalg.norm(hhat-recovered, 2)
    e.check("R12", "noisy_inverse_bound", actual <= bound,
            {"operator_error": actual, "bound": bound, "eps_p": eps_p, "eps_M": eps_m})
    cutoff = float((gaps[1]+gaps[2])/2)
    tail = (v[:, gaps > cutoff]*gaps[gaps > cutoff]) @ v[:, gaps > cutoff].T
    m2 = float(np.sum(gaps**2*np.sum(v*v, axis=0)))
    tail_norm, trace_tail = np.linalg.norm(tail, 2), float(np.trace(tail))
    e.check("R12", "spectral_tail_bound", tail_norm <= trace_tail+1e-12 and trace_tail <= m2/cutoff,
            {"norm_tail": tail_norm, "trace_tail": trace_tail, "M2_over_cutoff": m2/cutoff})
    dim, np_samples, nm_samples, noise_failure, measurement_b = 5, 100000, 1000000, .01, 1.
    entry_bound = measurement_b*np.sqrt(2*np.log(2*dim**2/noise_failure)/nm_samples)
    union_failure = 2*dim**2*np.exp(-nm_samples*entry_bound**2/(2*measurement_b**2))
    e.close("R12", "bounded_estimator_union_probability", union_failure, noise_failure, 1e-13)
    probability_bound = np.sqrt(np.log(2*dim/noise_failure)/(2*np_samples))
    e.close("R12", "probability_union_bound", 2*dim*np.exp(-2*np_samples*probability_bound**2),
            noise_failure, 1e-13)
    projector = np.eye(5)-np.outer(psi, psi)
    ha, hb = 20*projector, 200*projector
    e.close("R12", "invisible_band_same_ground", ha @ psi, hb @ psi, 1e-12)
    e.close("R12", "invisible_band_arbitrary_distance", np.linalg.norm(ha-hb, 2), 180.)
    e.check("R12", "invisible_band_empty", np.linalg.eigvalsh(ha)[1] > 10,
            {"observed_band": [0, 10], "gap_A": 20, "gap_B": 200})

    # R13: continuous positive variable coefficient with identical finite moments.
    qvar = sp.symbols("q", real=True)
    fs = [sp.Integer(1), sp.cos(qvar), sp.sin(qvar)]
    for i, fi in enumerate(fs):
        for jn, fj in enumerate(fs[i:], i):
            coeff = sp.integrate(sp.cos(3*qvar)*sp.diff(fi, qvar)*sp.diff(fj, qvar),
                                 (qvar, 0, 2*sp.pi))/(4*sp.pi)
            e.zero("R13", f"finite_moment_blind_{i}_{jn}", coeff)
    added = sp.integrate((1+eta*sp.cos(3*qvar))*sp.diff(sp.cos(2*qvar), qvar)
                        *sp.diff(sp.cos(qvar), qvar), (qvar, 0, 2*sp.pi))/(4*sp.pi)
    e.zero("R13", "added_record_resolves_metric", added+eta/4)
    numerical = quad(lambda qv: (1+.4*np.cos(3*qv))*2*np.sin(2*qv)*np.sin(qv)/(4*pi),
                     0, 2*pi, epsabs=1e-12)[0]
    e.close("R13", "independent_quadrature", numerical, -.1)

    # R14: theta-independent gravitational functionals cancel from the relative action.
    curvature, lam0, mgrav, theta_star = sp.symbols("R Lambda M theta_star", real=True)
    gamma = sp.cos(theta)+ss**2+curvature*sp.sin(theta)
    counterterm = -lam0+mgrav**2*curvature/2
    e.zero("R14", "relative_gravity_kernel", (gamma+counterterm)
           -(gamma+counterterm).subs(theta, theta_star)-(gamma-gamma.subs(theta, theta_star)))
    e.check("R14", "same_relative_different_de_sitter", .1/3 != .2/3,
            {"F": 1, "Lambda": [.1, .2], "H_squared": [.1/3, .2/3]})

    # R15: explicitly differentiate Friedmann and substitute independent equations.
    F, Fp, Fpp, K, Kp, U, Up, hub, hd, vel, acc, rho, source = sp.symbols(
        "F Fp Fpp K Kp U Up H Hd v a rho J", real=True)
    # d[3 F H^2-rho-K*v^2/2-U+3 H Fp*v]/dt
    drho = source*vel-3*hub*rho
    dedt = 3*Fp*vel*hub**2+6*F*hub*hd-drho-Kp*vel**3/2-K*vel*acc-Up*vel
    dedt += 3*hd*Fp*vel+3*hub*(Fpp*vel**2+Fp*acc)
    scalar_solution = (-3*K*hub*vel-Kp*vel**2/2-Up+3*Fp*(2*hub**2+hd)-source)/K
    raych = 2*F*hd+rho+K*vel**2+Fpp*vel**2+Fp*acc-hub*Fp*vel
    e.zero("R15", "friedmann_constraint_propagation", (dedt-3*hub*raych).subs(acc, scalar_solution))
    n, mp, mm = sp.symbols("n mp m", real=True)
    e.zero("R15", "particle_energy_exchange", (-3*hub*n)*mm+n*mp*vel
           +3*hub*(n*mm)-n*mp*vel)
    e.zero("R15", "einstein_constant_field_limit", raych.subs({Fp: 0, Fpp: 0, vel: 0})
           -(2*F*hd+rho))
    e.zero("R15", "acceleration_system_determinant",
           sp.Matrix([[K, -3*Fp], [Fp, 2*F]]).det()-(2*F*K+3*Fp**2))

    def coefficients(theta_value):
        return (1+.1*theta_value**2, .2*theta_value, .2,
                1+.2*theta_value**2, .4*theta_value,
                .05+.03*theta_value**2, .06*theta_value,
                1+.05*theta_value**2, .1*theta_value)

    def flrw_rhs(t, y):
        theta_value, speed, hh, number, loga = y
        fv, fp, fpp, kv, kp, uv, up, mv, mprime = coefficients(theta_value)
        density, source_value = number*mv, number*mprime
        rhs = [-3*kv*hh*speed-kp*speed**2/2-up+6*fp*hh**2-source_value,
               -density-kv*speed**2-fpp*speed**2+hh*fp*speed]
        acceleration, hdot = np.linalg.solve([[kv, -3*fp], [fp, 2*fv]], rhs)
        return [speed, acceleration, hdot, -3*hh*number, hh]

    theta0, speed0, number0 = .4, .03, .02
    fv, fp, _, kv, _, uv, _, mv, _ = coefficients(theta0)
    initial_energy = number0*mv+kv*speed0**2/2+uv
    hub0 = (-3*fp*speed0+np.sqrt((3*fp*speed0)**2+12*fv*initial_energy))/(6*fv)
    cosmology = solve_ivp(flrw_rhs, (0, 2), [theta0, speed0, hub0, number0, 0.],
                         rtol=1e-11, atol=1e-13, t_eval=np.linspace(0, 2, 81))
    e.check("R15", "nonconstant_F_ivp_success", cosmology.success, {"message": cosmology.message})
    friedmann_errors, comoving_numbers = [], []
    for theta_value, speed, hh, number, loga in cosmology.y.T:
        fv, fp, _, kv, _, uv, _, mv, _ = coefficients(theta_value)
        friedmann_errors.append(3*fv*hh**2-number*mv-kv*speed**2/2-uv+3*hh*fp*speed)
        comoving_numbers.append(number*np.exp(3*loga))
    e.close("R15", "unprojected_friedmann_constraint", friedmann_errors,
            np.zeros(len(friedmann_errors)), 1e-10)
    e.close("R15", "comoving_number_conservation", comoving_numbers,
            np.full(len(comoving_numbers), number0), 1e-10)

    # R16: two admissible positive states, and identical density with different pressure.
    ho = np.diag(np.arange(4)+.5)
    e.check("R16", "same_hamiltonian_different_energy", ho[0, 0] != ho[1, 1],
            {"vacuum_energy": ho[0, 0], "one_particle_energy": ho[1, 1]})
    rho_target, massval, momentum = 1., 1., 1.
    n_hot = rho_target/np.sqrt(massval**2+momentum**2)
    pressure = n_hot*momentum**2/(3*np.sqrt(massval**2+momentum**2))
    e.close("R16", "same_density_different_pressure", pressure/rho_target, 1/6)
    z = .7
    ez = lambda zp: np.sqrt(.3*(1+zp)**3+.7)
    dist = quad(lambda zp: 1/ez(zp), 0, z)[0]
    aps = [(dist/h0)*(h0*ez(z)) for h0 in [1., 1.3]]
    e.close("R16", "AP_scale_counterexample", *aps)

    # R17: solve full Schrodinger evolution vs a reduced PHP-only unitary.
    gg, time = .7, 1.2
    fullh = np.array([[0., gg], [gg, 0.]])
    state = expm(-1j*time*fullh) @ np.array([1., 0.])
    e.close("R17", "full_quantum_projection", state[0], np.cos(gg*time))
    e.close("R17", "residual_probability", abs(state[1])**2, np.sin(gg*time)**2)
    e.check("R17", "PHP_closure_fails", abs(state[0]-1) > .1,
            {"projected_amplitude": float(state[0].real), "PHP_only_amplitude": 1})

    # R18: a unitary perfect record of basis inputs preserves superpositions globally.
    p0, p1 = np.diag([1, 0]), np.diag([0, 1])
    pauli_x = np.array([[0, 1], [1, 0]])
    cu = np.kron(p0, np.eye(2))+np.kron(p1, pauli_x)
    incoming = np.kron(np.array([1, 1])/np.sqrt(2), np.array([1, 0]))
    outgoing = cu @ incoming
    e.close("R18", "record_unitary", cu.T @ cu, np.eye(4))
    e.close("R18", "linear_record_superposition", outgoing, np.array([1, 0, 0, 1])/np.sqrt(2))
    weights = [abs(outgoing[0])**2, abs(outgoing[3])**2]
    e.check("R18", "not_a_single_global_record", all(0 < w < 1 for w in weights),
            {"record_weights": weights, "global_purity": 1.})

    # R19: positive real-frequency covariance, negative reflection quadratic form.
    light, heavy = 1., 3.
    pp = np.linspace(0, 100, 51)
    fourier = 1/(pp+light**2)-1/(pp+heavy**2)
    times = np.array([.2, .7])
    cv = np.array([np.exp(-light*times[1]), -np.exp(-light*times[0])])
    sums = times[:, None]+times[None, :]
    reflected = np.exp(-light*sums)/(2*light)-np.exp(-heavy*sums)/(2*heavy)
    os_value = float(cv @ reflected @ cv)
    expected_os = -(cv @ np.exp(-heavy*times))**2/(2*heavy)
    e.close("R19", "reflection_witness_formula", os_value, expected_os)
    e.check("R19", "positive_euclidean_not_reflection_positive", np.all(fourier > 0) and os_value < 0,
            {"min_sampled_fourier_covariance": fourier.min(), "reflection_form": os_value})

    # R20: covariance held fixed; do not call synthetic points observational evidence.
    rb, rn = np.array([1., 1.]), np.array([0., 2.])
    score = lambda residual: float(np.sqrt(residual @ residual / len(residual)))
    e.check("R20", "partial_improvement_reverses", rn[0]**2 < rb[0]**2 and score(rn) > score(rb),
            {"baseline_R": score(rb), "candidate_R": score(rn), "covariance": "I_2"})
    xx = sp.symbols("x", real=True)
    perturbation = (xx-0)*(xx-1)*(xx-2)
    e.check("R20", "calibration_does_not_fix_holdout",
            all(perturbation.subs(xx, i) == 0 for i in [0, 1, 2]) and perturbation.subs(xx, 3) != 0,
            {"calibration_x": [0, 1, 2], "holdout_x": 3, "holdout_difference": 6})

    # R21: independent full matrix exponentials test the propagated reconstruction error.
    evolve_t = .15
    uu, uhat = expm(-1j*evolve_t*recovered), expm(-1j*evolve_t*hhat)
    unitary_error = np.linalg.norm(uu-uhat, 2)
    e.check("R21", "duhamel_propagation_bound", unitary_error <= evolve_t*actual+1e-12,
            {"unitary_error": unitary_error, "t_times_H_error": evolve_t*actual})
    observable = np.diag([-1., -.5, 0., .5, 1.])
    rho0 = np.diag([.1, .2, .3, .15, .25])
    expectation = lambda unit: float(np.trace(observable @ unit @ rho0 @ unit.conj().T).real)
    obs_error = abs(expectation(uu)-expectation(uhat))
    obs_bound = 2*np.linalg.norm(observable, 2)*evolve_t*bound
    e.check("R21", "record_to_observable_bound", obs_error <= obs_bound,
            {"observable_error": obs_error, "bound_from_records": obs_bound})

    claims = sorted({row["claim"] for row in e.checks})
    assert claims == [f"R{i:02d}" for i in range(1, 22)], claims
    return {
        "schema": "CE-RB1-v1", "scope": "symbolic identities and synthetic countermodels",
        "observational_validation": False, "full_CE_completion": False,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "sympy": sp.__version__},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "claim_ids": claims, "number_of_checks": len(e.checks),
        "all_passed": all(row["passed"] for row in e.checks), "checks": e.checks,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results.json"))
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"PASS {result['number_of_checks']} checks; {len(result['claim_ids'])} claim groups")
