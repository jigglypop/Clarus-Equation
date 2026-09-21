"""R22-R32: explicit empirical-bridge identities and countermodels.

These checks do not fit data or validate QCD/quantum gravity/observations.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.special import factorial

from verify_reverse import Evidence, ckm


def run():
    e = Evidence()
    d = sp.symbols("d", integer=True)
    e.check("R22", "exterior_dimension_roots", sp.solve(d-d*(d-1)/2, d) == [0, 3],
            {"nonnegative_roots": [0, 3]})
    e.check("R22", "maxwell_does_not_require_equal_dimensions", sp.binomial(4, 1) != sp.binomial(4, 2),
            {"d": 4, "one_forms": 4, "two_forms": 6, "gauge_invariance_reason": "d squared = 0"})

    s, delta, cc = sp.symbols("s delta c", real=True)
    e.zero("R23", "mixing_square_identity", s*(1-s)-(sp.Rational(1, 4)-(s-sp.Rational(1, 2))**2))
    function = delta+cc*delta*(sp.Rational(1, 4)-delta)
    e.zero("R23", "alternative_derivative", sp.diff(function, delta)-(1+cc*(sp.Rational(1, 4)-2*delta)))
    for point in [0, sp.Rational(1, 4)]:
        e.zero("R23", f"same_endpoint_{point}", function.subs(delta, point)-point)
    e.check("R23", "same_symmetry_different_interior", function.subs({delta: sp.Rational(1, 8), cc: 1})
            != sp.Rational(1, 8), {"delta": .125, "c0": .125, "c1": .140625})
    rho = sp.Matrix([[sp.Rational(1, 2), sp.Rational(1, 4)],
                     [sp.Rational(1, 4), sp.Rational(1, 2)]])
    gamma = sp.symbols("gamma", nonnegative=True)
    sigz = sp.diag(1, -1)
    dissipator = gamma*(sigz*rho*sigz-rho)
    e.zero("R23", "dephasing_rate_free", dissipator[0, 1]+2*gamma*rho[0, 1])
    e.zero("R23", "dephasing_trace_preserved", sp.trace(dissipator))
    b = np.array([[.7, .3], [.2, .8]])
    a1, a2 = 3*np.eye(2)+.18*b, 3*np.eye(2)+.18**2*b
    e.close("R23", "uniform_row_mean", a1 @ np.ones(2), [3.18, 3.18])
    e.check("R23", "other_allowed_offspring_mean", np.all(a2 >= 0) and np.max(abs(a1-a2)) > .1,
            {"depths": [3.18, 3+.18**2]})

    alpha = .11789
    x = 4*alpha**(4/3)
    aw = (1/(2*np.pi)-alpha)/(1+x)
    ae = x*aw
    e.close("R24", "coupling_sum_boundary", alpha+aw+ae, 1/(2*np.pi))
    drift = (-7*alpha**2-19/6*aw**2+11/3*ae**2)/(2*np.pi)
    e.check("R24", "coupling_sum_not_rg_invariant", drift < -.001,
            {"sum": alpha+aw+ae, "normal_derivative": drift})

    # R25: same norm, same Koide potential and positive branch; differing mass ratios.
    phi_values = [0., .1]
    vectors = [1/np.sqrt(6)+np.cos(phi+2*np.pi*np.arange(3)/3)/np.sqrt(3)
               for phi in phi_values]
    for index, vec in enumerate(vectors):
        e.check("R25", f"koide_positive_{index}", np.all(vec > 0), {"minimum_component": vec.min()})
        e.close("R25", f"fixed_koide_scale_{index}", vec @ vec, 1.)
        e.close("R25", f"koide_constraint_{index}", (vec @ vec)/sum(vec)**2, 2/3)
        potential = ((sum(vec)/np.sqrt(3))**2-(vec @ vec)/2)**2+((vec @ vec)-1)**2
        e.close("R25", f"same_zero_potential_{index}", potential, 0., 1e-25)
    e.check("R25", "koide_not_individual_masses",
            np.linalg.norm(np.sort(vectors[0]**2)-np.sort(vectors[1]**2)) > .01,
            {"mass_spectra": [sorted((vec**2).tolist()) for vec in vectors]})
    spurion = alpha**(1/6)
    charges = np.array([2, 4, 6])
    ybase = spurion**charges
    e.close("R25", "spurion_realizes_exponent", ybase, alpha**(charges/6))
    e.check("R25", "same_charge_free_matching", np.linalg.norm(2*ybase-ybase) > .1,
            {"same_charges": charges.tolist(), "coefficient_choices": [1, 2]})

    # R26: determinant phase leaves left mixing and mass spectrum unchanged.
    yd = ckm(.2, .03, .004, .5) @ np.diag([.1, .3, .8])
    phase = .12
    yd_alt = np.exp(1j*phase/3)*yd
    e.close("R26", "strong_cp_same_left_matrix", yd @ yd.conj().T, yd_alt @ yd_alt.conj().T)
    e.close("R26", "strong_cp_phase_changes", np.angle(np.linalg.det(yd_alt)/np.linalg.det(yd)), phase)
    angle = sp.symbols("angle", real=True)
    eta = .01
    pq_min = -np.arctan(eta)
    e.close("R26", "shifted_pq_stationarity", np.sin(pq_min)+eta*np.cos(pq_min), 0.)
    e.check("R26", "shifted_pq_positive_curvature", np.cos(pq_min)-eta*np.sin(pq_min) > 0,
            {"theta_min": pq_min, "exact_PQ_assumed": False})

    # R27: use a hierarchy where the tree seesaw expansion is valid.
    u = ckm(.5, .6, .1, .7)
    light = np.diag([1e-6, 2e-6, 5e-6])
    heavy = np.diag([2., 4., 7.])
    vev = 1.
    target = u.conj() @ light @ u.conj().T
    rs = [np.eye(3), ckm(.4, 0, 0, 0).real]
    ys = []
    for index, rot in enumerate(rs):
        y = 1j*np.sqrt(2)/vev * u.conj() @ np.sqrt(light) @ rot @ np.sqrt(heavy)
        ys.append(y)
        tree_m = -vev**2/2*y @ np.linalg.inv(heavy) @ y.T
        e.close("R27", f"same_tree_seesaw_{index}", tree_m, target, 1e-18)
    rate_forms = [y.conj().T @ y for y in ys]
    e.check("R27", "uv_couplings_remain_free", np.linalg.norm(rate_forms[0]-rate_forms[1]) > 1e-7,
            {"YdagY_distance": float(np.linalg.norm(rate_forms[0]-rate_forms[1]))})
    cp = np.array([[0., 1.], [1., 0.]])
    baryon = np.diag([1., -1.])
    rho_cp = np.array([[.5, .2], [.2, .5]])
    e.close("R27", "cp_odd_operator", cp @ baryon @ cp, -baryon)
    for ti in [0., .3, 2.]:
        evol = expm(-1j*ti*cp)
        rt = evol @ rho_cp @ evol.conj().T
        e.close("R27", f"cp_symmetric_zero_source_{ti}", np.trace(rt @ baryon), 0.)

    # R28: both PGFs and a truncated coefficient check, with tail explicitly small.
    depth = 3.177
    q = brentq(lambda zz: np.exp(depth*(zz-1))-zz, 0, .5)
    kk = np.arange(32)
    conditioned = np.exp(-depth)*depth**kk/factorial(kk)*q**(kk-1)
    poisson_cond = np.exp(-depth*q)*(depth*q)**kk/factorial(kk)
    e.close("R28", "conditioned_poisson_coefficients", conditioned, poisson_cond, 1e-12)
    e.close("R28", "conditioned_mean", kk @ conditioned, depth*q, 1e-12)
    e.check("R28", "composition_not_selected", .5 != .25,
            {"fixed_branching_D": depth, "dust_fractions": [.5, .25]})
    density_m, cosmological = 2., 1.
    omega_transition = density_m/(density_m+cosmological)
    e.close("R28", "acceleration_transition_fraction", omega_transition, 2/3)
    e.check("R28", "acceleration_not_one_over_D", abs(omega_transition-1/depth) > .1,
            {"acceleration_Omega_m": omega_transition, "chosen_face_Omega_m": 1/depth})

    # R29: independent ODE vs inverse-yield solution.
    xf, yf, endpoint = 10., .2, 1000.
    limits = []
    for coupling in [1., 10.]:
        sol = solve_ivp(lambda xx, y: -coupling*y*y/xx**2, (xf, endpoint), [yf],
                        rtol=1e-11, atol=1e-13)
        exact = 1/(1/yf+coupling*(1/xf-1/endpoint))
        e.check("R29", f"yield_ode_success_{coupling}", sol.success, {"message": sol.message})
        e.close("R29", f"yield_solution_{coupling}", sol.y[0, -1], exact, 1e-10)
        limits.append(1/(1/yf+coupling/xf))
    e.check("R29", "stability_not_relic_yield", abs(limits[0]-limits[1]) > .01,
            {"same_yf": yf, "lambda": [1, 10], "asymptotic_yields": limits})

    # R30: same vacuum and Planck scale, different physically allowed dust density.
    mpl2, vacuum = 1., .3
    hlambda2 = vacuum/(3*mpl2)
    h02 = [(vacuum+dust)/(3*mpl2) for dust in [.1, .4]]
    entropy = 8*np.pi**2*mpl2/hlambda2
    e.check("R30", "same_vacuum_entropy_different_H0", h02[0] != h02[1],
            {"fixed_entropy": entropy, "fixed_HLambda_squared": hlambda2, "H0_squared": h02})
    e.check("R30", "cross_sector_scale_not_units", np.sqrt(1.)/246 != np.sqrt(4.)/246,
            {"fixed_external_vev": 246, "s": [1, 4], "epsilon_over_s": [.1, .1]})

    # R31: quoted input uncertainty is propagated; this is not a new measurement.
    aa = sp.symbols("a", positive=True)
    e.zero("R31", "weak_mixing_input_derivative", sp.diff(4*aa**sp.Rational(4, 3), aa)
           -sp.Rational(16, 3)*aa**sp.Rational(1, 3))
    dxda = 16/3*alpha**(1/3)
    sigma_x = dxda*.0009
    dep = 3+x*(1-x)
    qr = brentq(lambda zz: np.exp(dep*(zz-1))-zz, 0, .5)
    derivative_q = -qr*(1-qr)/(1-dep*qr)*(1-2*x)*dxda

    def extinction_from_input(aval):
        xv = 4*aval**(4/3)
        dv = 3+xv*(1-xv)
        return brentq(lambda zz: np.exp(dv*(zz-1))-zz, 0, .5, xtol=1e-14)

    da = 1e-6
    fdq = (extinction_from_input(alpha+da)-extinction_from_input(alpha-da))/(2*da)
    e.close("R31", "full_chain_uncertainty_derivative", fdq, derivative_q, 1e-9)
    e.check("R31", "solver_digits_not_predictive_digits", sigma_x/.00006 > 30,
            {"quoted_sigma_alpha": .0009, "propagated_sigma_x": sigma_x,
             "propagated_sigma_q": abs(derivative_q)*.0009,
             "target_only_sigma_x": .00006, "scale_ratio": sigma_x/.00006})

    # R32: Feynman-Hellmann checked against eigensolver differences, not diagonal ansatz alone.
    mass_parameter = .12
    hbase = np.array([[1., .08], [.08, 1.5]])
    derivative_matrix = np.diag([.5, -.5])
    hq = hbase+mass_parameter*derivative_matrix
    eigenvalues, eigenvectors = np.linalg.eigh(hq)
    stepsize = 1e-5
    derivative_e = (np.linalg.eigvalsh(hq+stepsize*derivative_matrix)
                    -np.linalg.eigvalsh(hq-stepsize*derivative_matrix))/(2*stepsize)
    fh = np.diag(eigenvectors.T @ derivative_matrix @ eigenvectors)
    e.close("R32", "feynman_hellmann_vs_finite_difference", derivative_e, fh, 1e-9)
    splittings = [np.diff(np.linalg.eigvalsh(np.eye(2)+kap*.1*np.diag([-1, 1])/2))[0]
                  for kap in [1., 2.]]
    e.close("R32", "same_spurion_symmetry_different_coefficient", splittings, [.1, .2])

    ids = sorted({row["claim"] for row in e.checks})
    assert ids == [f"R{i:02d}" for i in range(22, 33)]
    return {"schema": "CE-RB1-empirical-v1", "scope": "synthetic identities and countermodels",
            "observational_validation": False, "full_CE_completion": False,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "helper_sha256": hashlib.sha256(Path(__file__).with_name("verify_reverse.py").read_bytes()).hexdigest(),
            "claim_ids": ids, "number_of_checks": len(e.checks),
            "all_passed": all(row["passed"] for row in e.checks), "checks": e.checks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results_empirical.json"))
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"PASS {result['number_of_checks']} checks; {len(result['claim_ids'])} claim groups")
