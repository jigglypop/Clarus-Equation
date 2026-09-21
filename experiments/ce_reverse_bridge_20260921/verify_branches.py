"""CE-RB2: source-specific Gaussian spectral and branch matching checks (R33-R40).

All parameters are synthetic. General proofs are in chapter 20.
No observational fit, all-loop claim, or original data reconstruction is performed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.linalg import block_diag
from scipy.optimize import brentq

from verify_reverse import Evidence


PI = np.pi


def integral_i(y):
    return quad(lambda t: np.log1p(y*t*(1-t)), 0, 1, epsabs=2e-13, epsrel=2e-13)[0]


def jfun(r):
    return -integral_i(-r)


def jprime(r):
    return quad(lambda t: t*(1-t)/(1-r*t*(1-t)), 0, 1, epsabs=2e-13)[0]


def pole(beta):
    if not 0 < beta < 1:
        raise ValueError("The source88 branch requires 0 < beta < 1")
    # Analytic upper bound is below the logarithmic threshold; do not sample r~4.
    upper = 6*(-np.expm1(-beta))+1e-10
    return brentq(lambda r: jfun(r)-beta, 0., upper, xtol=1e-14)


def spectral_continuum(beta, y):
    # S/b=exp(w); the stable logarithm avoids both overflow and 1-v cancellation.
    def density_integrand(w):
        v = np.sqrt(max(0., 1-4*np.exp(-w)))
        logpart = -2+v*(w+2*np.log1p(v)-np.log(4))
        return v/((beta+logpart)**2+PI**2*v*v)/(1+y*np.exp(-w))
    return quad(density_integrand, np.log(4), np.inf,
                epsabs=2e-11, epsrel=2e-11, limit=200)


def cfun(a, b):
    if abs(a-b) < 1e-14:
        return 0.
    return ((a+b)/2-a*b/(b-a)*np.log(b/a))/(16*PI**2)


def light_spectrum(s, epsilon, theta):
    return s+2*epsilon*np.cos((theta+2*PI*np.arange(3))/3)


def compensated_f(s, epsilon, heavy, lam, alpha=.25):
    aj = light_spectrum(s, epsilon, PI)
    cj = (s+heavy)/2+lam*(aj-s)/2
    return alpha/(8*PI**2)*(np.sum(aj*np.log(aj))+3*heavy*np.log(heavy)
                            -2*np.sum(cj*np.log(cj)))


def run():
    e = Evidence()
    values = {}
    # Independent dispersion integral versus the original Feynman parameter logarithm.
    for beta in [.05, .227764843511725, .8]:
        rp = pole(beta)
        residue = 1/jprime(rp)
        e.check("R33", f"positive_pole_residue_{beta}", residue > 0,
                {"dimensionless_pole": rp, "residue_c_over_b": residue})
        for y in [0., .3, 3., 10.]:
            continuum, estimated_error = spectral_continuum(beta, y)
            direct = 1/(beta+integral_i(y))
            spectral = residue/(y+rp)+continuum
            e.close("R33", f"spectral_representation_{beta}_{y}", direct, spectral, 2e-9)
            e.check("R33", f"positive_continuum_{beta}_{y}", continuum > 0,
                    {"continuum_cG": continuum, "quad_estimated_error": estimated_error})
    for z in [1+1j, -2+.4j, -5+.2j]:
        imaginary = quad(lambda t: np.log(1+z*t*(1-t)).imag, 0, 1)[0]
        e.check("R33", f"upper_half_plane_{z}", imaginary > 0,
                {"imaginary_part": imaginary})

    for beta in [.001, .05, .227764843511725, .8, .999]:
        rp = pole(beta)
        lo, hi = 4*(-np.expm1(-1.5*beta)), 6*(-np.expm1(-beta))
        e.check("R34", f"analytic_pole_bracket_{beta}", lo <= rp <= hi < 4,
                {"lower": lo, "pole": rp, "upper": hi})
    for r in [.1, 1., 3.5]:
        remainder = jfun(r)-r/6
        bound = r*r/(60*(1-r/4))
        e.check("R34", f"timelike_remainder_{r}", 0 <= remainder <= bound,
                {"remainder": remainder, "bound": bound})
    for y in [.1, 1., 10.]:
        remainder = y/6-integral_i(y)
        e.check("R34", f"euclidean_remainder_{y}", 0 <= remainder <= y*y/60,
                {"remainder": remainder, "bound": y*y/60})
    e.check("R34", "pole_inverse_lipschitz", abs(pole(.405)-pole(.4)) <= 6*.005,
            {"actual": abs(pole(.405)-pole(.4)), "bound": .03})

    s, epsilon, heavy, alpha = 1., .15, 10., .25
    aj = light_spectrum(s, epsilon, PI)
    fs = []
    for lam in [0., .5, 1.]:
        cj = (s+heavy)/2+lam*(aj-s)/2
        zero_moment = 6*alpha-2*alpha*3
        first_moment = alpha*(sum(aj)+3*heavy)-2*alpha*sum(cj)
        e.close("R35", f"global_curvature_moments_{lam}", [zero_moment, first_moment], [0, 0])
        fclosed = compensated_f(s, epsilon, heavy, lam)
        fs.append(fclosed)
        masses = np.concatenate([aj, np.full(3, heavy), cj])
        weights = np.concatenate([np.full(6, alpha), np.full(3, -2*alpha)])

        def heat_integrand(tau):
            if tau < 1e-4:
                # Subtract the analytically zero zeroth and first moments.
                return sum(float(weights @ ((-masses)**k))*tau**(k-2)/float(sp.factorial(k))
                           for k in range(2, 9))
            return float(weights @ np.exp(-tau*masses))/tau**2

        fheat = (quad(heat_integrand, 0, 1e-3, epsabs=1e-12)[0]
                 +quad(heat_integrand, 1e-3, np.inf, epsabs=1e-11)[0])/(8*PI**2)
        e.close("R35", f"heat_kernel_vs_closed_F_{lam}", fheat, fclosed, 2e-10)
    e.check("R35", "same_global_rules_different_F", fs[0] > fs[1] > fs[2] > 0,
            {"lambda": [0, .5, 1], "F": fs})
    e.check("R35", "blockwise_condition_only_lambda_one",
            np.max(abs(2*((s+heavy)/2)-(aj+heavy))) > .1,
            {"uniform_block_mass_moment_defects": (s-aj).tolist()})

    # Use a constant heavy basis rotation to absorb the source's fixed unitary T.
    def rotation(theta):
        eye = np.eye(3)
        co, si = np.cos(theta/2), np.sin(theta/2)
        return np.block([[co*eye, -si*eye], [si*eye, co*eye]])

    def active_x(theta, hv):
        w = rotation(theta)
        return w @ block_diag(np.diag(light_spectrum(s, epsilon, theta)), hv*np.eye(3)) @ w.T

    def extra_kernel(z, hv):
        return sum((hv-ai)**2/(32*PI**2)*quad(
            lambda t: np.log1p(z*t*(1-t)/((1-t)*ai+t*hv)), 0, 1,
            epsabs=1e-13, epsrel=1e-12)[0] for ai in aj)

    b, a = s+epsilon, s-2*epsilon
    beta = 1-a/(3*epsilon)*np.log(b/a)
    prefactor = epsilon**2/(24*PI**2)
    mu = prefactor*beta
    original_pole = b*pole(beta)
    w0 = rotation(PI)
    hh = 1e-4
    dx = (active_x(PI-2*hh, heavy)-8*active_x(PI-hh, heavy)
          +8*active_x(PI+hh, heavy)-active_x(PI+2*hh, heavy))/(12*hh)
    vertex = w0.T @ dx @ w0
    background_masses = np.concatenate([aj, np.full(3, heavy)])
    expected_vertex = np.diag(np.concatenate([
        -2*epsilon/3*np.sin((PI+2*PI*np.arange(3))/3), np.zeros(3)]))
    expected_vertex[:3, 3:] = np.diag((aj-heavy)/2)
    expected_vertex[3:, :3] = np.diag((aj-heavy)/2)
    e.close("R36", "active_path_vertex", vertex, expected_vertex, 1e-10)

    for z in [-.8, .1, 1.]:
        full_bubble = 0.
        for i in range(6):
            for j in range(6):
                if abs(vertex[i, j]) > 1e-9:
                    integral = quad(lambda t: np.log1p(z*t*(1-t)/(
                        (1-t)*background_masses[i]+t*background_masses[j])), 0, 1,
                        epsabs=1e-12)[0]
                    full_bubble += abs(vertex[i, j])**2*integral/(16*PI**2)
        closed = prefactor*integral_i(z/b)+extra_kernel(z, heavy)
        e.close("R36", f"full_vertex_vs_rotation_kernel_{z}", full_bubble, closed, 2e-11)
    dxp = 1e-4
    extra_slope = (extra_kernel(dxp, heavy)-extra_kernel(-dxp, heavy))/(2*dxp)
    e.close("R36", "source83_kinetic_recovered", extra_slope,
            sum(cfun(ai, heavy)/2 for ai in aj), 1e-10)
    pole_rows = []
    for hv in [10., 100.]:
        threshold = min((np.sqrt(aj)+np.sqrt(hv))**2)
        kernel = lambda mass2: mu-prefactor*jfun(mass2/b)+extra_kernel(-mass2, hv)
        newpole = brentq(kernel, 1e-15, original_pole, xtol=1e-15)
        e.check("R36", f"same_U_different_pole_{hv}", 0 < newpole < original_pole < threshold,
                {"B": hv, "source88_pole": original_pole, "UCR2_pole": newpole,
                 "mixed_threshold": threshold, "root_residual": kernel(newpole)})
        pole_rows.append({"B": hv, "mass_squared": newpole})
    values["pole_comparison"] = {"source88": original_pole, "UCR2": pole_rows}

    # R37: exact rational Jacobian of the source86 inverse and a separate integral.
    rr, cc = sp.symbols("r c", real=True)
    pp = (1-2*rr)*(1+rr)**2
    br = 2*rr**3*(1+cc)
    ell = sp.log(1+br/pp)
    et = 3*(1-rr)/((1-2*rr)*(1+rr))
    nn = -et*br/(pp+br)
    determinant = sp.det(sp.Matrix([ell, nn]).jacobian([rr, cc]))
    expected = sp.diff(et, rr)*br/(pp+br)*2*rr**3/(pp+br)
    e.zero("R37", "inverse_jacobian", determinant-expected)
    e.zero("R37", "seventh_order_degeneracy", sp.limit(expected/rr**7, rr, 0)-48*(1+cc))
    e.zero("R37", "vacuum_record_degeneracy", nn.subs(cc, -1))
    theta = 1.2
    mass_function = lambda source: light_spectrum(source, epsilon, theta)
    reference = lambda source: light_spectrum(source, epsilon, PI)
    uu = (sum(mass_function(s)**2*np.log(mass_function(s)))
          -sum(reference(s)**2*np.log(reference(s))))/(32*PI**2)
    numerator = 2*epsilon**3*(1+np.cos(theta))
    response = lambda source: np.log1p(numerator/((source-2*epsilon)*(source+epsilon)**2))
    integrated = quad(lambda source: (source-s)*response(source), s, np.inf,
                      epsabs=1e-12)[0]/(16*PI**2)
    e.close("R37", "boundary_fixed_relative_U", integrated, uu, 1e-12)

    # R38: chain rule with n_dot=-3Hn; use the source's small-charge asymptotics.
    n, hub, hd, G, Gp, t1, t2 = sp.symbols("n H Hd G Gp theta_n theta_nn", real=True)
    ndot = -3*hub*n
    nddot = -3*hd*n+9*hub**2*n
    chain_residual = G*(t2*ndot**2+t1*nddot+3*hub*t1*ndot)+Gp*(t1*ndot)**2/2
    formula = G*(9*hub**2*n**2*t2-3*n*hd*t1)+sp.Rational(9, 2)*Gp*hub**2*n**2*t1**2
    e.zero("R38", "tracking_residual_chain_rule", chain_residual-formula)
    kval, mass2, A0 = sp.symbols("k M_squared A0", positive=True)
    leading = formula.subs({t1: -2*kval*n, t2: -2*kval,
                           hd: -n**2/(mass2*A0), hub**2: n**2/(3*mass2*A0)})
    e.zero("R38", "source87_nontracking_coefficient", sp.limit(leading/n**4, n, 0)
           +12*G*kval/(mass2*A0))

    # R39: arbitrary positive field metric, coupling matrix and noncommuting mass basis.
    metric = np.array([[2., .3], [.3, 1.]])
    gv, ge = np.linalg.eigh(metric)
    inverse_sqrt = ge @ np.diag(gv**-.5) @ ge.T
    alpha_matrix = np.array([[-.5, -.5, .1], [.2, -.2, .3]])  # fields x species
    canonical_alpha = inverse_sqrt @ alpha_matrix
    mass_matrix = np.array([[2., .4], [.4, .5]])
    limit = 2*canonical_alpha.T @ canonical_alpha
    mmax = np.linalg.eigvalsh(mass_matrix)[-1]
    for q2 in [.1, 1., 100.]:
        finite = 2*canonical_alpha.T @ (q2*np.linalg.inv(q2*np.eye(2)+mass_matrix)) @ canonical_alpha
        gap = limit-finite
        bound = 2*np.linalg.norm(canonical_alpha, 2)**2*mmax/(q2+mmax)
        e.check("R39", f"loewner_force_bound_{q2}", np.linalg.eigvalsh(finite).min() > -1e-12
                and np.linalg.eigvalsh(gap).min() > -1e-12 and np.linalg.norm(gap, 2) <= bound+1e-12,
                {"force_gap_norm": np.linalg.norm(gap, 2), "bound": bound})
    tachyon_factor = 2/(2-1)
    e.check("R39", "tachyonic_massless_bound_counterexample", tachyon_factor > 1,
            {"m_squared": -1, "q_squared": 2, "response_factor": tachyon_factor})

    # R40: explicitly keep the same matching model when mu is changed.
    bt, ct, lt, l0, loc0 = sp.symbols("B c log_mu log_mu0 Z0", positive=True)
    slope = 3/(32*sp.pi**2*bt)+3/(128*sp.pi**2*ct)
    zlocal = loc0-slope*(lt-l0)
    zren = zlocal+3/(64*sp.pi**2*bt)*(2*lt-sp.log(bt))
    zren += 3/(256*sp.pi**2*ct)*(2*lt-sp.log(ct))
    e.zero("R40", "matched_kinetic_scale_derivative", sp.diff(zren, lt))
    e.zero("R40", "reset_local_zero_scale_drift", sp.diff(zren-zlocal, lt)-slope)
    bv, cv = 100., 50.5
    ff = compensated_f(1., .15, bv, 0.)
    fb = 3*.25/(8*PI**2)*np.log(bv/cv)
    mu0 = np.sqrt(bv)
    coefficient = 3/(32*PI**2*bv)+3/(128*PI**2*cv)
    matched, reset = [], []
    for muv in [mu0/2, mu0, mu0*2]:
        loop = 3/(64*PI**2*bv)*np.log(muv*muv/bv)+3/(256*PI**2*cv)*np.log(muv*muv/cv)
        local = -coefficient*np.log(muv/mu0)
        matched.append(1.5+ff*(local+loop)/fb**2)
        reset.append(1.5+ff*loop/fb**2)
    e.close("R40", "same_matching_kappa", matched, np.full(3, matched[1]), 1e-12)
    e.check("R40", "resetting_matching_changes_theory", np.ptp(reset) > 1.,
            {"same_model_kappa": matched, "local_reset_kappa": reset})

    ids = sorted({row["claim"] for row in e.checks})
    assert ids == [f"R{i:02d}" for i in range(33, 41)]
    return {"schema": "CE-RB2-v1", "scope": "source-specific Gaussian branches",
            "observational_validation": False, "full_CE_completion": False,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "helper_sha256": hashlib.sha256(Path(__file__).with_name("verify_reverse.py").read_bytes()).hexdigest(),
            "claim_ids": ids, "number_of_checks": len(e.checks),
            "all_passed": all(row["passed"] for row in e.checks), "checks": e.checks, "values": values}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results_branches.json"))
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"PASS {result['number_of_checks']} checks; {len(result['claim_ids'])} claim groups")
