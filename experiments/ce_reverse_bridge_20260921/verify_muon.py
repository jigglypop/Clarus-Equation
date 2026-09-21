"""CE-RB4: exact specified scalar-VP insertion, not the full muon anomaly."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.integrate import quad

from verify_reverse import Evidence


def integrate(function):
    return quad(function, 0., 1., epsabs=2e-13, epsrel=2e-10, limit=160)[0]


def scalar_a(r):
    # a / (alpha/pi)^2 for one charge-one complex scalar.
    return integrate(lambda y: (1-y)*integrate(
        lambda t: (1-2*t)**2*np.log1p(r*y*y*t*(1-t)/(1-y))))/4


def scalar_a_spectral(r):
    # Independent spectral variable v=sqrt(1-4M^2/S), after rational simplification.
    return r*integrate(lambda y: y*y*(1-y)*integrate(
        lambda v: v**4/(4*(1-y)+r*y*y*(1-v*v))))/6


def spectrum(s, eps, theta):
    return s+2*eps*np.cos((theta+2*np.pi*np.arange(3))/3)


def determinant(s, eps, theta):
    return s**3-3*s*eps**2+2*eps**3*np.cos(theta)


def relative_a(s, eps, theta, mu2):
    numerator = 2*eps**3*(1+np.cos(theta))
    def ell(mass):
        base = (mass-2*eps)*(mass+eps)**2
        return np.log1p(numerator/base)
    base_ell = ell(s)
    return integrate(lambda y: (1-y)*integrate(
        lambda t: (1-2*t)**2*(ell(s+mu2*y*y*t*(1-t)/(1-y))-base_ell)))/4


def remainder_bound(r):
    assert 0 < r < 1
    return r*r/1680*(np.log(14/r)+1)


def run():
    e = Evidence()
    values = {}
    t = sp.symbols("t")
    e.zero("R49", "bubble_first_moment", sp.integrate((1-2*t)**2*t*(1-t), (t, 0, 1))-sp.Rational(1, 30))
    e.zero("R49", "bubble_second_moment", sp.integrate((1-2*t)**2*t*t*(1-t)**2, (t, 0, 1))-sp.Rational(1, 210))
    e.zero("R49", "muon_leading_coefficient", sp.integrate(t*t, (t, 0, 1))/120-sp.Rational(1, 360))
    eta, r = sp.symbols("eta r", positive=True)
    bound = r*r/1680*sp.log(1/eta)+r*eta/120
    e.zero("R49", "optimal_split_location", sp.diff(bound, eta).subs(eta, r/14))
    e.zero("R49", "optimized_bound", bound.subs(eta, r/14)-r*r/1680*(sp.log(14/r)+1))
    rows = []
    for rv in [.001, .03, .3, .9, 3.]:
        actual = scalar_a(rv)
        spectral = scalar_a_spectral(rv)
        e.close("R49", f"direct_vs_spectral_{rv}", actual, spectral, 2e-12)
        leading = rv/360
        e.check("R49", f"positive_below_leading_{rv}", 0. < actual < leading,
                {"r": rv, "exact": actual, "leading": leading})
        row = {"r": rv, "exact": actual, "leading": leading, "remainder": leading-actual}
        if rv < 1:
            remainder = remainder_bound(rv)
            row["analytic_remainder_bound"] = remainder
            e.check("R49", f"heavy_remainder_bound_{rv}", leading-actual <= remainder,
                    {"observed_error": leading-actual, "bound": remainder})
        rows.append(row)
    values["normalized_scalar_insertion"] = rows

    # R50: algebraic resolvent/determinant difference and exact non-expanded integral.
    s, eps, c, shift = sp.symbols("s eps c k", real=True)
    dc = s**3-3*s*eps**2+2*eps**3*c
    dpi = (s-2*eps)*(s+eps)**2
    e.zero("R50", "determinant_difference", dc-dpi-2*eps**3*(1+c))
    e.zero("R50", "exact_log_derivative", sp.diff(sp.log(dc/dpi), s)-3*(s*s-eps*eps)*(1/dc-1/dpi))
    comparison = []
    sv, ev = 1., .15
    for theta in [0., 1.2, 2.8]:
        xs = spectrum(sv, ev, theta)
        reference = spectrum(sv, ev, np.pi)
        for kv in [.1, 2.]:
            direct_logs = np.sum(np.log1p(kv/xs))-np.sum(np.log1p(kv/reference))
            l0 = np.log(determinant(sv, ev, theta)/determinant(sv, ev, np.pi))
            lshift = np.log(determinant(sv+kv, ev, theta)/determinant(sv+kv, ev, np.pi))
            e.close("R50", f"shifted_determinant_identity_{theta}_{kv}", direct_logs, lshift-l0, 2e-14)
        for mu2 in [.03, 3.]:
            direct = sum(scalar_a(mu2/mass) for mass in xs)-sum(scalar_a(mu2/mass) for mass in reference)
            stable = relative_a(sv, ev, theta, mu2)
            e.close("R50", f"full_integral_difference_{theta}_{mu2}", direct, stable, 3e-12)
            e.check("R50", f"strict_original_branch_sign_{theta}_{mu2}", stable < 0.,
                    {"theta": theta, "mu_squared": mu2, "normalized_relative_a": stable})
            record = {"theta": theta, "mu_squared": mu2, "normalized_relative_a": stable}
            if mu2 < np.min(xs) and mu2 < np.min(reference):
                leading = mu2/360*(np.sum(1/xs)-np.sum(1/reference))
                error_bound = sum(remainder_bound(mu2/mass) for mass in np.concatenate([xs, reference]))
                e.check("R50", f"relative_heavy_approximation_bound_{theta}_{mu2}", abs(leading-stable) <= error_bound,
                        {"leading": float(leading), "exact": stable, "absolute_error_bound": error_bound})
                record["leading_difference"] = float(leading)
            comparison.append(record)
    e.close("R50", "same_reference_phase_zero", relative_a(sv, ev, np.pi, .3), 0., 1e-14)
    values["original_phase_differences"] = comparison

    ids = sorted({row["claim"] for row in e.checks})
    assert ids == ["R49", "R50"]
    return {"schema": "CE-RB4-v1", "scope": "one specified scalar vacuum-polarization insertion",
            "observational_validation": False, "full_CE_completion": False,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "helper_sha256": hashlib.sha256(Path(__file__).with_name("verify_reverse.py").read_bytes()).hexdigest(),
            "claim_ids": ids, "number_of_checks": len(e.checks), "all_passed": all(row["passed"] for row in e.checks),
            "checks": e.checks, "values": values}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results_muon.json"))
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"PASS {result['number_of_checks']} checks; {len(result['claim_ids'])} claim groups")
