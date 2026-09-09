"""Common three-complex-scalar spectrum: exact EM subset and sign audit.

No joint fit or cosmological forward model. Masses are in GeV. The fixed
benchmark is from external CE manuscript 21 at commit 05013ce3c653fc68c2fe5a4ab6294e95bd6c0a30.
The two integrals are equivalent representations of the same two-loop subset,
not a calculation of all electroweak or higher-loop contributions.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad

ALPHA = 1 / 137.035999084  # Fixed reproduction input, not a fitted parameter.
M_MU = 0.1056583755
Q2 = 16 / 3


def spectrum(e: float, theta: float) -> np.ndarray:
    if not (0 < e < 0.5 and math.isfinite(theta)):
        raise ValueError("Require 0 < epsilon/s < 1/2 and finite phase")
    return 1 + 2 * e * np.cos((theta + 2 * math.pi * np.arange(3)) / 3)


def exact_em(mass_scale: float, e: float, theta: float,
             *, representation: str = "feynman", relative: bool = False) -> float:
    """Finite-mass scalar-QED vacuum-polarization contribution.

    Integrate z=1-exp(-t), t in [0,80]. The omitted positive tail is bounded
    by < 6e-35 times the heavy-limit upper bound (not the exact integral).
    Relative mode evaluates theta minus pi without subtracting loop totals.
    """
    x = spectrum(e, theta)
    if not (math.isfinite(mass_scale) and mass_scale > 0):
        raise ValueError("positive finite mass scale required")
    if representation not in ("feynman", "spectral"):
        raise ValueError("unknown representation")
    if relative and representation != "feynman":
        raise ValueError("relative determinant kernel uses Feynman parameters")
    r = (M_MU / mass_scale) ** 2
    b0 = (1 - 2 * e) * (1 + e) ** 2
    k = 4 * e ** 3 * math.cos(theta / 2) ** 2

    def outer(t):
        ez = math.exp(-t)
        z = -math.expm1(-t)
        y = r * z * z / ez

        def inner(u):
            if representation == "spectral":
                return float(np.sum((y / x) * u**4 /
                                    (6 * (4 + (y / x) * (1 - u*u))))) * ez*ez / r
            c = y * u * (1 - u)
            if relative:
                increment = c * (3 * (1 - e*e) + 3*c + c*c)
                logterm = math.log1p(-k * increment / ((b0 + increment) * (b0 + k)))
            else:
                logterm = float(np.log1p(c / x).sum())
            return ez*ez * (1 - 2*u)**2 * logterm / (4*r)

        return quad(inner, 0, 1, epsabs=1e-15, epsrel=2e-10, limit=150)[0]

    integral = quad(outer, 0, 80, epsabs=1e-15, epsrel=2e-10,
                    points=[1, 5, 15, 30], limit=150)[0]
    return (ALPHA / math.pi)**2 * Q2 * r * integral


def run_audit():
    benchmark = dict(mass_scale=1000.0, e=0.025, theta=1.2)
    checks = []
    for mass, e, theta in [(1000., .025, 1.2), (1., .2, .3),
                           (.05, .4, 2.4), (10., .49, .8)]:
        direct = exact_em(mass, e, theta)
        dispersion = exact_em(mass, e, theta, representation="spectral")
        reference = exact_em(mass, e, math.pi)
        relative = exact_em(mass, e, theta, relative=True)
        heavy = (ALPHA / math.pi)**2 * M_MU**2 * Q2 / (360 * mass**2) * sum(1/spectrum(e, theta))
        checks.append(dict(mass_scale_GeV=mass, epsilon_over_s=e, theta=theta,
                           exact_em=direct, spectral_em=dispersion,
                           representation_relative_error=abs(direct-dispersion)/abs(direct),
                           relative_em=relative,
                           difference_check_relative_error=abs((direct-reference)-relative)/abs(relative),
                           heavy_upper_bound=float(heavy), below_heavy_bound=bool(direct <= heavy),
                           relative_nonpositive=relative <= 0))

    # Integer summaries in 1e-12 units avoid subtracting two ~0.001 totals.
    exp, exp_sd = 1165920715, 145
    sm, sm_sd = 1165920330, 620
    residual = (exp - sm) * 1e-12
    sigma = math.hypot(exp_sd, sm_sd) * 1e-12
    correction = checks[0]["exact_em"]
    phase_delta = checks[0]["relative_em"]
    score = lambda shift: abs(residual-shift)/sigma
    absolute_comparison = dict(baseline="WP2025 Standard Model",
        candidate="WP2025 plus fixed scalar EM subset", n=1,
        baseline_rmse=score(0), candidate_rmse=score(correction),
        delta_rmse=score(correction)-score(0), correction=correction,
        fraction_of_central_residual=correction/residual,
        fitted_parameters=0, independent_holdout=False)
    # A phase reference already contains the scalar sector. It is not SM.
    ref_correction = exact_em(1000., .025, math.pi)
    phase_comparison = dict(baseline="WP2025 plus scalar subset at theta=pi",
        candidate="same model at theta=1.2", n=1,
        baseline_rmse=score(ref_correction), candidate_rmse=score(correction),
        delta_rmse=score(correction)-score(ref_correction),
        phase_delta=phase_delta)
    result = dict(schema_version=1, role="conditional_common_spectrum_screen",
        benchmark=benchmark, alpha=ALPHA, muon_mass_GeV=M_MU, charge_squared_sum=Q2,
        sources={
            "experiment": "https://muon-g-2.fnal.gov/result2025.pdf",
            "theory": "https://muon-gm2-theory.illinois.edu/white-paper-25/",
            "comparison": "https://pdg.lbl.gov/2025/reviews/rpp2025-rev-g-2-muon-anom-mag-moment.pdf",
            "model_commit": "05013ce3c653fc68c2fe5a4ab6294e95bd6c0a30"},
        frozen_summary=dict(experiment_1e12=exp, experiment_sd_1e12=exp_sd,
            theory_1e12=sm, theory_sd_1e12=sm_sd,
            covariance_assumption="independent Gaussian experimental and theory summaries",
            residual=residual, combined_sigma=sigma),
        numerical_checks=checks, absolute_comparison=absolute_comparison,
        phase_comparison=phase_comparison,
        necessary_min_mass_upper_bound_GeV=math.sqrt(
            (ALPHA/math.pi)**2*M_MU**2*Q2/(120*residual)),
        bound_scope="To fill the positive central residual using only this EM subset; not a mass exclusion or fit",
        joint_rmse=None, scientific_success=False,
        missing=["absolute vacuum and common state", "dark-matter abundance and perturbations",
                 "Hubble and distance likelihood", "complete gauge/gravity and matter matching",
                 "atom, mass-ratio, and mixing predictions", "independent joint validation"])
    if not all(c["representation_relative_error"] < 1e-8 and
               c["difference_check_relative_error"] < 1e-6 and
               c["below_heavy_bound"] and c["relative_nonpositive"] for c in checks):
        raise ArithmeticError("independent integral or sign check failed")
    return result


if __name__ == "__main__":
    result = run_audit()
    Path(__file__).with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("absolute_comparison", "phase_comparison",
          "necessary_min_mass_upper_bound_GeV", "joint_rmse", "scientific_success")}, indent=2))
