"""Shared positive spectral matter force, matter-wave phase and amplitude screen.

Weak, static, unscreened conformal matter coupling A(Q)=exp(alpha Q/Mpl).
Here the background lambda=kappa*phi² is zero: source propagation uses D0.
This adds a supplied matter portal, not a derived CE geometry or cosmic solution.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad

from dimension_boundary_probe import BoundaryProbe


class MatterPortal:
    def __init__(self, *, beta=1., eta=.1, b=1., m=1., a=1.):
        self.probe = BoundaryProbe(beta=beta, eta=eta, b=b, m=m, a=a, coupling=0.)

    def profile(self, radius):
        """B(R)=E exp(-R sqrt(m²+aX)); F(R)=B-R B'. R=M_* r.

        Integrates the exact unbounded spectral measure via subordination;
        output is potential/force shape, before the common amplitude 2 alpha².
        """
        radius = float(radius)
        if not math.isfinite(radius) or radius < 0:
            raise ValueError("finite dimensionless radius>=0 required")
        if radius == 0:
            return {"potential_shape": 1., "force_shape": 1., "quadrature_errors": [0., 0.]}
        p = self.probe

        def integrand(u, force):
            if u == 0:
                return 0.
            t = radius*radius/(4*u)
            log_weight = (-u-p.m*p.m*t+p.kernel(p.beta+p.a*t)["log_heat_trace"]
                          -p.log_normalization)
            return math.exp(log_weight)*((2*math.sqrt(u)) if force else (1/math.sqrt(u)))/math.sqrt(math.pi)

        potential, ep = quad(lambda u: integrand(u, False), 0., math.inf, epsabs=2e-11, epsrel=2e-9)
        force, ef = quad(lambda u: integrand(u, True), 0., math.inf, epsabs=2e-11, epsrel=2e-9)
        return {"potential_shape": potential, "force_shape": force, "quadrature_errors": [ep, ef]}

    def point_source(self, radius_m, source_mass_kg, inverse_length_m, amplitude, *, newton_g=6.67430e-11):
        """Potential per test mass and inward acceleration in SI, point source.

        amplitude=2*alpha², inverse_length_m=M_* c/hbar if M_* is a mass.
        No extended-source, screening, calibration, or relativistic correction.
        """
        values = tuple(map(float, (radius_m, source_mass_kg, inverse_length_m, amplitude, newton_g)))
        radius_m, source_mass_kg, inverse_length_m, amplitude, newton_g = values
        if (not all(map(math.isfinite, values)) or any(x <= 0 for x in values[:3])
                or amplitude < 0 or newton_g <= 0):
            raise ValueError("finite positive radius,mass,inverse_length,G and amplitude>=0 required")
        shape = self.profile(radius_m*inverse_length_m)
        potential = -amplitude*newton_g*source_mass_kg*shape["potential_shape"]/radius_m
        acceleration = amplitude*newton_g*source_mass_kg*shape["force_shape"]/radius_m**2
        return {"potential_per_test_mass_m2_s2": potential, "inward_acceleration_m_s2": acceleration}

    def held_arm_phase(self, r1_m, r2_m, source_mass_kg, test_mass_kg, duration_s,
                       inverse_length_m, amplitude):
        """Additional propagation phase for fixed arms: -(V1-V2)*T/hbar.

        This is NOT the complete phase of a freely falling light-pulse instrument.
        """
        if not (math.isfinite(test_mass_kg) and math.isfinite(duration_s)
                and test_mass_kg > 0 and duration_s >= 0):
            raise ValueError("finite test mass>0 and duration>=0 required")
        hbar = 6.62607015e-34/(2*math.pi)
        v1 = self.point_source(r1_m, source_mass_kg, inverse_length_m, amplitude)["potential_per_test_mass_m2_s2"]
        v2 = self.point_source(r2_m, source_mass_kg, inverse_length_m, amplitude)["potential_per_test_mass_m2_s2"]
        return -test_mass_kg*(v1-v2)*duration_s/hbar


def common_amplitude_screen(residual, shape, covariance, groups, *, required_groups):
    """Exact SSE improvement intervals for f_s=f0+s*k with s>=0, fixed k,C.

    r=y-f0. This is a retrospective or training feasibility calculation, never
    permission to optimize on holdout. Covariance between groups is retained.
    Strict improvement is impossible for zero shape or B=k^T C^-1 r<=0.
    Returned infimum may lie at an excluded boundary of the strict interval.
    """
    r, k, c = (np.asarray(x, dtype=float) for x in (residual, shape, covariance))
    names, required = list(groups), list(required_groups)
    if (r.ndim != 1 or not r.size or k.shape != r.shape or c.shape != (r.size, r.size)
            or not all(np.isfinite(x).all() for x in (r, k, c))):
        raise ValueError("matching finite vectors and covariance required")
    if (len(names) != r.size or not required or len(set(required)) != len(required)
            or any(not isinstance(g, str) or not g for g in names+required)
            or set(names) != set(required)):
        raise ValueError("all and only the required nonempty groups must be present")
    if np.any(np.diag(c) <= 0):
        raise ValueError("positive covariance diagonal required")
    sd = np.sqrt(np.diag(c))
    correlation = c/np.outer(sd, sd)
    if not np.allclose(correlation, correlation.T, atol=1e-12, rtol=0):
        raise ValueError("symmetric covariance required")

    def block(indices):
        try:
            l = np.linalg.cholesky(correlation[np.ix_(indices, indices)])
        except np.linalg.LinAlgError as error:
            raise ValueError("positive definite covariance required") from error
        rw = np.linalg.solve(l, (r/sd)[indices])
        kw = np.linalg.solve(l, (k/sd)[indices])
        A, B = float(kw@kw), float(kw@rw)
        feasible = A > 0 and B > 0
        return {"quadratic_A": A, "alignment_B": B, "baseline_sse": float(rw@rw),
                "strict_improvement_possible": feasible,
                "open_upper_amplitude": 2*B/A if feasible else None}

    overall = block(np.arange(r.size))
    per_group = {g: block(np.flatnonzero(np.array(names) == g)) for g in required}
    feasible = overall["strict_improvement_possible"] and all(
        x["strict_improvement_possible"] for x in per_group.values())
    upper = min([overall["open_upper_amplitude"]]+[x["open_upper_amplitude"]
                for x in per_group.values()]) if feasible else None
    optimum = max(0., overall["alignment_B"]/overall["quadratic_A"]) if overall["quadratic_A"] else 0.
    constrained_infimum_at = min(optimum, upper) if feasible else None
    return {"overall": overall, "groups": per_group, "joint_strict_improvement_possible": feasible,
            "joint_open_interval": [0., upper] if feasible else None,
            "both_interval_endpoints_excluded": True,
            "overall_nonnegative_amplitude_optimum": optimum,
            "constrained_infimum_amplitude": constrained_infimum_at,
            "constrained_infimum_attained": feasible and 0 < constrained_infimum_at < upper,
            "scientific_success": False, "status": "fixed_shape_arithmetic_not_holdout_fit"}


def report():
    p = MatterPortal()
    source = json.loads((Path(__file__).with_name("hamilton2015_signed_summary.json")).read_text(encoding="utf-8"))
    cases = []
    for tag in ("corrected", "raw_before_systematic_corrections"):
        observation = source[tag]
        # k=1 is acceleration in the stated unit, not a inferred alpha² value.
        screen = common_amplitude_screen([observation["mean"]], [1.], [[observation["sigma"]**2]],
                                         ["published_marginal"], required_groups=["published_marginal"])
        cases.append({"input": tag, "role": "sign_screen" if tag == "corrected" else "invalid_raw_input_control",
                      "zero_baseline_standardized_residual": abs(observation["mean"])/observation["sigma"],
                      "unit_attractive_shift_standardized_residual": abs(1-observation["mean"])/observation["sigma"],
                      "screen": screen})
    return {"status": "conditional_matter_portal_and_seen_signed_marginal_screen",
            "shared_parameters": {"beta": 1., "eta": .1, "b": 1., "m": 1., "a": 1., "lambda_background": 0.},
            "profiles": [{"dimensionless_radius": r, **p.profile(r)} for r in (.01, .1, 1., 3., 10.)],
            "source_manifest": "hamilton2015_signed_summary.json", "published_marginal_cases": cases,
            "actual_geometry_forward_model_complete": False, "all_domain_rmse_reduced": False}


if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
