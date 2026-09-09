"""Continuous spectral-dimension candidate and joint residual arithmetic.

This is a mathematical screening tool, not a quantum/cosmology forward solver.
No observations are generated, fitted, or certified as independent here.
See paper/6_최신_연구/07_확률차원_검증기록/38_확률적_차원과_양자_거시_공동식.md.
"""

from __future__ import annotations

import json
import math

import numpy as np
from scipy.integrate import quad
from scipy.special import log_ndtr


def dimension_kernel(t, *, eta=0.1, b=1.0, d_ir=4.0):
    """Return log heat trace, effective dimension, and tilted continuum weight.

    t is positive dimensionless diffusion time, not cosmic time.
    The prior on u=(d-d_ir)/2 is (1-eta) delta_0 + eta half-normal(b).
    K(t)=t**(-d_ir/2) * [(1-eta)+eta*erfcx(log(t)/(2*sqrt(b)))].
    d_ir is supplied, not predicted. The trace is not a probability measure.
    """
    t, eta, b, d_ir = map(float, (t, eta, b, d_ir))
    if not all(map(math.isfinite, (t, eta, b, d_ir))):
        raise ValueError("finite parameters required")
    if t <= 0 or b <= 0 or d_ir <= 0 or not 0 <= eta <= 1:
        raise ValueError("t,b,d_ir > 0 and 0 <= eta <= 1 required")
    logt = math.log(t)
    if eta == 0:
        return {"log_heat_trace": -d_ir*logt/2, "dimension": d_ir,
                "continuum_weight": 0.0}
    a = logt/(2*math.sqrt(b))
    # log(erfcx(a)) without overflowing for large negative a.
    logz = math.log(2) + a*a + float(log_ndtr(-math.sqrt(2)*a))
    if a > 8:
        # Avoid subtracting nearly equal numbers in the truncated-normal mean.
        ratio = b/(logt*logt)
        den = quad(lambda v: math.exp(-v-ratio*v*v), 0, math.inf)[0]
        num = quad(lambda v: v*math.exp(-v-ratio*v*v), 0, math.inf)[0]
        mean_u = num/(logt*den)
    else:
        mean_u = math.exp(-logz)/math.sqrt(math.pi*b)-logt/(2*b)
    log_cont = math.log(eta)+logz
    log_mix = float(np.logaddexp(math.log1p(-eta), log_cont)) if eta < 1 else log_cont
    weight = math.exp(log_cont-log_mix)
    result = {"log_heat_trace": -d_ir*logt/2+log_mix,
              "dimension": d_ir+2*weight*mean_u, "continuum_weight": weight}
    if not all(map(math.isfinite, result.values())):
        raise ValueError("parameters exceed numerical range")
    return result


def joint_residuals(observed, baseline, candidate, covariance, groups, *, required_groups):
    """Full-covariance RMSE plus every group's marginal-covariance RMSE.

    All arrays use the same ordered observations and fixed covariance.
    Cross-group correlations are retained in the overall score. Group scores
    use principal covariance blocks, not arbitrary slices of whitened residuals.
    A numerical decrease is explicitly NOT an empirical-success certificate.
    """
    y, ref, pred = (np.asarray(x, dtype=float) for x in (observed, baseline, candidate))
    if y.ndim != 1 or y.size == 0 or ref.shape != y.shape or pred.shape != y.shape:
        raise ValueError("nonempty matching one-dimensional observations required")
    c = np.asarray(covariance, dtype=float)
    if c.shape != (y.size, y.size):
        raise ValueError("covariance shape does not match observations")
    if not all(np.isfinite(x).all() for x in (y, ref, pred, c)):
        raise ValueError("nonfinite observations, predictions, or covariance")
    if np.any(np.diag(c) <= 0):
        raise ValueError("positive variances required")
    # Check symmetry in correlation units to support heterogeneous units.
    sd = np.sqrt(np.diag(c))
    corr = c/np.outer(sd, sd)
    if not np.allclose(corr, corr.T, rtol=0, atol=1e-12):
        raise ValueError("symmetric covariance required")
    labels = list(groups)
    required = list(required_groups)
    if (len(labels) != y.size or not required or len(set(required)) != len(required)
            or any(not isinstance(g, str) or not g for g in labels+required)):
        raise ValueError("nonempty unique required groups and one label per datum required")
    if set(labels) != set(required):
        raise ValueError("all and only declared groups must be present")

    def score(indices):
        block = corr[np.ix_(indices, indices)]
        try:
            chol = np.linalg.cholesky(block)
        except np.linalg.LinAlgError as error:
            raise ValueError("positive definite covariance required; no automatic jitter") from error
        a = np.linalg.solve(chol, ((ref-y)/sd)[indices])
        z = np.linalg.solve(chol, ((pred-y)/sd)[indices])
        r0, r1 = float(np.linalg.norm(a)/math.sqrt(len(indices))), float(np.linalg.norm(z)/math.sqrt(len(indices)))
        return {"n": len(indices), "baseline_rmse": r0, "candidate_rmse": r1,
                "delta_rmse": r1-r0, "strictly_reduced": r1 < r0}

    overall = score(np.arange(y.size))
    per_group = {g: score(np.array([i for i, label in enumerate(labels) if label == g]))
                 for g in required}
    return {"overall": overall, "groups": per_group,
            "all_groups_reduced": all(g["strictly_reduced"] for g in per_group.values()),
            "joint_arithmetic_reduction": overall["strictly_reduced"] and all(
                g["strictly_reduced"] for g in per_group.values()),
            "scientific_success": False,
            "status": "arithmetic_only_requires_forward_model_and_independent_validation"}


REQUIRED_OBSERVATION_GROUPS = (
    "atom_interferometry", "particle_mass_ratios", "particle_mixing", "cosmology",
)


def scoped_joint_residuals(observed, baseline, candidate, covariance, groups):
    """User-selected atom + flavor + cosmology scope; no missing-domain pass.

    Mass ratios and mixing are separate guards so one cannot mask the other.
    Dataset releases, nuisance policies and independent validation remain open.
    """
    return joint_residuals(observed, baseline, candidate, covariance, groups,
                           required_groups=REQUIRED_OBSERVATION_GROUPS)


def relative_flat_action(cutoff, *, eta=0.1, b=1.0, s=3.0, epsilon=0.25,
                         phase=0.0, reference_phase=math.pi):
    """Finite-cutoff relative one-loop action DENSITY of three real bosons.

    Flat four-dimensional background, M=M_*=hbar=1. This is not a Friedmann
    solution or an absolute dark-energy density. The first two spectral moments
    cancel analytically. A power series avoids subtraction at small t.
    """
    cutoff, s, epsilon, phase, reference_phase = map(
        float, (cutoff, s, epsilon, phase, reference_phase))
    if not all(map(math.isfinite, (cutoff, s, epsilon, phase, reference_phase))):
        raise ValueError("finite parameters required")
    if not 0 < cutoff <= 1 or s <= 2*abs(epsilon):
        raise ValueError("0 < cutoff <= 1 and strictly positive masses required")
    dimension_kernel(1, eta=eta, b=b)
    angles = 2*math.pi*np.arange(3)
    masses = s+2*epsilon*np.cos((phase+angles)/3)
    reference = s+2*epsilon*np.cos((reference_phase+angles)/3)
    max_mass = max(masses.max(), reference.max())
    moments = [0., 0., 0.] + [float(np.sum(masses**k)-np.sum(reference**k))
                              for k in range(3, 19)]
    # The first non-cancelling moment has a simple exact expression.
    moments[3] = 6*epsilon**3*(math.cos(phase)-math.cos(reference_phase))

    def heat_difference(t):
        if t*max_mass < .1:
            return math.fsum((-t)**k*moments[k]/math.factorial(k) for k in range(3, 19))
        return float(np.sum(np.exp(-t*masses))-np.sum(np.exp(-t*reference)))

    def density_per_log_time(v):
        t = math.exp(v)
        difference = heat_difference(t)
        if difference == 0:
            return 0.
        log_value = (dimension_kernel(t, eta=eta, b=b)["log_heat_trace"]
                     +math.log(abs(difference))-math.log(2*(4*math.pi)**2))
        return -math.copysign(math.exp(log_value), difference)

    low, err_low = quad(density_per_log_time, math.log(cutoff), 0., epsabs=1e-12,
                        epsrel=1e-9, limit=200)
    high, err_high = quad(lambda t: density_per_log_time(math.log(t))/t, 1., math.inf,
                          epsabs=1e-12, epsrel=1e-9, limit=200)
    return {"relative_action_density": low+high, "quadrature_error": err_low+err_high}


if __name__ == "__main__":
    print(json.dumps({"status": "mathematical_candidate_not_observation_fit",
                      "parameters": {"eta": 0.1, "b": 1.0, "d_ir": 4.0},
                      "scales": [{"t": t, **dimension_kernel(t)}
                                 for t in (1e-12, 1e-6, 1.0, 1e6, 1e12)],
                      "flat_relative_action_cutoff_scan": [
                          {"cutoff": t, "eta": eta, **relative_flat_action(t, eta=eta)}
                          for eta in (0., .1) for t in (1e-3, 1e-6, 1e-12)]}, indent=2))
