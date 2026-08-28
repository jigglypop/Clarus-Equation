"""Fraction-layer regressions for the hard-constraint and uncertainty packages.

Executable companion of ``paper/9_등호이전/05k`` and ``05l``.  Monte Carlo
estimates with fixed seeds verify, on the finite/free-mode model:

- threshold scaling (05k theorem 3.2): a fixed survival fraction forces
  ``u_th = N/2 + z_q sqrt(N/2)``,
- path/energy fraction gap of order ``N^{-1/2}`` (05k theorem 3.3),
- per-mode Euclidean action moments ``(1/2, 1/2)`` in hbar units
  (05l theorem 2.2),
- mean-field ratio convergence ``1 + O(1/N_eff)`` (05l theorem 5.1),
- mode decomposition audit: exclusion of the ``N_eff = D_eff`` reading and
  the observation-forced lower bound ``N_eff >= ~445`` (05m theorems 3.2-3.3).

No physical readout is claimed here.  These are mathematical regressions.
"""

from __future__ import annotations

import json
import math

import numpy as np


def normal_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def normal_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def gamma_action_samples(shape: float, samples: int, seed: int) -> np.ndarray:
    """Total dimensionless action ``u = S_E / hbar`` of the free-mode model."""
    rng = np.random.default_rng(seed)
    return rng.gamma(shape=shape, scale=1.0, size=samples)


def threshold_fraction(shape: float, z: float, samples: int, seed: int) -> float:
    """Survival fraction at the scaled threshold ``u_th = shape + z sqrt(shape)``."""
    u = gamma_action_samples(shape, samples, seed)
    u_th = shape + z * math.sqrt(shape)
    return float(np.mean(u < u_th))


def path_energy_gap(shape: float, z: float, samples: int, seed: int) -> float:
    """Gap between path fraction and energy fraction at the scaled threshold."""
    u = gamma_action_samples(shape, samples, seed)
    u_th = shape + z * math.sqrt(shape)
    survive = u < u_th
    path_fraction = float(np.mean(survive))
    energy_fraction = float(np.sum(u[survive]) / np.sum(u))
    return abs(path_fraction - energy_fraction)


def mode_action_moments(samples: int, seed: int) -> tuple[float, float]:
    """Per-mode dimensionless action ``u_k = z^2 / 2`` mean and variance."""
    rng = np.random.default_rng(seed)
    u = 0.5 * rng.standard_normal(samples) ** 2
    return float(np.mean(u)), float(np.var(u))


def mean_field_ratio(n_modes: int, target_mean: float, samples: int, seed: int) -> float:
    """Ratio ``<e^{-Phi}> / e^{-<Phi>}`` for an intensive sum of modes."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((samples, n_modes))
    phi = (target_mean / n_modes) * np.sum(z**2, axis=1)
    return float(np.mean(np.exp(-phi)) / math.exp(-target_mean))


def gamma_mean_field_ratio(n_modes: float, target_mean: float) -> float:
    """Closed-form ratio ``<e^{-Phi}> / e^{-<Phi>}`` for the Gamma-mode benchmark.

    ``Phi ~ Gamma(k, theta)`` with ``k = n_modes / 2`` and mean ``target_mean``
    (05m theorem 3.2).  ``mean_field_ratio`` is its Monte Carlo counterpart.
    """
    k = n_modes / 2.0
    return math.exp(target_mean - k * math.log1p(target_mean / k))


def minimal_modes_for_budget(target_mean: float, rel_tolerance: float) -> int:
    """Smallest mode count whose benchmark ratio stays within ``1 + rel_tolerance``."""
    n = 2
    while gamma_mean_field_ratio(n, target_mean) > 1.0 + rel_tolerance:
        n += 1
    return n


def run_report() -> dict:
    z_boot = -1.66
    thresholds = {
        str(2 * k): {
            "fraction": threshold_fraction(k, z_boot, samples=200_000, seed=0),
            "normal_target": normal_cdf(z_boot),
        }
        for k in (32, 512)
    }

    z_gap = -1.0
    gaps = {}
    for k in (128, 512):
        gap = path_energy_gap(k, z_gap, samples=400_000, seed=1)
        gaps[str(2 * k)] = {
            "gap": gap,
            "gap_times_sqrt_shape": gap * math.sqrt(k),
            "predicted_constant": normal_pdf(z_gap),
        }

    mean, var = mode_action_moments(samples=200_000, seed=2)

    mean_field = {
        str(n): {
            "ratio": (ratio := mean_field_ratio(n, target_mean=3.0, samples=400_000, seed=3)),
            "excess_times_n": (ratio - 1.0) * n,
            "predicted_excess_times_n": 3.0**2,
        }
        for n in (16, 64, 256)
    }

    eps_sq = 0.04865
    d_eff = 3.17776
    phi_mean = (1.0 - eps_sq) * d_eff
    rel_budget = 0.0010 / 0.0486
    ratio_r2 = gamma_mean_field_ratio(d_eff, phi_mean)
    mode_audit = {
        "phi_mean": phi_mean,
        "ratio_at_n_eff_equal_d_eff": ratio_r2,
        "implied_omega_b": ratio_r2 * eps_sq,
        "observed_omega_b": 0.0486,
        "minimal_n_eff_within_budget": minimal_modes_for_budget(phi_mean, rel_budget),
        "correlation_cap": 2.0 * math.log1p(rel_budget) / (2.0 * phi_mean**2),
    }

    return {
        "threshold_scaling_05k_thm_3_2": thresholds,
        "path_energy_gap_05k_thm_3_3": gaps,
        "mode_action_moments_05l_thm_2_2": {"mean": mean, "var": var, "target": 0.5},
        "mean_field_ratio_05l_thm_5_1": mean_field,
        "mode_audit_05m_thm_3_2_3_3": mode_audit,
    }


def main() -> None:
    print(json.dumps(run_report(), indent=2))


if __name__ == "__main__":
    main()
