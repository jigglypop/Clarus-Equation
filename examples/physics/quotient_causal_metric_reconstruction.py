"""Finite certificates for quotient readout, causal order, and metric recovery.

This is deliberately a ceiling, not a CE metric derivation: a rank-one readout
can have a well-defined quotient coordinate while losing its full prior, and
causal order can determine at most conformal structure without further data.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math

import numpy as np

from examples.physics.causal_light_geometry_audit import conformal_counterexample


P0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)


def density(p: float, coherence: complex) -> np.ndarray:
    """Return a validated qubit state ``[[p,z],[z*,1-p]]``."""

    if not math.isfinite(p) or not 0.0 < p < 1.0:
        raise ValueError("p must be finite and lie in (0, 1)")
    if not math.isfinite(coherence.real) or not math.isfinite(coherence.imag):
        raise ValueError("coherence must be finite")
    state = np.array([[p, coherence], [coherence.conjugate(), 1.0 - p]], dtype=complex)
    if np.linalg.eigvalsh(state).min() < -1.0e-12:
        raise ValueError("density matrix must be positive semidefinite")
    return state


def luders_zero(state: np.ndarray) -> np.ndarray:
    """Subnormalised rank-one Lueders outcome ``P0 rho P0``."""

    return P0 @ state @ P0


def posterior_zero(state: np.ndarray) -> np.ndarray:
    outcome = luders_zero(state)
    probability = float(np.trace(outcome).real)
    if probability <= 0.0:
        raise ValueError("outcome zero has no posterior at zero probability")
    return outcome / probability


def quotient_coordinate(state: np.ndarray) -> float:
    return float(np.trace(luders_zero(state)).real)


def canonical_section(p: float) -> np.ndarray:
    return density(p, 0.0)


def conformal_sign(interval: tuple[float, ...], omega: float) -> float:
    """Mostly-plus interval sign after the positive scaling ``Omega**2 g``."""

    if not math.isfinite(omega) or omega <= 0.0:
        raise ValueError("Omega must be finite and positive")
    if len(interval) < 2:
        raise ValueError("an interval needs time and at least one spatial component")
    if any(not math.isfinite(value) for value in interval):
        raise ValueError("interval components must be finite")
    base = -interval[0] ** 2 + sum(value * value for value in interval[1:])
    return omega * omega * base


def volume_recovery(volume_ratio: float, *, n: int = 4) -> float:
    """Recover a supplied dimensionless conformal factor from ``Omega**n``."""

    if not math.isfinite(volume_ratio) or volume_ratio <= 0.0:
        raise ValueError("volume_ratio must be finite and positive")
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError("n must be an integer spacetime dimension of at least two")
    return volume_ratio ** (1.0 / n)


def z2_cycle(transports: tuple[int, int, int, int], lengths: tuple[float, ...]) -> dict[str, object]:
    """A C4 base whose fixed Z2 holonomy still leaves edge metric free."""

    if len(transports) != 4 or any(value not in (-1, 1) for value in transports):
        raise ValueError("Z2 transports must be exactly four values in {-1, +1}")
    if len(lengths) != 4 or any(not math.isfinite(value) or value <= 0.0 for value in lengths):
        raise ValueError("C4 lengths must be four finite positive values")
    return {"base": "C4", "holonomy": math.prod(transports), "perimeter": sum(lengths)}


@dataclass(frozen=True)
class QuotientCertificate:
    p: float
    coherences: tuple[complex, ...]
    prior_eigenvalues: tuple[tuple[float, float], ...]
    identical_subnormalised_readouts: bool
    identical_posteriors: bool
    distinct_priors: bool
    section_roundtrip: bool
    quotient_homeomorphism_conditions: dict[str, bool]
    controls: dict[str, bool]
    conformal: dict[str, object]
    z2_hidden_connection: dict[str, object]
    dimensions: dict[str, bool]
    accounting: dict[str, bool]
    status: dict[str, bool]


def certificate(*, p: float = 0.4, epsilon: float = 0.1, omega: float = 2.0, n: int = 4) -> QuotientCertificate:
    """Build the deterministic finite certificate with fail-closed inputs."""

    if not math.isfinite(epsilon) or not 0.0 < epsilon <= p < 1.0:
        raise ValueError("require finite 0 < epsilon <= p < 1")
    if not math.isfinite(omega) or omega <= 0.0:
        raise ValueError("Omega must be finite and positive")
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError("n must be an integer spacetime dimension of at least two")
    bound = math.sqrt(p * (1.0 - p))
    coherences = (0.0j, 0.5 * bound, 0.25j * bound)
    states = tuple(density(p, z) for z in coherences)
    readouts = tuple(luders_zero(state) for state in states)
    posteriors = tuple(posterior_zero(state) for state in states)
    eigens = tuple(tuple(float(x) for x in np.linalg.eigvalsh(state)) for state in states)
    section = canonical_section(p)
    ratio = omega**n
    intervals = {"timelike": (2.0, 1.0), "null": (1.0, 1.0), "spacelike": (1.0, 2.0)}
    signs = {name: conformal_sign(interval, omega) for name, interval in intervals.items()}
    base = {name: conformal_sign(interval, 1.0) for name, interval in intervals.items()}
    conformal = conformal_counterexample()
    plus = z2_cycle((1, 1, 1, 1), (1.0, 1.0, 1.0, 1.0))
    minus = z2_cycle((-1, 1, 1, 1), (1.0, 1.0, 1.0, 1.0))
    stretched = z2_cycle((1, 1, 1, 1), (1.0, 2.0, 1.0, 2.0))
    same_readout = all(np.allclose(readouts[0], item) for item in readouts[1:])
    same_posterior = all(np.allclose(posteriors[0], item) for item in posteriors[1:])
    distinct_priors = any(not np.allclose(states[0], item) for item in states[1:])
    quotient_conditions = {
        "finite_dimensional_density_state_space_compact": True,
        "luders_readout_continuous": True,
        "image_interval_pP0_hausdorff": True,
    }
    # This is only the compact-to-Hausdorff quotient-image theorem.  The
    # endpoint fibres p=0,1 shrink, so it does not provide a global smooth
    # fibre-bundle structure for the instrument state space.
    quotient_conditions["induced_quotient_to_image_homeomorphism"] = all(quotient_conditions.values())
    metric_counterexample = (
        conformal.causal_order_identical
        and conformal.minkowski_normalized_four_volume != conformal.de_sitter_normalized_four_volume
        and conformal.minkowski_normalized_ricci_scalar != conformal.de_sitter_normalized_ricci_scalar
    )
    recovered = volume_recovery(ratio, n=n)
    return QuotientCertificate(
        p=p,
        coherences=coherences,
        prior_eigenvalues=eigens,
        identical_subnormalised_readouts=same_readout,
        identical_posteriors=same_posterior,
        distinct_priors=distinct_priors,
        section_roundtrip=np.allclose(luders_zero(section), p * P0) and quotient_coordinate(section) == p,
        quotient_homeomorphism_conditions=quotient_conditions,
        controls={"posterior_sample_satisfies_p_ge_epsilon": p >= epsilon},
        conformal={
            "existing_counterexample_causal_order_identical": conformal.causal_order_identical,
            "existing_minkowski_volume": conformal.minkowski_normalized_four_volume,
            "existing_de_sitter_volume": conformal.de_sitter_normalized_four_volume,
            "existing_minkowski_ricci": conformal.minkowski_normalized_ricci_scalar,
            "existing_de_sitter_ricci": conformal.de_sitter_normalized_ricci_scalar,
            "n": n, "Omega": omega, "volume_ratio": ratio,
            "recovered_Omega": recovered,
            "causal_signs_unchanged": all((base[k] == 0.0 and signs[k] == 0.0) or base[k] * signs[k] > 0.0 for k in base),
        },
        z2_hidden_connection={"supplied_regular_bundle_control": True,
                              "instrument_connection_derived": False,
                              "plus": plus, "minus": minus, "stretched": stretched,
                              "same_base_different_holonomy": plus["holonomy"] != minus["holonomy"],
                              "fixed_holonomy_different_perimeter": plus["perimeter"] != stretched["perimeter"]},
        dimensions={"Omega_dimensionless": True, "volume_ratio_dimensionless": True,
                    "dimensionless_is_not_physical_derivation": True},
        accounting={"rn_weighting_used": False, "energy_or_stress_accounting_present": False},
        status={
            "full_map_injective": not (distinct_priors and same_readout),
            "induced_quotient_homeomorphism_conditional": all(quotient_conditions.values()),
            "homeomorphism_determines_metric": not metric_counterexample,
            "same_causal_order_different_full_metric_witness": metric_counterexample,
            "continuum_causal_order_to_conformal_theorem_proved": False,
            "distinguishing_continuum_assumptions_supplied": False,
            "volume_scale_recovered_for_supplied_toy": math.isclose(recovered, omega), "differentiable_structure_derived": False,
            "closed_posterior_domain_constructed": False,
            "instrument_fibers_global_bundle_derived": False,
            "quotient_smooth_manifold_derived": False,
            "metric_tensor_pullback_derived": False,
            "physical_causal_order_derived": False, "volume_law_derived": False,
            "levi_civita_dynamics_derived": False, "fold_stress_derived": False,
            "gr_lensing_backreaction_derived": False, "holdout_complete": False,
            "success_gates_5_to_8_complete": False,
        },
    )


def run() -> dict[str, object]:
    """Return JSON-safe output for a source-only inspection."""

    result = asdict(certificate())
    result["coherences"] = [[z.real, z.imag] for z in certificate().coherences]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p", type=float, default=0.4)
    args = parser.parse_args()
    print(json.dumps(asdict(certificate(p=args.p)), default=str, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
