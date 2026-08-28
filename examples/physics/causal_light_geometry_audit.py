"""Focused audits for the causal-light rendering hypothesis.

The module separates three statements that are easy to conflate:

1. causal order fixes null/conformal structure under continuum assumptions;
2. causal order alone does not fix volume, curvature, or dimension; and
3. a microscopic maximum update rate does not by itself imply Lorentz symmetry;
4. the null causal frontier can move at c while massive record carriers remain
   strictly subluminal.

The calculations below are counterexamples and toy reconstruction checks.  They
do not derive a growth dynamics, a photon U(1) sector, or the observed universe
from a zero-dimensional seed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import random
from typing import Sequence


Event = tuple[float, tuple[float, ...]]


@dataclass(frozen=True)
class ConformalCounterexample:
    """Dimensionless comparison on H*eta in [-2, -1] and a unit spatial cube."""

    causal_order_identical: bool
    minkowski_normalized_four_volume: float
    de_sitter_normalized_four_volume: float
    minkowski_normalized_ricci_scalar: float
    de_sitter_normalized_ricci_scalar: float


@dataclass(frozen=True)
class CountingVolumeAudit:
    """Toy recovery of a volume ratio after a counting law is supplied."""

    trials: int
    normalized_event_density: float
    minkowski_mean_count: float
    de_sitter_mean_count: float
    expected_volume_ratio: float
    recovered_volume_ratio: float


def causally_comparable(left: Event, right: Event, *, c: float = 1.0) -> bool:
    """Return whether two events are timelike or null related in flat coordinates."""

    if c <= 0.0:
        raise ValueError("c must be positive")
    if len(left[1]) != len(right[1]):
        raise ValueError("events must have the same spatial dimension")
    delta_t = abs(right[0] - left[0])
    distance_squared = sum(
        (right_value - left_value) ** 2
        for left_value, right_value in zip(left[1], right[1])
    )
    return c * c * delta_t * delta_t >= distance_squared


def causal_pairs(events: Sequence[Event], *, c: float = 1.0) -> set[tuple[int, int]]:
    """Return unordered index pairs joined by the causal order."""

    pairs: set[tuple[int, int]] = set()
    for left in range(len(events)):
        for right in range(left + 1, len(events)):
            if causally_comparable(events[left], events[right], c=c):
                pairs.add((left, right))
    return pairs


def massive_carrier_speed_ratio(momentum_to_mass_ratio: float) -> float:
    """Return ``v_group / c`` for a relativistic massive carrier.

    The input is the dimensionless ratio ``kappa = p / (m c)``.  From
    ``E**2 = p**2 c**2 + m**2 c**4``, the group-speed ratio is
    ``kappa / sqrt(1 + kappa**2)``.  It is strictly below one for every
    finite positive ``kappa``.  This is only a kinematic counterexample to the
    claim that every physical record must itself propagate exactly at c.
    """

    if momentum_to_mass_ratio < 0.0 or not math.isfinite(momentum_to_mass_ratio):
        raise ValueError("momentum_to_mass_ratio must be finite and non-negative")
    return momentum_to_mass_ratio / math.sqrt(1.0 + momentum_to_mass_ratio**2)


def conformal_counterexample() -> ConformalCounterexample:
    """Exhibit identical null order with unequal volume and curvature.

    Compare Minkowski space with the conformal de Sitter patch
    ``g_dS = (H eta)^-2 g_M`` on ``H eta in [-2, -1]``.  A positive conformal
    factor leaves causal signs and unparameterized null paths unchanged.  The
    reported quantities are the dimensionless combinations ``H**4 * V`` and
    ``R / H**2`` (with the coordinate cube normalized consistently).
    """

    events: tuple[Event, ...] = (
        (-1.9, (0.10, 0.10, 0.10)),
        (-1.6, (0.40, 0.10, 0.10)),
        (-1.3, (0.45, 0.40, 0.10)),
        (-1.1, (0.90, 0.90, 0.90)),
    )
    minkowski_order = causal_pairs(events)
    # Multiplication by Omega(eta)^2 > 0 cannot change the causal sign, so the
    # de Sitter conformal patch has the same coordinate causal relation.
    de_sitter_order = causal_pairs(events)

    eta_start = -2.0
    eta_end = -1.0
    minkowski_volume = eta_end - eta_start
    de_sitter_normalized_volume = (
        (-1.0 / (3.0 * eta_end**3))
        - (-1.0 / (3.0 * eta_start**3))
    )

    return ConformalCounterexample(
        causal_order_identical=minkowski_order == de_sitter_order,
        minkowski_normalized_four_volume=minkowski_volume,
        de_sitter_normalized_four_volume=de_sitter_normalized_volume,
        minkowski_normalized_ricci_scalar=0.0,
        de_sitter_normalized_ricci_scalar=12.0,
    )


def _poisson_count(mean: float, generator: random.Random) -> int:
    """Draw an exact Poisson count using unit-rate exponential arrivals."""

    if mean < 0.0 or not math.isfinite(mean):
        raise ValueError("Poisson mean must be finite and non-negative")
    elapsed = 0.0
    count = 0
    while True:
        elapsed += generator.expovariate(1.0)
        if elapsed > mean:
            return count
        count += 1


def counting_volume_audit(
    *,
    normalized_event_density: float = 120.0,
    trials: int = 1000,
    seed: int = 20260828,
) -> CountingVolumeAudit:
    """Show how calibrated counts distinguish conformally degenerate volumes.

    This adopts, rather than derives, ``N ~ Poisson(rho_c * V_4)``.  The two
    regions have the same conformal causal order used by
    :func:`conformal_counterexample`, but normalized four-volumes 1 and 7/24.
    """

    if normalized_event_density <= 0.0:
        raise ValueError("normalized_event_density must be positive")
    if trials < 1:
        raise ValueError("trials must be positive")

    generator = random.Random(seed)
    expected_ratio = 7.0 / 24.0
    minkowski_total = 0
    de_sitter_total = 0
    for _ in range(trials):
        minkowski_total += _poisson_count(normalized_event_density, generator)
        de_sitter_total += _poisson_count(
            normalized_event_density * expected_ratio,
            generator,
        )

    minkowski_mean = minkowski_total / trials
    de_sitter_mean = de_sitter_total / trials
    return CountingVolumeAudit(
        trials=trials,
        normalized_event_density=normalized_event_density,
        minkowski_mean_count=minkowski_mean,
        de_sitter_mean_count=de_sitter_mean,
        expected_volume_ratio=expected_ratio,
        recovered_volume_ratio=de_sitter_mean / minkowski_mean,
    )


def expected_ordering_fraction(spacetime_dimension: float) -> float:
    """Myrheim--Meyer ordering fraction for a flat causal diamond.

    If ``R`` is the number of causally comparable unordered pairs, this returns
    the large-sample expectation of ``R / comb(N, 2)``:

        Gamma(d + 1) Gamma(d / 2) / (2 Gamma(3 d / 2)).

    The often-quoted expression with denominator 4 is instead the large-sample
    relation density ``R / N**2``.  Keeping those conventions separate avoids a
    factor-of-two error in the inferred dimension.
    """

    if spacetime_dimension < 1.0:
        raise ValueError("spacetime_dimension must be at least one")
    log_fraction = (
        math.lgamma(spacetime_dimension + 1.0)
        + math.lgamma(spacetime_dimension / 2.0)
        - math.log(2.0)
        - math.lgamma(3.0 * spacetime_dimension / 2.0)
    )
    return math.exp(log_fraction)


def expected_relation_density(spacetime_dimension: float) -> float:
    """Return the large-sample expectation of R / N**2."""

    return expected_ordering_fraction(spacetime_dimension) / 2.0


def sprinkle_minkowski_diamond(
    spacetime_dimension: int,
    count: int,
    *,
    seed: int,
) -> list[Event]:
    """Uniformly sprinkle events into a unit flat Alexandrov interval.

    The interval has ``-1 <= t <= 1`` and spatial radius ``1 - abs(t)``.
    This samples a known manifold-like target; it does not generate a manifold
    from a singleton or test a causal-set growth law.
    """

    if spacetime_dimension < 2:
        raise ValueError("the sprinkling audit requires spacetime_dimension >= 2")
    if count < 1:
        raise ValueError("count must be positive")

    generator = random.Random(seed)
    spatial_dimension = spacetime_dimension - 1
    events: list[Event] = []
    for _ in range(count):
        # Cross-section volume is proportional to (1 - |t|) ** (d - 1).
        absolute_time = 1.0 - generator.random() ** (1.0 / spacetime_dimension)
        time = absolute_time if generator.random() < 0.5 else -absolute_time
        maximum_radius = 1.0 - absolute_time
        radius = maximum_radius * generator.random() ** (1.0 / spatial_dimension)

        direction = [generator.gauss(0.0, 1.0) for _ in range(spatial_dimension)]
        norm = math.sqrt(sum(component * component for component in direction))
        spatial = tuple(radius * component / norm for component in direction)
        events.append((time, spatial))
    return events


def ordering_fraction(events: Sequence[Event]) -> float:
    """Return R / comb(N, 2) for a finite causal sample."""

    count = len(events)
    if count < 2:
        raise ValueError("at least two events are required to estimate dimension")
    related = len(causal_pairs(events))
    return related / math.comb(count, 2)


def estimate_myrheim_meyer_dimension(
    observed_fraction: float,
    *,
    maximum_dimension: float = 32.0,
    iterations: int = 80,
) -> float:
    """Invert the monotone continuum ordering-fraction formula."""

    if not 0.0 < observed_fraction <= 1.0:
        raise ValueError("observed_fraction must lie in (0, 1]")
    if maximum_dimension <= 1.0:
        raise ValueError("maximum_dimension must exceed one")
    if observed_fraction < expected_ordering_fraction(maximum_dimension):
        raise ValueError("observed fraction implies a dimension above the search bound")

    lower = 1.0
    upper = maximum_dimension
    for _ in range(iterations):
        midpoint = (lower + upper) / 2.0
        if expected_ordering_fraction(midpoint) > observed_fraction:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def square_lattice_angular_frequency(
    wavevector: Sequence[float],
    *,
    lattice_spacing: float = 1.0,
    limiting_speed: float = 1.0,
) -> float:
    """Nearest-neighbour lattice dispersion used as a preferred-frame control."""

    if not wavevector:
        raise ValueError("wavevector must contain at least one component")
    if lattice_spacing <= 0.0 or limiting_speed <= 0.0:
        raise ValueError("lattice_spacing and limiting_speed must be positive")
    sine_sum = sum(
        math.sin(component * lattice_spacing / 2.0) ** 2
        for component in wavevector
    )
    return 2.0 * limiting_speed * math.sqrt(sine_sum) / lattice_spacing


def lattice_directional_split(
    wavenumber: float,
    *,
    lattice_spacing: float = 1.0,
    limiting_speed: float = 1.0,
) -> float:
    """Relative axis/diagonal frequency split at fixed 2D |k|."""

    if wavenumber <= 0.0:
        raise ValueError("wavenumber must be positive")
    axis = square_lattice_angular_frequency(
        (wavenumber, 0.0),
        lattice_spacing=lattice_spacing,
        limiting_speed=limiting_speed,
    )
    diagonal_component = wavenumber / math.sqrt(2.0)
    diagonal = square_lattice_angular_frequency(
        (diagonal_component, diagonal_component),
        lattice_spacing=lattice_spacing,
        limiting_speed=limiting_speed,
    )
    return abs(axis - diagonal) / ((axis + diagonal) / 2.0)


def run_audit(*, count: int = 800, seed: int = 20260828) -> dict[str, object]:
    """Run the focused counterexample and reconstruction checks."""

    target_dimension = 4
    events = sprinkle_minkowski_diamond(target_dimension, count, seed=seed)
    observed = ordering_fraction(events)
    estimated = estimate_myrheim_meyer_dimension(observed)
    conformal = conformal_counterexample()
    counting = counting_volume_audit(seed=seed)

    return {
        "claim_boundary": {
            "light_only_volume_curvature": "refuted_by_conformal_counterexample",
            "order_plus_number_route": "toy_reconstruction_only",
            "zero_dimensional_origin": "not_tested",
        },
        "conformal_counterexample": asdict(conformal),
        "counting_volume_audit": asdict(counting),
        "dimension_reconstruction": {
            "sample_count": count,
            "seed": seed,
            "target_spacetime_dimension": target_dimension,
            "continuum_ordering_fraction": expected_ordering_fraction(target_dimension),
            "observed_ordering_fraction": observed,
            "estimated_spacetime_dimension": estimated,
        },
        "lattice_control": {
            "low_wavenumber_directional_split": lattice_directional_split(0.01),
            "finite_wavenumber_directional_split": lattice_directional_split(1.5),
            "interpretation": "a maximum update speed alone does not ensure Lorentz symmetry",
        },
        "record_frontier_control": {
            "null_outer_front_speed_ratio": 1.0,
            "massive_carrier_speed_ratio_at_p_over_mc_1": massive_carrier_speed_ratio(1.0),
            "stored_record_speed_ratio": 0.0,
            "interpretation": "the causal outer envelope can be null while records are timelike",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260828)
    arguments = parser.parse_args()
    print(json.dumps(run_audit(count=arguments.count, seed=arguments.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
