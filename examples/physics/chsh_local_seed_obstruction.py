"""Finite CHSH obstruction for a setting-independent local common seed.

This module keeps three maps separate.

* A fixed-setting quantum projective instrument produces the singlet CHSH
  probability box and is CPTP for each declared setting pair.
* A supplied inverse-CDF seed can be lifted from the coarse label ``a`` to a
  fine coordinate ``(a, r)``.  The lift is a weighted-measure bijection, but
  with the usual interval and finite-coproduct topologies it is not a
  homeomorphism when more than one outcome has positive probability.
* A setting-independent common-past seed with factorized local responses is
  a convex mixture of sixteen deterministic strategies and obeys CHSH <= 2.
  It therefore cannot reproduce the finite singlet box with CHSH = 2 sqrt(2).

The Bell obstruction is deliberately narrow.  It does not exclude a global
or contextual fine-state bijection, and operational no-signalling of this
finite probability box is not a derivation of relativistic QFT
microcausality.  No seed, outcome, probability, or CHSH score is assigned an
energy or spacetime scale here.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from itertools import product
import json
import math
from typing import Sequence

import numpy as np

from examples.physics.local_born_selection_kernel import (
    build_seed_partition,
    select_partition_cell,
)


DEFAULT_TOLERANCE = 1.0e-12
OUTCOMES = (-1, 1)
SETTINGS = (0, 1)
CHSH_PATTERN = np.array([[-1.0, -1.0], [-1.0, 1.0]])


def _positive_tolerance(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    return value


def _isotropic_parameter(value: float) -> float:
    parameter = float(value)
    if not math.isfinite(parameter) or not 0.0 <= parameter <= 1.0:
        raise ValueError("eta must be finite and lie in [0, 1]")
    return parameter


def isotropic_chsh_box(eta: float) -> np.ndarray:
    """Return the PR-oriented isotropic box ``P_eta``.

    ``P_eta(a,b|x,y) = (1 + a*b*eta*c_xy)/4`` with
    ``c = (-1,-1,-1,+1)``.  Thus its absolute CHSH value is ``4*eta``;
    ``eta=1`` is a PR box, not a quantum singlet box.
    """

    visibility = _isotropic_parameter(eta)
    probabilities = np.zeros((2, 2, 2, 2), dtype=np.float64)
    for x in SETTINGS:
        for y in SETTINGS:
            for a_index, a in enumerate(OUTCOMES):
                for b_index, b in enumerate(OUTCOMES):
                    probabilities[x, y, a_index, b_index] = 0.25 * (
                        1.0 + a * b * visibility * CHSH_PATTERN[x, y]
                    )
    return probabilities


def singlet_density() -> np.ndarray:
    """Return ``(|01>-|10>)(<01|-<10|)/2`` in the computational basis."""

    vector = np.array([0.0, 1.0, -1.0, 0.0], dtype=np.complex128) / math.sqrt(2.0)
    return np.outer(vector, vector.conj())


def chsh_observables() -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Return ``A=(Z,X)`` and ``B=((Z+X)/sqrt(2),(Z-X)/sqrt(2))``."""

    x_pauli = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    z_pauli = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    alice = (z_pauli, x_pauli)
    bob = (
        (z_pauli + x_pauli) / math.sqrt(2.0),
        (z_pauli - x_pauli) / math.sqrt(2.0),
    )
    return alice, bob


def _projector(observable: np.ndarray, outcome: int) -> np.ndarray:
    if outcome not in OUTCOMES:
        raise ValueError("outcome must be -1 or +1")
    return 0.5 * (np.eye(2, dtype=np.complex128) + outcome * observable)


@dataclass(frozen=True)
class ProjectiveInstrumentAudit:
    probabilities: np.ndarray
    maximum_projector_residual: float
    maximum_completeness_residual: float
    minimum_choi_eigenvalue: float
    maximum_posterior_trace_residual: float


def quantum_projective_instrument_audit() -> ProjectiveInstrumentAudit:
    """Audit all four fixed-setting joint projective instruments."""

    density = singlet_density()
    alice, bob = chsh_observables()
    probabilities = np.zeros((2, 2, 2, 2), dtype=np.float64)
    maximum_projector_residual = 0.0
    maximum_completeness_residual = 0.0
    minimum_choi_eigenvalue = math.inf
    maximum_posterior_trace_residual = 0.0
    joint_identity = np.eye(4, dtype=np.complex128)

    for x in SETTINGS:
        for y in SETTINGS:
            operators: list[np.ndarray] = []
            completeness = np.zeros((4, 4), dtype=np.complex128)
            for a_index, a in enumerate(OUTCOMES):
                for b_index, b in enumerate(OUTCOMES):
                    operator = np.kron(
                        _projector(alice[x], a),
                        _projector(bob[y], b),
                    )
                    operators.append(operator)
                    maximum_projector_residual = max(
                        maximum_projector_residual,
                        float(np.linalg.norm(operator @ operator - operator, ord=2)),
                        float(np.linalg.norm(operator - operator.conj().T, ord=2)),
                    )
                    completeness += operator.conj().T @ operator
                    operation = operator @ density @ operator.conj().T
                    probability = float(np.trace(operation).real)
                    probabilities[x, y, a_index, b_index] = probability
                    if probability > 0.0:
                        posterior = operation / probability
                        maximum_posterior_trace_residual = max(
                            maximum_posterior_trace_residual,
                            abs(float(np.trace(posterior).real) - 1.0),
                            abs(float(np.trace(posterior).imag)),
                        )

            maximum_completeness_residual = max(
                maximum_completeness_residual,
                float(np.linalg.norm(completeness - joint_identity, ord=2)),
            )
            choi = np.zeros((16, 16), dtype=np.complex128)
            for operator in operators:
                vector = operator.reshape(-1, order="F")
                choi += np.outer(vector, vector.conj())
            minimum_choi_eigenvalue = min(
                minimum_choi_eigenvalue,
                float(np.linalg.eigvalsh(choi).min()),
            )

    return ProjectiveInstrumentAudit(
        probabilities=probabilities,
        maximum_projector_residual=maximum_projector_residual,
        maximum_completeness_residual=maximum_completeness_residual,
        minimum_choi_eigenvalue=minimum_choi_eigenvalue,
        maximum_posterior_trace_residual=maximum_posterior_trace_residual,
    )


def box_correlations(probabilities: np.ndarray) -> tuple[float, float, float, float]:
    """Return ``(E00,E01,E10,E11)`` after strict shape validation."""

    box = np.asarray(probabilities, dtype=np.float64)
    if box.shape != (2, 2, 2, 2) or not np.isfinite(box).all():
        raise ValueError("probability box must be finite with shape (2, 2, 2, 2)")
    correlations: list[float] = []
    for x in SETTINGS:
        for y in SETTINGS:
            correlations.append(
                math.fsum(
                    a * b * float(box[x, y, a_index, b_index])
                    for a_index, a in enumerate(OUTCOMES)
                    for b_index, b in enumerate(OUTCOMES)
                )
            )
    return tuple(correlations)  # type: ignore[return-value]


def chsh_scores(probabilities: np.ndarray) -> tuple[float, float]:
    """Return the oriented facet score and the usual absolute CHSH score."""

    correlations = np.asarray(box_correlations(probabilities)).reshape(2, 2)
    facet_score = float(np.sum(CHSH_PATTERN * correlations))
    standard_expression = float(
        correlations[0, 0]
        + correlations[0, 1]
        + correlations[1, 0]
        - correlations[1, 1]
    )
    return facet_score, abs(standard_expression)


@dataclass(frozen=True)
class BoxAudit:
    minimum_probability: float
    maximum_normalization_residual: float
    maximum_no_signalling_residual: float
    maximum_unbiased_marginal_residual: float


def audit_probability_box(probabilities: np.ndarray) -> BoxAudit:
    """Audit nonnegativity, per-context normalization, and no-signalling."""

    box = np.asarray(probabilities, dtype=np.float64)
    if box.shape != (2, 2, 2, 2) or not np.isfinite(box).all():
        raise ValueError("probability box must be finite with shape (2, 2, 2, 2)")
    normalization_residual = max(
        abs(float(np.sum(box[x, y])) - 1.0) for x in SETTINGS for y in SETTINGS
    )
    no_signalling_residual = 0.0
    unbiased_residual = 0.0
    for x in SETTINGS:
        for a_index in range(2):
            marginals = tuple(float(np.sum(box[x, y, a_index, :])) for y in SETTINGS)
            no_signalling_residual = max(
                no_signalling_residual, abs(marginals[0] - marginals[1])
            )
            unbiased_residual = max(
                unbiased_residual, *(abs(value - 0.5) for value in marginals)
            )
    for y in SETTINGS:
        for b_index in range(2):
            marginals = tuple(float(np.sum(box[x, y, :, b_index])) for x in SETTINGS)
            no_signalling_residual = max(
                no_signalling_residual, abs(marginals[0] - marginals[1])
            )
            unbiased_residual = max(
                unbiased_residual, *(abs(value - 0.5) for value in marginals)
            )
    return BoxAudit(
        minimum_probability=float(box.min()),
        maximum_normalization_residual=normalization_residual,
        maximum_no_signalling_residual=no_signalling_residual,
        maximum_unbiased_marginal_residual=unbiased_residual,
    )


def deterministic_local_strategies() -> tuple[tuple[int, int, int, int], ...]:
    """Return all ``(A0,A1,B0,B1)`` deterministic response assignments."""

    return tuple(product(OUTCOMES, repeat=4))


def deterministic_facet_score(strategy: Sequence[int]) -> int:
    """Return the PR-oriented CHSH facet score of one local strategy."""

    values = tuple(strategy)
    if len(values) != 4 or any(value not in OUTCOMES for value in values):
        raise ValueError("strategy must contain four outcomes in {-1, +1}")
    a0, a1, b0, b1 = values
    correlations = np.array(
        [[a0 * b0, a0 * b1], [a1 * b0, a1 * b1]], dtype=np.float64
    )
    return int(np.sum(CHSH_PATTERN * correlations))


def local_boundary_strategies() -> tuple[tuple[int, int, int, int], ...]:
    """Return the eight deterministic vertices on the chosen ``S=2`` facet."""

    return tuple(
        strategy
        for strategy in deterministic_local_strategies()
        if deterministic_facet_score(strategy) == 2
    )


def deterministic_mixture_box(
    strategies: Sequence[Sequence[int]],
) -> np.ndarray:
    """Return the uniform box generated by declared deterministic strategies."""

    declared = tuple(tuple(strategy) for strategy in strategies)
    if not declared:
        raise ValueError("at least one deterministic strategy is required")
    for strategy in declared:
        deterministic_facet_score(strategy)
    probabilities = np.zeros((2, 2, 2, 2), dtype=np.float64)
    weight = 1.0 / len(declared)
    for a0, a1, b0, b1 in declared:
        alice = (a0, a1)
        bob = (b0, b1)
        for x in SETTINGS:
            for y in SETTINGS:
                a_index = OUTCOMES.index(alice[x])
                b_index = OUTCOMES.index(bob[y])
                probabilities[x, y, a_index, b_index] += weight
    return probabilities


@dataclass(frozen=True)
class FineSeedCoordinate:
    outcome_index: int
    residual_coordinate: float
    interval: tuple[float, float]
    interval_probability: float


def lift_seed_coordinate(
    probabilities: Sequence[float],
    seed: float,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> FineSeedCoordinate:
    """Lift coarse inverse-CDF output to ``(outcome, residual coordinate)``."""

    partition = build_seed_partition(probabilities, tolerance=tolerance)
    outcome_index = select_partition_cell(partition, seed)
    start, end = partition.intervals[outcome_index]
    width = end - start
    if width <= 0.0:
        raise ArithmeticError("selected probability fibre must have positive width")
    residual = (float(seed) - start) / width
    if residual >= 1.0 and residual <= 1.0 + 10.0 * tolerance:
        residual = math.nextafter(1.0, 0.0)
    if not 0.0 <= residual < 1.0:
        raise ArithmeticError("residual coordinate must lie in [0, 1)")
    return FineSeedCoordinate(
        outcome_index=outcome_index,
        residual_coordinate=residual,
        interval=(start, end),
        interval_probability=partition.cell_probabilities[outcome_index],
    )


def invert_seed_coordinate(
    probabilities: Sequence[float],
    outcome_index: int,
    residual_coordinate: float,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> float:
    """Invert the declared fine seed coordinate on a positive fibre."""

    partition = build_seed_partition(probabilities, tolerance=tolerance)
    if (
        isinstance(outcome_index, bool)
        or not isinstance(outcome_index, int)
        or not 0 <= outcome_index < len(partition.intervals)
    ):
        raise ValueError("outcome_index must select a declared probability fibre")
    residual = float(residual_coordinate)
    if not math.isfinite(residual) or not 0.0 <= residual < 1.0:
        raise ValueError("residual_coordinate must be finite and lie in [0, 1)")
    start, end = partition.intervals[outcome_index]
    width = end - start
    if width <= 0.0 or partition.cell_probabilities[outcome_index] <= 0.0:
        raise ValueError("zero-probability fibres are empty")
    seed = start + width * residual
    if seed >= end:
        seed = math.nextafter(end, start)
    if select_partition_cell(partition, seed) != outcome_index:
        raise ArithmeticError("fine coordinate inverse left its declared fibre")
    return seed


def usual_coproduct_seed_lift_is_homeomorphism(
    probabilities: Sequence[float],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> bool:
    """Return the connectedness verdict for the usual declared topologies.

    With one positive fibre the affine coordinate map is a homeomorphism.
    With two or more positive fibres, ``[0,1)`` is connected while the finite
    coproduct of fibres is disconnected, so the bijection is not a
    homeomorphism.  A transported topology would make this tautological and
    is not used here.
    """

    partition = build_seed_partition(probabilities, tolerance=tolerance)
    positive_fibres = sum(value > 0.0 for value in partition.cell_probabilities)
    return positive_fibres == 1


@dataclass(frozen=True)
class ChshLocalSeedCertificate:
    quantum_correlations: tuple[float, float, float, float]
    quantum_oriented_facet_score: float
    quantum_absolute_chsh_score: float
    quantum_formula_residual: float
    quantum_minimum_probability: float
    quantum_normalization_residual: float
    quantum_no_signalling_residual: float
    quantum_unbiased_marginal_residual: float
    maximum_projector_residual: float
    maximum_instrument_completeness_residual: float
    minimum_instrument_choi_eigenvalue: float
    maximum_posterior_trace_residual: float
    deterministic_strategy_count: int
    deterministic_facet_scores: tuple[int, ...]
    maximum_deterministic_absolute_chsh_score: float
    local_boundary_strategy_count: int
    local_boundary_strategies: tuple[tuple[int, int, int, int], ...]
    local_boundary_probability_residual: float
    local_boundary_no_signalling_residual: float
    local_boundary_unbiased_marginal_residual: float
    pr_minimum_probability: float
    pr_normalization_residual: float
    pr_no_signalling_residual: float
    ns_local_fraction: float
    ns_nonlocal_fraction: float
    local_fraction_chsh_upper_bound: float
    local_fraction_upper_bound_residual: float
    local_pr_decomposition_residual: float
    seed_context_probabilities: tuple[float, ...]
    seed_positive_fibre_count: int
    maximum_seed_lift_round_trip_residual: float
    maximum_seed_fibre_measure_residual: float
    usual_coproduct_seed_lift_homeomorphism: bool
    coarse_seed_readout_many_to_one: bool
    dimensions: dict[str, bool]
    accounting: dict[str, bool]
    boundaries: dict[str, bool]
    alternatives: dict[str, bool]
    status: dict[str, bool]


def certificate(
    *, tolerance: float = DEFAULT_TOLERANCE
) -> ChshLocalSeedCertificate:
    """Build the E28 finite CHSH and fine-seed-boundary certificate."""

    tol = _positive_tolerance(tolerance)
    instrument = quantum_projective_instrument_audit()
    quantum_box = instrument.probabilities
    expected_quantum_box = isotropic_chsh_box(1.0 / math.sqrt(2.0))
    quantum_formula_residual = float(
        np.max(np.abs(quantum_box - expected_quantum_box))
    )
    quantum_audit = audit_probability_box(quantum_box)
    quantum_correlations = box_correlations(quantum_box)
    quantum_facet_score, quantum_chsh = chsh_scores(quantum_box)

    strategies = deterministic_local_strategies()
    facet_scores = tuple(deterministic_facet_score(strategy) for strategy in strategies)
    maximum_local_chsh = float(max(abs(value) for value in facet_scores))
    boundary_strategies = local_boundary_strategies()
    local_boundary_box = deterministic_mixture_box(boundary_strategies)
    expected_local_boundary = isotropic_chsh_box(0.5)
    local_boundary_probability_residual = float(
        np.max(np.abs(local_boundary_box - expected_local_boundary))
    )
    local_boundary_audit = audit_probability_box(local_boundary_box)

    pr_box = isotropic_chsh_box(1.0)
    pr_audit = audit_probability_box(pr_box)
    local_fraction = 2.0 - math.sqrt(2.0)
    nonlocal_fraction = math.sqrt(2.0) - 1.0
    reconstructed_quantum = (
        local_fraction * local_boundary_box + nonlocal_fraction * pr_box
    )
    decomposition_residual = float(
        np.max(np.abs(reconstructed_quantum - quantum_box))
    )
    local_fraction_upper_bound = (4.0 - quantum_chsh) / 2.0
    local_fraction_upper_bound_residual = abs(
        local_fraction_upper_bound - local_fraction
    )

    seed_probabilities = tuple(float(value) for value in quantum_box[0, 0].reshape(-1))
    seed_partition = build_seed_partition(seed_probabilities, tolerance=tol)
    seed_probes = tuple(
        start + fraction * (end - start)
        for (start, end), probability in zip(
            seed_partition.intervals, seed_partition.cell_probabilities
        )
        if probability > 0.0
        for fraction in (0.25, 0.75)
    )
    seed_coordinates = tuple(
        lift_seed_coordinate(seed_probabilities, seed, tolerance=tol)
        for seed in seed_probes
    )
    seed_round_trip_residual = max(
        abs(
            invert_seed_coordinate(
                seed_probabilities,
                coordinate.outcome_index,
                coordinate.residual_coordinate,
                tolerance=tol,
            )
            - seed
        )
        for seed, coordinate in zip(seed_probes, seed_coordinates)
    )
    seed_fibre_measure_residual = max(
        abs((end - start) - probability)
        for (start, end), probability in zip(
            seed_partition.intervals, seed_partition.cell_probabilities
        )
    )
    positive_fibre_count = sum(
        probability > 0.0 for probability in seed_partition.cell_probabilities
    )
    homeomorphism = usual_coproduct_seed_lift_is_homeomorphism(
        seed_probabilities, tolerance=tol
    )
    coarse_many_to_one = any(
        left.outcome_index == right.outcome_index
        and left.residual_coordinate != right.residual_coordinate
        for left, right in zip(seed_coordinates[::2], seed_coordinates[1::2])
    )

    numerical_limit = 50.0 * tol
    cptp_certified = (
        instrument.maximum_projector_residual <= numerical_limit
        and instrument.maximum_completeness_residual <= numerical_limit
        and instrument.minimum_choi_eigenvalue >= -numerical_limit
        and instrument.maximum_posterior_trace_residual <= numerical_limit
    )
    quantum_box_certified = (
        quantum_formula_residual <= numerical_limit
        and quantum_audit.minimum_probability >= -numerical_limit
        and quantum_audit.maximum_normalization_residual <= numerical_limit
        and quantum_audit.maximum_no_signalling_residual <= numerical_limit
        and quantum_audit.maximum_unbiased_marginal_residual <= numerical_limit
        and abs(quantum_chsh - 2.0 * math.sqrt(2.0)) <= numerical_limit
    )
    local_no_go_certified = (
        len(strategies) == 16
        and set(facet_scores) == {-2, 2}
        and maximum_local_chsh <= 2.0 + numerical_limit
        and quantum_chsh > 2.0 + numerical_limit
    )
    local_fraction_certified = (
        len(boundary_strategies) == 8
        and local_boundary_probability_residual <= numerical_limit
        and local_boundary_audit.maximum_no_signalling_residual <= numerical_limit
        and pr_audit.minimum_probability >= -numerical_limit
        and pr_audit.maximum_no_signalling_residual <= numerical_limit
        and decomposition_residual <= numerical_limit
        and local_fraction_upper_bound_residual <= numerical_limit
    )
    fine_seed_bijection_certified = (
        positive_fibre_count == 4
        and seed_round_trip_residual <= numerical_limit
        and seed_fibre_measure_residual <= numerical_limit
        and coarse_many_to_one
    )

    return ChshLocalSeedCertificate(
        quantum_correlations=quantum_correlations,
        quantum_oriented_facet_score=quantum_facet_score,
        quantum_absolute_chsh_score=quantum_chsh,
        quantum_formula_residual=quantum_formula_residual,
        quantum_minimum_probability=quantum_audit.minimum_probability,
        quantum_normalization_residual=quantum_audit.maximum_normalization_residual,
        quantum_no_signalling_residual=quantum_audit.maximum_no_signalling_residual,
        quantum_unbiased_marginal_residual=(
            quantum_audit.maximum_unbiased_marginal_residual
        ),
        maximum_projector_residual=instrument.maximum_projector_residual,
        maximum_instrument_completeness_residual=(
            instrument.maximum_completeness_residual
        ),
        minimum_instrument_choi_eigenvalue=instrument.minimum_choi_eigenvalue,
        maximum_posterior_trace_residual=(
            instrument.maximum_posterior_trace_residual
        ),
        deterministic_strategy_count=len(strategies),
        deterministic_facet_scores=facet_scores,
        maximum_deterministic_absolute_chsh_score=maximum_local_chsh,
        local_boundary_strategy_count=len(boundary_strategies),
        local_boundary_strategies=boundary_strategies,
        local_boundary_probability_residual=local_boundary_probability_residual,
        local_boundary_no_signalling_residual=(
            local_boundary_audit.maximum_no_signalling_residual
        ),
        local_boundary_unbiased_marginal_residual=(
            local_boundary_audit.maximum_unbiased_marginal_residual
        ),
        pr_minimum_probability=pr_audit.minimum_probability,
        pr_normalization_residual=pr_audit.maximum_normalization_residual,
        pr_no_signalling_residual=pr_audit.maximum_no_signalling_residual,
        ns_local_fraction=local_fraction,
        ns_nonlocal_fraction=nonlocal_fraction,
        local_fraction_chsh_upper_bound=local_fraction_upper_bound,
        local_fraction_upper_bound_residual=local_fraction_upper_bound_residual,
        local_pr_decomposition_residual=decomposition_residual,
        seed_context_probabilities=seed_probabilities,
        seed_positive_fibre_count=positive_fibre_count,
        maximum_seed_lift_round_trip_residual=seed_round_trip_residual,
        maximum_seed_fibre_measure_residual=seed_fibre_measure_residual,
        usual_coproduct_seed_lift_homeomorphism=homeomorphism,
        coarse_seed_readout_many_to_one=coarse_many_to_one,
        dimensions={
            "probabilities_and_marginals_dimensionless": True,
            "eta_and_local_fraction_dimensionless": True,
            "outcomes_settings_correlations_and_chsh_dimensionless": True,
            "seed_and_residual_coordinate_dimensionless": True,
            "no_mass_energy_length_or_time_scale_introduced": True,
        },
        accounting={
            "each_setting_probability_box_normalized_once": True,
            "local_and_pr_mixture_weights_sum_to_one": math.isclose(
                local_fraction + nonlocal_fraction, 1.0, abs_tol=numerical_limit
            ),
            "weighted_fibre_measure_uses_born_probability_once": True,
            "coarse_and_fine_seed_probabilities_not_double_counted": True,
            "unselected_probabilities_not_added_as_energy_or_stress": True,
            "seed_or_hidden_coordinate_carries_energy": False,
        },
        boundaries={
            "isotropic_parameter_one_is_pr_box_not_singlet": True,
            "local_fraction_scenario_is_fixed_two_setting_binary_outcome": True,
            "local_fraction_remainder_class_is_nonsignalling": True,
            "bell_assumes_setting_independent_seed_distribution": True,
            "bell_assumes_factorized_local_response": True,
            "global_or_contextual_fine_bijection_not_excluded": True,
            "conditional_joint_inverse_cdf_is_not_local_factorization": True,
            "fine_seed_residual_is_not_a_derived_physical_hidden_path": True,
            "zero_probability_fibres_are_empty": True,
            "usual_interval_and_finite_coproduct_topologies_declared": True,
            "transported_topology_not_used_as_physical_evidence": True,
            "finite_discrete_observation_label_space_is_zero_dimensional": True,
            "zero_dimensional_readout_is_not_spacetime_dimension": True,
            "measure_bijection_does_not_earn_metric_pullback": True,
            "operational_no_signalling_is_not_qft_microcausality": True,
            "timelike_domino_limited_to_future_cone_pointer_propagation": True,
            "timelike_domino_not_spacelike_bell_correlation_generator": True,
        },
        alternatives={
            "global_contextual_joint_rule_route_open": True,
            "ontic_nonlocal_operational_no_signalling_route_open": True,
            "measurement_dependent_or_retrocausal_route_open": True,
            "timelike_durable_pointer_route_open": True,
            "boundary_glued_representation_invariant_topology_route_open": True,
        },
        status={
            "fixed_setting_quantum_projective_instruments_cptp": cptp_certified,
            "finite_singlet_chsh_box_certified": quantum_box_certified,
            "finite_box_operational_no_signalling_certified": (
                quantum_audit.maximum_no_signalling_residual <= numerical_limit
            ),
            "setting_independent_local_factorization_excluded_for_box": (
                local_no_go_certified
            ),
            "nonsignalling_remainder_local_fraction_certified": (
                local_fraction_certified
            ),
            "fine_seed_weighted_measure_bijection_formula_certified": (
                fine_seed_bijection_certified
            ),
            "usual_topology_homeomorphism_counterexample": (
                fine_seed_bijection_certified and not homeomorphism
            ),
            "usual_topology_homeomorphism_derived": False,
            "physical_seed_law_derived": False,
            "objective_single_outcome_selection_derived": False,
            "durable_physical_pointer_derived": False,
            "relativistic_qft_microcausality_derived": False,
            "full_lightcone_no_controllable_influence_gate_complete": False,
            "spacetime_topology_metric_or_curvature_derived": False,
            "fold_stress_or_gravity_derived": False,
            "mass_dependent_probability_deformation_derived": False,
            "independent_holdout_complete": False,
            "success_gates_1_to_8_complete": False,
        },
    )


def run() -> dict[str, object]:
    return asdict(certificate())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
