"""Finite global-ledger audit for the PR-oriented singlet CHSH box.

The set-level bookkeeping question and the probability question are kept
separate here.

For every fixed measurement context ``(x, y)``, a deterministic global atom
``(A0, A1, B0, B1)`` can be rearranged bijectively into

``((A_x, B_y), (A_{1-x}, B_{1-y}))``.

The first pair is the visible readout and the second pair is the hidden
ledger.  Dropping the hidden pair is a four-to-one projection; retaining it
is a bijection (and a homeomorphism when both finite spaces carry the
discrete topology).

That bijection does not supply a setting-independent *positive measure* on
the sixteen atoms.  For the isotropic box

``P_eta(a,b|x,y) = (1 + a*b*eta*c_xy)/4``

with ``c=(-1,-1,-1,+1)``, the symmetric signed extension

``q_eta(lambda) = (1 + eta*F_lambda)/16``

reconstructs every context, where ``F_lambda`` is the oriented deterministic
CHSH score and therefore equals ``+2`` or ``-2``.  At the quantum value
``eta=1/sqrt(2)``, any such extension has l1 norm at least ``sqrt(2)`` and
negative mass at least ``(sqrt(2)-1)/2``.  The symmetric extension saturates
that bound.  Replacing this representative by its normalized absolute value
produces a different positive local box with half the target correlations.

Signed weights in this module are linear-representation coefficients.  They
are not observed probabilities, negative frequencies, energy, stress, a
metric volume, or a physical hidden-path law.  The result is confined to the
declared 2x2 binary scenario and does not constitute a general metric or
gravity no-go.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from fractions import Fraction
import json
import math
from typing import Sequence

import numpy as np

from examples.physics.chsh_local_seed_obstruction import (
    CHSH_PATTERN,
    OUTCOMES,
    SETTINGS,
    audit_probability_box,
    box_correlations,
    chsh_scores,
    deterministic_facet_score,
    deterministic_local_strategies,
    isotropic_chsh_box,
    quantum_projective_instrument_audit,
)


DEFAULT_TOLERANCE = 1.0e-12
QUANTUM_ETA = 1.0 / math.sqrt(2.0)


def _positive_tolerance(value: float) -> float:
    tolerance = float(value)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    return tolerance


def _visibility(value: float) -> float:
    eta = float(value)
    if not math.isfinite(eta) or not 0.0 <= eta <= 1.0:
        raise ValueError("eta must be finite and lie in [0, 1]")
    return eta


def _context_value(value: int, *, name: str) -> int:
    if isinstance(value, bool) or value not in SETTINGS:
        raise ValueError(f"{name} must be 0 or 1")
    return int(value)


def _outcome_pair(values: Sequence[int], *, name: str) -> tuple[int, int]:
    pair = tuple(values)
    if len(pair) != 2 or any(value not in OUTCOMES for value in pair):
        raise ValueError(f"{name} must contain two outcomes in {{-1, +1}}")
    return pair  # type: ignore[return-value]


def _global_assignment(values: Sequence[int]) -> tuple[int, int, int, int]:
    assignment = tuple(values)
    if len(assignment) != 4 or any(value not in OUTCOMES for value in assignment):
        raise ValueError("assignment must contain four outcomes in {-1, +1}")
    return assignment  # type: ignore[return-value]


def global_assignments() -> tuple[tuple[int, int, int, int], ...]:
    """Return the sixteen counterfactual atoms ``(A0,A1,B0,B1)``."""

    return deterministic_local_strategies()


@dataclass(frozen=True)
class ContextLedgerCoordinate:
    """One full finite ledger coordinate for a fixed context."""

    context: tuple[int, int]
    visible_outcomes: tuple[int, int]
    hidden_outcomes: tuple[int, int]


def lift_context_ledger(
    assignment: Sequence[int], x: int, y: int
) -> ContextLedgerCoordinate:
    """Rearrange a global atom into visible and hidden outcome pairs."""

    a0, a1, b0, b1 = _global_assignment(assignment)
    x_value = _context_value(x, name="x")
    y_value = _context_value(y, name="y")
    alice = (a0, a1)
    bob = (b0, b1)
    return ContextLedgerCoordinate(
        context=(x_value, y_value),
        visible_outcomes=(alice[x_value], bob[y_value]),
        hidden_outcomes=(alice[1 - x_value], bob[1 - y_value]),
    )


def invert_context_ledger(
    coordinate: ContextLedgerCoordinate,
) -> tuple[int, int, int, int]:
    """Invert a declared full ledger coordinate exactly."""

    if not isinstance(coordinate, ContextLedgerCoordinate):
        raise TypeError("coordinate must be a ContextLedgerCoordinate")
    x = _context_value(coordinate.context[0], name="x")
    y = _context_value(coordinate.context[1], name="y")
    visible_a, visible_b = _outcome_pair(
        coordinate.visible_outcomes, name="visible_outcomes"
    )
    hidden_a, hidden_b = _outcome_pair(
        coordinate.hidden_outcomes, name="hidden_outcomes"
    )
    alice = [0, 0]
    bob = [0, 0]
    alice[x] = visible_a
    alice[1 - x] = hidden_a
    bob[y] = visible_b
    bob[1 - y] = hidden_b
    return alice[0], alice[1], bob[0], bob[1]


def deterministic_oriented_scores() -> tuple[int, ...]:
    """Return ``F_lambda`` for all atoms; every value is ``-2`` or ``+2``."""

    return tuple(deterministic_facet_score(atom) for atom in global_assignments())


def symmetric_signed_global_extension(eta: float) -> tuple[float, ...]:
    """Return ``q_eta(lambda)=(1+eta*F_lambda)/16``."""

    visibility = _visibility(eta)
    return tuple(
        (1.0 + visibility * score) / 16.0
        for score in deterministic_oriented_scores()
    )


def _finite_weights(weights: Sequence[float]) -> tuple[float, ...]:
    declared = tuple(float(value) for value in weights)
    if len(declared) != 16 or not all(math.isfinite(value) for value in declared):
        raise ValueError("weights must contain sixteen finite values")
    return declared


def marginalize_global_weights(weights: Sequence[float]) -> np.ndarray:
    """Project signed or positive atom weights into all four contexts."""

    declared = _finite_weights(weights)
    probabilities = np.zeros((2, 2, 2, 2), dtype=np.float64)
    for weight, assignment in zip(declared, global_assignments()):
        a0, a1, b0, b1 = assignment
        alice = (a0, a1)
        bob = (b0, b1)
        for x in SETTINGS:
            for y in SETTINGS:
                a_index = OUTCOMES.index(alice[x])
                b_index = OUTCOMES.index(bob[y])
                probabilities[x, y, a_index, b_index] += weight
    return probabilities


def total_variation_norm(weights: Sequence[float]) -> float:
    """Return the dimensionless signed ``l1`` norm."""

    return math.fsum(abs(value) for value in _finite_weights(weights))


def negative_mass(weights: Sequence[float]) -> float:
    """Return ``sum(max(-q_lambda,0))`` for a signed normalized extension."""

    return math.fsum(max(-value, 0.0) for value in _finite_weights(weights))


def normalized_absolute_weights(weights: Sequence[float]) -> tuple[float, ...]:
    """Return ``|q|/||q||_1``; this is a new positive model."""

    declared = _finite_weights(weights)
    norm = math.fsum(abs(value) for value in declared)
    if norm <= 0.0:
        raise ValueError("absolute weights must have positive total mass")
    return tuple(abs(value) / norm for value in declared)


def context_cells() -> tuple[tuple[int, int, int, int], ...]:
    """Return row labels ``(x,y,a,b)`` for the marginal incidence matrix."""

    return tuple(
        (x, y, a, b)
        for x in SETTINGS
        for y in SETTINGS
        for a in OUTCOMES
        for b in OUTCOMES
    )


def marginal_incidence_matrix() -> np.ndarray:
    """Return the exact 0/1 map from sixteen atoms to sixteen context cells."""

    matrix = np.zeros((16, 16), dtype=np.int64)
    for row, (x, y, a, b) in enumerate(context_cells()):
        for column, assignment in enumerate(global_assignments()):
            a0, a1, b0, b1 = assignment
            alice = (a0, a1)
            bob = (b0, b1)
            matrix[row, column] = int(alice[x] == a and bob[y] == b)
    return matrix


def exact_rational_rank(matrix: np.ndarray) -> int:
    """Compute a small matrix rank by exact rational row reduction."""

    values = np.asarray(matrix)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("matrix must be a nonempty two-dimensional array")
    if not np.isfinite(values.astype(np.float64)).all():
        raise ValueError("matrix entries must be finite")
    rows = [
        [Fraction(str(values[row, column])) for column in range(values.shape[1])]
        for row in range(values.shape[0])
    ]
    rank = 0
    for column in range(values.shape[1]):
        pivot = next(
            (row for row in range(rank, len(rows)) if rows[row][column] != 0),
            None,
        )
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        pivot_value = rows[rank][column]
        rows[rank] = [value / pivot_value for value in rows[rank]]
        for row in range(len(rows)):
            if row == rank or rows[row][column] == 0:
                continue
            factor = rows[row][column]
            rows[row] = [
                value - factor * pivot_entry
                for value, pivot_entry in zip(rows[row], rows[rank])
            ]
        rank += 1
        if rank == len(rows):
            break
    return rank


def walsh_kernel_vectors() -> dict[str, tuple[int, ...]]:
    """Return seven unobserved Walsh directions of the marginal map."""

    vectors: dict[str, list[int]] = {
        "A0A1": [],
        "B0B1": [],
        "A0A1B0": [],
        "A0A1B1": [],
        "A0B0B1": [],
        "A1B0B1": [],
        "A0A1B0B1": [],
    }
    for a0, a1, b0, b1 in global_assignments():
        vectors["A0A1"].append(a0 * a1)
        vectors["B0B1"].append(b0 * b1)
        vectors["A0A1B0"].append(a0 * a1 * b0)
        vectors["A0A1B1"].append(a0 * a1 * b1)
        vectors["A0B0B1"].append(a0 * b0 * b1)
        vectors["A1B0B1"].append(a1 * b0 * b1)
        vectors["A0A1B0B1"].append(a0 * a1 * b0 * b1)
    return {name: tuple(values) for name, values in vectors.items()}


def quantum_kernel_perturbed_extension(delta: float) -> tuple[float, ...]:
    """Return ``q_quantum + delta*A0*A1/16``.

    All context marginals are unchanged.  The extension remains an l1
    minimizer on the closed interval ``|delta| <= sqrt(2)-1``; it is not a
    positive probability distribution on that interval except at no point,
    because the quantum CHSH target lies outside the local polytope.
    """

    parameter = float(delta)
    if not math.isfinite(parameter):
        raise ValueError("delta must be finite")
    base = symmetric_signed_global_extension(QUANTUM_ETA)
    direction = walsh_kernel_vectors()["A0A1"]
    return tuple(
        weight + parameter * value / 16.0
        for weight, value in zip(base, direction)
    )


def swap_opposite_score_weights(weights: Sequence[float]) -> tuple[float, ...]:
    """Apply one atom bijection that is not a marginal-incidence automorphism."""

    permuted = list(_finite_weights(weights))
    scores = deterministic_oriented_scores()
    negative_index = scores.index(-2)
    positive_index = scores.index(2)
    permuted[negative_index], permuted[positive_index] = (
        permuted[positive_index],
        permuted[negative_index],
    )
    return tuple(permuted)


@dataclass(frozen=True)
class ContextualGlobalSectionCertificate:
    eta: float
    atom_count: int
    context_count: int
    full_ledger_round_trip_failures: int
    minimum_full_ledger_unique_image_count: int
    minimum_visible_projection_fibre_size: int
    maximum_visible_projection_fibre_size: int
    deterministic_oriented_scores: tuple[int, ...]
    incidence_rank: int
    incidence_nullity: int
    maximum_walsh_kernel_residual: int
    target_correlations: tuple[float, float, float, float]
    target_oriented_score: float
    target_absolute_chsh_score: float
    target_minimum_probability: float
    target_normalization_residual: float
    target_no_signalling_residual: float
    parent_instrument_probability_residual: float
    signed_weights: tuple[float, ...]
    signed_weight_sum: float
    signed_normalization_residual: float
    signed_minimum_weight: float
    signed_maximum_weight: float
    signed_negative_atom_count: int
    signed_positive_atom_count: int
    signed_context_marginal_residual: float
    signed_l1_norm: float
    signed_l1_lower_bound: float
    signed_l1_saturation_residual: float
    signed_negative_mass: float
    signed_negative_mass_lower_bound: float
    positive_global_chsh_gap: float
    delta_minimizer_half_width: float
    delta_witness: float
    delta_context_marginal_residual: float
    delta_l1_residual: float
    endpoint_maximum_context_marginal_residual: float
    endpoint_maximum_l1_residual: float
    endpoint_minimum_absolute_weight: float
    minimum_beyond_interval_l1_excess: float
    raw_absolute_mass: float
    normalized_absolute_mass: float
    normalized_absolute_correlations: tuple[float, float, float, float]
    normalized_absolute_oriented_score: float
    normalized_absolute_target_residual: float
    normalized_absolute_no_signalling_residual: float
    permutation_sum_residual: float
    permutation_l1_residual: float
    permutation_negative_mass_residual: float
    permutation_target_marginal_residual: float
    dimensions: dict[str, bool]
    accounting: dict[str, bool]
    boundaries: dict[str, bool]
    alternatives: dict[str, bool]
    status: dict[str, bool]


def certificate(
    *, tolerance: float = DEFAULT_TOLERANCE
) -> ContextualGlobalSectionCertificate:
    """Build the E29 finite global-section and signed-measure certificate."""

    tol = _positive_tolerance(tolerance)
    assignments = global_assignments()
    contexts = tuple((x, y) for x in SETTINGS for y in SETTINGS)

    round_trip_failures = 0
    unique_image_counts: list[int] = []
    visible_fibre_sizes: list[int] = []
    for x, y in contexts:
        coordinates = tuple(lift_context_ledger(atom, x, y) for atom in assignments)
        round_trip_failures += sum(
            invert_context_ledger(coordinate) != atom
            for atom, coordinate in zip(assignments, coordinates)
        )
        unique_image_counts.append(len(set(coordinates)))
        for visible_a in OUTCOMES:
            for visible_b in OUTCOMES:
                visible_fibre_sizes.append(
                    sum(
                        coordinate.visible_outcomes == (visible_a, visible_b)
                        for coordinate in coordinates
                    )
                )

    incidence = marginal_incidence_matrix()
    incidence_rank = exact_rational_rank(incidence)
    kernel_vectors = walsh_kernel_vectors()
    maximum_kernel_residual = max(
        int(np.max(np.abs(incidence @ np.asarray(vector, dtype=np.int64))))
        for vector in kernel_vectors.values()
    )

    target = isotropic_chsh_box(QUANTUM_ETA)
    target_audit = audit_probability_box(target)
    target_correlations = box_correlations(target)
    target_oriented_score, target_absolute_score = chsh_scores(target)
    instrument = quantum_projective_instrument_audit()
    parent_instrument_residual = float(
        np.max(np.abs(instrument.probabilities - target))
    )

    signed = symmetric_signed_global_extension(QUANTUM_ETA)
    signed_sum = math.fsum(signed)
    signed_box = marginalize_global_weights(signed)
    signed_marginal_residual = float(np.max(np.abs(signed_box - target)))
    signed_l1 = total_variation_norm(signed)
    signed_l1_lower_bound = max(1.0, target_oriented_score / 2.0)
    signed_negativity = negative_mass(signed)
    signed_negativity_lower_bound = 0.5 * (signed_l1_lower_bound - 1.0)

    delta_half_width = math.sqrt(2.0) - 1.0
    delta_witness = 0.5 * delta_half_width
    delta_extension = quantum_kernel_perturbed_extension(delta_witness)
    delta_box = marginalize_global_weights(delta_extension)
    endpoints = tuple(
        quantum_kernel_perturbed_extension(sign * delta_half_width)
        for sign in (-1.0, 1.0)
    )
    beyond_extensions = tuple(
        quantum_kernel_perturbed_extension(sign * 1.1 * delta_half_width)
        for sign in (-1.0, 1.0)
    )

    raw_absolute_mass = signed_l1
    absolute_weights = normalized_absolute_weights(signed)
    absolute_mass = math.fsum(absolute_weights)
    absolute_box = marginalize_global_weights(absolute_weights)
    absolute_correlations = box_correlations(absolute_box)
    absolute_oriented_score, _ = chsh_scores(absolute_box)
    absolute_audit = audit_probability_box(absolute_box)

    permuted = swap_opposite_score_weights(signed)
    permuted_box = marginalize_global_weights(permuted)

    numerical_limit = 50.0 * tol
    full_ledger_bijection = (
        round_trip_failures == 0
        and min(unique_image_counts) == len(assignments)
    )
    visible_projection_many_to_one = (
        min(visible_fibre_sizes) == 4 and max(visible_fibre_sizes) == 4
    )
    signed_extension_certified = (
        abs(signed_sum - 1.0) <= numerical_limit
        and signed_marginal_residual <= numerical_limit
        and sum(value < 0.0 for value in signed) == 8
        and sum(value > 0.0 for value in signed) == 8
    )
    minimum_norm_certified = (
        abs(signed_l1 - signed_l1_lower_bound) <= numerical_limit
        and abs(signed_negativity - signed_negativity_lower_bound)
        <= numerical_limit
    )
    delta_nonunique_minimizer_certified = (
        float(np.max(np.abs(delta_box - target))) <= numerical_limit
        and abs(total_variation_norm(delta_extension) - signed_l1)
        <= numerical_limit
        and max(
            float(np.max(np.abs(marginalize_global_weights(item) - target)))
            for item in endpoints
        )
        <= numerical_limit
        and max(abs(total_variation_norm(item) - signed_l1) for item in endpoints)
        <= numerical_limit
        and min(total_variation_norm(item) for item in beyond_extensions)
        > signed_l1 + numerical_limit
    )
    absolute_replacement_changes_target = (
        float(np.max(np.abs(absolute_box - target))) > numerical_limit
        and np.allclose(
            np.asarray(absolute_correlations),
            0.5 * np.asarray(target_correlations),
            atol=numerical_limit,
            rtol=0.0,
        )
        and abs(absolute_oriented_score - math.sqrt(2.0)) <= numerical_limit
    )
    permutation_invariants_preserved = (
        abs(math.fsum(permuted) - signed_sum) <= numerical_limit
        and abs(total_variation_norm(permuted) - signed_l1) <= numerical_limit
        and abs(negative_mass(permuted) - signed_negativity) <= numerical_limit
    )
    permutation_changes_marginals = (
        float(np.max(np.abs(permuted_box - target))) > numerical_limit
    )

    dimensions = {
        "eta_is_dimensionless": True,
        "born_probabilities_are_dimensionless": True,
        "signed_global_weights_are_dimensionless": True,
        "facet_scores_and_l1_norm_are_dimensionless": True,
        "negativity_is_dimensionless": True,
        "no_energy_length_time_or_mass_scale_introduced": True,
    }
    accounting = {
        "each_context_born_box_normalized_once": (
            target_audit.maximum_normalization_residual <= numerical_limit
        ),
        "full_ledger_relabels_each_atom_once": full_ledger_bijection,
        "visible_projection_does_not_add_hidden_weights": True,
        "signed_extension_is_an_alternative_linear_representation": True,
        "signed_and_absolute_models_are_not_added_together": True,
        "absolute_replacement_is_explicitly_renormalized": (
            abs(absolute_mass - 1.0) <= numerical_limit
        ),
        "signed_weight_not_added_as_energy_or_stress": True,
        "signed_or_hidden_atom_carries_energy": False,
    }
    boundaries = {
        "full_visible_plus_hidden_ledger_is_bijective": full_ledger_bijection,
        "finite_discrete_full_ledger_bijection_is_homeomorphism": (
            full_ledger_bijection
        ),
        "visible_readout_alone_is_many_to_one": visible_projection_many_to_one,
        "set_bijection_does_not_imply_measure_preservation": True,
        "positive_global_measure_failure_is_not_bijection_failure": True,
        "global_state_destruction_not_inferred": True,
        "signed_weight_is_not_observed_probability_or_frequency": True,
        "signed_weight_is_not_negative_energy_or_stress": True,
        "finite_discrete_zero_dimensionality_is_not_spacetime_dimension": True,
        "absolute_value_result_uses_symmetric_delta_zero_representative": True,
        "absolute_value_result_is_not_general_metric_measure_or_gravity_no_go": True,
        "atom_permutation_preserves_marginals_only_if_incidence_is_respected": True,
        "fine_and_global_section_results_are_limited_to_2x2_binary_scenario": True,
        "signed_extension_is_not_a_quantum_channel_or_selection_dynamics": True,
        "operational_no_signalling_is_not_qft_microcausality": True,
    }
    alternatives = {
        "context_dependent_per_setting_instrument_or_ledger": True,
        "measurement_dependent_or_retrocausal_route": True,
        "ontically_nonlocal_but_operationally_no_signalling_route": True,
        "future_lightcone_pointer_domino_only": True,
        "independent_representation_invariant_geometry_and_measure_law": True,
    }
    status = {
        "full_context_ledger_set_bijection_certified": full_ledger_bijection,
        "visible_projection_many_to_one_certified": visible_projection_many_to_one,
        "incidence_rank_nine_nullity_seven_certified": (
            incidence_rank == 9
            and 16 - incidence_rank == 7
            and maximum_kernel_residual == 0
        ),
        "all_context_signed_extension_certified": signed_extension_certified,
        "positive_setting_independent_global_probability_excluded_for_target": (
            target_oriented_score > 2.0 + numerical_limit
            and set(deterministic_oriented_scores()) == {-2, 2}
        ),
        "minimum_signed_l1_and_negativity_certified": minimum_norm_certified,
        "minimum_signed_extension_is_nonunique": (
            delta_nonunique_minimizer_certified
        ),
        "symmetric_absolute_replacement_changes_born_marginals": (
            absolute_replacement_changes_target
        ),
        "arbitrary_atom_bijection_need_not_preserve_physical_marginals": (
            permutation_invariants_preserved and permutation_changes_marginals
        ),
        "fixed_context_parent_instruments_remain_cptp": (
            parent_instrument_residual <= numerical_limit
            and instrument.maximum_completeness_residual <= numerical_limit
            and instrument.minimum_choi_eigenvalue >= -numerical_limit
        ),
        "finite_target_operational_no_signalling_certified": (
            target_audit.maximum_no_signalling_residual <= numerical_limit
        ),
        "physical_hidden_path_or_seed_law_derived": False,
        "objective_single_outcome_selection_derived": False,
        "relativistic_qft_microcausality_derived": False,
        "full_lightcone_no_controllable_influence_gate_complete": False,
        "spacetime_metric_volume_or_gravity_derived": False,
        "mass_dependent_probability_deformation_derived": False,
        "independent_holdout_complete": False,
        "success_gates_1_to_8_complete": False,
    }

    return ContextualGlobalSectionCertificate(
        eta=QUANTUM_ETA,
        atom_count=len(assignments),
        context_count=len(contexts),
        full_ledger_round_trip_failures=round_trip_failures,
        minimum_full_ledger_unique_image_count=min(unique_image_counts),
        minimum_visible_projection_fibre_size=min(visible_fibre_sizes),
        maximum_visible_projection_fibre_size=max(visible_fibre_sizes),
        deterministic_oriented_scores=deterministic_oriented_scores(),
        incidence_rank=incidence_rank,
        incidence_nullity=16 - incidence_rank,
        maximum_walsh_kernel_residual=maximum_kernel_residual,
        target_correlations=target_correlations,
        target_oriented_score=target_oriented_score,
        target_absolute_chsh_score=target_absolute_score,
        target_minimum_probability=target_audit.minimum_probability,
        target_normalization_residual=target_audit.maximum_normalization_residual,
        target_no_signalling_residual=target_audit.maximum_no_signalling_residual,
        parent_instrument_probability_residual=parent_instrument_residual,
        signed_weights=signed,
        signed_weight_sum=signed_sum,
        signed_normalization_residual=abs(signed_sum - 1.0),
        signed_minimum_weight=min(signed),
        signed_maximum_weight=max(signed),
        signed_negative_atom_count=sum(value < 0.0 for value in signed),
        signed_positive_atom_count=sum(value > 0.0 for value in signed),
        signed_context_marginal_residual=signed_marginal_residual,
        signed_l1_norm=signed_l1,
        signed_l1_lower_bound=signed_l1_lower_bound,
        signed_l1_saturation_residual=abs(signed_l1 - signed_l1_lower_bound),
        signed_negative_mass=signed_negativity,
        signed_negative_mass_lower_bound=signed_negativity_lower_bound,
        positive_global_chsh_gap=target_oriented_score - 2.0,
        delta_minimizer_half_width=delta_half_width,
        delta_witness=delta_witness,
        delta_context_marginal_residual=float(np.max(np.abs(delta_box - target))),
        delta_l1_residual=abs(total_variation_norm(delta_extension) - signed_l1),
        endpoint_maximum_context_marginal_residual=max(
            float(np.max(np.abs(marginalize_global_weights(item) - target)))
            for item in endpoints
        ),
        endpoint_maximum_l1_residual=max(
            abs(total_variation_norm(item) - signed_l1) for item in endpoints
        ),
        endpoint_minimum_absolute_weight=min(
            abs(value) for item in endpoints for value in item
        ),
        minimum_beyond_interval_l1_excess=(
            min(total_variation_norm(item) for item in beyond_extensions) - signed_l1
        ),
        raw_absolute_mass=raw_absolute_mass,
        normalized_absolute_mass=absolute_mass,
        normalized_absolute_correlations=absolute_correlations,
        normalized_absolute_oriented_score=absolute_oriented_score,
        normalized_absolute_target_residual=float(
            np.max(np.abs(absolute_box - target))
        ),
        normalized_absolute_no_signalling_residual=(
            absolute_audit.maximum_no_signalling_residual
        ),
        permutation_sum_residual=abs(math.fsum(permuted) - signed_sum),
        permutation_l1_residual=abs(total_variation_norm(permuted) - signed_l1),
        permutation_negative_mass_residual=abs(
            negative_mass(permuted) - signed_negativity
        ),
        permutation_target_marginal_residual=float(
            np.max(np.abs(permuted_box - target))
        ),
        dimensions=dimensions,
        accounting=accounting,
        boundaries=boundaries,
        alternatives=alternatives,
        status=status,
    )


def run() -> dict[str, object]:
    """Return a JSON-serializable E29 certificate."""

    return asdict(certificate())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run(), indent=args.indent, sort_keys=True))


if __name__ == "__main__":
    main()
