from fractions import Fraction

import pytest

from examples.physics.linearized_spin2_acceptance import (
    _multiply,
    _rank,
    audit_linearized_spin2_acceptance,
    harmonic_constraint_matrix,
    massive_fierz_pauli_constraint_matrix,
    residual_gauge_matrix,
    transverse_traceless_basis,
)


def test_harmonic_constraint_kernel_has_dimension_six() -> None:
    constraints = harmonic_constraint_matrix()

    assert len(constraints) == 4
    assert len(constraints[0]) == 10
    assert _rank(constraints) == 4
    assert 10 - _rank(constraints) == 6


def test_four_residual_gauge_directions_lie_in_harmonic_kernel() -> None:
    constraints = harmonic_constraint_matrix()
    gauge = residual_gauge_matrix()

    assert _rank(gauge) == 4
    assert _multiply(constraints, gauge) == (
        (Fraction(0),) * 4,
        (Fraction(0),) * 4,
        (Fraction(0),) * 4,
        (Fraction(0),) * 4,
    )


def test_declared_null_direction_has_zero_minkowski_norm() -> None:
    momentum_up = (1, 0, 0, 1)
    momentum_down = (-1, 0, 0, 1)

    assert sum(up * down for up, down in zip(momentum_up, momentum_down)) == 0


def test_plus_and_cross_are_independent_tt_quotient_representatives() -> None:
    constraints = harmonic_constraint_matrix()
    gauge = residual_gauge_matrix()
    tt_basis = transverse_traceless_basis()
    gauge_plus_tt = tuple(
        tuple(gauge[row][column] for column in range(4)) + tt_basis[row]
        for row in range(10)
    )

    assert _multiply(constraints, tt_basis) == (
        (Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0)),
    )
    assert _rank(gauge_plus_tt) == 6
    assert _rank(gauge_plus_tt) == _rank(gauge) + _rank(tt_basis)


def test_massive_fierz_pauli_negative_control_has_five_polarizations() -> None:
    constraints = massive_fierz_pauli_constraint_matrix()

    assert _rank(constraints) == 5
    assert 10 - _rank(constraints) == 5


def test_acceptance_audit_closes_only_linearized_supplied_action() -> None:
    audit = audit_linearized_spin2_acceptance()

    assert audit.symmetric_tensor_component_count == 10
    assert audit.null_momentum_norm_squared == 0
    assert audit.harmonic_constraint_rank == 4
    assert audit.harmonic_solution_dimension == 6
    assert audit.residual_gauge_rank == 4
    assert audit.gauge_image_within_harmonic_kernel
    assert audit.harmonic_kernel_spanned_by_gauge_plus_tt
    assert audit.physical_quotient_dimension == 2
    assert audit.transverse_traceless_representative_count == 2
    assert audit.massive_fierz_pauli_polarization_count == 5
    assert audit.arithmetic_spin2_plus_one_scalar_count == 3
    assert audit.exact_two_polarization_quotient_closed
    assert audit.status == "EXACT_LINEARIZED_FIERZ_PAULI_TWO_POLARIZATION_GATE_CLOSED"
    assert not audit.nonlinear_diffeomorphism_symmetry_derived
    assert not audit.refinement_limit_kernel_derived
    assert not audit.refinement_uniform_ward_identities_proved
    assert not audit.extra_poles_excluded_for_a_microscopic_model
    assert not audit.einstein_hilbert_dominance_from_ce_proved
    assert audit.claim_ceiling.endswith("NOT_MICROSCOPIC_EH_DERIVATION")


def test_exact_rank_rejects_noncomposable_matrices() -> None:
    with pytest.raises(ValueError, match="do not compose"):
        _multiply(((Fraction(1), Fraction(2)),), ((Fraction(1),),))


def test_exact_rank_handles_dependent_rows() -> None:
    matrix = (
        (Fraction(1), Fraction(2), Fraction(3)),
        (Fraction(2), Fraction(4), Fraction(6)),
        (Fraction(0), Fraction(1), Fraction(1)),
    )

    assert _rank(matrix) == 2
