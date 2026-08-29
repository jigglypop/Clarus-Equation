from fractions import Fraction
import math

import numpy as np
import pytest

from examples.physics.lattice_fierz_pauli_refinement import (
    linearized_bianchi_divergence_matrix,
    linearized_einstein_symbol,
    linearized_gauge_direction_matrix,
)
from examples.physics.two_derivative_spin2_uniqueness import (
    _rank,
    audit_two_derivative_spin2_uniqueness,
    gauge_and_bianchi_coefficient_constraints,
    gauge_and_self_adjoint_coefficient_constraints,
    gauge_coefficient_constraints,
    general_two_derivative_spin2_symbol,
)


def test_gauge_invariance_alone_leaves_two_coefficient_directions() -> None:
    constraints = gauge_coefficient_constraints()

    assert _rank(constraints) == 3
    assert 5 - _rank(constraints) == 2


def test_adding_bianchi_identity_leaves_one_unique_ray() -> None:
    constraints = gauge_and_bianchi_coefficient_constraints()
    ray = (Fraction(1), Fraction(-1), Fraction(1), Fraction(1), Fraction(-1))

    assert _rank(constraints) == 4
    assert 5 - _rank(constraints) == 1
    assert all(
        sum(entry * value for entry, value in zip(row, ray)) == 0
        for row in constraints
    )


def test_adding_action_self_adjointness_also_leaves_the_same_unique_ray() -> None:
    constraints = gauge_and_self_adjoint_coefficient_constraints()
    ray = (Fraction(1), Fraction(-1), Fraction(1), Fraction(1), Fraction(-1))

    assert _rank(constraints) == 4
    assert 5 - _rank(constraints) == 1
    assert all(
        sum(entry * value for entry, value in zip(row, ray)) == 0
        for row in constraints
    )


@pytest.mark.parametrize(
    "momentum",
    (
        (1.2, 0.3, -0.4, 0.8),
        (0.7, -0.2, 0.5, 0.1),
        (2.0, 0.4, 0.3, -0.9),
    ),
)
def test_unique_ray_matches_twice_linearized_einstein_symbol(
    momentum: tuple[float, float, float, float],
) -> None:
    symbol = general_two_derivative_spin2_symbol(
        momentum, (1.0, -1.0, 1.0, 1.0, -1.0)
    )

    assert symbol == pytest.approx(2.0 * linearized_einstein_symbol(momentum))


def test_unique_ray_obeys_gauge_and_bianchi_identities() -> None:
    momentum = (1.1, -0.4, 0.2, 0.6)
    symbol = general_two_derivative_spin2_symbol(
        momentum, (1.0, -1.0, 1.0, 1.0, -1.0)
    )

    assert np.linalg.norm(symbol @ linearized_gauge_direction_matrix(momentum)) == pytest.approx(
        0.0, abs=1.0e-12
    )
    assert np.linalg.norm(
        linearized_bianchi_divergence_matrix(momentum) @ symbol
    ) == pytest.approx(0.0, abs=1.0e-12)


def test_gauge_only_counterexample_fails_bianchi_identity() -> None:
    momentum = (1.3, 0.2, -0.5, 0.7)
    symbol = general_two_derivative_spin2_symbol(
        momentum, (1.0, -1.0, 1.0, 0.0, 0.0)
    )

    assert np.linalg.norm(symbol @ linearized_gauge_direction_matrix(momentum)) == pytest.approx(
        0.0, abs=1.0e-12
    )
    assert np.linalg.norm(
        linearized_bianchi_divergence_matrix(momentum) @ symbol
    ) > 1.0e-6


@pytest.mark.parametrize("second_action_parameter", (-2.0, 0.0, 0.5, 1.0, 2.0))
def test_full_gauge_family_is_self_adjoint_only_on_fierz_pauli_ray(
    second_action_parameter: float,
) -> None:
    momentum = (1.2, 0.3, -0.4, 0.8)
    coefficients = (
        1.0,
        -1.0,
        1.0,
        second_action_parameter,
        -second_action_parameter,
    )
    symbol = general_two_derivative_spin2_symbol(momentum, coefficients)
    eta = (-1.0, 1.0, 1.0, 1.0)
    components = tuple(
        (first, second)
        for first in range(4)
        for second in range(first, 4)
    )
    weights = np.asarray(
        [
            (1.0 if first == second else 2.0) * eta[first] * eta[second]
            for first, second in components
        ]
    )
    residual = np.linalg.norm(
        np.diag(weights) @ symbol - (np.diag(weights) @ symbol).T
    )

    if second_action_parameter == 1.0:
        assert residual == pytest.approx(0.0, abs=1.0e-12)
    else:
        assert residual > 1.0e-6


def test_negative_overall_ray_preserves_identities_but_does_not_fix_sign() -> None:
    momentum = (1.0, -0.2, 0.4, 0.7)
    negative_symbol = general_two_derivative_spin2_symbol(
        momentum, (-1.0, 1.0, -1.0, -1.0, 1.0)
    )

    assert negative_symbol == pytest.approx(-2.0 * linearized_einstein_symbol(momentum))
    assert np.linalg.norm(
        negative_symbol @ linearized_gauge_direction_matrix(momentum)
    ) == pytest.approx(0.0, abs=1.0e-12)
    assert np.linalg.norm(
        linearized_bianchi_divergence_matrix(momentum) @ negative_symbol
    ) == pytest.approx(0.0, abs=1.0e-12)


def test_coefficient_perturbation_breaks_gauge_identity() -> None:
    momentum = (1.0, 0.2, 0.3, 0.6)
    symbol = general_two_derivative_spin2_symbol(
        momentum, (1.0, -1.0, 1.0, 1.0, -0.9)
    )

    assert np.linalg.norm(symbol @ linearized_gauge_direction_matrix(momentum)) > 1.0e-6


def test_uniqueness_audit_keeps_microscopic_and_nonlinear_claims_false() -> None:
    audit = audit_two_derivative_spin2_uniqueness()

    assert audit.gauge_constraint_rank == 3
    assert audit.gauge_invariant_family_dimension == 2
    assert audit.gauge_bianchi_constraint_rank == 4
    assert audit.gauge_bianchi_family_dimension == 1
    assert audit.gauge_self_adjoint_constraint_rank == 4
    assert audit.gauge_self_adjoint_family_dimension == 1
    assert audit.unique_coefficient_ray == (1, -1, 1, 1, -1)
    assert audit.exact_unique_fierz_pauli_ray_within_ansatz
    assert audit.gauge_invariance_alone_is_insufficient
    assert (
        audit.conditional_unique_fierz_pauli_quadratic_kernel_ray_within_declared_operator_ansatz
    )
    assert audit.conditional_quadratic_action_is_fp_up_to_boundary_and_overall_scale
    assert audit.status == "UNIQUE_TWO_DERIVATIVE_MASSLESS_SPIN2_KERNEL_RAY_CLOSED"
    assert not audit.positive_overall_normalization_fixed
    assert not audit.microscopic_effective_kernel_proved_to_lie_in_ansatz
    assert not audit.higher_derivative_and_nonlocal_terms_excluded_microscopically
    assert not audit.nonlinear_einstein_completion_derived_from_ce
    assert audit.claim_ceiling.endswith("NOT_MICROSCOPIC_EH_DOMINANCE")


@pytest.mark.parametrize(
    "momentum,coefficients",
    (
        ((1.0, 2.0, 3.0), (1.0, -1.0, 1.0, 1.0, -1.0)),
        ((1.0, 0.0, 0.0, 1.0), (1.0, -1.0)),
        ((math.inf, 0.0, 0.0, 1.0), (1.0, -1.0, 1.0, 1.0, -1.0)),
    ),
)
def test_invalid_inputs_are_rejected(
    momentum: tuple[float, ...], coefficients: tuple[float, ...]
) -> None:
    with pytest.raises(ValueError):
        general_two_derivative_spin2_symbol(momentum, coefficients)


@pytest.mark.parametrize("tolerance", (0.0, -1.0, math.inf, math.nan))
def test_invalid_audit_tolerance_is_rejected(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        audit_two_derivative_spin2_uniqueness(tolerance=tolerance)
