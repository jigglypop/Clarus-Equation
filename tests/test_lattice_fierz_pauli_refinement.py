import math

import numpy as np
import pytest

from examples.physics.lattice_fierz_pauli_refinement import (
    audit_lattice_fierz_pauli_refinement,
    central_difference_dimensionless_momentum,
    linearized_bianchi_divergence_matrix,
    linearized_einstein_symbol,
    linearized_gauge_direction_matrix,
)


def test_continuum_fierz_pauli_symbol_annihilates_gauge_directions() -> None:
    momentum = (1.2, 0.3, -0.4, 0.8)
    symbol = linearized_einstein_symbol(momentum)
    gauge = linearized_gauge_direction_matrix(momentum)

    assert np.linalg.norm(symbol @ gauge) == pytest.approx(0.0, abs=1.0e-12)


def test_continuum_symbol_obeys_linearized_bianchi_identity() -> None:
    momentum = (0.9, -0.2, 0.6, 0.1)
    symbol = linearized_einstein_symbol(momentum)
    divergence = linearized_bianchi_divergence_matrix(momentum)

    assert np.linalg.norm(divergence @ symbol) == pytest.approx(0.0, abs=1.0e-12)


def test_central_difference_symbol_obeys_cubic_error_bound() -> None:
    momentum = np.asarray((1.1, -0.7, 0.3, -0.2))
    spacing = 0.2
    lattice = central_difference_dimensionless_momentum(
        momentum, lattice_spacing_over_reference_length=spacing
    )

    assert np.all(
        np.abs(lattice - momentum)
        <= spacing**2 * np.abs(momentum) ** 3 / 6.0 + 1.0e-15
    )


def test_halving_spacing_has_second_order_free_symbol_convergence() -> None:
    momentum = (0.9, 0.2, -0.3, 0.5)
    continuum = linearized_einstein_symbol(momentum)
    errors = []
    for spacing in (0.2, 0.1, 0.05):
        lattice_momentum = central_difference_dimensionless_momentum(
            momentum, lattice_spacing_over_reference_length=spacing
        )
        errors.append(
            np.linalg.norm(linearized_einstein_symbol(lattice_momentum) - continuum)
        )

    assert errors[1] < errors[0] / 3.9
    assert errors[2] < errors[1] / 3.9


def test_null_ray_remains_null_and_has_two_polarizations_at_finite_spacing() -> None:
    audit = audit_lattice_fierz_pauli_refinement(
        lattice_spacing_over_reference_length=0.2,
        compact_dimensionless_momentum_bound=2.0,
        null_ray_frequency=0.8,
    )

    assert audit.null_ray_lattice_norm_squared == pytest.approx(0.0, abs=1.0e-12)
    assert audit.null_ray_harmonic_constraint_rank == 4
    assert audit.null_ray_residual_gauge_rank == 4
    assert audit.null_ray_physical_quotient_dimension == 2
    assert audit.low_momentum_null_ray_two_polarization_gate_preserved


def test_refinement_audit_closes_only_declared_free_compact_family() -> None:
    audit = audit_lattice_fierz_pauli_refinement(
        lattice_spacing_over_reference_length=0.1,
        compact_dimensionless_momentum_bound=2.0,
    )

    assert audit.all_momentum_arguments_dimensionless
    assert audit.central_difference_error_bound_satisfied
    assert audit.sample_fierz_pauli_error_within_uniform_analytic_bound
    assert (
        audit.fierz_pauli_symbol_frobenius_error
        <= audit.compact_uniform_fierz_pauli_symbol_error_bound
    )
    assert audit.algebraic_gauge_and_bianchi_identities_preserved
    assert audit.quadratic_action_symbol_self_adjoint
    assert audit.weighted_action_self_adjoint_residual == pytest.approx(
        0.0, abs=1.0e-12
    )
    assert audit.compact_uniform_free_symbol_limit_closed
    assert audit.low_momentum_component_doubler_window
    assert audit.status == "GAUGE_PRESERVING_FREE_FIERZ_PAULI_COMPACT_REFINEMENT_CLOSED"
    assert not audit.global_lattice_doublers_excluded
    assert not audit.geometric_or_spin_foam_refinement_derived
    assert not audit.interacting_renormalized_limit_proved
    assert not audit.nonlinear_constraint_algebra_proved
    assert not audit.einstein_hilbert_dominance_from_ce_proved
    assert audit.claim_ceiling.endswith("NOT_SPINFOAM_EH_LIMIT")


@pytest.mark.parametrize(
    "spacing,bound,frequency,tolerance",
    (
        (0.0, 1.0, 0.5, 1.0e-10),
        (0.1, 0.0, 0.5, 1.0e-10),
        (0.1, 1.0, 0.0, 1.0e-10),
        (0.1, 1.0, 0.5, 0.0),
        (math.inf, 1.0, 0.5, 1.0e-10),
    ),
)
def test_invalid_positive_inputs_are_rejected(
    spacing: float, bound: float, frequency: float, tolerance: float
) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        audit_lattice_fierz_pauli_refinement(
            lattice_spacing_over_reference_length=spacing,
            compact_dimensionless_momentum_bound=bound,
            null_ray_frequency=frequency,
            tolerance=tolerance,
        )


def test_momentum_outside_compact_box_is_rejected() -> None:
    with pytest.raises(ValueError, match="inside"):
        audit_lattice_fierz_pauli_refinement(
            lattice_spacing_over_reference_length=0.1,
            compact_dimensionless_momentum_bound=1.0,
            generic_dimensionless_momentum_up=(1.1, 0.0, 0.0, 0.0),
        )


def test_alias_free_component_window_is_required() -> None:
    with pytest.raises(ValueError, match="pi/2"):
        audit_lattice_fierz_pauli_refinement(
            lattice_spacing_over_reference_length=1.0,
            compact_dimensionless_momentum_bound=2.0,
            generic_dimensionless_momentum_up=(0.5, 0.0, 0.0, 0.0),
        )


@pytest.mark.parametrize(
    "spacing,bound",
    ((0.2, 1.0), (0.1, 2.0), (0.05, 3.0)),
)
def test_explicit_uniform_polynomial_bound_covers_each_compact_sample(
    spacing: float, bound: float
) -> None:
    audit = audit_lattice_fierz_pauli_refinement(
        lattice_spacing_over_reference_length=spacing,
        compact_dimensionless_momentum_bound=bound,
        generic_dimensionless_momentum_up=(
            0.8 * bound,
            -0.6 * bound,
            0.4 * bound,
            -0.2 * bound,
        ),
        null_ray_frequency=0.5 * bound,
    )

    assert (
        audit.fierz_pauli_symbol_frobenius_error
        <= audit.compact_uniform_fierz_pauli_symbol_error_bound + 1.0e-12
    )


def test_uniform_bound_decays_quadratically_at_fixed_compact_box() -> None:
    coarse = audit_lattice_fierz_pauli_refinement(
        lattice_spacing_over_reference_length=0.2,
        compact_dimensionless_momentum_bound=1.0,
        generic_dimensionless_momentum_up=(0.8, 0.2, -0.4, 0.7),
        null_ray_frequency=0.5,
    )
    fine = audit_lattice_fierz_pauli_refinement(
        lattice_spacing_over_reference_length=0.1,
        compact_dimensionless_momentum_bound=1.0,
        generic_dimensionless_momentum_up=(0.8, 0.2, -0.4, 0.7),
        null_ray_frequency=0.5,
    )

    assert fine.compact_uniform_fierz_pauli_symbol_error_bound == pytest.approx(
        coarse.compact_uniform_fierz_pauli_symbol_error_bound / 4.0
    )


def test_null_frequency_must_be_inside_the_same_compact_window() -> None:
    with pytest.raises(ValueError, match="null-ray frequency"):
        audit_lattice_fierz_pauli_refinement(
            lattice_spacing_over_reference_length=0.1,
            compact_dimensionless_momentum_bound=1.0,
            generic_dimensionless_momentum_up=(0.8, 0.2, -0.4, 0.7),
            null_ray_frequency=1.1,
        )


def test_central_difference_has_a_global_doubler_zero_outside_low_window() -> None:
    lattice = central_difference_dimensionless_momentum(
        (math.pi, 0.0, 0.0, 0.0),
        lattice_spacing_over_reference_length=1.0,
    )

    assert lattice == pytest.approx((0.0, 0.0, 0.0, 0.0), abs=1.0e-12)


@pytest.mark.parametrize("spacing", (0.3, 0.17, 0.08))
def test_lattice_symbol_preserves_gauge_and_bianchi_at_multiple_spacings(
    spacing: float,
) -> None:
    momentum = central_difference_dimensionless_momentum(
        (1.0, 0.3, -0.5, 0.7),
        lattice_spacing_over_reference_length=spacing,
    )
    symbol = linearized_einstein_symbol(momentum)
    gauge = linearized_gauge_direction_matrix(momentum)
    divergence = linearized_bianchi_divergence_matrix(momentum)

    assert np.linalg.norm(symbol @ gauge) == pytest.approx(0.0, abs=1.0e-11)
    assert np.linalg.norm(divergence @ symbol) == pytest.approx(0.0, abs=1.0e-11)


@pytest.mark.parametrize(
    "momentum,spacing,bound",
    (
        ((0.9, 0.1, -0.2, 0.4), 0.2, 1.0),
        ((1.5, -0.7, 0.3, 1.1), 0.1, 2.0),
        ((-0.4, 0.6, 0.8, -0.2), 0.3, 1.0),
    ),
)
def test_entrywise_thirteen_b_delta_bound(
    momentum: tuple[float, float, float, float], spacing: float, bound: float
) -> None:
    continuum = np.asarray(momentum)
    lattice = central_difference_dimensionless_momentum(
        continuum, lattice_spacing_over_reference_length=spacing
    )
    delta = float(np.max(np.abs(lattice - continuum)))
    entry_error = float(
        np.max(
            np.abs(
                linearized_einstein_symbol(lattice)
                - linearized_einstein_symbol(continuum)
            )
        )
    )

    assert entry_error <= 13.0 * bound * delta + 1.0e-12
