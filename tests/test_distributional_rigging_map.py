import math

import numpy as np
import pytest

from examples.physics.distributional_rigging_map import (
    audit_distributional_rigging_map,
    branch_functional,
    generalized_branch_coefficients,
    rigging_pairing,
    zero_refinement_embedding,
)
from examples.physics.curved_sector_thimble import dimensionless_curved_regge_phase


def test_generalized_branch_truncation_has_linearly_growing_norm() -> None:
    coefficients = generalized_branch_coefficients(
        7, dimensionless_phase_increment=0.37
    )

    assert np.vdot(coefficients, coefficients).real == pytest.approx(7.0)


def test_zero_refinement_embeddings_are_isometric_and_compositional() -> None:
    vector = np.asarray((1.0 + 2.0j, -3.0j))
    direct = zero_refinement_embedding(vector, refined_dimension=6)
    via_four = zero_refinement_embedding(
        zero_refinement_embedding(vector, refined_dimension=4), refined_dimension=6
    )

    assert direct == pytest.approx(via_four)
    assert np.vdot(direct, direct) == pytest.approx(np.vdot(vector, vector))


def test_finite_support_functional_is_eventually_constant_under_refinement() -> None:
    vector = (1.0 + 1.0j, 2.0 - 0.5j, -0.2)
    embedded = zero_refinement_embedding(vector, refined_dimension=11)

    assert branch_functional(
        embedded, dimensionless_phase_increment=0.41
    ) == pytest.approx(
        branch_functional(vector, dimensionless_phase_increment=0.41)
    )


def test_rigging_pairing_is_antilinear_linear_hermitian_and_positive() -> None:
    first = np.asarray((1.0 + 2.0j, -0.5j, 3.0))
    second = np.asarray((-2.0j, 1.5, 0.25 + 0.1j))
    scalar = 0.7 - 1.3j
    phase = -0.22

    pairing = rigging_pairing(
        first, second, dimensionless_phase_increment=phase
    )
    assert rigging_pairing(
        scalar * first, second, dimensionless_phase_increment=phase
    ) == pytest.approx(scalar.conjugate() * pairing)
    assert rigging_pairing(
        first, scalar * second, dimensionless_phase_increment=phase
    ) == pytest.approx(scalar * pairing)
    assert pairing == pytest.approx(
        rigging_pairing(second, first, dimensionless_phase_increment=phase).conjugate()
    )
    diagonal = rigging_pairing(
        first, first, dimensionless_phase_increment=phase
    )
    assert diagonal.imag == pytest.approx(0.0)
    assert diagonal.real >= 0.0


def test_explicit_null_vector_is_removed_by_the_quotient() -> None:
    phase = 0.31
    null_vector = (1.0, -np.exp(1j * phase))

    assert branch_functional(
        null_vector, dimensionless_phase_increment=phase
    ) == pytest.approx(0.0j, abs=1.0e-12)
    assert rigging_pairing(
        null_vector, (2.0, -1.0j), dimensionless_phase_increment=phase
    ) == pytest.approx(0.0j, abs=1.0e-12)


def test_distributional_audit_closes_exact_model_but_not_gr_claims() -> None:
    audit = audit_distributional_rigging_map(
        3, 5, 9, dimensionless_phase_increment=0.43
    )

    assert audit.coarse_truncation_norm_squared == pytest.approx(3.0)
    assert audit.refined_truncation_norm_squared == pytest.approx(9.0)
    assert audit.embedded_truncation_difference_norm_squared == pytest.approx(6.0)
    assert audit.direct_system_identity_composition_isometric
    assert audit.cylindrical_consistency_exact
    assert audit.finite_truncations_fail_hilbert_norm_cauchy_criterion
    assert audit.finite_support_distributional_limit_eventually_constant
    assert audit.rigging_pairing_hermitian_positive_semidefinite
    assert audit.rigging_gram_rank == 1
    assert audit.quotient_dimension == 1
    assert audit.physical_completion_isomorphic_to_complex
    assert audit.unit_norm_coarse_probe_pairing == pytest.approx(1.0 / 3.0)
    assert audit.unit_norm_refined_probe_pairing == pytest.approx(1.0 / 9.0)
    assert audit.unit_norm_truncation_pairings_break_cylindrical_consistency
    assert audit.status == "EXACT_DISTRIBUTIONAL_RIGGING_MAP_MODEL_CLOSED"
    assert not audit.topological_limit_excluded
    assert not audit.geometric_refinement_defined
    assert not audit.eprl_amplitude_used
    assert not audit.renormalized_cutoff_removal_proved
    assert not audit.einstein_hilbert_dominance_proved
    assert not audit.anomaly_free_constraint_algebra_proved
    assert not audit.exactly_two_graviton_helicities_proved
    assert audit.claim_ceiling.endswith("NOT_4D_GR_CONTINUUM")


def test_unit_norm_truncation_functional_has_inverse_square_root_scaling() -> None:
    phase = 0.27
    for dimension in (2, 5, 11):
        unit_vector = generalized_branch_coefficients(
            dimension, dimensionless_phase_increment=phase
        ) / math.sqrt(dimension)
        first_basis = np.zeros(dimension, dtype=complex)
        first_basis[0] = 1.0

        assert np.vdot(unit_vector, first_basis) == pytest.approx(
            np.exp(-1j * phase) / math.sqrt(dimension)
        )


def test_dimensionless_curved_regge_phase_can_feed_declared_branch_phase() -> None:
    curved = dimensionless_curved_regge_phase(
        (0.5, 1.25),
        (0.2, -0.4),
        cosmological_constant_times_reference_length_squared=3.0,
        four_volume_over_reference_length_fourth=0.1,
        inverse_dimensionless_gravitational_coupling=2.0,
    )
    audit = audit_distributional_rigging_map(
        2,
        4,
        7,
        dimensionless_phase_increment=curved.dimensionless_curved_regge_phase,
    )

    assert audit.dimensionless_phase_increment == pytest.approx(-1.4)
    assert audit.phase_increment_declared_dimensionless
    assert audit.status == "EXACT_DISTRIBUTIONAL_RIGGING_MAP_MODEL_CLOSED"
    assert not audit.geometric_refinement_defined


@pytest.mark.parametrize(
    "dimensions",
    ((0, 2, 3), (2, 2, 3), (3, 2, 4), (1, 3, 3)),
)
def test_invalid_refinement_dimensions_are_rejected(
    dimensions: tuple[int, int, int],
) -> None:
    with pytest.raises(ValueError):
        audit_distributional_rigging_map(
            *dimensions, dimensionless_phase_increment=0.0
        )


@pytest.mark.parametrize("phase", (math.inf, -math.inf, math.nan))
def test_nonfinite_phase_is_rejected(phase: float) -> None:
    with pytest.raises(ValueError, match="phase"):
        generalized_branch_coefficients(3, dimensionless_phase_increment=phase)


def test_embedding_cannot_reduce_dimension() -> None:
    with pytest.raises(ValueError, match="cannot be smaller"):
        zero_refinement_embedding((1.0, 2.0), refined_dimension=1)


@pytest.mark.parametrize("coarse,refined", ((1, 2), (2, 7), (5, 13)))
def test_generalized_truncation_difference_is_exactly_new_label_count(
    coarse: int, refined: int
) -> None:
    phase = 0.73
    coarse_coefficients = generalized_branch_coefficients(
        coarse, dimensionless_phase_increment=phase
    )
    refined_coefficients = generalized_branch_coefficients(
        refined, dimensionless_phase_increment=phase
    )
    difference = refined_coefficients - zero_refinement_embedding(
        coarse_coefficients, refined_dimension=refined
    )

    assert np.vdot(difference, difference).real == pytest.approx(refined - coarse)


def test_quotient_coordinate_is_surjective_and_gram_is_rank_one() -> None:
    phase = -0.19
    preimage_of_one = (np.exp(1j * phase), 0.0, 0.0)
    basis = np.eye(3, dtype=complex)
    gram = np.asarray(
        [
            [
                rigging_pairing(
                    basis[row], basis[column], dimensionless_phase_increment=phase
                )
                for column in range(3)
            ]
            for row in range(3)
        ]
    )

    assert branch_functional(
        preimage_of_one, dimensionless_phase_increment=phase
    ) == pytest.approx(1.0)
    assert np.linalg.matrix_rank(gram, tol=1.0e-12) == 1


@pytest.mark.parametrize("tolerance", (0.0, -1.0, math.inf, math.nan))
def test_invalid_audit_tolerance_is_rejected(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        audit_distributional_rigging_map(
            1,
            2,
            3,
            dimensionless_phase_increment=0.0,
            tolerance=tolerance,
        )


def test_invalid_vectors_are_rejected() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        branch_functional((), dimensionless_phase_increment=0.0)
    with pytest.raises(ValueError, match="finite"):
        branch_functional((complex(math.inf, 0.0),), dimensionless_phase_increment=0.0)
