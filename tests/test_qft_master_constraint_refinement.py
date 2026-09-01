import numpy as np
import pytest

from examples.physics.qft_master_constraint_refinement import (
    audit_master_constraint_refinement,
    master_constraint,
    zero_projector,
)
from examples.physics.distributional_rigging_map import zero_refinement_embedding


def test_e60_zero_projector_and_master_have_exact_finite_kernel():
    projector = zero_projector(7, dimensionless_phase_increment=0.31)
    master = master_constraint(7, dimensionless_phase_increment=0.31)

    assert np.linalg.norm(projector @ projector - projector) < 1.0e-12
    assert np.linalg.norm(projector - projector.T.conj()) < 1.0e-12
    assert np.min(np.linalg.eigvalsh(master)) > -1.0e-12
    assert np.count_nonzero(np.abs(np.linalg.eigvalsh(master)) < 1.0e-12) == 1


def test_e60_normalized_projector_dies_but_renormalized_pairing_survives():
    audit = audit_master_constraint_refinement(
        3, 11, dimensionless_phase_increment=0.43
    )

    assert audit.finite_kernels_nontrivial
    assert audit.finite_kernel_embedding_inconsistent
    assert audit.normalized_projector_limit_trivial_on_probe
    assert audit.renormalized_forms_cylindrically_consistent
    assert audit.normalization_linear_in_dimension
    assert audit.renormalized_pairing_residual < 1.0e-12
    assert audit.required_scale_ratio == pytest.approx(11.0 / 3.0)
    assert audit.status == 'EXACT_MASTER_CONSTRAINT_DISTRIBUTIONAL_TOY_CLOSED'
    assert not audit.gravity_regulator_defined
    assert not audit.original_hda_anomaly_checked
    assert not audit.continuum_physical_hilbert_proved


def test_e60_renormalized_pairing_is_cylindrical_for_general_test_vectors():
    phase = -0.27
    first = np.array([1.0 + 0.2j, -0.4j, 0.7])
    second = np.array([-0.3, 1.1j, 0.5 - 0.6j])
    coarse = first.size
    coarse_projector = zero_projector(
        coarse, dimensionless_phase_increment=phase
    )
    coarse_pairing = coarse * np.vdot(first, coarse_projector @ second)

    for refined in (7, 19, 53):
        refined_projector = zero_projector(
            refined, dimensionless_phase_increment=phase
        )
        first_refined = zero_refinement_embedding(first, refined_dimension=refined)
        second_refined = zero_refinement_embedding(second, refined_dimension=refined)
        refined_pairing = refined * np.vdot(
            first_refined, refined_projector @ second_refined
        )
        assert refined_pairing == pytest.approx(coarse_pairing, abs=1.0e-12)


def test_e60_zero_projector_has_inverse_square_root_strong_loss_rate():
    probe = np.array([1.0, -0.2j, 0.4 + 0.1j])
    phase = 0.19
    scaled_norms = []
    for refined in (16, 64, 256, 1024):
        projector = zero_projector(refined, dimensionless_phase_increment=phase)
        embedded = zero_refinement_embedding(probe, refined_dimension=refined)
        scaled_norms.append(np.sqrt(refined) * np.linalg.norm(projector @ embedded))

    assert scaled_norms == pytest.approx([scaled_norms[0]] * len(scaled_norms))


def test_e60_finite_kernel_limit_depends_on_refinement_embedding():
    coarse_kernel = np.array([1.0, 0.0, 0.0])
    refined_projector = np.diag([1.0, 0.0, 0.0, 0.0])
    preserving = np.zeros((4, 3))
    preserving[:3, :3] = np.eye(3)
    shifting = np.zeros((4, 3))
    shifting[1:, :] = np.eye(3)

    assert np.linalg.norm(
        refined_projector @ preserving @ coarse_kernel - preserving @ coarse_kernel
    ) == 0.0
    assert np.linalg.norm(refined_projector @ shifting @ coarse_kernel) == 0.0
    assert np.linalg.norm(shifting @ coarse_kernel) == 1.0


def test_e60_continuous_zero_fiber_requires_the_density_normalization():
    epsilon = 0.25
    finite_limits = []
    raw_limits = []
    overnormalized_limits = []
    for dimension in (200, 400, 800, 1600):
        spectrum = (np.arange(dimension) + 0.5) / dimension
        window_weight = np.count_nonzero(spectrum <= epsilon) / dimension
        raw_limits.append(window_weight)
        finite_limits.append(window_weight / epsilon)
        overnormalized_limits.append(window_weight / epsilon**2)

    assert raw_limits[-1] == pytest.approx(epsilon, abs=1.0 / 1600)
    assert finite_limits[-1] == pytest.approx(1.0, abs=1.0 / 400)
    assert overnormalized_limits[-1] == pytest.approx(1.0 / epsilon, abs=1.0 / 100)


@pytest.mark.parametrize('dimensions', ((0, 3), (3, 3), (4, 2)))
def test_e60_invalid_refinement_dimensions_fail_closed(dimensions):
    with pytest.raises(ValueError):
        audit_master_constraint_refinement(
            *dimensions, dimensionless_phase_increment=0.0
        )
