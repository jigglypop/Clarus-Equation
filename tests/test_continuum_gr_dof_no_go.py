import numpy as np
import pytest

from examples.physics.continuum_gr_dof_no_go import (
    MINKOWSKI_METRIC,
    continuum_gr_dof_no_go,
    massless_spin_two_polarization_count,
    massless_tt_basis_4d,
    massive_spin_two_polarization_count,
    massive_traceless_transverse_basis_4d,
)


def _minkowski_trace(tensor: np.ndarray) -> float:
    return float(np.sum(MINKOWSKI_METRIC * tensor))


def test_covariant_r_squared_counterexample_deletes_unique_gr_implication() -> None:
    audit = continuum_gr_dof_no_go(alpha_over_reference_length_squared=2.0)

    assert audit.scalaron_mass_squared_times_reference_length_squared == pytest.approx(
        1.0 / 12.0
    )
    assert audit.einstein_hilbert_polarizations == 2
    assert audit.r_plus_r_squared_polarizations == 3
    assert audit.shared_flat_stationary_background
    assert audit.shared_finite_flat_face_data
    assert audit.both_actions_diffeomorphism_invariant
    assert not audit.unique_continuum_action_follows
    assert not audit.exactly_two_dof_follow
    assert audit.status == "FINITE_FACE_TO_UNIQUE_CONTINUUM_GR_IMPLICATION_DISPROVED"


def test_massless_tt_basis_has_exactly_plus_and_cross() -> None:
    basis = massless_tt_basis_4d()
    null_covector = np.array((-1.0, 0.0, 0.0, 1.0))

    assert massless_spin_two_polarization_count(4) == 2
    assert len(basis) == 2
    for tensor in basis:
        np.testing.assert_allclose(tensor, tensor.T)
        assert _minkowski_trace(tensor) == pytest.approx(0.0)
        np.testing.assert_allclose(null_covector @ tensor, np.zeros(4))
    assert np.linalg.matrix_rank(np.stack([item.ravel() for item in basis])) == 2


def test_massive_rest_frame_basis_has_five_independent_polarizations() -> None:
    basis = massive_traceless_transverse_basis_4d()
    timelike_covector = np.array((-1.0, 0.0, 0.0, 0.0))

    assert massive_spin_two_polarization_count(4) == 5
    assert len(basis) == 5
    for tensor in basis:
        np.testing.assert_allclose(tensor, tensor.T)
        assert _minkowski_trace(tensor) == pytest.approx(0.0)
        np.testing.assert_allclose(timelike_covector @ tensor, np.zeros(4))
    assert np.linalg.matrix_rank(np.stack([item.ravel() for item in basis])) == 5


@pytest.mark.parametrize("alpha", [0.0, -1.0, float("inf")])
def test_counterexample_requires_nontachyonic_finite_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        continuum_gr_dof_no_go(alpha)
