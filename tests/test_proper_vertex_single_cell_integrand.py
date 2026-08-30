from __future__ import annotations

import math

import numpy as np
import pytest

from examples.physics.proper_vertex_one_to_five_ls_intertwiners import (
    spin_coherent_state_coefficients,
)
from examples.physics.proper_vertex_single_cell_integrand import (
    certify_proper_vertex_single_cell_integrand,
    evaluate_cp1_proper_face_pairing,
    evaluate_proper_vertex_coefficient_at_frames,
    spin_j_linear_epsilon_dual,
    spin_state_homogeneous_polynomial,
)
from examples.physics.proper_vertex_single_cell_kernel import (
    certify_proper_vertex_single_cell_kernel,
)


def test_linear_epsilon_dual_and_homogeneous_polynomial() -> None:
    spin = 2
    xi = np.asarray((0.6 + 0.2j, math.sqrt(0.6)), dtype=complex)
    xi /= np.linalg.norm(xi)
    state = np.asarray(spin_coherent_state_coefficients(spin, xi))
    epsilon_xi = np.asarray((-xi[1], xi[0]))
    expected = np.asarray(spin_coherent_state_coefficients(spin, epsilon_xi))
    assert np.allclose(
        spin_j_linear_epsilon_dual(state, spin), expected, atol=1.0e-12
    )
    z = np.asarray(((0.7 + 0.1j,), (0.2 - 0.3j,)))
    assert spin_state_homogeneous_polynomial(state, spin, z)[0] == pytest.approx(
        complex(np.dot(xi, z[:, 0])) ** (2 * spin), abs=1.0e-12
    )
    w = z[:, 0]
    source_polynomial = spin_state_homogeneous_polynomial(
        state, spin, np.conjugate(w[:, None])
    )[0]
    assert source_polynomial == pytest.approx(
        complex(np.vdot(w, xi)) ** (2 * spin), abs=1.0e-12
    )
    target_polynomial = spin_state_homogeneous_polynomial(
        spin_j_linear_epsilon_dual(state, spin), spin, w[:, None]
    )[0]
    j_xi = np.asarray((-np.conjugate(xi[1]), np.conjugate(xi[0])))
    assert target_polynomial == pytest.approx(
        complex(np.vdot(j_xi, w)) ** (2 * spin), abs=1.0e-12
    )


def test_identity_frame_pairing_matches_cp1_normalization() -> None:
    spin = 2
    gamma = 0.274
    xi = np.asarray((1.0, 0.0), dtype=complex)
    j_xi = np.asarray((0.0, 1.0), dtype=complex)
    source = np.asarray(spin_coherent_state_coefficients(spin, xi))
    target = np.asarray(spin_coherent_state_coefficients(spin, j_xi))
    value = evaluate_cp1_proper_face_pairing(
        spin=spin,
        gamma=gamma,
        source_state=source,
        target_state_after_projector=target,
        source_frame=np.eye(2),
        target_frame=np.eye(2),
        number_u=24,
        number_phi=48,
    )
    expected = math.sqrt(1.0 + gamma**2) / complex(1.0, -gamma)
    assert value == pytest.approx(expected, abs=2.0e-13)


def test_ten_cp1_propagators_and_pointwise_integrand_converge() -> None:
    certificate = certify_proper_vertex_single_cell_integrand()

    assert certificate.face_propagator_count == 10
    assert certificate.all_ten_cp1_face_pairings_numerically_evaluated
    assert certificate.all_ten_proper_projectors_inserted_on_eq53_target_endpoint
    assert certificate.all_integer_spin_crossing_signs_trivial
    assert certificate.cartan_dual_classical_frame_tuple_used
    assert not certificate.finite_spin_stationary_point_certified
    assert certificate.pointwise_root_gauge_fixed_coefficient_function_quadrature_estimated
    assert certificate.max_face_quadrature_absolute_difference < 2.0e-8
    assert certificate.max_projector_insertion_value_difference < 1.0e-10
    assert abs(certificate.pointwise_proper_vertex_coefficient_function) > 0.0


def test_arbitrary_frame_evaluator_recomputes_projectors_and_matches_fixed_tuple() -> None:
    kernel = certify_proper_vertex_single_cell_kernel()
    frames = dict(kernel.critical_point.gauge_fixed_frames)
    direct = evaluate_proper_vertex_coefficient_at_frames(
        frames=frames,
        kernel_contract=kernel,
        number_u=44,
        number_phi=88,
    )
    fixed = certify_proper_vertex_single_cell_integrand()

    assert direct.face_count == 10
    assert direct.orientation_degenerate_face_count == 0
    assert direct.minimum_absolute_normalized_orientation_determinant > 0.0
    assert direct.all_frames_sl2c
    assert direct.all_nondegenerate_projectors_recomputed_from_supplied_frames
    assert direct.projector_ranks == tuple(
        face.spin for face in kernel.face_kernel_contracts
    )
    assert direct.coefficient_product == pytest.approx(
        fixed.pointwise_proper_vertex_coefficient_function, abs=2.0e-13
    )


def test_arbitrary_frame_evaluator_rejects_wrong_frame_domain() -> None:
    kernel = certify_proper_vertex_single_cell_kernel()
    frames = dict(kernel.critical_point.gauge_fixed_frames)
    frames.pop(next(iter(frames)))
    with pytest.raises(ValueError, match='five kernel labels'):
        evaluate_proper_vertex_coefficient_at_frames(
            frames=frames, kernel_contract=kernel
        )


def test_pointwise_coefficient_is_root_gauge_independent_after_refixing() -> None:
    kernel = certify_proper_vertex_single_cell_kernel()
    frames = dict(kernel.critical_point.gauge_fixed_frames)
    reference = evaluate_proper_vertex_coefficient_at_frames(
        frames=frames,
        kernel_contract=kernel,
        number_u=44,
        number_phi=88,
    ).coefficient_product
    for new_root in sorted(frames):
        left = np.linalg.inv(frames[new_root])
        refixed = {label: left @ frame for label, frame in frames.items()}
        value = evaluate_proper_vertex_coefficient_at_frames(
            frames=refixed,
            kernel_contract=kernel,
            number_u=44,
            number_phi=88,
        ).coefficient_product
        assert value == pytest.approx(reference, abs=1.0e-13)


def test_pointwise_integrand_ceiling_stops_before_haar_integral() -> None:
    with pytest.raises(ValueError, match='refine'):
        certify_proper_vertex_single_cell_integrand(
            coarse_shape=(20, 40), fine_shape=(20, 60)
        )

    certificate = certify_proper_vertex_single_cell_integrand()
    assert certificate.cp1_real_dimension_per_face == 2
    assert certificate.gauge_fixed_sl2c_real_dimension == 24
    assert not certificate.product_haar_density_factor_included
    assert not certificate.noncompact_sl2c_haar_measure_materialized
    assert not certificate.noncompact_sl2c_haar_integral_evaluated
    assert not certificate.proper_eprl_five_vertex_amplitude_derived
    assert not certificate.proper_eprl_multicell_hessian_computed
    assert certificate.status.endswith('QUADRATURE_ESTIMATED')
    assert certificate.claim_ceiling.endswith('CLASSICAL_FRAMES_ONLY')
