from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from examples.physics.eprl_y_gamma_and_proper_projector import (
    evaluate_y_gamma_coherent_state,
    identity_sl2c,
    proper_orientation_sign,
    proper_positive_spectral_projector,
    relative_sl2c,
    spin_generators,
)
from examples.physics.proper_vertex_one_to_five_frame_lifts import (
    hermitian_sl2c_boost_lift,
)


@pytest.mark.parametrize('spin', (Fraction(0), Fraction(1, 2), Fraction(1), Fraction(3, 2)))
def test_spin_generators_close_su2_and_casimir(spin: Fraction) -> None:
    generators = spin_generators(spin)

    assert len(generators.magnetic_numbers) == int(2 * spin + 1)
    assert generators.hermiticity_residual < 2.0e-14
    assert generators.commutator_residual < 2.0e-14
    assert generators.casimir_residual < 2.0e-14


def test_y_gamma_coherent_homogeneous_value_matches_published_formula() -> None:
    spin = Fraction(3, 2)
    gamma = 0.274
    xi = np.asarray((1.0, 1.0j)) / np.sqrt(2.0)
    z_value = np.asarray((2.0 - 1.0j, -0.5 + 0.25j))
    evaluation = evaluate_y_gamma_coherent_state(spin, gamma, xi, z_value)

    expected_polynomial = np.sqrt(float(2 * spin + 1) / np.pi) * (
        np.dot(xi, z_value) ** int(2 * spin)
    )
    expected_radial = complex(np.vdot(z_value, z_value).real) ** complex(
        -1.0 - float(spin), gamma * float(spin)
    )
    assert evaluation.coherent_polynomial_value == pytest.approx(expected_polynomial)
    assert evaluation.radial_factor == pytest.approx(expected_radial)
    assert evaluation.embedded_value == pytest.approx(expected_radial * expected_polynomial)
    assert evaluation.principal_series_k == spin
    assert evaluation.principal_series_p == pytest.approx(gamma * float(spin))
    assert evaluation.su2_lowest_type_selected
    assert not evaluation.full_principal_series_representation_materialized


def test_proper_projector_uses_strictly_positive_spectrum() -> None:
    rapidity = 0.4
    boost = hermitian_sl2c_boost_lift(
        (np.cosh(rapidity), 0.0, 0.0, np.sinh(rapidity))
    )
    positive = proper_positive_spectral_projector(Fraction(2), 1, boost)
    negative = proper_positive_spectral_projector(Fraction(2), -1, boost)

    assert positive.positive_eigenvalue_count == 2
    assert positive.zero_eigenvalue_count == 1
    assert positive.nontrivial_generator
    assert negative.positive_eigenvalue_count == 2
    assert np.allclose(
        positive.projector + negative.projector,
        np.eye(5) - np.diag((0.0, 0.0, 1.0, 0.0, 0.0)),
        atol=2.0e-14,
        rtol=0.0,
    )
    for result in (positive, negative):
        assert result.hermiticity_residual < 2.0e-14
        assert result.projector_idempotence_residual < 2.0e-14
        assert result.projector_hermiticity_residual < 2.0e-14
        assert result.strictly_positive_interval_used

    trivial = proper_positive_spectral_projector(
        Fraction(2), 1, identity_sl2c()
    )
    assert not trivial.nontrivial_generator
    assert trivial.trace_vector_norm == pytest.approx(0.0)
    assert trivial.positive_eigenvalue_count == 0
    assert trivial.zero_eigenvalue_count == 5
    assert np.allclose(trivial.projector, np.zeros((5, 5)))


def test_beta_sign_is_order_lift_sign_and_common_left_invariant() -> None:
    directions = (
        (0.17, 1.0, 0.0, 0.0),
        (0.19, 0.0, 1.0, 0.0),
        (0.23, 0.0, 0.0, 1.0),
        (0.29, 1.0, 1.0, 1.0),
    )
    frames = {0: identity_sl2c()}
    for label, (rapidity, x_value, y_value, z_value) in enumerate(directions, start=1):
        direction = np.asarray((x_value, y_value, z_value), dtype=float)
        direction /= np.linalg.norm(direction)
        future = np.concatenate(
            ((np.cosh(rapidity),), np.sinh(rapidity) * direction)
        )
        frames[label] = hermitian_sl2c_boost_lift(future)
    reference = proper_orientation_sign(frames, 0, 1)

    common_left = hermitian_sl2c_boost_lift(
        (np.cosh(0.11), np.sinh(0.11), 0.0, 0.0)
    )
    transformed = {
        label: common_left @ ((-1.0 if label % 2 else 1.0) * frame)
        for label, frame in frames.items()
    }
    changed = proper_orientation_sign(transformed, 0, 1)
    assert reference.nondegenerate
    assert reference.beta in (-1, 1)
    assert reference.common_order_reversal_invariant
    assert changed.beta == reference.beta


def test_operator_input_domains_and_relative_convention() -> None:
    with pytest.raises(ValueError, match='half-integer'):
        spin_generators(Fraction(1, 3))
    with pytest.raises(ValueError, match='gamma'):
        evaluate_y_gamma_coherent_state(
            Fraction(1), np.inf, (1.0, 0.0), (1.0, 0.0)
        )
    with pytest.raises(ValueError, match='beta'):
        proper_positive_spectral_projector(Fraction(1), 0, identity_sl2c())

    source = hermitian_sl2c_boost_lift(
        (np.cosh(0.1), np.sinh(0.1), 0.0, 0.0)
    )
    target = hermitian_sl2c_boost_lift(
        (np.cosh(0.2), 0.0, np.sinh(0.2), 0.0)
    )
    assert np.allclose(
        relative_sl2c(source, target),
        np.linalg.solve(source, target),
        atol=2.0e-14,
        rtol=0.0,
    )
