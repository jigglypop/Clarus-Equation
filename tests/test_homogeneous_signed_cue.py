from __future__ import annotations

from dataclasses import fields
import math

import numpy as np
import pytest

import reality_stone.clarus as clarus
from reality_stone.clarus.covariant_metric_flow import CovariantMetricState
from reality_stone.clarus.homogeneous_signed_cue import (
    HomogeneousSignedCue,
    HomogeneousSignedCueCertificate,
    HomogeneousSignedCueState,
    SignedCueReadout,
)


def _assert_no_negative_zero(state: HomogeneousSignedCueState) -> None:
    factor = np.asarray(state.factor, dtype=np.float64)
    assert not np.any(np.signbit(factor[factor == 0.0]))


@pytest.mark.parametrize(
    "reference",
    [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 1.0e-14, -1.0e-14]),
    ],
)
@pytest.mark.parametrize("sign", [-1, +1])
def test_exact_axes_and_both_signs_have_costs_two_and_four(
    reference: np.ndarray,
    sign: int,
) -> None:
    model = HomogeneousSignedCue(3)
    reference = reference / np.linalg.norm(reference)

    state = model.write_cue(model.identity_state(), reference, sign)
    readout = model.readout(state, reference)
    correct_index = readout.action_signs.index(sign)
    wrong_index = 1 - correct_index

    assert isinstance(readout, SignedCueReadout)
    assert readout.selected_sign == sign
    assert readout.unique
    assert readout.costs[correct_index] == pytest.approx(2.0, abs=3e-13)
    assert readout.costs[wrong_index] == pytest.approx(4.0, abs=3e-13)
    assert readout.wrong_minus_selected_margin == pytest.approx(2.0, abs=4e-13)
    _assert_no_negative_zero(state)


def test_one_write_matches_independent_rank_one_formula() -> None:
    model = HomogeneousSignedCue(3)
    reference = np.array([2.0, -1.0, 2.0]) / 3.0
    sign = -1
    cue = np.append(sign * reference, 1.0)
    expected = np.eye(4) + 0.5 * np.outer(cue, cue)

    state = model.write_cue(model.identity_state(), reference, sign)

    np.testing.assert_allclose(model.metric(state), expected, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("singular_value", [0.25, 4.0])
@pytest.mark.parametrize("sign", [-1, +1])
def test_embedded_chart_endpoint_preserves_costs_and_action(
    singular_value: float,
    sign: int,
) -> None:
    model = HomogeneousSignedCue(3)
    reference = np.array([2.0, -1.0, 2.0]) / 3.0
    chart = np.array(
        [
            [0.0, -singular_value, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0 / singular_value],
        ]
    )
    lift = np.eye(4)
    lift[:3, :3] = chart
    inverse = np.linalg.inv(lift)
    transformed_initial_metric = inverse.T @ inverse

    original = model.write_cue(model.identity_state(), reference, sign)
    transformed_initial = model.make_state_from_metric(transformed_initial_metric)
    transformed = model.write_cue(transformed_initial, chart @ reference, sign)
    original_readout = model.readout(original, reference)
    transformed_readout = model.readout(transformed, chart @ reference)

    assert transformed_readout.selected_sign == original_readout.selected_sign == sign
    _assert_no_negative_zero(original)
    _assert_no_negative_zero(transformed_initial)
    _assert_no_negative_zero(transformed)
    np.testing.assert_allclose(
        transformed_readout.costs,
        original_readout.costs,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        model.metric(transformed),
        inverse.T @ model.metric(original) @ inverse,
        rtol=2e-12,
        atol=2e-12,
    )


def test_snapshot_roundtrip_is_exact_and_state_has_one_factor_field() -> None:
    model = HomogeneousSignedCue(3)
    state = model.write_cue(model.identity_state(), [1.0, 0.0, 0.0], +1)

    snapshot = model.snapshot(state)
    restored = model.from_snapshot(snapshot)

    assert tuple(field.name for field in fields(HomogeneousSignedCueState)) == ("factor",)
    assert snapshot == state
    assert restored == state
    _assert_no_negative_zero(snapshot)
    _assert_no_negative_zero(restored)
    np.testing.assert_array_equal(
        np.asarray(restored.factor, dtype=np.float64).view(np.uint64),
        np.asarray(state.factor, dtype=np.float64).view(np.uint64),
    )


def test_strict_control_is_exactly_sign_even_in_hex_serialization() -> None:
    model = HomogeneousSignedCue(3)
    references = (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, -0.0, 1.0]),
        np.array([0.31, -0.52, 0.79]),
    )
    for reference in references:
        reference = reference / np.linalg.norm(reference)
        positive = model.strict_write(model.strict_identity_state(), reference)
        negative = model.strict_write(model.strict_identity_state(), -reference)

        positive_bytes = model.serialize_strict_state(positive)
        negative_bytes = model.serialize_strict_state(negative)

        assert positive_bytes == negative_bytes
        assert b"-0x0.0p+0" not in positive_bytes
        assert model.strict_terminal_distribution(positive, reference) == (
            0.5,
            0.5,
        )
        assert model.strict_terminal_distribution(negative, reference) == (
            0.5,
            0.5,
        )


def test_deleting_homogeneous_coordinate_kills_sign_readout() -> None:
    model = HomogeneousSignedCue(3)
    reference = np.array([2.0, -1.0, 2.0]) / 3.0
    spatial_metrics = []
    action_costs = []
    for sign in (-1, +1):
        state = model.write_cue(model.identity_state(), reference, sign)
        spatial = model.metric(state)[:3, :3]
        spatial_metrics.append(spatial)
        action_costs.append(
            (
                float((-reference) @ spatial @ (-reference)),
                float(reference @ spatial @ reference),
            )
        )

    np.testing.assert_array_equal(spatial_metrics[0], spatial_metrics[1])
    assert action_costs[0][0] == pytest.approx(action_costs[0][1])
    assert action_costs[1][0] == pytest.approx(action_costs[1][1])
    assert action_costs[0] == pytest.approx(action_costs[1])


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda model: model.write_cue(model.identity_state(), [0.0, 0.0, 0.0], 1), "nonzero"),
        (lambda model: model.write_cue(model.identity_state(), [1.0, math.nan, 0.0], 1), "finite"),
        (lambda model: model.write_cue(model.identity_state(), [1.0, 0.0], 1), "shape"),
        (
            lambda model: model.write_cue(model.identity_state(), [1.0, 0.0, 0.0], 0),
            r"-1 or \+1",
        ),
        (lambda model: model.readout(model.identity_state(), [math.inf, 0.0, 0.0]), "finite"),
        (
            lambda model: model.write_cue(model.identity_state(), [2.0, 0.0, 0.0], 1),
            "prediction p=2",
        ),
        (
            lambda model: model.strict_write(model.strict_identity_state(), [0.0, 0.0, 0.0]),
            "nonzero",
        ),
    ],
)
def test_invalid_inputs_are_rejected(call: object, message: str) -> None:
    model = HomogeneousSignedCue(3)
    with pytest.raises(ValueError, match=message):
        call(model)  # type: ignore[operator]


def test_wrong_state_types_and_invalid_dimension_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive built-in integer"):
        HomogeneousSignedCue(True)
    model = HomogeneousSignedCue(3)
    with pytest.raises(ValueError, match="HomogeneousSignedCueState"):
        model.metric(CovariantMetricState(((1.0,),)))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="CovariantMetricState"):
        model.strict_write(model.identity_state(), [1.0, 0.0, 0.0])  # type: ignore[arg-type]


def test_certificate_discloses_extra_coordinates_and_narrow_scope() -> None:
    model = HomogeneousSignedCue(3)
    certificate = model.certificate(model.identity_state())

    assert isinstance(certificate, HomogeneousSignedCueCertificate)
    assert certificate.persistent_state_field_count == 1
    assert certificate.ambient_real_state_coordinates == 10
    assert certificate.original_metric_degrees_of_freedom == 6
    assert certificate.added_ambient_coordinates == 4
    assert certificate.packed_covector_coordinates == 3
    assert certificate.packed_scalar_coordinates == 1
    assert certificate.optimizer_state_field_count == 0
    assert certificate.registered_prewrite_prediction == 2.0
    assert certificate.prewrite_prediction_tolerance_multiplier == 256.0
    assert certificate.declared_chart_group == "diag(GL(d),1)_only"
    assert certificate.embedded_spatial_chart_covariance_in_exact_arithmetic
    assert not certificate.general_ambient_gl_semantics_verified
    assert certificate.homogeneous_splitting_is_extra_structure
    assert not certificate.general_delayed_credit_verified
    assert not certificate.infinite_scc_intelligence_growth_verified
    assert not certificate.biological_fidelity_verified
    assert not certificate.cosmological_identity_verified
    assert not certificate.agi_evidence


def test_public_exports_are_the_production_types() -> None:
    assert clarus.HomogeneousSignedCueState is HomogeneousSignedCueState
    assert clarus.SignedCueReadout is SignedCueReadout
    assert clarus.HomogeneousSignedCueCertificate is HomogeneousSignedCueCertificate
    assert clarus.HomogeneousSignedCue is HomogeneousSignedCue
