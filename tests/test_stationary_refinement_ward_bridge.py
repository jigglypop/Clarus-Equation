from __future__ import annotations

import math

import numpy as np
import pytest

from examples.physics.stationary_refinement_ward_bridge import (
    audit_stationary_refinement_ward_bridge,
    certify_linear_refinement_ward_pullback,
    composed_action_hessian,
)


def test_nonlinear_embedding_has_the_exact_chain_rule_extra_term() -> None:
    result = composed_action_hessian(
        (3.0,), ((5.0,),), ((2.0,),), (((4.0,),),)
    )

    assert np.allclose(result, np.asarray(((32.0,),)))


def test_stationarity_removes_nonlinear_embedding_extra_term() -> None:
    result = composed_action_hessian(
        (0.0,), ((5.0,),), ((2.0,),), (((4.0,),),)
    )

    assert np.allclose(result, np.asarray(((20.0,),)))


def test_linear_embedding_needs_no_stationarity_for_hessian_pullback() -> None:
    result = composed_action_hessian(
        (3.0,), ((5.0,),), ((2.0,),), (((0.0,),),)
    )

    assert np.allclose(result, np.asarray(((20.0,),)))


def test_exact_linear_refinement_inherits_the_fine_ward_identity() -> None:
    fine_hessian = np.diag((2.0, 3.0, 0.0, 5.0))
    embedding = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0),
        )
    )
    coarse_hessian = embedding.T @ fine_hessian @ embedding
    coarse_gauge = np.asarray(((0.0,), (0.0,), (1.0,)))
    fine_gauge = embedding @ coarse_gauge
    certificate = certify_linear_refinement_ward_pullback(
        coarse_hessian,
        fine_hessian,
        embedding,
        coarse_gauge,
        fine_gauge,
        ((1.0,),),
    )

    assert certificate.exact_hessian_pullback
    assert certificate.exact_gauge_intertwining
    assert certificate.exact_fine_ward_identity
    assert certificate.exact_fine_left_ward_identity
    assert certificate.exact_coarse_ward_identity
    assert certificate.exact_coarse_left_ward_identity
    assert certificate.coarse_ward_triangle_bound == pytest.approx(0.0)
    assert certificate.exact_residual_decomposition_error == pytest.approx(0.0)


def test_approximate_inputs_obey_the_exact_residual_decomposition_bound() -> None:
    fine_hessian = np.diag((2.0, 3.0, 0.4, 5.0))
    embedding = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0),
        )
    )
    exact_pullback = embedding.T @ fine_hessian @ embedding
    coarse_hessian = exact_pullback + np.diag((0.01, 0.0, 0.0))
    coarse_gauge = np.asarray(((0.0,), (0.0,), (1.0,)))
    fine_gauge = embedding @ coarse_gauge + np.asarray(
        ((0.0,), (0.0,), (0.0,), (0.02,))
    )
    certificate = certify_linear_refinement_ward_pullback(
        coarse_hessian,
        fine_hessian,
        embedding,
        coarse_gauge,
        fine_gauge,
        ((1.0,),),
    )

    assert certificate.exact_residual_decomposition_error < 1.0e-12
    assert certificate.coarse_ward_within_triangle_bound
    assert (
        certificate.coarse_ward_residual
        <= certificate.coarse_ward_triangle_bound + 1.0e-12
    )
    assert not certificate.exact_fine_ward_identity
    assert not certificate.exact_coarse_ward_identity


@pytest.mark.parametrize(
    "momentum",
    (
        (1.2, 0.3, -0.4, 0.8),
        (0.7, -0.2, 0.5, 0.1),
        (2.0, 0.4, 0.3, -0.9),
    ),
)
def test_spin2_duplicated_refinement_witness_closes_all_declared_identities(
    momentum: tuple[float, float, float, float],
) -> None:
    audit = audit_stationary_refinement_ward_bridge(momentum)

    assert audit.coarse_field_dimension == 10
    assert audit.fine_field_dimension == 20
    assert audit.embedding_is_linear
    assert audit.gauge_generators_are_field_independent
    assert audit.embedding_isometry_residual < 1.0e-10
    assert audit.average_projector_idempotence_residual < 1.0e-10
    assert audit.relative_projector_idempotence_residual < 1.0e-10
    assert audit.relative_projector_embedding_residual < 1.0e-10
    assert audit.constructed_quadratic_action_cylindrical_residual < 1.0e-10
    assert audit.exact_linear_refinement_hessian_pullback_closed
    assert audit.conditional_nonlinear_pullback_at_stationary_point_closed
    assert audit.fine_ward_identity_inherited_by_coarse_kernel
    assert audit.nonlinear_off_stationary_counterexample_closed
    assert audit.certificate.exact_hessian_pullback
    assert audit.certificate.exact_gauge_intertwining
    assert audit.certificate.exact_fine_ward_identity
    assert audit.certificate.exact_fine_left_ward_identity
    assert audit.certificate.exact_coarse_ward_identity
    assert audit.certificate.exact_coarse_left_ward_identity
    assert audit.status == "CONDITIONAL_REFINEMENT_HESSIAN_WARD_BRIDGE_CLOSED"


def test_audit_keeps_microscopic_claims_false() -> None:
    audit = audit_stationary_refinement_ward_bridge()

    assert not audit.rigging_pairing_cylindricity_implies_action_hessian_consistency
    assert not audit.microscopic_spin_foam_effective_action_supplied
    assert not audit.proper_vertex_multicell_hessian_refinement_derived
    assert not audit.ce_effective_kernel_proved_to_lie_in_two_derivative_ansatz
    assert audit.claim_ceiling.endswith("NOT_MICROSCOPIC_KERNEL_DERIVATION")


@pytest.mark.parametrize(
    "arguments,message",
    (
        (
            {
                "fine_gradient": (1.0,),
                "fine_hessian": ((1.0, 0.0), (0.0, 1.0)),
                "embedding_jacobian": ((1.0,),),
                "embedding_second_derivatives": (((0.0,),),),
            },
            "fine_hessian shape",
        ),
        (
            {
                "fine_gradient": (1.0,),
                "fine_hessian": ((1.0,),),
                "embedding_jacobian": ((1.0,),),
                "embedding_second_derivatives": (((math.inf,),),),
            },
            "embedding_second_derivatives",
        ),
    ),
)
def test_composed_hessian_rejects_invalid_shapes_and_values(
    arguments: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        composed_action_hessian(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "keyword,value,message",
    (
        ("dimensionless_momentum_up", (1.0, 2.0, 3.0), "four values"),
        ("dimensionless_momentum_up", (1.0, 0.0, math.nan, 1.0), "finite"),
        ("refinement_relative_mode_scale", 0.0, "positive"),
        ("tolerance", math.inf, "tolerance"),
    ),
)
def test_audit_rejects_invalid_inputs(
    keyword: str, value: object, message: str
) -> None:
    arguments: dict[str, object] = {}
    arguments[keyword] = value
    with pytest.raises(ValueError, match=message):
        audit_stationary_refinement_ward_bridge(**arguments)  # type: ignore[arg-type]
