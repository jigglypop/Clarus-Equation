from dataclasses import replace

import numpy as np
import pytest

from examples.physics.qft_reference_flrw_brst_one_loop_admission import (
    ABS_TOL,
    audit_breaking,
    audit_complex,
    evaluate_finite_one_loop_st_admission_gate,
    finite_one_loop_st_contract,
    finite_st_complex,
    validate_contract,
)


def test_finite_st_admission_distinguishes_exact_anomaly_and_open_breaking() -> None:
    receipt = evaluate_finite_one_loop_st_admission_gate()

    assert receipt.loop_order_label == 1
    assert 'reference scales' in receipt.coefficient_normalization
    assert receipt.ranks == (1, 1, 1)
    assert receipt.cohomology_dimensions == (0, 1, 1, 0)
    assert receipt.nilpotency_minus_one_residual < ABS_TOL
    assert receipt.nilpotency_zero_residual < ABS_TOL

    assert receipt.exact_breaking.closed
    assert receipt.exact_breaking.removable
    assert receipt.exact_breaking.renormalized_breaking_norm < ABS_TOL
    assert np.allclose(receipt.exact_breaking.counterterm, (0.0, 3.0 / 8.0, 0.0))

    assert receipt.anomaly_control.closed
    assert not receipt.anomaly_control.removable
    assert receipt.anomaly_control.image_distance > 0.9
    assert not receipt.nonclosed_control.closed
    assert not receipt.nonclosed_control.removable
    assert receipt.nonclosed_control.closure_residual > 0.9

    assert receipt.wrong_counterterm_sign_residual > 1.0e-2
    assert receipt.nonnilpotent_control_residual > 1.0e-4
    assert receipt.nonnilpotent_control_to_tolerance_ratio > 1.0e6
    assert receipt.basis_change_classification_invariant
    assert receipt.basis_nonzero_quotient_coordinate == pytest.approx(0.60)
    assert receipt.basis_covariance_residual < 1.0e-10
    assert receipt.minimum_retained_singular_to_threshold_ratio > 1.0e6
    assert receipt.rank_tolerance_sweep_invariant
    assert receipt.rank_ambiguity_control_detected
    assert receipt.declared_finite_one_loop_st_admission_gate_passed


def test_contract_fails_closed_on_any_unsupported_one_loop_promotion() -> None:
    contract = finite_one_loop_st_contract()
    validate_contract(contract)

    for field in (
        'breaking_derived_from_loop_integral',
        'uv_regulator_supplied',
        'continuum_counterterm_basis_complete',
        'local_counterterm_coefficients_computed',
        'regulator_independence_computed',
        'continuum_local_brst_cohomology_computed',
        'bv_measure_laplacian_computed',
        'ctp_doubling_computed',
        'positive_physical_hilbert_computed',
        'nonperturbative_m2_passed',
    ):
        with pytest.raises(ValueError, match='unsupported'):
            validate_contract(replace(contract, **{field: True}))

    with pytest.raises(ValueError, match='provenance'):
        validate_contract(replace(contract, breaking_provenance=''))
    with pytest.raises(ValueError, match='computed loop coefficients'):
        validate_contract(
            replace(
                contract,
                operator_coefficient_status=(
                    'computed', *contract.operator_coefficient_status[1:]
                ),
            )
        )


def test_complex_and_breaking_inputs_fail_closed() -> None:
    complex_ = finite_st_complex()
    malformed_b_one = complex_.b_one.copy()
    malformed_b_one[0, 0] = 1.0e-3
    malformed = replace(complex_, b_one=malformed_b_one)
    malformed_audit = audit_complex(malformed)

    assert malformed_audit.nilpotency_zero_residual > 1.0e-4
    with pytest.raises(ValueError, match='ghost-number-one'):
        audit_breaking(complex_, np.zeros(2))
    with pytest.raises(ValueError, match='finite'):
        audit_breaking(complex_, np.array([0.0, np.nan, 0.0]))
