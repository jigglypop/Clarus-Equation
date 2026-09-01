from dataclasses import replace
from fractions import Fraction

import pytest

from examples.physics.qft_m2_quantum_admission_closure import (
    SOURCE_TRANSCRIPTION_SHA256,
    evaluate_m2_admission_closure_gate,
    m2_admission_closure_contract,
    matrix_l1,
    matrix_multiply,
    source_payload_sha256,
    validate_contract,
)


@pytest.fixture(scope='module')
def receipt():
    return evaluate_m2_admission_closure_gate()


def test_finite_reference_lane_is_separated_from_m2(receipt) -> None:
    assert receipt.upstream_contract_verified
    assert receipt.finite_reference_lane_passed
    assert receipt.parent_promotion_rejected
    assert not receipt.field_content_matches
    assert 'matter scalar chi' in receipt.m1_field_content
    assert 'four dimensionless' in receipt.m1_field_content
    assert receipt.missing_evidence_count == 6
    assert not receipt.m1_specific_quantum_m2_passed
    assert receipt.quantum_m2_incomplete
    assert not receipt.m3_to_m9_unlocked
    assert not receipt.model_abandoned
    assert receipt.declared_m2_admission_closure_gate_passed


def test_nilpotency_does_not_imply_positive_physical_norm(receipt) -> None:
    assert receipt.negative_norm_q_squared_residual == '0'
    assert receipt.negative_norm_cohomology_dimension == 1
    assert receipt.negative_physical_norm == '-1'
    assert receipt.nilpotency_does_not_imply_positivity
    zero_q = ((Fraction(0),),)
    assert matrix_l1(matrix_multiply(zero_q, zero_q)) == 0


def test_finite_sector_closure_does_not_fix_extensions(receipt) -> None:
    assert receipt.finite_tested_commutator_residual_l1 == '0'
    assert receipt.nonclosing_extension_commutator_l1 == '4'
    assert receipt.finite_sector_closure_does_not_imply_full_closure
    assert 'dimensionless normalized' in receipt.finite_counterexample_scope[1]
    assert 'no M1 operator regulator or anomaly' in (
        receipt.finite_counterexample_scope[3]
    )


def test_alternative_route_is_selected_without_abandonment(receipt) -> None:
    assert len(receipt.alternative_routes) == 4
    assert receipt.selected_next_route == 'M1-specific-perturbative-BV-BRST'
    assert receipt.selected_next_route in receipt.alternative_routes
    assert not receipt.actual_m1_anomaly_computed
    assert not receipt.all_quantizations_no_go_proved
    assert not receipt.positive_physical_hilbert_proved


def test_evidence_and_claim_contract_fail_closed() -> None:
    contract = m2_admission_closure_contract()
    validate_contract(contract)
    assert source_payload_sha256(contract) == SOURCE_TRANSCRIPTION_SHA256
    changes = (
        {'source_transcription_sha256': '0' * 64},
        {'m1_field_content': contract.e69_reference_field_content},
        {'required_evidence': contract.required_evidence[:-1]},
        {
            'finite_counterexample_scope': (
                *contract.finite_counterexample_scope[:-1],
                'actual M1 anomaly',
            )
        },
        {'forbidden_parent_promotions': contract.forbidden_parent_promotions[:-1]},
        {'selected_next_route': 'unsupported'},
        {'finite_reference_lane_passed': False},
        {'parent_promotion_rejected': False},
        {'m1_specific_quantum_m2_passed': True},
        {'model_abandoned': True},
        {'m3_to_m9_unlocked': True},
        {'actual_m1_anomaly_computed': True},
        {'all_quantizations_no_go_proved': True},
        {'positive_physical_hilbert_proved': True},
    )
    for change in changes:
        with pytest.raises(ValueError):
            validate_contract(replace(contract, **change))


def test_status_is_admission_closure_not_m2_proof(receipt) -> None:
    assert receipt.derivation_status == (
        'finite_reference_lane_closed_quantum_m2_incomplete'
    )
    assert len(receipt.forbidden_parent_promotions) == 4
    assert all(
        marker in ' '.join(receipt.forbidden_parent_promotions)
        for marker in ('no-go', 'Hilbert', 'quantum M2', 'all-loop')
    )
