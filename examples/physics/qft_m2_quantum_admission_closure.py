'''Formal M2 admission closure after the finite E54--E69 evidence lane.

This gate separates a passed finite reference reconstruction from the missing
M1-specific quantum constraint algebra and positive physical Hilbert space.
It includes two finite logical counterexamples to invalid parent implications.
It does not calculate an M1 anomaly or prove a no-go for all quantizations.
'''

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib

from examples.physics.qft_reference_flrw_operator_trace_synthesis import (
    SOURCE_TRANSCRIPTION_SHA256 as E69_I_HASH,
    operator_trace_synthesis_contract,
    validate_contract as validate_e69_i_contract,
)


M2_TARGET = (
    '[C_A[f],C_B[g]]psi=i*hbar*C_C[f^C_AB(f,g)]psi on one common '
    'dense invariant domain after regulator removal; nontrivial positive '
    'completed physical inner product required'
)
M1_FIELD_CONTENT = (
    'Einstein gravity plus one matter scalar chi and four dimensionless '
    'Klein-Gordon reference scalars X^0,...,X^3'
)
E69_REFERENCE_FIELD_CONTENT = (
    'Einstein-Hilbert gravity plus one minimally coupled massless quantum scalar'
)
REQUIRED_EVIDENCE = (
    'common dense invariant domain for all regulated constraints and products',
    'fixed regulator ordering adjoints and structure-function prescription',
    'all-smearing regulator-removal commutator identity in a stated topology',
    'nontrivial joint kernel quotient rigging map or reduced physical sector',
    'positive completed physical inner product and observable adjoints',
    'M1-specific BRST-BV transformations regulator and ST-QME breaking',
)
EVIDENCE_STATUS = (
    ('classical ADM HDA', 'passed-E54-classical-only'),
    ('common quantum domain', 'missing'),
    ('M1 quantum constraint operators', 'missing'),
    ('regulator-removal commutator', 'missing'),
    ('M1-specific one-loop ST-QME', 'missing'),
    ('nontrivial physical kernel or rigging map', 'missing'),
    ('positive physical inner product', 'missing'),
    ('finite E69 reference reconstruction', 'passed-reference-only'),
)
ALTERNATIVE_ROUTES = (
    'Dirac-RAQ-master-constraint',
    'M1-specific-perturbative-BV-BRST',
    'nondegenerate-four-clock-reduced-quantization',
    'discrete-refinement-continuum-restoration',
)
SELECTED_NEXT_ROUTE = 'M1-specific-perturbative-BV-BRST'
COUNTEREXAMPLE_FORMULA = (
    'Q=0 on a one-dimensional nondegenerate Hermitian form [-1] is nilpotent '
    'with H0 dimension one but has no positive-norm nonzero state; a finite '
    'zero tested block admits a nonclosing 2x2 direct-sum extension relative '
    'to the stipulated zero-RHS target with entrywise l1=4'
)
FINITE_COUNTEREXAMPLE_SCOPE = (
    'exact rational finite matrices',
    'dimensionless normalized Hermitian-form value and entrywise l1 norm',
    'zero-RHS abelian target for the direct-sum extension',
    'logical non-implications only; no M1 operator regulator or anomaly',
)
FORBIDDEN_PARENT_PROMOTIONS = (
    'finite counterexample implies a continuum or all-quantization no-go',
    'finite positive Gram matrix implies positive completed physical Hilbert space',
    'finite BRST cohomology dimension implies full quantum M2',
    'tree or finite ST identity implies all-loop ST-QME',
)
UPSTREAM_HASHES = (('E69-I', E69_I_HASH),)
SOURCE_TRANSCRIPTION_SHA256 = (
    '10bee0b3b0e1c95c4945d0a61e198bfa1c29e80f0075b2ce8a83101e79096f26'
)


@dataclass(frozen=True)
class M2AdmissionClosureContract:
    m2_target: str
    m1_field_content: str
    e69_reference_field_content: str
    required_evidence: tuple[str, ...]
    evidence_status: tuple[tuple[str, str], ...]
    alternative_routes: tuple[str, ...]
    selected_next_route: str
    counterexample_formula: str
    finite_counterexample_scope: tuple[str, ...]
    forbidden_parent_promotions: tuple[str, ...]
    upstream_hashes: tuple[tuple[str, str], ...]
    source_transcription_sha256: str
    finite_reference_lane_passed: bool
    parent_promotion_rejected: bool
    m1_specific_quantum_m2_passed: bool
    model_abandoned: bool
    m3_to_m9_unlocked: bool
    actual_m1_anomaly_computed: bool
    all_quantizations_no_go_proved: bool
    positive_physical_hilbert_proved: bool
    derivation_status: str


def m2_admission_closure_contract() -> M2AdmissionClosureContract:
    return M2AdmissionClosureContract(
        m2_target=M2_TARGET,
        m1_field_content=M1_FIELD_CONTENT,
        e69_reference_field_content=E69_REFERENCE_FIELD_CONTENT,
        required_evidence=REQUIRED_EVIDENCE,
        evidence_status=EVIDENCE_STATUS,
        alternative_routes=ALTERNATIVE_ROUTES,
        selected_next_route=SELECTED_NEXT_ROUTE,
        counterexample_formula=COUNTEREXAMPLE_FORMULA,
        finite_counterexample_scope=FINITE_COUNTEREXAMPLE_SCOPE,
        forbidden_parent_promotions=FORBIDDEN_PARENT_PROMOTIONS,
        upstream_hashes=UPSTREAM_HASHES,
        source_transcription_sha256=SOURCE_TRANSCRIPTION_SHA256,
        finite_reference_lane_passed=True,
        parent_promotion_rejected=True,
        m1_specific_quantum_m2_passed=False,
        model_abandoned=False,
        m3_to_m9_unlocked=False,
        actual_m1_anomaly_computed=False,
        all_quantizations_no_go_proved=False,
        positive_physical_hilbert_proved=False,
        derivation_status='finite_reference_lane_closed_quantum_m2_incomplete',
    )


def canonical_source_payload(contract: M2AdmissionClosureContract) -> str:
    separator = chr(44)
    evidence = separator.join(
        f'{name}:{status}' for name, status in contract.evidence_status
    )
    upstream = separator.join(
        f'{name}:{value}' for name, value in contract.upstream_hashes
    )
    return '|'.join(
        (
            f'target={contract.m2_target}',
            f'm1={contract.m1_field_content}',
            f'reference={contract.e69_reference_field_content}',
            f'required={separator.join(contract.required_evidence)}',
            f'evidence={evidence}',
            f'routes={separator.join(contract.alternative_routes)}',
            f'selected={contract.selected_next_route}',
            f'counterexamples={contract.counterexample_formula}',
            f'counterexample_scope={separator.join(contract.finite_counterexample_scope)}',
            f'forbidden_promotions={separator.join(contract.forbidden_parent_promotions)}',
            f'upstream={upstream}',
        )
    )


def source_payload_sha256(contract: M2AdmissionClosureContract) -> str:
    return hashlib.sha256(
        canonical_source_payload(contract).encode('utf-8')
    ).hexdigest()


def validate_contract(contract: M2AdmissionClosureContract) -> None:
    frozen = (
        contract.m2_target == M2_TARGET,
        contract.m1_field_content == M1_FIELD_CONTENT,
        contract.e69_reference_field_content == E69_REFERENCE_FIELD_CONTENT,
        contract.required_evidence == REQUIRED_EVIDENCE,
        contract.evidence_status == EVIDENCE_STATUS,
        contract.alternative_routes == ALTERNATIVE_ROUTES,
        contract.selected_next_route == SELECTED_NEXT_ROUTE,
        contract.counterexample_formula == COUNTEREXAMPLE_FORMULA,
        contract.finite_counterexample_scope == FINITE_COUNTEREXAMPLE_SCOPE,
        contract.forbidden_parent_promotions == FORBIDDEN_PARENT_PROMOTIONS,
        contract.upstream_hashes == UPSTREAM_HASHES,
    )
    if not all(frozen):
        raise ValueError('M2 evidence, counterexample, or route contract changed')
    if (
        contract.source_transcription_sha256 != SOURCE_TRANSCRIPTION_SHA256
        or source_payload_sha256(contract) != SOURCE_TRANSCRIPTION_SHA256
    ):
        raise ValueError('M2 admission closure hash mismatch')
    if not (
        contract.finite_reference_lane_passed
        and contract.parent_promotion_rejected
    ):
        raise ValueError('finite evidence and invalid promotion must be separated')
    unsupported = (
        contract.m1_specific_quantum_m2_passed,
        contract.model_abandoned,
        contract.m3_to_m9_unlocked,
        contract.actual_m1_anomaly_computed,
        contract.all_quantizations_no_go_proved,
        contract.positive_physical_hilbert_proved,
    )
    if any(unsupported):
        raise ValueError('unsupported M2 pass, abandonment, or no-go promotion')
    if contract.derivation_status != (
        'finite_reference_lane_closed_quantum_m2_incomplete'
    ):
        raise ValueError('this gate is an admission closure only')


def matrix_multiply(
    left: tuple[tuple[Fraction, ...], ...],
    right: tuple[tuple[Fraction, ...], ...],
) -> tuple[tuple[Fraction, ...], ...]:
    if not left or len(left[0]) != len(right):
        raise ValueError('matrix dimensions do not compose')
    return tuple(
        tuple(
            sum(
                (
                    left[row][inner] * right[inner][column]
                    for inner in range(len(right))
                ),
                Fraction(0),
            )
            for column in range(len(right[0]))
        )
        for row in range(len(left))
    )


def matrix_subtract(
    left: tuple[tuple[Fraction, ...], ...],
    right: tuple[tuple[Fraction, ...], ...],
) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(
            left[row][column] - right[row][column]
            for column in range(len(left[row]))
        )
        for row in range(len(left))
    )


def matrix_l1(matrix: tuple[tuple[Fraction, ...], ...]) -> Fraction:
    return sum(
        (abs(value) for row in matrix for value in row),
        Fraction(0),
    )


def zero_matrix(dimension: int) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(Fraction(0) for _ in range(dimension))
        for _ in range(dimension)
    )


@dataclass(frozen=True)
class M2AdmissionClosureReceipt:
    source_transcription_sha256: str
    upstream_hashes: tuple[tuple[str, str], ...]
    upstream_contract_verified: bool
    m1_field_content: str
    e69_reference_field_content: str
    field_content_matches: bool
    required_evidence: tuple[str, ...]
    evidence_status: tuple[tuple[str, str], ...]
    missing_evidence_count: int
    negative_norm_q_squared_residual: str
    negative_norm_cohomology_dimension: int
    negative_physical_norm: str
    nilpotency_does_not_imply_positivity: bool
    finite_tested_commutator_residual_l1: str
    nonclosing_extension_commutator_l1: str
    finite_sector_closure_does_not_imply_full_closure: bool
    finite_counterexample_scope: tuple[str, ...]
    forbidden_parent_promotions: tuple[str, ...]
    alternative_routes: tuple[str, ...]
    selected_next_route: str
    finite_reference_lane_passed: bool
    parent_promotion_rejected: bool
    m1_specific_quantum_m2_passed: bool
    quantum_m2_incomplete: bool
    model_abandoned: bool
    m3_to_m9_unlocked: bool
    actual_m1_anomaly_computed: bool
    all_quantizations_no_go_proved: bool
    positive_physical_hilbert_proved: bool
    derivation_status: str
    declared_m2_admission_closure_gate_passed: bool


def evaluate_m2_admission_closure_gate() -> M2AdmissionClosureReceipt:
    contract = m2_admission_closure_contract()
    validate_contract(contract)
    validate_e69_i_contract(operator_trace_synthesis_contract())

    q_operator = ((Fraction(0),),)
    q_squared = matrix_multiply(q_operator, q_operator)
    negative_norm = Fraction(-1)

    tested_a = zero_matrix(2)
    tested_b = zero_matrix(2)
    tested_commutator = matrix_subtract(
        matrix_multiply(tested_a, tested_b),
        matrix_multiply(tested_b, tested_a),
    )
    extension_a = (
        (Fraction(0), Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(0), Fraction(1)),
        (Fraction(0), Fraction(0), Fraction(1), Fraction(0)),
    )
    extension_b = (
        (Fraction(0), Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(0), Fraction(-1)),
    )
    extension_commutator = matrix_subtract(
        matrix_multiply(extension_a, extension_b),
        matrix_multiply(extension_b, extension_a),
    )
    tested_residual = matrix_l1(tested_commutator)
    extension_residual = matrix_l1(extension_commutator)
    field_content_matches = (
        contract.m1_field_content == contract.e69_reference_field_content
    )
    missing_count = sum(
        status == 'missing' for _, status in contract.evidence_status
    )
    nilpotency_counterexample = (
        matrix_l1(q_squared) == 0
        and negative_norm < 0
    )
    extension_counterexample = (
        tested_residual == 0 and extension_residual == 4
    )
    declared_passed = all(
        (
            not field_content_matches,
            missing_count == 6,
            nilpotency_counterexample,
            extension_counterexample,
            contract.finite_reference_lane_passed,
            contract.parent_promotion_rejected,
            not contract.m1_specific_quantum_m2_passed,
            not contract.model_abandoned,
            not contract.m3_to_m9_unlocked,
            contract.selected_next_route in contract.alternative_routes,
        )
    )
    return M2AdmissionClosureReceipt(
        source_transcription_sha256=contract.source_transcription_sha256,
        upstream_hashes=contract.upstream_hashes,
        upstream_contract_verified=True,
        m1_field_content=contract.m1_field_content,
        e69_reference_field_content=contract.e69_reference_field_content,
        field_content_matches=field_content_matches,
        required_evidence=contract.required_evidence,
        evidence_status=contract.evidence_status,
        missing_evidence_count=missing_count,
        negative_norm_q_squared_residual=str(matrix_l1(q_squared)),
        negative_norm_cohomology_dimension=1,
        negative_physical_norm=str(negative_norm),
        nilpotency_does_not_imply_positivity=nilpotency_counterexample,
        finite_tested_commutator_residual_l1=str(tested_residual),
        nonclosing_extension_commutator_l1=str(extension_residual),
        finite_sector_closure_does_not_imply_full_closure=(
            extension_counterexample
        ),
        finite_counterexample_scope=contract.finite_counterexample_scope,
        forbidden_parent_promotions=contract.forbidden_parent_promotions,
        alternative_routes=contract.alternative_routes,
        selected_next_route=contract.selected_next_route,
        finite_reference_lane_passed=contract.finite_reference_lane_passed,
        parent_promotion_rejected=contract.parent_promotion_rejected,
        m1_specific_quantum_m2_passed=(
            contract.m1_specific_quantum_m2_passed
        ),
        quantum_m2_incomplete=True,
        model_abandoned=contract.model_abandoned,
        m3_to_m9_unlocked=contract.m3_to_m9_unlocked,
        actual_m1_anomaly_computed=contract.actual_m1_anomaly_computed,
        all_quantizations_no_go_proved=(
            contract.all_quantizations_no_go_proved
        ),
        positive_physical_hilbert_proved=(
            contract.positive_physical_hilbert_proved
        ),
        derivation_status=contract.derivation_status,
        declared_m2_admission_closure_gate_passed=declared_passed,
    )
