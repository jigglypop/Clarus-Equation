from dataclasses import replace

import numpy as np
import pytest

from examples.physics.qft_brst_admission import (
    audit_breaking,
    audit_flat_null_gravity_sector,
    audit_physical_form,
    cohomology_dimensions,
    deformed_ward_residual,
    flat_tree_contract,
    gauge_fermion_deformation_class_residual,
    linearized_diffeomorphism_map,
    linearized_einstein_operator,
    quartet_complex,
    scalar_kinetic_gram,
    transverse_traceless_basis,
    validate_contract,
)


def test_tree_contract_is_complete_but_fails_closed_on_m2_scope() -> None:
    contract = flat_tree_contract()
    validate_contract(contract)

    assert contract.loop_order == 0
    assert contract.background_on_shell
    assert contract.linearized_tree_ward_identity_computed
    assert not contract.reference_patch_nondegenerate
    assert not contract.loop_anomaly_cohomology_computed
    assert not contract.nonperturbative_m2_passed


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('gauge_fixing_fermion', ''),
        ('loop_order', -1),
        ('eft_operator_dimension_max', 1),
        ('renormalization_scale_over_planck', 0.0),
    ],
)
def test_contract_validation_fails_closed(field: str, value: object) -> None:
    contract = replace(flat_tree_contract(), **{field: value})
    with pytest.raises(ValueError):
        validate_contract(contract)


def test_contract_rejects_an_unsupported_full_m2_flag() -> None:
    contract = replace(flat_tree_contract(), nonperturbative_m2_passed=True)
    with pytest.raises(ValueError, match='cannot pass full M2'):
        validate_contract(contract)


def test_quartet_has_one_physical_h_zero_class_and_no_h_one() -> None:
    complex_ = quartet_complex()

    assert np.linalg.norm(complex_.d_zero @ complex_.d_minus_one) < 1.0e-12
    assert cohomology_dimensions(complex_) == (0, 1, 0)
    assert gauge_fermion_deformation_class_residual(complex_, 3.75) < 1.0e-12


def test_exact_breaking_is_removable_by_declared_counterterm() -> None:
    complex_ = quartet_complex()
    audit = audit_breaking(complex_, np.array([1.0]))

    assert audit.closed
    assert audit.removable
    assert audit.counterterm_residual < 1.0e-12


def test_closed_nonexact_anomaly_fails_the_gate() -> None:
    complex_ = quartet_complex(include_anomaly=True)
    audit = audit_breaking(complex_, np.array([0.0, 1.0]))

    assert cohomology_dimensions(complex_) == (0, 1, 1)
    assert audit.closed
    assert not audit.removable
    assert audit.counterterm_residual > 0.99


def test_nilpotency_and_cohomology_do_not_imply_positive_norm() -> None:
    complex_ = quartet_complex()
    positive = audit_physical_form(complex_, np.diag([1.0, 1.0, 0.0]))
    negative = audit_physical_form(complex_, np.diag([-1.0, 1.0, 0.0]))
    non_descending = audit_physical_form(complex_, np.diag([1.0, 1.0, 1.0]))

    assert positive.descends_to_cohomology
    assert positive.positive
    assert negative.descends_to_cohomology
    assert not negative.positive
    assert not non_descending.descends_to_cohomology
    assert not non_descending.positive


def test_physical_form_rejects_a_nonnilpotent_complex() -> None:
    complex_ = quartet_complex()
    malformed = replace(complex_, d_minus_one=np.array([[0.0], [1.0], [0.0]]))
    with pytest.raises(ValueError, match='not nilpotent'):
        audit_physical_form(malformed, np.diag([1.0, 1.0, 0.0]))


def test_linearized_einstein_map_has_exact_null_gauge_image() -> None:
    momentum = np.array([1.0, 0.0, 0.0, 1.0])
    equation = linearized_einstein_operator(momentum)
    gauge = linearized_diffeomorphism_map(momentum)

    assert equation.shape == (10, 10)
    assert gauge.shape == (10, 4)
    assert np.linalg.norm(equation @ gauge) < 1.0e-12


def test_flat_null_sector_has_two_positive_tt_modes_and_five_scalars() -> None:
    audit = audit_flat_null_gravity_sector()

    assert audit.equation_rank == 4
    assert audit.solution_dimension == 6
    assert audit.gauge_rank == 4
    assert audit.quotient_dimension == 2
    assert audit.tt_equation_residual < 1.0e-12
    assert audit.tt_gauge_overlap_residual < 1.0e-12
    assert np.allclose(audit.tt_gram_eigenvalues, (1.0, 1.0))
    assert np.allclose(audit.scalar_gram_eigenvalues, np.ones(5))
    assert audit.five_scalar_modes_positive
    assert audit.total_free_physical_mode_count == 7
    assert audit.tree_gate_passed
    assert not audit.reference_patch_nondegenerate
    assert not audit.loop_anomaly_cohomology_computed
    assert not audit.nonperturbative_m2_passed


def test_tt_basis_is_explicitly_two_dimensional() -> None:
    tt = transverse_traceless_basis()

    assert tt.shape == (10, 2)
    assert np.linalg.matrix_rank(tt) == 2
    with pytest.raises(ValueError, match='null momentum'):
        transverse_traceless_basis(np.array([1.0, 0.0, 0.0, 0.0]))


def test_rotated_null_momentum_gets_its_own_tt_frame() -> None:
    momentum = np.array([1.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0])
    audit = audit_flat_null_gravity_sector(momentum, mu_x_over_k_ref=0.25)

    assert audit.tree_gate_passed
    assert audit.tt_equation_residual < 1.0e-12
    assert audit.tt_gauge_overlap_residual < 1.0e-12
    assert np.allclose(
        audit.scalar_gram_eigenvalues,
        (0.25**2, 0.25**2, 0.25**2, 0.25**2, 1.0),
    )


def test_scalar_kinetic_gram_fails_at_the_degenerate_mu_x_limit() -> None:
    assert np.all(np.linalg.eigvalsh(scalar_kinetic_gram(0.1)) > 0.0)
    with pytest.raises(ValueError, match='positive'):
        scalar_kinetic_gram(0.0)


def test_small_noninvariant_deformation_breaks_the_tree_ward_identity() -> None:
    assert deformed_ward_residual(0.0) < 1.0e-12
    assert deformed_ward_residual(1.0e-3) > 1.0e-3


def test_massless_audit_rejects_nonnul_or_zero_momentum() -> None:
    with pytest.raises(ValueError, match='null momentum'):
        audit_flat_null_gravity_sector(np.array([1.0, 0.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match='nonzero'):
        audit_flat_null_gravity_sector(np.zeros(4))
