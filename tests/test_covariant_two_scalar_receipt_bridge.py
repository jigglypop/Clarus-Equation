from __future__ import annotations

import numpy as np
import pytest

from examples.physics.covariant_two_scalar_receipt_bridge import (
    certify_allocation_nonidentifiability,
    two_scalar_exchange_receipt,
)


MINKOWSKI = (
    (-1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
REST_OBSERVER = (1.0, 0.0, 0.0, 0.0)


def _on_shell_inputs(coupling: float = 0.3) -> dict[str, object]:
    phi = 0.8
    psi = -0.4
    mass_phi = 1.2
    mass_psi = 0.7
    interaction_d_phi = coupling * phi * psi * psi
    interaction_d_psi = coupling * phi * phi * psi
    return {
        "metric_covariant": MINKOWSKI,
        "observer_contravariant": REST_OBSERVER,
        "phi": phi,
        "psi": psi,
        "gradient_phi_covector": (0.5, 0.0, 0.0, 0.0),
        "gradient_psi_covector": (-0.6, 0.0, 0.0, 0.0),
        "box_phi": mass_phi * mass_phi * phi + interaction_d_phi,
        "box_psi": mass_psi * mass_psi * psi + interaction_d_psi,
        "mass_phi": mass_phi,
        "mass_psi": mass_psi,
        "coupling": coupling,
        "reference_mass_scale": 1.0,
    }


def test_on_shell_split_current_closes_total_ward_identity() -> None:
    receipt = two_scalar_exchange_receipt(
        **_on_shell_inputs(),
        allocation_fraction=0.37,
    )

    assert receipt.interaction_energy_density == pytest.approx(0.01536)
    assert receipt.exchange_current_phi_covector == pytest.approx(
        (-0.0049536, 0.0, 0.0, 0.0)
    )
    assert receipt.exchange_current_psi_covector == pytest.approx(
        (0.0049536, 0.0, 0.0, 0.0)
    )
    assert receipt.phi_sector_divergence_covector == pytest.approx(
        receipt.exchange_current_phi_covector
    )
    assert receipt.psi_sector_divergence_covector == pytest.approx(
        receipt.exchange_current_psi_covector
    )
    assert receipt.dimensionless_eom_residual < 1.0e-12
    assert receipt.dimensionless_interaction_allocation_residual < 1.0e-12
    assert receipt.dimensionless_total_divergence < 1.0e-12
    assert receipt.dimensionless_ward_identity_residual < 1.0e-12
    assert receipt.dimensionless_complementarity_residual < 1.0e-12
    assert receipt.metric_signature == (-1, 1, 1, 1)
    assert receipt.dimensions_pass
    assert receipt.interaction_energy_counted_once
    assert receipt.field_mass_dimension == 1
    assert receipt.interaction_mass_dimension == 4
    assert receipt.current_mass_dimension == 5
    assert receipt.normalized_residual_mass_dimension == 0
    assert receipt.on_shell_within_tolerance
    assert receipt.total_stress_conserved_on_shell
    assert receipt.covariant_action_exchange_current_derived
    assert not receipt.interaction_allocation_dynamically_selected
    assert not receipt.domino_receipt_to_action_derived
    assert not receipt.covariant_matching_current_derived
    assert not receipt.record_to_gravity_source_derived


def test_same_action_and_interaction_density_do_not_select_a_unique_current() -> None:
    certificate = certify_allocation_nonidentifiability(**_on_shell_inputs())

    assert certificate.alpha_zero_receipt.interaction_energy_density == pytest.approx(
        certificate.alpha_one_receipt.interaction_energy_density
    )
    assert certificate.alpha_zero_receipt.exchange_current_phi_covector == pytest.approx(
        (0.0192, 0.0, 0.0, 0.0)
    )
    assert certificate.alpha_one_receipt.exchange_current_phi_covector == pytest.approx(
        (-0.04608, 0.0, 0.0, 0.0)
    )
    assert certificate.dimensionless_interaction_density_difference == pytest.approx(0.0)
    assert certificate.dimensionless_current_difference == pytest.approx(0.06528)
    assert (
        certificate.dimensionless_total_interaction_allocation_difference
        == pytest.approx(0.0)
    )
    assert certificate.same_action_and_interaction_density
    assert certificate.currents_distinct
    assert certificate.total_stress_alpha_invariant
    assert certificate.unique_current_claim_refuted
    assert certificate.supplied_allocation_required
    assert not certificate.domino_receipt_to_action_derived
    assert not certificate.physical_source_derived


def test_zero_coupling_recovers_decoupled_exchange_limit() -> None:
    receipt = two_scalar_exchange_receipt(
        **_on_shell_inputs(coupling=0.0),
        allocation_fraction=0.61,
    )

    assert receipt.interaction_energy_density == 0.0
    assert receipt.exchange_current_phi_covector == (0.0, 0.0, 0.0, 0.0)
    assert receipt.exchange_current_psi_covector == (0.0, 0.0, 0.0, 0.0)
    assert receipt.zero_coupling_exchange_vanishes
    assert receipt.total_stress_conserved_on_shell


def test_off_shell_input_keeps_ward_identity_but_not_conservation_claim() -> None:
    inputs = _on_shell_inputs()
    inputs["box_phi"] = float(inputs["box_phi"]) + 0.1
    receipt = two_scalar_exchange_receipt(
        **inputs,
        allocation_fraction=0.37,
    )

    assert receipt.dimensionless_eom_residual == pytest.approx(0.1)
    assert receipt.dimensionless_ward_identity_residual < 1.0e-12
    assert receipt.dimensionless_total_divergence > 0.0
    assert not receipt.on_shell_within_tolerance
    assert not receipt.total_stress_conserved_on_shell
    with pytest.raises(ValueError, match="on-shell"):
        certify_allocation_nonidentifiability(**inputs)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("allocation_fraction", 1.1, "allocation_fraction"),
        ("coupling", -0.1, "coupling"),
        ("observer_contravariant", (2.0, 0.0, 0.0, 0.0), "unit timelike"),
        ("metric_covariant", np.eye(4), "Lorentzian signature"),
    ],
)
def test_invalid_action_contract_fails_closed(
    field: str,
    value: object,
    message: str,
) -> None:
    inputs = _on_shell_inputs()
    inputs[field] = value
    inputs.setdefault("allocation_fraction", 0.5)

    with pytest.raises((ValueError, ArithmeticError), match=message):
        two_scalar_exchange_receipt(**inputs)
