from __future__ import annotations

import math

import pytest

from examples.physics.finite_barrier_mode_leakage import (
    certify_finite_barrier_mode_leakage,
    exact_rectangular_barrier_transmission_probability,
    gaussian_overlap_amplitude,
)


def _certificate(**overrides: object):
    inputs: dict[str, object] = {
        "mode_count": 3,
        "sigma": 1.0,
        "center_spacing": 5.0,
        "delta_basis": 0.01,
        "nonrelativistic_mass": 1.0,
        "barrier_height": 8.0,
        "incident_energy": 2.0,
        "barrier_width": 2.0,
        "delta_leak": 0.01,
        "ideal_projected_hopping": 2.0,
        "projected_hamiltonian_norm_error": 0.001,
        "delta_dyn": 0.01,
    }
    inputs.update(overrides)
    return certify_finite_barrier_mode_leakage(**inputs)


def test_gaussian_convention_overlap_and_required_spacing_are_amplitudes() -> None:
    assert gaussian_overlap_amplitude(sigma=2.0, center_spacing=4.0) == pytest.approx(math.exp(-1.0))
    certificate = _certificate()
    assert certificate.gaussian_overlap_amplitude == pytest.approx(math.exp(-25.0 / 4.0))
    assert certificate.basis_amplitude_budget == pytest.approx(3.0 * math.exp(-25.0 / 4.0))
    assert certificate.required_center_spacing == pytest.approx(2.0 * math.sqrt(math.log(300.0)))
    assert certificate.basis_amplitude_budget_holds
    boundary = _certificate(center_spacing=certificate.required_center_spacing)
    assert boundary.basis_amplitude_budget == pytest.approx(
        boundary.basis_amplitude_target, rel=1.0e-12
    )
    deep_separation = _certificate(center_spacing=100.0)
    assert math.isfinite(deep_separation.gaussian_log_overlap_amplitude)
    assert math.isfinite(deep_separation.basis_log_amplitude_budget)
    assert deep_separation.gaussian_overlap_amplitude == 0.0
    assert deep_separation.gaussian_overlap_numerically_underflowed
    assert deep_separation.basis_amplitude_budget_holds


def test_exact_barrier_probability_and_exponential_certificate_are_separate() -> None:
    certificate = _certificate()
    kappa = math.sqrt(12.0)
    expected = 1.0 / (1.0 + 64.0 / 48.0 * math.sinh(2.0 * kappa) ** 2)
    assert exact_rectangular_barrier_transmission_probability(
        nonrelativistic_mass=1.0, barrier_height=8.0, incident_energy=2.0, barrier_width=2.0
    ) == pytest.approx(expected)
    assert certificate.barrier_transmission_probability == pytest.approx(expected)
    assert certificate.exponential_regime_holds
    assert certificate.exponential_probability_upper is not None
    assert certificate.barrier_transmission_probability <= certificate.exponential_probability_upper
    assert certificate.exact_required_barrier_width > 0.0
    assert certificate.exponential_required_barrier_width > 0.0
    exact_boundary = _certificate(barrier_width=certificate.exact_required_barrier_width)
    assert exact_boundary.barrier_probability_budget == pytest.approx(
        exact_boundary.barrier_probability_target, rel=1.0e-12
    )
    exponential_boundary = _certificate(
        barrier_width=certificate.exponential_required_barrier_width
    )
    assert exponential_boundary.exponential_regime_holds
    assert exponential_boundary.exponential_probability_upper is not None
    assert (
        3.0 * exponential_boundary.exponential_probability_upper
        <= exponential_boundary.barrier_probability_target * (1.0 + 1.0e-12)
    )


def test_below_regime_and_deep_barrier_stay_finite_in_the_log_domain() -> None:
    below_regime = _certificate(barrier_width=0.01)
    assert not below_regime.exponential_regime_holds
    assert below_regime.exponential_probability_upper is None
    assert 0.0 < below_regime.barrier_transmission_probability < 1.0
    deep = _certificate(
        barrier_width=1.0e6,
        delta_leak=1.0e-300,
    )
    assert math.isfinite(deep.barrier_log_transmission_probability)
    assert math.isfinite(deep.exact_required_barrier_width)
    assert math.isfinite(deep.exponential_required_barrier_width)
    assert deep.barrier_transmission_probability >= 0.0
    assert deep.barrier_probability_numerically_underflowed


def test_budgets_have_distinct_types_and_are_never_aggregated() -> None:
    certificate = _certificate()
    assert certificate.error_type_tuple == (
        "basis_amplitude", "barrier_probability", "projected_operator_norm"
    )
    assert certificate.basis_amplitude_budget_holds
    assert certificate.barrier_probability_budget_holds
    assert certificate.projected_operator_norm_budget_holds
    assert not hasattr(certificate, "aggregate_error")


def test_ideal_swap_and_finite_duhamel_telescoping_witness() -> None:
    exact = _certificate(projected_hamiltonian_norm_error=0.0)
    assert exact.ideal_swap_time == pytest.approx(math.pi / 4.0)
    assert exact.ideal_swap_probability == pytest.approx(1.0)
    assert exact.ideal_unitarity_residual < 1.0e-12
    assert exact.ideal_swap_phase_residual < 1.0e-12
    assert exact.single_step_operator_difference < 1.0e-12
    assert exact.repeated_step_operator_difference < 1.0e-12
    perturbed = _certificate(projected_hamiltonian_norm_error=0.01)
    assert perturbed.single_step_operator_difference <= perturbed.single_step_duhamel_bound_raw + 1.0e-11
    assert perturbed.repeated_step_operator_difference <= perturbed.repeated_step_telescoping_bound_raw + 1.0e-11
    assert perturbed.single_step_duhamel_bound_clipped <= 2.0
    assert perturbed.repeated_step_telescoping_bound_clipped <= 2.0


def test_dimensions_and_scope_flags_are_explicitly_limited() -> None:
    certificate = _certificate()
    assert certificate.dimensions_pass
    assert certificate.sigma_mass_dimension == -1
    assert certificate.energy_mass_dimension == 1
    assert certificate.kappa_mass_dimension == 1
    assert certificate.barrier_width_mass_dimension == -1
    assert certificate.transmission_probability_mass_dimension == 0
    assert certificate.identities_and_finite_witness_only
    false_physical_flags = (
        certificate.e15_modes_derived,
        certificate.kg_to_schrodinger_projection_derived,
        certificate.rectangular_barrier_represents_periodic_lattice,
        certificate.barrier_or_wkb_to_hopping_derived,
        certificate.finite_barrier_exact_localization,
        certificate.autonomous_dwell_time_derived,
        certificate.scattering_instrument_or_energy_receipt_derived,
        certificate.repeated_cptp_or_fresh_ancilla_derived,
        certificate.causal_or_strict_front_derived,
        certificate.qft_microcausality_or_no_signalling_derived,
        certificate.gr_source_derived,
        certificate.selection_derived,
        certificate.gates_5_to_8_closed,
    )
    assert not any(false_physical_flags)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mode_count", 0, "mode_count"),
        ("sigma", 0.0, "sigma"),
        ("delta_basis", 1.0, "delta_basis"),
        ("incident_energy", 8.0, "incident_energy"),
        ("projected_hamiltonian_norm_error", -0.1, "projected_hamiltonian_norm_error"),
        ("delta_dyn", 0.0, "delta_dyn"),
    ],
)
def test_invalid_contract_fails_closed(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _certificate(**{field: value})
