from __future__ import annotations

import math

import numpy as np
import pytest

from examples.physics.finite_barrier_mode_leakage import exact_rectangular_barrier_transmission_probability
from examples.physics.single_energy_barrier_instrument import certify_single_energy_barrier_instrument


def _certificate(**overrides: object):
    arguments: dict[str, object] = dict(nonrelativistic_mass=1.0, barrier_height=8.0, incident_energy=2.0, barrier_width=2.0)
    arguments.update(overrides)
    return certify_single_energy_barrier_instrument(**arguments)


def test_spot_amplitudes_and_e16_transmission_agree() -> None:
    certificate = _certificate()
    assert certificate.k == pytest.approx(2.0)
    assert certificate.kappa == pytest.approx(math.sqrt(12.0))
    assert certificate.transmission_probability == pytest.approx(
        exact_rectangular_barrier_transmission_probability(nonrelativistic_mass=1.0, barrier_height=8.0, incident_energy=2.0, barrier_width=2.0)
    )
    assert certificate.transmission_e16_residual < 1.0e-12
    assert abs(certificate.transmission_amplitude) ** 2 == pytest.approx(certificate.transmission_probability)
    assert certificate.reflection_probability_residual < 1.0e-12
    assert certificate.transmission_amplitude == pytest.approx(0.0014696395355568164 - 0.0008484951524735881j)
    assert certificate.reflection_amplitude == pytest.approx(-0.4999985601078058 - 0.8660245724606968j)
    assert certificate.transmission_probability == pytest.approx(2.879784388242834e-6)
    assert certificate.reflection_amplitude / certificate.transmission_amplitude == pytest.approx(-589.2768610995332j)


def test_scattering_identities_and_port_instrument_for_arbitrary_density_matrix() -> None:
    rho = np.array(((0.4, 0.3 + 0.1j), (0.3 - 0.1j, 0.6)))
    certificate = _certificate(rho=rho)
    assert certificate.coefficient_identity_residual < 1.0e-12
    assert certificate.cross_amplitude_residual < 1.0e-12
    assert certificate.scattering_unitarity_residual < 1.0e-12
    assert certificate.kraus_completeness_residual < 1.0e-12
    assert certificate.record_isometry.shape == (4, 2)
    assert certificate.record_isometry_residual < 1.0e-12
    assert certificate.choi_minimum_eigenvalue >= -1.0e-12
    assert certificate.output_trace_residual < 1.0e-12
    assert certificate.output_minimum_eigenvalue >= -1.0e-12
    assert sum(certificate.output_port_probabilities) == pytest.approx(1.0)
    assert certificate.nonselective_energy_residual == pytest.approx(0.0)
    assert certificate.energy_intertwining_residual < 1.0e-12
    assert certificate.nonselective_shell_energy_residual < 1.0e-12
    assert certificate.isometric_shell_energy_residual < 1.0e-12
    assert certificate.input_shell_energy_expectation == pytest.approx(certificate.nonselective_output_shell_energy_expectation)
    assert certificate.input_shell_energy_expectation == pytest.approx(certificate.isometric_total_output_energy_expectation)
    assert certificate.final_shell_hamiltonian.shape == (4, 4)


def test_left_only_input_has_only_the_stated_reflection_transmission_labels() -> None:
    certificate = _certificate()
    assert certificate.left_input_port0_reflection_probability == pytest.approx(certificate.reflection_probability)
    assert certificate.left_input_port1_transmission_probability == pytest.approx(certificate.transmission_probability)
    assert abs(certificate.conventional_coordinate_transmission_amplitude) == pytest.approx(abs(certificate.transmission_amplitude))


def test_deep_finite_barrier_preserves_log_semantics_under_display_underflow() -> None:
    certificate = _certificate(barrier_width=1.0e6)
    assert math.isfinite(certificate.log_transmission_probability)
    assert certificate.transmission_probability == 0.0
    assert certificate.transmission_probability_numerically_underflowed
    assert certificate.reflection_probability == pytest.approx(1.0)


def test_subnormal_transmission_is_not_mislabelled_as_underflow() -> None:
    # For this supplied barrier log(T) is near -720: it is subnormal but nonzero.
    certificate = _certificate(barrier_width=104.0)
    assert certificate.log_transmission_probability == pytest.approx(-720.0, abs=5.0)
    assert certificate.transmission_probability > 0.0
    assert not certificate.transmission_probability_numerically_underflowed


def test_dimensions_scope_and_no_hopping_relation() -> None:
    certificate = _certificate()
    assert certificate.dimensions_pass
    assert certificate.width_mass_dimension == -1
    assert certificate.wavenumber_mass_dimension == 1
    assert certificate.kappa_mass_dimension == 1
    assert certificate.dimensionless_barrier_width_mass_dimension == 0
    assert not hasattr(certificate, "ideal_projected_hopping")
    assert all((
        certificate.conditional_single_energy_scattering_unitarity,
        certificate.output_port_cptp_instrument,
        certificate.prepared_record_isometry,
        certificate.elastic_degenerate_energy_bookkeeping,
        certificate.one_sided_port_label_statement,
    ))
    assert not any((
        certificate.physical_observation_or_selection_derived,
        certificate.general_reflection_transmission_labels_derived,
        certificate.wavepacket_or_energy_spread_derived,
        certificate.autonomous_detector_derived,
        certificate.durable_record_reset_or_battery_derived,
        certificate.physical_non_degenerate_record_energy_receipt_derived,
        certificate.repeated_fresh_ancilla_cptp_derived,
        certificate.causal_front_derived, certificate.qft_or_gr_derived,
        certificate.e17_j_transmission_relation_derived,
        certificate.residual_prediction_derived, certificate.gates_3_to_8_closed,
    ))


@pytest.mark.parametrize("field,value", [
    ("nonrelativistic_mass", 0.0), ("barrier_height", 0.0),
    ("incident_energy", 8.0), ("barrier_width", 0.0),
    ("rho", np.eye(3)), ("rho", np.array(((1.0, 1.0), (0.0, 0.0)))),
    ("rho", np.array(((1.2, 0.0), (0.0, -0.2)))),
])
def test_invalid_domain_or_density_matrix_fails_closed(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        _certificate(**{field: value})
