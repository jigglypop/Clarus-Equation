import math

import pytest

from reality_stone.clarus.thin_shell_defect_reality import (
    C,
    G,
    audit_static_schwarzschild_thin_shell,
    audit_minimal_elastic_defect,
    audit_quantum_negative_layer,
    audit_relaxed_internal_mode,
    audit_floquet_radial_control,
    barotropic_radial_stability,
)


def test_flat_exterior_one_metre_shell_has_exact_junction_stress() -> None:
    audit = audit_static_schwarzschild_thin_shell()

    expected_sigma = -(C**4) / (2.0 * math.pi * G)
    assert math.isclose(audit.surface_energy_j_m2, expected_sigma, rel_tol=1e-14)
    assert math.isclose(
        audit.tangential_pressure_n_m,
        abs(expected_sigma) / 2.0,
        rel_tol=1e-14,
    )
    assert audit.schwarzschild_mass_kg == 0.0
    assert 450.0 < audit.shell_mass_earth < 453.0


@pytest.mark.parametrize("lapse", [1.0, 0.1, 1.0e-6, 1.0e-12])
def test_scale_free_conformal_edge_eos_never_matches(lapse: float) -> None:
    audit = audit_static_schwarzschild_thin_shell(lapse=lapse)

    assert audit.surface_energy_j_m2 < 0.0
    assert audit.tangential_pressure_n_m > 0.0
    assert audit.conformal_edge_pressure_n_m < 0.0
    assert not audit.conformal_eos_match
    assert not audit.reality_pass


def test_near_horizon_reduces_energy_but_diverges_pressure_ratio() -> None:
    flat = audit_static_schwarzschild_thin_shell(lapse=1.0)
    near = audit_static_schwarzschild_thin_shell(lapse=1.0e-12)

    assert math.isclose(
        abs(near.surface_energy_j_m2) / abs(flat.surface_energy_j_m2),
        1.0e-6,
        rel_tol=1e-14,
    )
    assert near.pressure_to_abs_energy_ratio > 2.4e11
    assert near.tangential_pressure_n_m > flat.tangential_pressure_n_m


def test_required_edge_degrees_are_gravitationally_large() -> None:
    audit = audit_static_schwarzschild_thin_shell()

    assert 6.0e68 < audit.required_effective_degrees < 6.2e68
    assert 0.39 < audit.species_cutoff_to_radius < 0.41


@pytest.mark.parametrize("lapse", [1.0e-6, 0.1, 0.2, 1.0 / 3.0, 0.5, 1.0])
@pytest.mark.parametrize("eta", [0.0, 0.25, 0.5, 1.0])
def test_no_causal_barotropic_sound_speed_is_radially_stable(
    lapse: float,
    eta: float,
) -> None:
    audit = barotropic_radial_stability(lapse, eta)

    assert audit.causal_gradient_stable
    assert not audit.radially_stable
    assert not audit.causal_stable_overlap_exists


def test_radial_stability_requires_pathological_sound_response() -> None:
    near_horizon = barotropic_radial_stability(0.1, 4.0)
    far = barotropic_radial_stability(0.5, -2.0)

    assert near_horizon.radially_stable
    assert not near_horizon.causal_gradient_stable
    assert near_horizon.stability_threshold is not None
    assert near_horizon.stability_threshold > 1.0

    assert far.radially_stable
    assert not far.causal_gradient_stable
    assert far.stability_threshold is not None
    assert far.stability_threshold < 0.0


@pytest.mark.parametrize("shear_speed", [0.0, 0.25, 1.0])
def test_causal_shear_modulus_cannot_change_spherical_instability(
    shear_speed: float,
) -> None:
    audit = audit_minimal_elastic_defect(
        lapse=0.5,
        bulk_sound_speed_squared=1.0,
        shear_sound_speed_squared=shear_speed,
    )

    assert audit.spherical_mode_shear_strain_norm == 0.0
    assert not audit.shear_changes_radial_mode
    assert audit.elastic_characteristics_causal
    assert not audit.radial_mode_stable
    assert not audit.minimal_elastic_reality_pass


def test_negative_tension_brane_match_is_ghostly_and_radially_unstable() -> None:
    audit = audit_minimal_elastic_defect(lapse=1.0 / 3.0)

    assert audit.negative_tension_eos_match
    assert audit.negative_tension_bending_kinetic_sign == -1
    assert audit.negative_tension_bending_ghost
    assert math.isclose(
        audit.radial_potential_curvature_times_radius_squared,
        -2.0,
        rel_tol=1e-14,
    )
    assert not audit.radial_mode_stable


def test_one_species_quantum_layer_is_forced_to_extreme_uv_scale() -> None:
    audit = audit_quantum_negative_layer()

    assert 2.4e-24 < audit.maximum_negative_layer_thickness_m < 2.6e-24
    assert 7.5e16 < audit.ultraviolet_energy_ev < 8.5e16
    assert audit.sampling_time_s < 1.0e-32
    assert not audit.flat_space_qei_is_direct_boundary_proof
    assert audit.boundary_completion_required
    assert not audit.one_layer_reality_pass


def test_species_relaxes_qei_thickness_only_by_cube_root() -> None:
    one = audit_quantum_negative_layer()
    trillion = audit_quantum_negative_layer(effective_species=1.0e12)

    assert math.isclose(
        trillion.maximum_negative_layer_thickness_m
        / one.maximum_negative_layer_thickness_m,
        1.0e4,
        rel_tol=1e-14,
    )


@pytest.mark.parametrize("mixing", [0.0, 0.1, 1.0, 10.0])
def test_passive_stable_internal_mode_cannot_cure_radial_tachyon(
    mixing: float,
) -> None:
    audit = audit_relaxed_internal_mode(
        bare_radial_curvature=-2.0,
        internal_mode_curvature=3.0,
        mixing=mixing,
    )

    assert audit.relaxed_effective_radial_curvature <= -2.0
    assert not audit.passive_mixing_stiffens_radial_mode
    assert not audit.radially_stable_after_relaxation
    assert audit.active_or_nonadiabatic_control_required


def test_direct_stiffness_not_mixing_is_what_can_close_radial_gate() -> None:
    marginal = audit_relaxed_internal_mode(-2.0, 4.0, 2.0, direct_radial_stiffness=3.0)
    stable = audit_relaxed_internal_mode(-2.0, 4.0, 2.0, direct_radial_stiffness=3.1)

    assert marginal.relaxed_effective_radial_curvature == 0.0
    assert math.isclose(marginal.minimum_direct_stiffness_required, 3.0)
    assert not marginal.radially_stable_after_relaxation
    assert stable.radially_stable_after_relaxation


def test_floquet_monodromy_preserves_phase_space_volume() -> None:
    audit = audit_floquet_radial_control(0.05, 0.1)

    assert math.isclose(audit.monodromy_determinant, 1.0, rel_tol=1e-10)


def test_fast_drive_can_stabilize_but_is_not_a_static_source() -> None:
    audit = audit_floquet_radial_control(0.05, 0.1)

    assert audit.averaged_curvature > 0.0
    assert audit.exact_floquet_stable
    assert audit.high_frequency_control_regime
    assert audit.control_pass
    assert audit.drive_is_continuously_required
    assert not audit.stable_after_drive_loss
    assert not audit.supplies_static_negative_stress


def test_drive_below_averaged_threshold_does_not_stabilize() -> None:
    audit = audit_floquet_radial_control(0.05, 0.05)

    assert audit.averaged_curvature < 0.0
    assert not audit.exact_floquet_stable
    assert not audit.control_pass


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"radius_m": 0.0}, "radius_m"),
        ({"lapse": 0.0}, "lapse"),
        ({"lapse": 1.1}, "lapse"),
        ({"casimir_coefficient": 0.0}, "casimir_coefficient"),
    ],
)
def test_invalid_inputs_are_rejected(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        audit_static_schwarzschild_thin_shell(**kwargs)
