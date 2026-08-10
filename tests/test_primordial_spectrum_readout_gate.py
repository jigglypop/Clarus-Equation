from examples.physics.primordial_spectrum_readout_gate import (
    D_EFF,
    OBS_AS_1E9,
    OBS_AS_SIGMA_1E9,
    inferred_geometry_exponent,
    readouts,
)


def by_name(name: str):
    return next(item for item in readouts() if item.name == name)


def test_total_fixed_point_response_is_rejected():
    item = by_name("total fixed-point response")
    assert item.status == "reject"
    assert item.as_1e9 > 7
    assert item.sigma_offset > 100


def test_unprojected_residual_drive_is_still_too_large():
    item = by_name("local residual drive")
    assert item.status == "reject"
    assert item.as_1e9 > 5
    assert item.sigma_offset > 100


def test_phase_projection_gets_scale_but_not_precision():
    item = by_name("phase projected drive")
    assert item.status == "candidate"
    assert 2.2 < item.as_1e9 < 2.4
    assert item.sigma_offset > 3


def test_integer_geometry_projection_passes_without_fitted_exponent():
    item = by_name("integer geometry projected drive")
    assert item.status == "pass"
    assert abs(item.sigma_offset) < 2


def test_effective_geometry_projection_passes_and_is_not_observed_exponent_fit():
    item = by_name("effective geometry projected drive")
    assert item.status == "pass"
    assert abs(item.sigma_offset) < 1

    gamma_eff = D_EFF / (D_EFF + 1)
    gamma_obs = inferred_geometry_exponent()
    assert abs(gamma_eff - gamma_obs) > 1e-3


def test_observational_snapshot_matches_canonical_linear_planck_conversion():
    assert OBS_AS_1E9 == 2.099
    assert OBS_AS_SIGMA_1E9 == 0.029
