"""Conditional growth/calibration checks, not evidence of observational success."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]/"verify"
saved = sys.path[:]
try:
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("dimension_growth_bridge", HERE/"dimension_growth_bridge.py")
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
finally:
    sys.path[:] = saved


def test_zero_amplitude_matches_independent_hypergeometric_gr():
    result = model.GrowthBridge().solve(0.)
    assert np.max(np.abs(np.array(result["D_ratio_to_GR"])-1)) < 2e-9
    assert np.max(np.abs(np.array(result["Dprime_ratio_to_GR"])-1)) < 2e-9


def test_fixed_background_comparison_and_initial_amplitude():
    bridge = model.GrowthBridge()
    result = bridge.solve(.02, convention="fixed_Einstein_G")
    assert min(result["D_ratio_to_GR"]) > 1-2e-9
    assert min(result["Dprime_ratio_to_GR"]) > 1-2e-9
    assert result["today_D_ratio"] > 1.01
    assert result["initial_D_Dprime"] == bridge.solve(0.)["initial_D_Dprime"]
    assert result["today_normalization_multiplier_if_refitted"] < 1


def test_calibrated_G_changes_background_and_invalidates_fixed_H_ordering():
    bridge = model.GrowthBridge()
    result = bridge.solve(.02)
    assert result["omega_E0"] == pytest.approx(.315/1.02)
    assert result["lambda_density_fraction_input"] == pytest.approx(1-.315/1.02)
    # Heavy scalar means a negligible response; reduced Einstein gravity then
    # yields smaller D with this stated background/initial-data convention.
    heavy = model.GrowthBridge(mass_over_h0=1e6).solve(.02)
    assert heavy["today_D_ratio"] < 1
    assert heavy["minimum_k_physical_over_H"] > 10
    assert result["minimum_mass_over_H"] > 10
    assert result["maximum_tracking_abs_ln_A_estimate"] < .001
    assert result["scientific_success"] is False
    assert result["survey_fsigma8"] is None


def test_moment_and_solar_bounds_handle_unbounded_spectrum():
    bridge = model.GrowthBridge()
    x, w, _ = bridge.probe.positive_quadrature(n_u=24, n_x=32)
    d = bridge.probe.m**2+bridge.probe.a*x
    assert np.sum(w*d) == pytest.approx(bridge.mean_mass_squared, rel=1e-9)
    for radius in (.01, .5, 2.):
        y = radius*np.sqrt(d)
        assert np.sum(w*(-np.expm1(-y))) <= radius*np.sqrt(bridge.mean_mass_squared)
        assert np.sum(w*(1-(1+y)*np.exp(-y))) <= radius**2*bridge.mean_mass_squared/2
    bound = bridge.solar_and_lab_bounds(s=model.cassini_band()["maximum_s"])
    assert bound["solar_1_minus_B_upper"] < 2e-11
    assert bound["absolute_calibrated_lab_force_fraction_upper"] < 1e-52


def test_cassini_band_and_point_residual_direction():
    band = model.cassini_band()
    s = band["maximum_s"]
    prediction = -2*s/(1+s)
    assert prediction == pytest.approx(band["lower_gamma_minus_one"])
    assert s == pytest.approx(1.203973238e-5, rel=2e-7)
    assert abs((2.1e-5-prediction)/2.3e-5) > abs(2.1e-5/2.3e-5)


@pytest.mark.parametrize("kwargs", [{"s": -1}, {"s": .1, "f_cal": 2},
                                    {"s": .1, "convention": "hidden"}])
def test_invalid_conventions_rejected(kwargs):
    with pytest.raises(ValueError):
        model.GrowthBridge().solve(**kwargs)
