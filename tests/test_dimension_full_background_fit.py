"""Independent GR limit and data-score checks for direct nonlinear profiles."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]/"verify"
saved = sys.path[:]
try:
    sys.path.insert(0,str(HERE))
    spec = importlib.util.spec_from_file_location("dimension_full_background_fit",HERE/"dimension_full_background_fit.py")
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
finally:
    sys.path[:] = saved


def test_zero_coupling_background_evaluation_matches_independent_gr_distance_fit():
    from dimension_bao_comparison import distance_shape
    d = model.combined_data()
    evaluated = model.evaluate_background(d,omega_local=.32,s=0.,eta=.1)
    reference = model.score(d["y"],d["cov"],distance_shape(d["z"],d["kind"],.32))
    assert evaluated["chi2"] == pytest.approx(reference["chi2"],abs=1e-10)
    assert evaluated["prediction"] == pytest.approx(reference["prediction"],abs=2e-10)
    assert evaluated["fitted_nuisance_count"] == 1  # only A at fixed density


def test_written_profile_scores_match_full_covariance_and_local_minimum_checks():
    r = json.loads((HERE/"dimension_full_background_fit.json").read_text())
    d = model.combined_data()
    base = r["baseline_GR_same_two_nuisances"]["chi2"]
    for candidate in r["candidates"]:
        residual = d["y"]-np.array(candidate["prediction"])
        independent = residual@np.linalg.solve(d["cov"],residual)
        assert candidate["chi2"] == pytest.approx(independent,abs=1e-11)
        assert candidate["fitted_nuisance_count"] == 2
        assert min(candidate["neighbor_chi2"]) > candidate["chi2"]
        assert abs(candidate["refinement_delta_chi2"]) < 1e-10
        delta = candidate["chi2"]-base
        assert candidate["delta_BAO_chi2_to_equally_fitted_GR"] == pytest.approx(delta,abs=1e-12)
        assert delta > 0
    assert not r["scientific_success"] and r["all_domain_rmse"] is None
