"""Source integrity, distance integrals, GLS and nuisance-accounting checks."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parents[1]/"verify"
saved = sys.path[:]
try:
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("dimension_bao_comparison", HERE/"dimension_bao_comparison.py")
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
finally:
    sys.path[:] = saved


def test_pinned_sources_and_no_duplicate_dr12_overlap_bin():
    blocks, sources = model.load_blocks()
    assert len(sources) == 6
    assert sum(len(b["z"]) for b in blocks) == 8
    assert .61 not in blocks[0]["z"]
    assert any(abs(b["cov"][0, 1]) > 0 for b in blocks)


def test_source_tampering_fails_closed(tmp_path):
    first = model.ASSETS[0][1]
    (tmp_path/first).write_text("modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        model.load_blocks(tmp_path)


def test_distance_matches_exact_matter_only_limit():
    z = np.array([.38, .38, 1.48, 1.48])
    kinds = ["DM_over_rs", "DH_over_rs"]*2
    computed = model.distance_shape(z, kinds, 1.)
    expected = [2*(1-1/np.sqrt(1+.38)), (1+.38)**-1.5,
                2*(1-1/np.sqrt(1+1.48)), (1+1.48)**-1.5]
    assert computed == pytest.approx(expected, rel=1e-12)


def test_gls_profile_matches_independent_minimization_and_permutation():
    b = model.load_blocks()[0][0]
    v = model.distance_shape(b["z"], b["kind"], .315)
    result = model.score(b["y"], b["cov"], v)
    def objective(a):
        r = b["y"]-a*v
        return r@np.linalg.solve(b["cov"], r)
    fit = minimize_scalar(objective, bracket=(29., 31.))
    assert result["amplitude_c_over_H0_rd"] == pytest.approx(fit.x, rel=1e-8)
    assert result["chi2"] == pytest.approx(fit.fun, rel=1e-10)
    perm = [2, 0, 3, 1]
    other = model.score(b["y"][perm], b["cov"][np.ix_(perm, perm)], v[perm])
    assert other["chi2"] == pytest.approx(result["chi2"], rel=1e-12)
    assert result["fitted_nuisance_count"] == 1
    assert result["whitened_rmse"]**2 == pytest.approx(result["chi2"]/4)


def test_report_does_not_turn_partial_bao_gain_into_joint_success():
    result = model.report()
    fixed = [r for r in result["results"] if r["mode"] == "fixed_H0_rd"]
    changes = [r["comparisons"][1]["delta_chi2_to_GR"] for r in fixed]
    assert changes[0] > 0 and changes[1] > 0 and changes[2] < 0 and changes[3] > 0
    for r in result["results"]:
        assert r["comparisons"][2]["cassini_nominal_band_compatible"] is False
    assert result["quantum_rmse"] is None and result["all_domain_rmse"] is None
    assert result["scientific_success"] is False
