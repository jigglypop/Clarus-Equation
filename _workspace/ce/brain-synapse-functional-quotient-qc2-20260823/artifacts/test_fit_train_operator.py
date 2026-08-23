from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


MODULE_PATH = Path(__file__).with_name("fit_train_operator.py")
SPEC = importlib.util.spec_from_file_location("fit_train_operator", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
fit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fit
SPEC.loader.exec_module(fit)


def test_exact_outer_fold_salt_and_group_integrity():
    groups = np.asarray(["a", "a", "b", "c", "d", "e", "f"])
    folds = fit.fold_assignments(groups, fit.OUTER_FOLD_SALT, fit.FOLDS)
    assert fit.OUTER_FOLD_SALT == "BA-SRM3-OUTER-FOLD-V1:"
    assert folds[0] == folds[1]
    expected = fit.MODEL.deterministic_fold(
        "a", "BA-SRM3-OUTER-FOLD-V1:", 5
    )
    assert folds[0] == expected


def test_theta_selection_recovers_valid_candidate_on_grouped_toy():
    rng = np.random.default_rng(21)
    groups = np.asarray([f"g{i}" for i in range(25) for _ in range(2)])
    numeric = rng.normal(size=(50, 6))
    categorical = np.asarray([["a" if i % 2 else "b"] for i in range(50)])
    target = np.column_stack(
        [numeric[:, 0] + 0.05 * rng.normal(size=50) + offset for offset in range(4)]
    )
    selected, table, receipt = fit.select_theta_cv(
        numeric,
        categorical,
        target,
        groups,
        dimensions=(2, 4),
        ells=(1.0,),
        ridges=(1e-2,),
    )
    assert selected["dimension"] in (2, 4)
    assert len(table) == 2
    assert receipt["fold_salt"] == fit.OUTER_FOLD_SALT
    prediction = fit.crossfit_fixed_theta(
        numeric, categorical, target, groups, selected
    )
    assert prediction.shape == target.shape
    assert np.all(np.isfinite(prediction))


def test_gamma_uses_raw_residual_covariance_and_is_deterministic():
    rng = np.random.default_rng(5)
    residual = rng.normal(size=(80, 6)) * np.arange(1, 7)[None, :]
    selected, table = fit.select_gamma(residual)
    assert selected["gamma"] in fit.GAMMAS
    assert len(table) == len(fit.GAMMAS)
    selected2, _ = fit.select_gamma(residual)
    assert selected2 == selected


def test_rank_bootstrap_seed_and_quantiles_are_reproducible():
    ratios = np.linspace(1e-5, 1e-2, 30)
    groups = np.asarray([f"s{i // 3}" for i in range(30)])
    first = fit.rank_bootstrap(ratios, groups, "ex", replicates=50)
    second = fit.rank_bootstrap(ratios, groups, "ex", replicates=50)
    assert first == second
    assert first["q025"] <= first["q500"] <= first["q975"]


def test_model_module_is_hash_pinned():
    assert len(fit.EXPECTED_MODEL_MODULE_SHA256) == 64
    assert fit.MODEL.RANK_RELATIVE_TOL == 1e-4


def test_nested_outer_fold_target_cannot_affect_its_fit():
    rng = np.random.default_rng(110)
    groups = np.asarray([f"slice-{i:03d}" for i in range(120)])
    numeric = rng.normal(size=(120, 7))
    categorical = np.asarray(
        [["a" if i % 2 else "b"] for i in range(120)], dtype=str
    )
    target = np.column_stack(
        [numeric[:, 0] + 0.2 * rng.normal(size=120) + j for j in range(4)]
    )
    outer = fit.fold_assignments(groups, fit.OUTER_FOLD_SALT, fit.FOLDS)
    assert set(outer.tolist()) == set(range(fit.FOLDS))
    base = fit.nested_outer_fit(
        numeric,
        categorical,
        target,
        groups,
        dimensions=(2,),
        ells=(1.0,),
        ridges=(1e-2,),
    )
    perturbed = target.copy()
    perturbed[outer == 0] += 10_000.0
    changed = fit.nested_outer_fit(
        numeric,
        categorical,
        perturbed,
        groups,
        dimensions=(2,),
        ells=(1.0,),
        ridges=(1e-2,),
    )
    np.testing.assert_array_equal(base[0][outer == 0], changed[0][outer == 0])
    np.testing.assert_array_equal(base[1][0], changed[1][0])
    np.testing.assert_array_equal(base[2][0], changed[2][0])
    assert base[3][0]["inner_fold_salt"] == "BA-SRM3-INNER-R-V1:0:"
    assert base[3][0]["outer_target_used_in_fit"] is False


def test_selected_dimension_cannot_silently_downgrade():
    fit.require_dimension(np.zeros((5, 4)), 4, "toy")
    with pytest.raises(fit.FitFailure, match="below selected d=5"):
        fit.require_dimension(np.zeros((5, 4)), 5, "toy")


def test_dataset_validator_rejects_missing_or_wrong_shape():
    data = {
        "numeric": np.zeros((4, 98)),
        "categorical": np.full((4, 5), "x"),
        "target": np.ones((4, 16)),
        "sequence_key": np.asarray(["a", "b", "c", "d"]),
        "slice_ext_id": np.asarray(["s1", "s2", "s3", "s4"]),
        "synapse_type": np.asarray(["ex", "ex", "in", "in"]),
        "numeric_feature_names": np.full(98, "n"),
        "categorical_feature_names": np.full(5, "c"),
        "target_names": np.full(16, "y"),
    }
    fit.validate_dataset(data)
    missing = dict(data)
    del missing["target_names"]
    with pytest.raises(fit.FitFailure, match="dataset key mismatch"):
        fit.validate_dataset(missing)
    wrong = dict(data)
    wrong["numeric"] = np.zeros((4, 97))
    with pytest.raises(fit.FitFailure, match="dataset shape mismatch"):
        fit.validate_dataset(wrong)


def test_operator_rank_gate_never_unlocks_development():
    passed = fit.operator_stage_gate(True)
    assert passed["status"] == "PASS_TRAIN_OPERATOR_RANK_GATE"
    assert passed["train_geometry_controls_unlock"] is True
    assert passed["development_unlock"] is False
    stopped = fit.operator_stage_gate(False)
    assert stopped["status"] == "STOP_TRAIN_OPERATOR_RANK"
    assert stopped["train_geometry_controls_unlock"] is False
    assert stopped["development_unlock"] is False
