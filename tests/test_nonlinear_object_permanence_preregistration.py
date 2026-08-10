from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "experiments" / "preregistration" / "nonlinear_object_permanence_v1.json"
)
SPEC_PATH = ROOT / "docs" / "7_AGI" / "27_Nonlinear_Object_Permanence.md"


def _config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_preregistration_is_locked_before_implementation() -> None:
    config = _config()

    assert config["schema_version"] == 1
    assert config["status"] == "locked_pre_implementation"
    assert config["registered_on"] == "2026-08-10"
    assert config["rollout_horizons"] == [1, 5, 20, 100]


def test_evaluation_seed_splits_are_nonempty_and_disjoint() -> None:
    splits = _config()["splits"]
    assert isinstance(splits, dict)

    seed_sets = []
    for name, values in splits.items():
        assert values, f"empty seed split: {name}"
        seed_set = set(values)
        assert len(seed_set) == len(values), f"duplicate seed in split: {name}"
        seed_sets.append((name, seed_set))

    for index, (left_name, left) in enumerate(seed_sets):
        for right_name, right in seed_sets[index + 1 :]:
            assert left.isdisjoint(right), f"seed leakage: {left_name} vs {right_name}"


def test_gate_thresholds_match_the_locked_spec() -> None:
    config = _config()
    g2 = config["g2_gate"]
    g3 = config["g3_gate"]
    assert isinstance(g2, dict)
    assert isinstance(g3, dict)

    assert g2["local_chart_rmse_reduction_vs_persistence_at_20"] == 0.20
    assert g2["local_chart_rmse_reduction_vs_global_linear_at_20"] == 0.20
    assert g2["minimum_seed_wins_out_of_5"] == 4
    assert g2["intervention_sign_accuracy_min"] == 0.90
    assert g3["hidden_rmse_reduction_vs_persistence"] == 0.25
    assert g3["reappearance_error_reduction_vs_monolithic"] == 0.10
    assert g3["identity_switch_rate_max"] == 0.05


def test_human_spec_points_to_the_machine_readable_contract() -> None:
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "nonlinear_object_permanence_v1.json" in spec
    assert "PRE-IMPLEMENTATION / LOCKED V1" in spec
    assert "test를 보고 임계값" in spec
