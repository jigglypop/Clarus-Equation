"""Focused synthetic checks for the frozen neural-geometry candidate runner."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np


def load_runner(path: Path):
    spec = importlib.util.spec_from_file_location("e17_candidate_tournament", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load tournament module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    artifact_root = Path(__file__).resolve().parents[1]
    run_root = artifact_root.parent
    parser.add_argument(
        "--runner", default=artifact_root / "e17_candidate_tournament.py", type=Path
    )
    parser.add_argument(
        "--output",
        default=Path(__file__).with_name("candidate_math_fixture_output_v2.2.json"),
        type=Path,
    )
    args = parser.parse_args()
    runner = load_runner(args.runner)

    j0 = np.array([[2.0]])
    j1 = np.array([[3.0]])
    q0 = np.array([[5.0]])
    q1 = np.array([[7.0]])
    correct_time_varying = j1 @ q0 @ j1.T + q1
    wrong_initial_product = q0 + j0 @ q1 @ j0.T
    assert np.allclose(correct_time_varying, [[52.0]])
    assert np.allclose(wrong_initial_product, [[33.0]])
    assert not np.allclose(correct_time_varying, wrong_initial_product)

    model = runner.LinearModel(
        j=np.array([[0.7, 0.1], [-0.2, 0.5]]),
        bias=np.array([0.1, -0.05]),
        q=np.array([[0.4, 0.1], [0.1, 0.3]]),
        fit_transitions=100,
    )
    horizon = runner.horizon_model(model, 5)
    direct = np.zeros((2, 2))
    power = np.eye(2)
    for _ in range(5):
        direct += power @ model.q @ power.T
        power = power @ model.j
    assert np.allclose(horizon.reachability, direct)

    covariance = np.array([[2.0, 0.3], [0.3, 1.2]])
    metric = np.linalg.inv(covariance)
    chart = np.array([[1.4, 0.2], [-0.1, 0.8]])
    delta = np.array([0.4, -0.7])
    transformed_covariance = chart @ covariance @ chart.T
    transformed_metric = np.linalg.inv(transformed_covariance)
    transformed_delta = chart @ delta
    chart_error = abs(
        float(delta @ metric @ delta)
        - float(transformed_delta @ transformed_metric @ transformed_delta)
    )
    assert chart_error < 1e-12

    low_rank_metric, optimizer = runner.fit_low_rank_precision(
        np.array([[1.0, 0.2], [0.2, 0.8]]), rank=1, penalty=0.01
    )
    low_rank_minimum = float(np.min(np.linalg.eigvalsh(low_rank_metric)))
    assert low_rank_minimum > 0
    assert optimizer["parameter_count"] == 4

    extreme_model = runner.LinearModel(
        j=np.diag([1000.0, -1000.0]),
        bias=np.zeros(2),
        q=np.eye(2),
        fit_transitions=100,
    )
    extreme_horizon = runner.horizon_model(extreme_model, 1)
    zero_ridge_extreme_code = None
    try:
        runner.deformation_metric(
            "S14",
            {"tau": 0.1, "lambda_g": 0.0},
            extreme_model,
            extreme_horizon,
            1,
        )
    except runner.CandidateFailure as error:
        zero_ridge_extreme_code = error.code
    assert zero_ridge_extreme_code == "INELIGIBLE_SINGULAR"
    stable_softplus_metric = runner.deformation_metric(
        "S14",
        {"tau": 0.1, "lambda_g": 1e-6},
        extreme_model,
        extreme_horizon,
        1,
    )
    assert np.isfinite(stable_softplus_metric).all()
    assert float(np.min(np.linalg.eigvalsh(stable_softplus_metric))) > 0

    s7_h1_status = runner.deformation_static_ineligibility("S7-H", 1)
    assert s7_h1_status == "INELIGIBLE_TAUTOLOGY"
    assert runner.deformation_static_ineligibility("S7-H", 5) is None

    field_gates = runner.condition_field_gates(
        np.array([1.0, -0.5]), np.array([0.2, 0.5, 0.8])
    )
    zero_field_key = runner.tuple_key({"lambda_g": 0.0})
    positive_field_key = runner.tuple_key({"lambda_g": 1e-6})
    assert field_gates["S8"][zero_field_key]["status"] == "INELIGIBLE_SINGULAR"
    assert field_gates["S9"][zero_field_key]["status"] == "INELIGIBLE_SINGULAR"
    assert field_gates["S8"][zero_field_key]["minimum_eigenvalue"] == 0.0
    assert field_gates["S9"][zero_field_key]["minimum_eigenvalue"] == 0.0
    assert field_gates["S8"][positive_field_key]["status"] == "ELIGIBLE"
    assert field_gates["S9"][positive_field_key]["status"] == "ELIGIBLE"

    nonfinite_serialization_rejected = False
    try:
        runner.to_jsonable(float("nan"))
    except ValueError:
        nonfinite_serialization_rejected = True
    assert nonfinite_serialization_rejected
    with tempfile.TemporaryDirectory() as directory:
        exclusive_path = Path(directory) / "exclusive.json"
        runner.dump_json(exclusive_path, {"flag": True})
        boolean_serialized_as_boolean = (
            json.loads(exclusive_path.read_text(encoding="utf-8"))["flag"] is True
        )
        exclusive_create_rejected = False
        try:
            runner.dump_json(exclusive_path, {"flag": False})
        except FileExistsError:
            exclusive_create_rejected = True
    assert boolean_serialized_as_boolean
    assert exclusive_create_rejected

    triangle = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    resistance = runner.effective_resistance(triangle)
    diffusion = runner.diffusion_distance(triangle, 1)
    assert np.isfinite(resistance).all() and np.isfinite(diffusion).all()
    disconnected_code = None
    try:
        runner.effective_resistance(
            np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        )
    except runner.CandidateFailure as error:
        disconnected_code = error.code
    assert disconnected_code == "INELIGIBLE_GRAPH_DISCONNECTED"

    zero_residual = np.zeros((10, 2))
    unit_nlpd = runner.gaussian_nlpd(zero_residual, np.eye(2))
    inflated_nlpd = runner.gaussian_nlpd(zero_residual, 100.0 * np.eye(2))
    assert inflated_nlpd > unit_nlpd

    left = (np.array([[0.0], [0.0]]),)
    right = (np.array([[2.0], [2.0]]),)
    simple_w2 = runner.empirical_w2(left, right, np.eye(1))
    assert abs(simple_w2 - 2.0) < 1e-12

    registry = runner.validate_registry(
        run_root / "00-contract.md",
        artifact_root / "candidate-equation-registry.md",
        artifact_root / "candidate-equation-registry.json",
    )
    assert len(registry["candidate_ids"]) == 27

    output = {
        "status": "PASS",
        "synthetic_only": True,
        "time_varying_correct": float(correct_time_varying[0, 0]),
        "time_varying_wrong_initial_product": float(wrong_initial_product[0, 0]),
        "chart_quadratic_error": chart_error,
        "s13_minimum_eigenvalue": low_rank_minimum,
        "s13_optimizer_iterations": optimizer["iterations"],
        "s14_extreme_maximum": float(np.max(stable_softplus_metric)),
        "s14_zero_ridge_extreme_code": zero_ridge_extreme_code,
        "s7_h1_status": s7_h1_status,
        "s8_zero_ridge_rank_gate": field_gates["S8"][zero_field_key]["status"],
        "s9_zero_ridge_rank_gate": field_gates["S9"][zero_field_key]["status"],
        "nonfinite_serialization_rejected": nonfinite_serialization_rejected,
        "boolean_serialized_as_boolean": boolean_serialized_as_boolean,
        "exclusive_create_rejected": exclusive_create_rejected,
        "graph_resistance_maximum": float(np.max(resistance)),
        "graph_diffusion_maximum": float(np.max(diffusion)),
        "disconnected_reason_code": disconnected_code,
        "unit_zero_residual_nlpd": unit_nlpd,
        "inflated_zero_residual_nlpd": inflated_nlpd,
        "simple_w2": simple_w2,
        "registry_candidate_count": len(registry["candidate_ids"]),
        "registry_markdown_sha256": registry["markdown_sha256"],
        "registry_json_sha256": registry["json_sha256"],
    }
    with args.output.open("w", encoding="utf-8", newline="\n") as target:
        json.dump(output, target, indent=2, sort_keys=True)
        target.write("\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
