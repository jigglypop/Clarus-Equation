"""Focused verifier for frozen v3 results, trace linkage, and non-outcome fixtures."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
from run_synthetic_suite import BASELINES, D, KINDS, NODES, STEPS, TEST, TRAJ, candidate_sign_fixture, curvature_fixture, first_passage, force_pushforward_fixture, h_value, holm, scalar_curvature, pullback_metric, chart_metric, sign_p

P = Path(__file__).resolve().with_name("synthetic-v3-results.json")
TRACE = Path(__file__).resolve().with_name("synthetic-v3-traces.npz")
COUNTS = {"metric": 5, "direct_vQ": 6, "gain_noise": 5, "noise_only": 5, "euclidean": 4, "flat_pullback": 5}
def finite(value):
    if isinstance(value, dict): return all(finite(v) for v in value.values())
    if isinstance(value, list): return all(finite(v) for v in value)
    return not isinstance(value, (int, float)) or math.isfinite(value)
def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()
def fixture_only():
    result = curvature_fixture(); assert result["euclidean_max_abs_R"] < 1e-3 and result["pullback_max_abs_R"] < 1e-3 and result["conformal_max_abs_R"] > 1e-3
    print(json.dumps({"status": "PASS_CURVATURE_FIXTURE", **result}, sort_keys=True))
def sign_fixture_only():
    result = candidate_sign_fixture(); assert result["metric_e3_factor"] == result["expected"]
    print(json.dumps({"status": "PASS_SIGN_FIXTURE", **result}, sort_keys=True))
def force_fixture_only():
    result = force_pushforward_fixture(); print(json.dumps({"status": "PASS_FORCE_FIXTURE", **result}, sort_keys=True))
def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--curvature-fixture", action="store_true"); parser.add_argument("--candidate-sign-fixture", action="store_true"); parser.add_argument("--force-fixture", action="store_true"); args = parser.parse_args()
    if args.curvature_fixture: fixture_only(); return
    if args.candidate_sign_fixture: sign_fixture_only(); return
    if args.force_fixture: force_fixture_only(); return
    data = json.loads(P.read_text(encoding="utf-8")); assert data["schema"] == "nrm-synthetic-v3" and finite(data) and sha256(TRACE) == data["trace_sha256"]
    frozen = data["frozen"]; assert frozen["parameter_counts"] == COUNTS
    with np.load(TRACE, allow_pickle=False) as trace:
        assert trace["train_paths"].shape == (6, 20, 24, 321, 3) and trace["test_paths"].shape == (6, 20, 24, 321, 3)
        assert trace["train_W"].shape == (6, 20, NODES, NODES) and trace["test_W"].shape == (6, 20, NODES, NODES)
        assert trace["train_phi"].shape == (6, 20) and trace["test_phi"].shape == (6, 20)
        assert all(np.isfinite(trace[name]).all() for name in trace.files)
        assert np.array_equal(trace["test_directions"], np.tile(np.eye(D)[2], (6, 20, 1)))
        assert np.array_equal(trace["train_directions"], np.asarray([[np.eye(D)[i % 2] for i in range(TEST)] for _ in KINDS]))
        assert all(np.isclose(trace[phase + "_phi"][g, i], trace[phase + "_W"][g, i].sum() / (NODES * (NODES - 1))) for phase in ("train", "test") for g in range(6) for i in range(20))
        assert all(int(trace["train_seeds"][g, i]) == 11000 + 1000 * g + i and int(trace["test_seeds"][g, i]) == 21000 + 1000 * g + i for g in range(6) for i in range(20))
        by_kind = {kind: [] for kind in KINDS}
        for record in data["records"]:
            by_kind[record["generator"]].append(record); gi, i = KINDS.index(record["generator"]), record["test_circuit"]
            assert record["fit_id"] == f"{record['generator']}-{i}" and record["train_seed"] == int(trace["train_seeds"][gi, i]) and record["test_seed"] == int(trace["test_seeds"][gi, i])
            assert record["held_direction"] == [0., 0., 1.] and np.isclose(record["phi"], trace["test_phi"][gi, i])
            assert record["test_first_passage"] == first_passage(trace["test_paths"][gi, i]) and record["first_passage_threshold"] == .8
            assert set(record["scores_by_trajectory"]) == {"metric", *BASELINES} and all(len(v) == TRAJ for v in record["scores_by_trajectory"].values())
            assert {name: fit["parameter_count"] for name, fit in record["fits"].items()} == COUNTS and all(np.all(np.asarray(fit["Kdiag"]) >= frozen["K_floor"]) and fit["sigma2"] > 0 for fit in record["fits"].values())
            truth, mesh = record["evaluator_truth"], record["evaluator_truth"]["mesh"]
            points, matrices = np.asarray(mesh["points"]), [np.asarray(g) for g in mesh["g_matrices"]]
            if truth["mobility"] == "anisotropic_coupled": expected_mesh = [np.diag((1., 1., math.exp(1.2 * h))) for h in h_value(record["phi"], points)]
            elif truth["mobility"] == "flat_pullback": expected_mesh = [pullback_metric(.22)(point) for point in points]
            elif truth["mobility"] == "conformal_coupled": expected_mesh = [math.exp(h) * np.eye(D) for h in h_value(record["phi"], points)]
            else: expected_mesh = [np.eye(D) for _ in points]
            assert len(matrices) == len(expected_mesh) and all(np.linalg.eigvalsh(g).min() > 0 and np.allclose(g, expected) for g, expected in zip(matrices, expected_mesh))
        assert all(len(rows) == TEST and sorted(r["test_circuit"] for r in rows) == list(range(TEST)) for rows in by_kind.values())
        evaluator, g1 = data["evaluator"], by_kind["G1_metric"]
        recovery = [abs(float(r["fits"]["metric"]["param"]) - 1.2) / 1.2 <= .25 and float(r["fits"]["metric"]["param"]) > 0 for r in g1]
        assert recovery == evaluator["g1_recovery"] and sum(recovery) >= 18
        g1_p = {base: sign_p(np.array([np.mean(r["scores_by_trajectory"][base]) - np.mean(r["scores_by_trajectory"]["metric"]) for r in g1]), 501 + j) for j, base in enumerate(BASELINES)}
        assert all(0 <= evaluator["g1_pvalues"][k] <= 1 and abs(g1_p[k] - evaluator["g1_pvalues"][k]) < 1e-15 for k in BASELINES) and holm(g1_p) == evaluator["g1_holm"] and all(evaluator["g1_holm"].values())
        fp = evaluator["false_positive_rows"]; assert len(fp) == 100 and len({(r["generator"], r["test_circuit"]) for r in fp}) == 100
        for row in fp:
            source = next(r for r in by_kind[row["generator"]] if r["test_circuit"] == row["test_circuit"])
            expected = {base: sign_p(np.array(source["scores_by_trajectory"][base]) - np.array(source["scores_by_trajectory"]["metric"]), 7000 + 100 * KINDS.index(row["generator"]) + 5 * row["test_circuit"] + j) for j, base in enumerate(BASELINES)}
            assert all(0 <= row["pvalues"][k] <= 1 and abs(expected[k] - row["pvalues"][k]) < 1e-15 for k in expected) and holm(expected) == row["holm"] and any(row["holm"].values()) == row["any_reject"] and all(row["holm"].values()) == row["full_promotion"]
        assert sum(row["any_reject"] for row in fp) <= 5
        g3 = by_kind["G3_flat_pullback"]; c_recovery = [abs(float(r["fits"]["flat_pullback"]["param"]) - .22) <= .05 for r in g3]
        assert c_recovery == evaluator["g3_c_recovery"] and sum(c_recovery) >= 18
        curvature = evaluator["curvature"]; assert len(curvature) == TEST
        for row in curvature:
            c = float(g3[row["test_circuit"]]["fits"]["flat_pullback"]["param"]); point = np.asarray(row["mesh_point"]); values = [abs(scalar_curvature(pullback_metric(c), point))] + [abs(scalar_curvature(chart_metric(pullback_metric(c), np.asarray(matrix)), point)) for matrix in row["affine_matrices"]]
            actual_conditions = [float(np.linalg.cond(matrix)) for matrix in row["affine_matrices"]]
            assert abs(c - row["fitted_c"]) < 1e-12 and len(row["affine_matrices"]) == 3 and np.allclose(actual_conditions, row["affine_chart_condition_numbers"], atol=1e-12) and all(np.isfinite(condition) and condition <= 5 for condition in actual_conditions)
            assert np.allclose(values, row["values"], atol=1e-8, rtol=1e-8) and abs(max(values) - row["max_abs_R"]) < 1e-8 and (max(values) > 1e-3) == row["false_positive"]
        assert sum(row["false_positive"] for row in curvature) <= 1
    fixture_only(); sign_fixture_only(); force_fixture_only(); print(json.dumps({"status": "PASS"}, sort_keys=True))
if __name__ == "__main__": main()
