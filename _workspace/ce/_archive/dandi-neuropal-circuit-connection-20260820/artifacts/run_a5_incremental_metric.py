"""A5 separates inherited geometry L0 from activity-induced deformation dL."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

import run_a4_dynamic_metric as a4


ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / "_workspace/ce/dandi-neuropal-circuit-connection-20260820"
MANIFEST = RUN / "artifacts/source-manifest-v4.json"
DEV_RESULT = RUN / "artifacts/a5-development-result.json"
CONFIRM_RESULT = RUN / "artifacts/a5-confirmation-result.json"
RIDGE = 1.0e-2
EPS = 1.0e-12


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def design_receipt(design: np.ndarray) -> tuple[int, float]:
    rms = np.sqrt(np.mean(design * design, axis=0))
    normalized = design / np.maximum(rms, EPS)
    singular = np.linalg.svd(normalized, compute_uv=False)
    rank = int(np.linalg.matrix_rank(normalized, tol=1.0e-10))
    ratio = float(singular[-1] / singular[0]) if singular[0] > 0 else 0.0
    return rank, ratio


def constrained_ridge(
    design: np.ndarray,
    target: np.ndarray,
    constrained_columns: tuple[int, ...],
) -> tuple[np.ndarray, tuple[int, ...]]:
    p = design.shape[1]
    unconstrained = [j for j in range(p) if j not in constrained_columns]
    best_beta = None
    best_active: tuple[int, ...] = ()
    best_objective = float("inf")
    for mask in range(1 << len(constrained_columns)):
        active = tuple(constrained_columns[j] for j in range(len(constrained_columns)) if (mask >> j) & 1)
        free = unconstrained + list(active)
        beta = np.zeros(p, dtype=float)
        if free:
            d = design[:, free]
            solved = np.linalg.solve(d.T @ d + RIDGE * np.eye(len(free)), d.T @ target)
            beta[free] = solved
        if any(beta[j] < -1.0e-12 for j in active):
            continue
        beta[np.abs(beta) < 1.0e-12] = 0.0
        residual = target - design @ beta
        objective = float(residual @ residual + RIDGE * (beta @ beta))
        if objective < best_objective:
            best_objective = objective
            best_beta = beta
            best_active = active
    if best_beta is None:
        raise RuntimeError("CONSTRAINED_RIDGE_FAILURE")
    return best_beta, best_active


def fit_from_features(
    z: np.ndarray,
    idx: np.ndarray,
    base_feature: np.ndarray,
    delta_feature: np.ndarray | None,
) -> dict[str, Any]:
    base_scale = float(np.sqrt(np.mean(base_feature * base_feature)))
    if not np.isfinite(base_scale) or base_scale <= 1.0e-10:
        raise RuntimeError("UNIDENTIFIABLE_BASE_GEOMETRY")
    blocks = [z[idx], base_feature / base_scale]
    scales = [1.0, base_scale]
    if delta_feature is not None:
        delta_scale = float(np.sqrt(np.mean(delta_feature * delta_feature)))
        if not np.isfinite(delta_scale) or delta_scale <= 1.0e-10:
            raise RuntimeError("CONTROL_DEGENERATE")
        blocks.append(delta_feature / delta_scale)
        scales.append(delta_scale)
    y = z[idx + 1]
    means = [block.mean(axis=0) for block in blocks]
    ymean = y.mean(axis=0)
    design = np.column_stack([(block - mean).reshape(-1) for block, mean in zip(blocks, means)])
    target = (y - ymean).reshape(-1)
    rank, ratio = design_receipt(design)
    if rank < len(blocks) or ratio < 0.05:
        raise RuntimeError("CONTROL_DEGENERATE" if delta_feature is not None else "UNIDENTIFIABLE_BASE_GEOMETRY")
    constrained = tuple(range(1, len(blocks)))
    beta, active = constrained_ridge(design, target, constrained)
    intercept = ymean - sum(beta[j] * means[j] for j in range(len(blocks)))
    return {
        "beta": beta,
        "intercept": intercept,
        "scales": scales,
        "rank": rank,
        "singular_value_ratio": ratio,
        "active_nonnegative_columns": list(active),
    }


def predict_from_features(
    z: np.ndarray,
    idx: np.ndarray,
    base_feature: np.ndarray,
    delta_feature: np.ndarray | None,
    model: dict[str, Any],
) -> np.ndarray:
    blocks = [z[idx], base_feature / model["scales"][1]]
    if delta_feature is not None:
        blocks.append(delta_feature / model["scales"][2])
    return model["intercept"] + sum(model["beta"][j] * blocks[j] for j in range(len(blocks)))


def base_and_delta_features(
    prep: dict[str, Any],
    idx: np.ndarray,
    base_geom: dict[str, Any],
    arm_geom: dict[str, Any],
    alpha: np.ndarray,
    activation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    zero_base = np.zeros_like(base_geom["base_weight"])
    base, _ = a4.graph_feature(prep["z"], activation, idx, base_geom, zero_base)
    zero_arm = np.zeros_like(alpha)
    arm_static, _ = a4.graph_feature(prep["z"], activation, idx, arm_geom, zero_arm)
    arm_full, metric_receipt = a4.graph_feature(prep["z"], activation, idx, arm_geom, alpha)
    return base, arm_full - arm_static, metric_receipt


def evaluate(prep: dict[str, Any]) -> dict[str, Any]:
    bidx = a4.a3.pair_indices(prep["b_domain"], a4.STATE_SHIFT, 1)
    tidx = a4.a3.pair_indices(prep["test_domain"], a4.STATE_SHIFT, 1)
    try:
        arms, construction_receipt = a4.prepare_arms(prep)
        base_geom, _, base_activation = arms["fixed_geometry"]
        zero = np.zeros_like(base_geom["base_weight"])
        base_b, _ = a4.graph_feature(prep["z"], base_activation, bidx, base_geom, zero)
        base_t, _ = a4.graph_feature(prep["z"], base_activation, tidx, base_geom, zero)
        no_def = fit_from_features(prep["z"], bidx, base_b, None)

        models = {}
        train_features = {}
        test_features = {}
        metric_receipts = {}
        for name in ("a4", "edge_shuffle", "time_shift", "state_shift", "identity_shuffle", "phase_randomized"):
            arm_geom, alpha, activation = arms[name]
            _, delta_b, metric_b = base_and_delta_features(prep, bidx, base_geom, arm_geom, alpha, activation)
            _, delta_t, _ = base_and_delta_features(prep, tidx, base_geom, arm_geom, alpha, activation)
            try:
                model = fit_from_features(prep["z"], bidx, base_b, delta_b)
            except RuntimeError as exc:
                if name != "a4" and str(exc) == "CONTROL_DEGENERATE":
                    return {
                        "status": "CONTROL_DEGENERATE",
                        "degenerate_control": name,
                        "asset_id": prep["asset_id"],
                    }
                raise
            models[name] = model
            train_features[name] = delta_b
            test_features[name] = delta_t
            metric_receipts[name] = {
                **metric_b,
                "delta_to_base_rms_ratio": float(model["scales"][2] / no_def["scales"][1]),
            }

        if metric_receipts["a4"]["h_variance"] <= 1.0e-10:
            raise RuntimeError("UNIDENTIFIABLE_GRAPH_TERM")

        base_train_prediction = predict_from_features(prep["z"], bidx, base_b, None, no_def)
        variance = np.maximum(np.mean((prep["z"][bidx + 1] - base_train_prediction) ** 2, axis=0), 1.0e-8)
        base_test_prediction = predict_from_features(prep["z"], tidx, base_t, None, no_def)
        base_error = prep["z"][tidx + 1] - base_test_prediction
        arm_rows = {
            "no_deformation": {
                "delta_log_score": 0.0,
                "beta": no_def["beta"].tolist(),
                "scales": no_def["scales"],
                "rank": no_def["rank"],
                "singular_value_ratio": no_def["singular_value_ratio"],
                "active_nonnegative_columns": no_def["active_nonnegative_columns"],
            }
        }
        for name, model in models.items():
            prediction = predict_from_features(prep["z"], tidx, base_t, test_features[name], model)
            error = prep["z"][tidx + 1] - prediction
            delta = float(np.mean(-0.5 * ((error * error - base_error * base_error) / variance)))
            output_name = "a5" if name == "a4" else name
            arm_rows[output_name] = {
                "delta_log_score": delta,
                "beta": model["beta"].tolist(),
                "scales": model["scales"],
                "rank": model["rank"],
                "singular_value_ratio": model["singular_value_ratio"],
                "active_nonnegative_columns": model["active_nonnegative_columns"],
                "metric_receipt": metric_receipts[name],
            }
        return {
            "status": "ADMISSIBLE",
            "asset_id": prep["asset_id"],
            "lindi_sha256": prep["lindi_sha256"],
            "official_asset_sha256_declared": prep["official_asset_sha256"],
            "provenance_status": prep["provenance_status"],
            "trace_shape": list(prep["x"].shape),
            "eligible_neurons": prep["eligible_neurons"],
            "excluded_neurons": prep["excluded_neurons"],
            "missing_fraction_raw": prep["missing_fraction_raw"],
            "construction_pairs": int(len(bidx)),
            "test_pairs": int(len(tidx)),
            "construction_receipt": construction_receipt,
            "arms": arm_rows,
        }
    except RuntimeError as exc:
        if str(exc) not in {"NO_DYNAMIC_EDGE", "UNIDENTIFIABLE_GRAPH_TERM", "UNIDENTIFIABLE_BASE_GEOMETRY", "CONTROL_DEGENERATE"}:
            raise
        return {"status": str(exc), "asset_id": prep["asset_id"]}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    admissible = [row for row in rows if row["status"] == "ADMISSIBLE"]
    names = ("a5", "no_deformation", "edge_shuffle", "time_shift", "state_shift", "identity_shuffle", "phase_randomized")
    means = {
        name: float(np.mean([row["arms"][name]["delta_log_score"] for row in admissible])) if admissible else None
        for name in names
    }
    return {
        "worm_count": len(rows),
        "admissible_count": len(admissible),
        "positive_a5_count": sum(row["arms"]["a5"]["delta_log_score"] > 0 for row in admissible),
        "mean_delta": means,
    }


def sign_flip_p(values: list[float]) -> float:
    observed = float(np.mean(values))
    stats = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        stats.append(float(np.mean(np.asarray(signs) * np.asarray(values))))
    return float(np.mean(np.asarray(stats) >= observed - 1.0e-15))


def max_stat_adjusted_p(rows: list[dict[str, Any]]) -> dict[str, float]:
    controls = ("no_deformation", "edge_shuffle", "time_shift", "state_shift", "identity_shuffle", "phase_randomized")
    diffs = {
        name: np.asarray([row["arms"]["a5"]["delta_log_score"] - row["arms"][name]["delta_log_score"] for row in rows])
        for name in controls
    }
    observed = {name: float(np.mean(values)) for name, values in diffs.items()}
    null_max = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(rows)):
        sign = np.asarray(signs)
        null_max.append(max(float(np.mean(sign * values)) for values in diffs.values()))
    null_max = np.asarray(null_max)
    return {name: float(np.mean(null_max >= stat - 1.0e-15)) for name, stat in observed.items()}


def run(stage: str) -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assets = manifest["assets"][:3] if stage == "development" else manifest["assets"][3:]
    if stage == "confirmation":
        if not DEV_RESULT.exists():
            raise RuntimeError("confirmation requires development result")
        dev = json.loads(DEV_RESULT.read_text(encoding="utf-8"))
        if not dev["decision"]["pass_development"]:
            raise RuntimeError("confirmation sealed because development did not pass")
    rows = [evaluate(a4.split_and_standardize(a4.load_asset_verified(asset))) for asset in assets]
    agg = aggregate(rows)
    a5_mean = agg["mean_delta"]["a5"]
    controls = [value for name, value in agg["mean_delta"].items() if name != "a5"]
    if stage == "development":
        passed = bool(
            agg["admissible_count"] == 3
            and agg["positive_a5_count"] >= 2
            and a5_mean > 0
            and all(a5_mean > value for value in controls)
        )
        decision = {"pass_development": passed, "confirmation_open": passed}
    else:
        admissible = [row for row in rows if row["status"] == "ADMISSIBLE"]
        values = [row["arms"]["a5"]["delta_log_score"] for row in admissible]
        p = sign_flip_p(values) if len(values) == 5 else 1.0
        adjusted = max_stat_adjusted_p(admissible) if len(admissible) == 5 else {}
        decision = {
            "exact_sign_flip_p": p,
            "max_stat_adjusted_p": adjusted,
            "pass_a5": bool(
                agg["admissible_count"] == 5
                and all(value > 0 for value in values)
                and p < 0.05
                and all(a5_mean > value for value in controls)
                and adjusted
                and all(value < 0.05 for value in adjusted.values())
            ),
        }
    return {
        "status": "COMPLETE",
        "stage": stage,
        "claim_ceiling": "OBSERVATIONAL_INCREMENTAL_GRAPH_FEATURE",
        "source_manifest_sha256": file_sha256(MANIFEST),
        "a4_dependency_sha256": file_sha256(Path(a4.__file__)),
        "a3_dependency_sha256": file_sha256(Path(a4.a3.__file__)),
        "script_sha256": file_sha256(Path(__file__)),
        "rows": rows,
        "aggregate": agg,
        "decision": decision,
    }


def self_test() -> None:
    rng = np.random.default_rng(9)
    design = rng.normal(size=(500, 3))
    target = design @ np.array([0.8, 0.3, 0.2]) + rng.normal(scale=0.01, size=500)
    beta, active = constrained_ridge(design, target, (1, 2))
    assert beta[1] >= 0 and beta[2] >= 0
    assert set(active) == {1, 2}
    target_negative = design @ np.array([0.8, -0.3, 0.2])
    beta_negative, _ = constrained_ridge(design, target_negative, (1, 2))
    assert beta_negative[1] == 0 and beta_negative[2] >= 0
    print("PASS_A5_SELF_TEST")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("development", "confirmation"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.stage:
        parser.error("--stage required")
    result = run(args.stage)
    output = DEV_RESULT if args.stage == "development" else CONFIRM_RESULT
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "aggregate": result["aggregate"], "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
