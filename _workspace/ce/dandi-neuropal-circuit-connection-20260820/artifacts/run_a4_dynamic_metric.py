"""A4 state-dependent graph-metric test on a fresh DANDI 000541 cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import run_a3_connection_operator as a3


ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / "_workspace/ce/dandi-neuropal-circuit-connection-20260820"
MANIFEST = RUN / "artifacts/source-manifest-v3.json"
DEV_RESULT = RUN / "artifacts/a4-development-result.json"
CONFIRM_RESULT = RUN / "artifacts/a4-confirmation-result.json"
RIDGE = 1.0e-2
SEED = 20260820
SHIFT_LAGS = (17, 31, 47)
STATE_SHIFT = 31
EPS = 1.0e-12


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def load_asset_verified(asset: dict[str, Any]) -> dict[str, Any]:
    lindi_path = a3.DATA / f"000541-{asset['id']}.nwb.lindi.json"
    metadata = json.loads(lindi_path.read_text(encoding="utf-8"))["generationMetadata"]
    expected = (asset["id"], asset["path"], int(asset["size"]))
    observed = (metadata["assetId"], metadata["assetPath"], int(metadata["assetSize"]))
    if observed != expected:
        raise RuntimeError(f"LINDI provenance mismatch: expected={expected}, observed={observed}")
    raw = a3.load_asset(asset)
    raw["official_asset_sha256"] = asset["sha256"]
    raw["provenance_status"] = "PARTIAL_RANGE_PROVENANCE"
    return raw


def split_and_standardize(raw: dict[str, Any]) -> dict[str, Any]:
    x = raw["x"]
    n_time = x.shape[0]
    times = raw["start"] + np.arange(n_time) / raw["rate"]
    allowed = np.ones(n_time, dtype=bool)
    for onset, offset in zip(raw["stim_start"], raw["stim_stop"]):
        allowed &= ~((times >= onset - 5.0) & (times <= offset + 20.0))
    k_end = int(math.floor(0.2 * n_time))
    b_start = k_end + 8
    b_end = int(math.floor(0.7 * n_time)) - 8
    test_start = int(math.floor(0.7 * n_time)) + 8
    k_idx = np.flatnonzero(allowed & (np.arange(n_time) < k_end))
    b_domain = allowed & (np.arange(n_time) >= b_start) & (np.arange(n_time) < b_end)
    test_domain = allowed & (np.arange(n_time) >= test_start) & (np.arange(n_time) < n_time - 1)
    labels = np.asarray(raw["labels"], dtype=object)
    coords = raw["coords"]
    keep = (labels != "") & np.isfinite(coords).all(axis=1) & np.isfinite(x).all(axis=0)
    if keep.sum() < 20 or len(k_idx) < 40:
        raise RuntimeError(f"insufficient complete data {raw['asset_id']}: neurons={keep.sum()}, K={len(k_idx)}")
    x = x[:, keep]
    coords = coords[keep]
    labels = labels[keep]
    mu = np.median(x[k_idx], axis=0)
    scale = 1.4826 * np.median(np.abs(x[k_idx] - mu[None, :]), axis=0)
    good_scale = np.isfinite(mu) & np.isfinite(scale) & (scale > 1.0e-6)
    x, coords, labels = x[:, good_scale], coords[good_scale], labels[good_scale]
    mu, scale = mu[good_scale], scale[good_scale]
    if x.shape[1] < 20:
        raise RuntimeError(f"insufficient scaled data {raw['asset_id']}: neurons={x.shape[1]}")
    z = (x - mu[None, :]) / scale[None, :]
    km = z[k_idx].mean(axis=0)
    _, _, vt = np.linalg.svd(z[k_idx] - km, full_matrices=False)
    rank = min(3, vt.shape[0])
    basis = vt[:rank]
    centered = z - km
    residual = centered - (centered @ basis.T) @ basis
    return {
        **raw,
        "x": x,
        "z": z,
        "residual": residual,
        "coords": coords,
        "labels": labels.tolist(),
        "allowed": allowed,
        "b_domain": b_domain,
        "test_domain": test_domain,
        "k_count": int(len(k_idx)),
        "eligible_neurons": int(z.shape[1]),
        "excluded_neurons": int(raw["x"].shape[1] - z.shape[1]),
        "missing_fraction_raw": float(np.mean(~np.isfinite(raw["x"]))),
    }


def geometry(coords: np.ndarray, k: int = 6) -> dict[str, np.ndarray | float]:
    n = len(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    adj = np.zeros((n, n), dtype=bool)
    node_ids = np.arange(n)
    for i in range(n):
        order = np.lexsort((node_ids, dist[i]))
        adj[i, order[: min(k, n - 1)]] = True
    adj |= adj.T
    np.fill_diagonal(adj, False)
    edge_dist = dist[np.triu(adj, 1)]
    lref = float(np.median(edge_dist))
    if not np.isfinite(lref) or lref <= 0:
        raise RuntimeError("invalid spatial reference length")
    ell = np.zeros_like(dist)
    ell[adj] = dist[adj] / lref
    base_weight = np.zeros_like(dist)
    base_weight[adj] = 1.0 / np.maximum(ell[adj] ** 2, 1.0e-8)
    return {"adj": adj, "ell": ell, "base_weight": base_weight, "lref": lref}


def standardized_cross(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x) < 3 or x.shape != y.shape:
        raise RuntimeError("insufficient correlation rows")
    x0 = x - x.mean(axis=0)
    y0 = y - y.mean(axis=0)
    xs = np.sqrt(np.sum(x0 * x0, axis=0))
    ys = np.sqrt(np.sum(y0 * y0, axis=0))
    denom = np.maximum(xs[:, None] * ys[None, :], EPS)
    return (x0.T @ y0) / denom


def correlation_parts(residual: np.ndarray, domain: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    idx = np.flatnonzero(domain)
    zero = standardized_cross(residual[idx], residual[idx])
    shifted = []
    counts = {"zero": int(len(idx))}
    for tau in SHIFT_LAGS:
        valid = domain.copy()
        valid[:tau] = False
        valid &= np.roll(domain, tau)
        ti = np.flatnonzero(valid)
        counts[str(tau)] = int(len(ti))
        cross = standardized_cross(residual[ti], residual[ti - tau])
        shifted.append((cross + cross.T) / 2.0)
    return (zero + zero.T) / 2.0, np.mean(shifted, axis=0), counts


def normalize_alpha(raw: np.ndarray, adj: np.ndarray, require_positive: bool) -> tuple[np.ndarray, float]:
    alpha = np.maximum(raw, 0.0) * adj
    alpha = (alpha + alpha.T) / 2.0
    np.fill_diagonal(alpha, 0.0)
    positive = alpha[np.triu(adj, 1) & (alpha > 0)]
    if len(positive) == 0 or not np.isfinite(np.median(positive)) or np.median(positive) <= 0:
        if require_positive:
            raise RuntimeError("NO_DYNAMIC_EDGE")
        return np.zeros_like(alpha), 0.0
    median = float(np.median(positive))
    return alpha / median, median


def build_alpha(
    prep: dict[str, Any],
    geom: dict[str, Any],
    residual: np.ndarray | None = None,
    shifted_control: bool = False,
    require_positive: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    r = prep["residual"] if residual is None else residual
    zero, shift, counts = correlation_parts(r, prep["b_domain"])
    raw = shift if shifted_control else zero - shift
    alpha, median = normalize_alpha(raw, geom["adj"], require_positive=require_positive)
    edges = np.triu(geom["adj"], 1)
    return alpha, {
        "positive_normalizer": median,
        "positive_edge_fraction": float(np.mean(alpha[edges] > 0)),
        "max_normalized_strength": float(np.max(alpha[edges])) if np.any(edges) else 0.0,
        "correlation_sample_counts": counts,
    }


def phase_randomize_blocks(residual: np.ndarray, domain: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros_like(residual)
    edges = np.diff(np.r_[False, domain, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    for start, stop in zip(starts, stops):
        block = residual[start:stop]
        if len(block) < 4:
            continue
        spectrum = np.fft.rfft(block, axis=0)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=spectrum.shape)
        phase[0] = 0.0
        if len(block) % 2 == 0:
            phase[-1] = 0.0
        out[start:stop] = np.fft.irfft(spectrum * np.exp(1j * phase), n=len(block), axis=0)
    return out


def shuffle_alpha_by_length(alpha: np.ndarray, geom: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    upper = np.argwhere(np.triu(geom["adj"], 1))
    lengths = np.asarray([geom["ell"][i, j] for i, j in upper])
    cuts = np.quantile(lengths, (1.0 / 3.0, 2.0 / 3.0))
    bins = np.digitize(lengths, cuts, right=True)
    values = np.asarray([alpha[i, j] for i, j in upper])
    shuffled = values.copy()
    for b in range(3):
        where = np.flatnonzero(bins == b)
        shuffled[where] = values[rng.permutation(where)] if len(where) else values[where]
    out = np.zeros_like(alpha)
    for (i, j), value in zip(upper, shuffled):
        out[i, j] = out[j, i] = value
    return out


def graph_feature(
    z: np.ndarray,
    activation: np.ndarray,
    idx: np.ndarray,
    geom: dict[str, Any],
    alpha: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    base = geom["base_weight"]
    edge_mask = np.triu(geom["adj"], 1)
    rows = np.empty((len(idx), z.shape[1]), dtype=float)
    h_values = []
    clipped = 0
    total = 0
    for row, t in enumerate(idx):
        raw_h = alpha * np.outer(activation[t], activation[t])
        h = np.clip(raw_h, 0.0, 4.0)
        w = base * np.exp(h)
        rows[row] = -np.sum(w * (z[t, :, None] - z[t, None, :]), axis=1)
        hv = h[edge_mask]
        h_values.append(hv)
        clipped += int(np.sum(raw_h[edge_mask] >= 4.0))
        total += len(hv)
    hv_all = np.concatenate(h_values) if h_values else np.zeros(0)
    receipt = {
        "h_mean": float(np.mean(hv_all)) if len(hv_all) else 0.0,
        "h_variance": float(np.var(hv_all)) if len(hv_all) else 0.0,
        "clipping_fraction": float(clipped / total) if total else 0.0,
        "mean_edge_length_ratio": float(np.mean(np.exp(-0.5 * hv_all))) if len(hv_all) else 1.0,
        "mean_conductance_ratio": float(np.mean(np.exp(hv_all))) if len(hv_all) else 1.0,
    }
    return rows, receipt


def fit_model(
    z: np.ndarray,
    activation: np.ndarray,
    idx: np.ndarray,
    geom: dict[str, Any],
    alpha: np.ndarray,
) -> dict[str, Any]:
    graph, metric_receipt = graph_feature(z, activation, idx, geom, alpha)
    graph_scale = float(np.sqrt(np.mean(graph * graph)))
    if not np.isfinite(graph_scale) or graph_scale <= 1.0e-10:
        raise RuntimeError("UNIDENTIFIABLE_GRAPH_TERM")
    blocks = [z[idx], graph / graph_scale]
    y = z[idx + 1]
    means = [block.mean(axis=0) for block in blocks]
    ymean = y.mean(axis=0)
    design = np.column_stack([(block - mean).reshape(-1) for block, mean in zip(blocks, means)])
    target = (y - ymean).reshape(-1)
    col_rms = np.sqrt(np.mean(design * design, axis=0))
    normalized_design = design / np.maximum(col_rms, EPS)
    singular = np.linalg.svd(normalized_design, compute_uv=False)
    sv_ratio = float(singular[-1] / singular[0]) if singular[0] > 0 else 0.0
    rank = int(np.linalg.matrix_rank(normalized_design, tol=1.0e-10))
    if rank < 2 or sv_ratio < 0.05:
        raise RuntimeError("UNIDENTIFIABLE_GRAPH_TERM")
    beta = np.linalg.solve(design.T @ design + RIDGE * np.eye(2), design.T @ target)
    constrained = False
    if beta[1] < 0:
        d0 = design[:, :1]
        beta0 = np.linalg.solve(d0.T @ d0 + RIDGE * np.eye(1), d0.T @ target)
        beta = np.array([float(beta0[0]), 0.0])
        constrained = True
    intercept = ymean - beta[0] * means[0] - beta[1] * means[1]
    return {
        "beta": beta,
        "intercept": intercept,
        "graph_scale": graph_scale,
        "rank": rank,
        "singular_value_ratio": sv_ratio,
        "nonnegative_constraint_active": constrained,
        "metric_receipt": metric_receipt,
    }


def predict_model(
    z: np.ndarray,
    activation: np.ndarray,
    idx: np.ndarray,
    geom: dict[str, Any],
    alpha: np.ndarray,
    model: dict[str, Any],
) -> np.ndarray:
    graph, _ = graph_feature(z, activation, idx, geom, alpha)
    beta = model["beta"]
    return model["intercept"] + beta[0] * z[idx] + beta[1] * graph / model["graph_scale"]


def prepare_arms(prep: dict[str, Any]) -> tuple[dict[str, tuple[dict[str, Any], np.ndarray, np.ndarray]], dict[str, Any]]:
    z = prep["z"]
    activation = sigmoid(z - 2.5)
    geom = geometry(prep["coords"])
    alpha, alpha_receipt = build_alpha(prep, geom, require_positive=True)
    rng_seed = int(hashlib.sha256(prep["asset_id"].encode()).hexdigest()[:8], 16) ^ SEED
    rng = np.random.default_rng(rng_seed)

    edge_shuffle = shuffle_alpha_by_length(alpha, geom, rng)
    time_alpha, time_receipt = build_alpha(prep, geom, shifted_control=True, require_positive=False)
    shifted_activation = np.zeros_like(activation)
    shifted_activation[STATE_SHIFT:] = activation[:-STATE_SHIFT]

    coord_perm = rng.permutation(z.shape[1])
    identity_geom = geometry(prep["coords"][coord_perm])
    identity_alpha, identity_receipt = build_alpha(prep, identity_geom, require_positive=False)

    phase_residual = phase_randomize_blocks(prep["residual"], prep["b_domain"], rng)
    phase_alpha, phase_receipt = build_alpha(prep, geom, residual=phase_residual, require_positive=False)

    arms = {
        "a4": (geom, alpha, activation),
        "fixed_geometry": (geom, np.zeros_like(alpha), activation),
        "edge_shuffle": (geom, edge_shuffle, activation),
        "time_shift": (geom, time_alpha, activation),
        "state_shift": (geom, alpha, shifted_activation),
        "identity_shuffle": (identity_geom, identity_alpha, activation),
        "phase_randomized": (geom, phase_alpha, activation),
    }
    receipt = {
        "threshold_raw_definition": "mu_i + 2.5 * (1.4826 * MAD_i)",
        "soft_activation_mean_B": float(np.mean(activation[prep["b_domain"]])),
        "alpha": alpha_receipt,
        "spatial_reference_length": geom["lref"],
        "edge_count": int(np.sum(np.triu(geom["adj"], 1))),
        "control_alpha": {
            "time_shift": time_receipt,
            "identity_shuffle": identity_receipt,
            "phase_randomized": phase_receipt,
        },
    }
    return arms, receipt


def evaluate(prep: dict[str, Any]) -> dict[str, Any]:
    bidx = a3.pair_indices(prep["b_domain"], STATE_SHIFT, 1)
    tidx = a3.pair_indices(prep["test_domain"], STATE_SHIFT, 1)
    try:
        arms, construction_receipt = prepare_arms(prep)
        models = {}
        for name, (geom, alpha, activation) in arms.items():
            models[name] = fit_model(prep["z"], activation, bidx, geom, alpha)
        if models["a4"]["metric_receipt"]["h_variance"] <= 1.0e-10:
            raise RuntimeError("UNIDENTIFIABLE_GRAPH_TERM")
        base_geom, base_alpha, base_activation = arms["fixed_geometry"]
        base_model = models["fixed_geometry"]
        base_train = predict_model(prep["z"], base_activation, bidx, base_geom, base_alpha, base_model)
        variance = np.maximum(np.mean((prep["z"][bidx + 1] - base_train) ** 2, axis=0), 1.0e-8)
        base_test = predict_model(prep["z"], base_activation, tidx, base_geom, base_alpha, base_model)
        base_error = prep["z"][tidx + 1] - base_test
        arm_rows = {}
        for name, (geom, alpha, activation) in arms.items():
            model = models[name]
            prediction = predict_model(prep["z"], activation, tidx, geom, alpha, model)
            error = prep["z"][tidx + 1] - prediction
            delta = float(np.mean(-0.5 * ((error * error - base_error * base_error) / variance)))
            arm_rows[name] = {
                "delta_log_score": delta,
                "beta": model["beta"].tolist(),
                "graph_scale": model["graph_scale"],
                "rank": model["rank"],
                "singular_value_ratio": model["singular_value_ratio"],
                "nonnegative_constraint_active": model["nonnegative_constraint_active"],
                "metric_receipt": model["metric_receipt"],
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
        if str(exc) not in {"NO_DYNAMIC_EDGE", "UNIDENTIFIABLE_GRAPH_TERM"}:
            raise
        return {
            "status": str(exc),
            "asset_id": prep["asset_id"],
            "lindi_sha256": prep["lindi_sha256"],
            "official_asset_sha256_declared": prep["official_asset_sha256"],
            "provenance_status": prep["provenance_status"],
            "trace_shape": list(prep["x"].shape),
            "eligible_neurons": prep["eligible_neurons"],
            "construction_pairs": int(len(bidx)),
            "test_pairs": int(len(tidx)),
        }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    admissible = [row for row in rows if row["status"] == "ADMISSIBLE"]
    names = ("a4", "fixed_geometry", "edge_shuffle", "time_shift", "state_shift", "identity_shuffle", "phase_randomized")
    means = {
        name: float(np.mean([row["arms"][name]["delta_log_score"] for row in admissible]))
        if admissible
        else None
        for name in names
    }
    positive = sum(row["arms"]["a4"]["delta_log_score"] > 0 for row in admissible)
    return {
        "worm_count": len(rows),
        "admissible_count": len(admissible),
        "positive_a4_count": positive,
        "mean_delta": means,
    }


def sign_flip_p(values: list[float]) -> float:
    observed = sum(values)
    count = 0
    for mask in range(1 << len(values)):
        total = sum(v if (mask >> i) & 1 else -v for i, v in enumerate(values))
        count += total >= observed - 1.0e-15
    return count / (1 << len(values))


def run(stage: str) -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assets = manifest["assets"][:3] if stage == "development" else manifest["assets"][3:]
    if stage == "confirmation":
        if not DEV_RESULT.exists():
            raise RuntimeError("confirmation requires development result")
        dev = json.loads(DEV_RESULT.read_text(encoding="utf-8"))
        if not dev["decision"]["pass_development"]:
            raise RuntimeError("confirmation sealed because development did not pass")
    rows = [evaluate(split_and_standardize(load_asset_verified(asset))) for asset in assets]
    agg = aggregate(rows)
    controls = [value for name, value in agg["mean_delta"].items() if name != "a4"]
    a4_mean = agg["mean_delta"]["a4"]
    if stage == "development":
        passed = bool(
            agg["admissible_count"] == 3
            and agg["positive_a4_count"] >= 2
            and a4_mean > 0
            and all(a4_mean > value for value in controls)
        )
        decision = {"pass_development": passed, "confirmation_open": passed}
    else:
        values = [row["arms"]["a4"]["delta_log_score"] for row in rows if row["status"] == "ADMISSIBLE"]
        p = sign_flip_p(values) if len(values) == 5 else 1.0
        decision = {
            "exact_sign_flip_p": p,
            "pass_a4": bool(
                agg["admissible_count"] == 5
                and all(value > 0 for value in values)
                and p < 0.05
                and all(a4_mean > value for value in controls)
            ),
        }
    return {
        "status": "COMPLETE",
        "stage": stage,
        "claim_ceiling": "OBSERVATIONAL_STATE_DEPENDENT_GRAPH_PREDICTOR",
        "source_manifest_sha256": file_sha256(MANIFEST),
        "a3_dependency_sha256": file_sha256(Path(a3.__file__)),
        "script_sha256": file_sha256(Path(__file__)),
        "rows": rows,
        "aggregate": agg,
        "decision": decision,
    }


def self_test() -> None:
    rng = np.random.default_rng(7)
    coords = rng.normal(size=(14, 3))
    geom = geometry(coords)
    alpha = rng.uniform(0.0, 2.0, size=(14, 14))
    alpha = (alpha + alpha.T) / 2.0 * geom["adj"]
    np.fill_diagonal(alpha, 0.0)
    z = rng.normal(size=(180, 14))
    activation = sigmoid(z - 2.5)
    idx = np.arange(32, 140)
    graph, receipt = graph_feature(z, activation, idx, geom, alpha)
    model = fit_model(z, activation, idx, geom, alpha)
    prediction = predict_model(z, activation, idx, geom, alpha, model)
    h = np.clip(alpha * np.outer(activation[40], activation[40]), 0.0, 4.0)
    w = geom["base_weight"] * np.exp(h)
    lap = np.diag(w.sum(axis=1)) - w
    assert graph.shape == prediction.shape == (len(idx), 14)
    assert np.linalg.eigvalsh(lap).min() > -1.0e-9
    assert receipt["mean_edge_length_ratio"] <= 1.0
    assert receipt["mean_conductance_ratio"] >= 1.0
    assert model["beta"][1] >= 0.0
    print("PASS_A4_SELF_TEST")


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
