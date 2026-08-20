"""Frozen CloudCell heterogeneous-threshold/circuit-geometry validation.

This is an observational predictive test.  It never interprets fitted lag
weights as physical parent receipts or structural synapses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / "_workspace/ce/cloudcell-heterogeneous-circuit-geometry-20260820"
AUDIT = ROOT / "_workspace/ce/cloudcell-real-brain-metric-routing-20260820/artifacts/cloudcell-input-audit.json"
VALIDATION_JSON = RUN / "artifacts/validation-result.json"
CONFIRMATION_JSON = RUN / "artifacts/confirmation-result.json"
PRIMARY = {
    "BrainScanner20200310_141211",
    "BrainScanner20200310_142022",
    "BrainScanner20170424_105620",
    "BrainScanner20170610_105634",
    "BrainScanner20180709_100433",
}
GFP = {"BrainScanner20210503_122703"}
RIDGE = 1.0e-2
EPS = 1.0e-12
SEED = 20260820


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def recordings() -> list[dict[str, Any]]:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    rows = []
    for dataset in audit["datasets"]:
        for rec in dataset["recordings"]:
            if rec["recording"] in PRIMARY | GFP:
                rows.append(rec)
    names = {r["recording"] for r in rows}
    expected = PRIMARY | GFP
    if names != expected:
        raise RuntimeError(f"frozen recording mismatch: {sorted(expected - names)}")
    return sorted(rows, key=lambda r: r["recording"])


def robust_calibration(x: np.ndarray, k_idx: np.ndarray, common: bool = False) -> tuple[np.ndarray, np.ndarray]:
    cal = x[:, k_idx]
    if common:
        mu0 = float(np.nanmedian(cal))
        mad0 = float(1.4826 * np.nanmedian(np.abs(cal - mu0)))
        mu = np.full(x.shape[0], mu0)
        scale = np.full(x.shape[0], max(mad0, 1.0e-6))
    else:
        mu = np.nanmedian(cal, axis=1)
        scale = 1.4826 * np.nanmedian(np.abs(cal - mu[:, None]), axis=1)
        scale = np.maximum(scale, 1.0e-6)
    return mu, scale


def red_channel_residual(green: np.ndarray, red: np.ndarray, k_idx: np.ndarray) -> np.ndarray:
    residual = np.full_like(green, np.nan, dtype=float)
    for i in range(green.shape[0]):
        mask = np.isfinite(green[i, k_idx]) & np.isfinite(red[i, k_idx])
        if mask.sum() < 8:
            continue
        design = np.column_stack([red[i, k_idx][mask], np.ones(mask.sum())])
        beta, intercept = np.linalg.lstsq(design, green[i, k_idx][mask], rcond=None)[0]
        finite = np.isfinite(green[i]) & np.isfinite(red[i])
        residual[i, finite] = green[i, finite] - beta * red[i, finite] - intercept
    return residual


def prepare(rec: dict[str, Any], common: bool = False) -> dict[str, Any]:
    mat = loadmat(ROOT / rec["mat_path"], variable_names=["gRaw", "rRaw", "XYZcoord"])
    green = np.asarray(mat["gRaw"], dtype=float)[:, : rec["usable_timepoints"]]
    red = np.asarray(mat["rRaw"], dtype=float)[:, : rec["usable_timepoints"]]
    xyz = np.asarray(mat["XYZcoord"], dtype=float)
    policy = rec["clock_window_policy"]["admissible_anchor_indices"]
    train = np.asarray(policy["train"], dtype=int)
    cut = max(1, len(train) // 3)
    k_idx = train[:cut]
    b_idx = train[cut:]
    v_idx = np.asarray(policy["validation"], dtype=int)
    t_idx = np.asarray(policy["test"], dtype=int)
    x = red_channel_residual(green, red, k_idx)
    mu, scale = robust_calibration(x, k_idx, common=common)
    construction_samples = np.unique(np.concatenate([k_idx, b_idx, b_idx + 1]))
    finite_fraction = np.mean(np.isfinite(x[:, construction_samples]), axis=1)
    valid_neuron = (
        (finite_fraction >= 0.75)
        & np.isfinite(xyz).all(axis=1)
        & np.isfinite(mu)
        & np.isfinite(scale)
    )
    if valid_neuron.sum() < 8:
        raise RuntimeError(f"{rec['recording']}: fewer than 8 eligible neurons")
    z = (x[valid_neuron] - mu[valid_neuron, None]) / scale[valid_neuron, None]
    z = np.where(np.isfinite(z), z, 0.0)
    return {
        "name": rec["recording"],
        "signal_class": rec["signal_class"],
        "z": z,
        "xyz": xyz[valid_neuron],
        "k": k_idx,
        "b": b_idx,
        "v": v_idx,
        "t": t_idx,
        "eligible_neurons": int(valid_neuron.sum()),
        "excluded_neurons": int((~valid_neuron).sum()),
    }


def ridge_fit(z: np.ndarray, idx: np.ndarray, penalty: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    x = z[:, idx].T
    y = z[:, idx + 1].T
    xm = x.mean(axis=0)
    ym = y.mean(axis=0)
    xc = x - xm
    yc = y - ym
    d = x.shape[1]
    gram = xc.T @ xc
    rhs = xc.T @ yc
    if penalty is None:
        coef_source_target = np.linalg.solve(gram + RIDGE * np.eye(d), rhs)
        w = coef_source_target.T
    else:
        w = np.empty((d, d))
        for i in range(d):
            diag = RIDGE / np.clip(penalty[i], 0.5, 2.0)
            w[i] = np.linalg.solve(gram + np.diag(diag), rhs[:, i])
    intercept = ym - w @ xm
    return w, intercept


def knn_graph(xyz: np.ndarray, k: int = 6) -> tuple[np.ndarray, np.ndarray, float]:
    diff = xyz[:, None, :] - xyz[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    nn = np.argpartition(dist, kth=min(k, len(xyz) - 1) - 1, axis=1)[:, :k]
    adj = np.zeros_like(dist, dtype=bool)
    for i, js in enumerate(nn):
        adj[i, js] = True
    adj |= adj.T
    np.fill_diagonal(adj, False)
    scale = float(np.median(dist[adj]))
    if not math.isfinite(scale) or scale <= 0:
        raise RuntimeError("invalid spatial scale")
    normalized = dist / scale
    np.fill_diagonal(normalized, 0.0)
    return adj, normalized, scale


def scc_cycle_edges(alpha: np.ndarray, adj: np.ndarray) -> list[tuple[int, int, float]]:
    d = len(alpha)
    directed = np.zeros((d, d), dtype=np.int8)  # row=source, col=target for csgraph
    for i in range(d):
        for j in range(i + 1, d):
            if not adj[i, j]:
                continue
            # alpha[target, source]
            a_ij = alpha[i, j]
            a_ji = alpha[j, i]
            if a_ji > a_ij or (a_ji == a_ij and i < j):
                directed[i, j] = 1
            else:
                directed[j, i] = 1
    count, labels = connected_components(csr_matrix(directed), directed=True, connection="strong")
    sizes = np.bincount(labels, minlength=count)
    edges: list[tuple[int, int, float]] = []
    for source, target in zip(*np.nonzero(directed)):
        if labels[source] == labels[target] and sizes[labels[source]] >= 2:
            edges.append((int(source), int(target), float(alpha[target, source])))
    return edges


def source_from_edges(d: int, edges: list[tuple[int, int, float]]) -> np.ndarray:
    s = np.zeros(d)
    for source, target, strength in edges:
        s[source] += strength
        s[target] += strength
    positive = s[s > 0]
    if len(edges) < 2 or positive.size == 0:
        return np.zeros(d)
    s /= float(positive.mean())
    return s - s.mean()


def geometry(z: np.ndarray, idx: np.ndarray, xyz: np.ndarray, mode: str = "cycle") -> dict[str, Any]:
    w0, _ = ridge_fit(z, idx)
    denom = np.sum(np.abs(w0), axis=1, keepdims=True) + EPS
    pi = np.abs(w0) / denom
    z0 = z[:, idx]
    z1 = z[:, idx + 1]
    q0 = (z0 >= 2.5) * np.minimum(1.0, np.maximum(z0 - 2.5, 0.0) / 2.5)
    q1 = (z1 >= 2.5) * np.minimum(1.0, np.maximum(z1 - 2.5, 0.0) / 2.5)
    coactivity = np.einsum("it,jt->ij", q1, q0) / len(idx)
    alpha = pi * coactivity
    adj, dist, coordinate_scale = knn_graph(xyz)
    cycle_edges = scc_cycle_edges(alpha, adj)

    if mode == "shuffle":
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(len(alpha))
        cycle_edges = scc_cycle_edges(alpha[:, perm], adj)
    elif mode == "noncycle":
        strengths = sorted((e[2] for e in cycle_edges), reverse=True)
        candidates = []
        for i in range(len(alpha)):
            for j in range(i + 1, len(alpha)):
                if not adj[i, j]:
                    continue
                source, target = (i, j) if xyz[i, 0] <= xyz[j, 0] else (j, i)
                candidates.append((source, target))
        candidates.sort()
        cycle_edges = [(*candidates[n % len(candidates)], strength) for n, strength in enumerate(strengths)] if candidates else []
    elif mode == "time_shift":
        shifted = np.roll(q0, 37, axis=1)
        alpha_shift = pi * (np.einsum("it,jt->ij", q1, shifted) / len(idx))
        cycle_edges = scc_cycle_edges(alpha_shift, adj)

    s = source_from_edges(len(alpha), cycle_edges)
    degree = adj.sum(axis=1).astype(float)
    invsqrt = np.zeros_like(degree)
    invsqrt[degree > 0] = 1.0 / np.sqrt(degree[degree > 0])
    lap = np.eye(len(alpha)) - invsqrt[:, None] * adj.astype(float) * invsqrt[None, :]
    kfield = np.linalg.solve(np.eye(len(alpha)) + lap, -0.5 * s)
    exponent = -((np.exp((kfield[:, None] + kfield[None, :]) / 2.0) - 1.0) * dist**2)
    exponent[~np.isfinite(exponent)] = 0.0
    route = np.clip(np.exp(np.clip(exponent, -50.0, 50.0)), 0.5, 2.0)
    np.fill_diagonal(route, 1.0)
    return {
        "w0": w0,
        "route": route,
        "cycle_edge_count": len(cycle_edges),
        "cycle_node_count": len({u for e in cycle_edges for u in e[:2]}),
        "coordinate_scale": coordinate_scale,
        "k_min": float(kfield.min()),
        "k_max": float(kfield.max()),
        "route_min": float(route.min()),
        "route_max": float(route.max()),
        "laplacian_min_eigenvalue": float(np.linalg.eigvalsh(lap).min()),
    }


def score(z: np.ndarray, fit_idx: np.ndarray, eval_idx: np.ndarray, w: np.ndarray, intercept: np.ndarray, base_w: np.ndarray, base_b: np.ndarray) -> float:
    xb = z[:, fit_idx].T
    yb = z[:, fit_idx + 1].T
    base_resid = yb - (xb @ base_w.T + base_b)
    variance = np.maximum(np.mean(base_resid * base_resid, axis=0), 1.0e-8)
    x = z[:, eval_idx].T
    y = z[:, eval_idx + 1].T
    err = y - (x @ w.T + intercept)
    err0 = y - (x @ base_w.T + base_b)
    return float(np.mean(-0.5 * ((err * err - err0 * err0) / variance)))


def one_recording(prep: dict[str, Any], eval_role: str, use_a2: bool) -> dict[str, Any]:
    z, b_idx = prep["z"], prep["b"]
    eval_idx = prep[eval_role]
    base_w, base_b = ridge_fit(z, b_idx)
    geom = geometry(z, b_idx, prep["xyz"], "cycle")
    if use_a2:
        model_w, model_b = ridge_fit(z, b_idx, geom["route"])
    else:
        model_w = base_w * geom["route"]
        model_b = base_b.copy()
    delta = score(z, b_idx, eval_idx, model_w, model_b, base_w, base_b)
    pred_base = prep["z"][:, eval_idx].T @ base_w.T + base_b
    pred_model = prep["z"][:, eval_idx].T @ model_w.T + model_b

    controls = {}
    for mode in ("shuffle", "noncycle", "time_shift"):
        cg = geometry(z, b_idx, prep["xyz"], mode)
        if use_a2:
            cw, cb = ridge_fit(z, b_idx, cg["route"])
        else:
            cw, cb = base_w * cg["route"], base_b
        controls[mode] = {
            "delta": score(z, b_idx, eval_idx, cw, cb, base_w, base_b),
            "cycle_edge_count": cg["cycle_edge_count"],
        }

    common = prepare(next(r for r in recordings() if r["recording"] == prep["name"]), common=True)
    common_w, common_b = ridge_fit(common["z"], common["b"])
    common_g = geometry(common["z"], common["b"], common["xyz"], "cycle")
    if use_a2:
        common_mw, common_mb = ridge_fit(common["z"], common["b"], common_g["route"])
    else:
        common_mw, common_mb = common_w * common_g["route"], common_b
    controls["common_threshold"] = {
        "delta": score(common["z"], common["b"], common[eval_role], common_mw, common_mb, common_w, common_b),
        "cycle_edge_count": common_g["cycle_edge_count"],
    }
    spectral_base = float(np.max(np.abs(np.linalg.eigvals(base_w))))
    spectral_model = float(np.max(np.abs(np.linalg.eigvals(model_w))))
    return {
        "recording": prep["name"],
        "signal_class": prep["signal_class"],
        "eligible_neurons": prep["eligible_neurons"],
        "excluded_neurons": prep["excluded_neurons"],
        "delta_log_score": delta,
        "controls": controls,
        "geometry": {k: v for k, v in geom.items() if k not in {"w0", "route"}},
        "receipt": {
            "frobenius_ratio": float(np.linalg.norm(model_w) / max(np.linalg.norm(base_w), EPS)),
            "spectral_radius_ratio": spectral_model / max(spectral_base, EPS),
            "prediction_variance_ratio": float(np.var(pred_model) / max(np.var(pred_base), EPS)),
        },
    }


def sign_flip_p(values: list[float]) -> float:
    obs = float(sum(values))
    n = len(values)
    ge = 0
    for mask in range(1 << n):
        total = sum(v if (mask >> i) & 1 else -v for i, v in enumerate(values))
        ge += total >= obs - 1.0e-15
    return ge / (1 << n)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary = [r for r in rows if r["recording"] in PRIMARY]
    vals = [r["delta_log_score"] for r in primary]
    gfp = next(r for r in rows if r["recording"] in GFP)
    shuffle = [r["controls"]["shuffle"]["delta"] for r in primary]
    return {
        "primary_recording_count": len(primary),
        "positive_recording_count": sum(v > 0 for v in vals),
        "mean_delta_log_score": float(np.mean(vals)),
        "median_delta_log_score": float(np.median(vals)),
        "one_sided_exact_sign_flip_p": sign_flip_p(vals),
        "mean_shuffle_delta": float(np.mean(shuffle)),
        "gfp_delta": gfp["delta_log_score"],
    }


def run(stage: str) -> dict[str, Any]:
    if stage == "validation":
        eval_role = "v"
        use_a2 = False
    else:
        if not VALIDATION_JSON.exists():
            raise RuntimeError("confirmation requires frozen validation-result.json")
        validation = json.loads(VALIDATION_JSON.read_text(encoding="utf-8"))
        use_a2 = bool(validation["decision"]["activate_a2"])
        eval_role = "t"
    rows = [one_recording(prepare(rec), eval_role, use_a2) for rec in recordings()]
    agg = aggregate(rows)
    result: dict[str, Any] = {
        "status": "COMPLETE",
        "stage": stage,
        "formula": "A2_ANISOTROPIC_RIDGE" if use_a2 else "A1_POST_FIT_GAIN",
        "parent_receipt_gate": "BLOCKED_PARENT_RECEIPT",
        "claim_ceiling": "OBSERVATIONAL_HELDOUT_PREDICTIVE_FEATURE_ONLY",
        "source": {
            "audit_sha256": sha256(AUDIT),
            "script_sha256": sha256(Path(__file__)),
            "seed": SEED,
        },
        "rows": rows,
        "aggregate": agg,
    }
    if stage == "validation":
        activate = not (agg["positive_recording_count"] >= 4 and agg["mean_delta_log_score"] > 0)
        result["decision"] = {
            "a1_pass": not activate,
            "activate_a2": activate,
            "rule": "activate iff fewer than 4/5 positive or mean delta <= 0",
        }
    else:
        a = agg
        result["decision"] = {
            "pass_predictive_feature": bool(
                a["positive_recording_count"] >= 4
                and a["mean_delta_log_score"] > 0
                and a["one_sided_exact_sign_flip_p"] < 0.05
                and a["mean_delta_log_score"] > a["mean_shuffle_delta"]
                and a["gfp_delta"] < a["mean_delta_log_score"]
            )
        }
    return result


def self_test() -> None:
    rng = np.random.default_rng(7)
    z = rng.normal(size=(10, 180))
    xyz = rng.normal(size=(10, 3))
    idx = np.arange(20, 130)
    w, b = ridge_fit(z, idx)
    geom = geometry(z, idx, xyz)
    assert w.shape == (10, 10) and b.shape == (10,)
    assert geom["route"].shape == (10, 10)
    assert np.all((geom["route"] >= 0.5) & (geom["route"] <= 2.0))
    assert geom["laplacian_min_eigenvalue"] > -1.0e-9
    wr, br = ridge_fit(z, idx, geom["route"])
    assert np.isfinite(score(z, idx, np.arange(140, 170), wr, br, w, b))
    print("PASS_SELF_TEST")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("validation", "confirmation"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.stage:
        parser.error("--stage is required unless --self-test is used")
    result = run(args.stage)
    out = VALIDATION_JSON if args.stage == "validation" else CONFIRMATION_JSON
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "aggregate": result["aggregate"], "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
