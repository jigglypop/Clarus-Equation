"""A3 low-dimensional connection-operator test on DANDI 000541.

The route is observational.  Development and confirmation worms are separated
by the frozen source-manifest order; confirmation cannot run unless development
passes its preregistered gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import lindi
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / "_workspace/ce/dandi-neuropal-circuit-connection-20260820"
DATA = ROOT / "data/external/dandi"
MANIFEST = RUN / "artifacts/source-manifest-v2.json"
DEV_RESULT = RUN / "artifacts/a3-development-result.json"
CONFIRM_RESULT = RUN / "artifacts/a3-confirmation-result.json"
RIDGE = 1.0e-2
SEED = 20260820
EPS = 1.0e-12
SHIFT_LAGS = (17, 31, 47)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean_strings(values) -> list[str]:
    out = []
    for x in np.asarray(values).reshape(-1):
        x = x.item() if isinstance(x, np.generic) else x
        if isinstance(x, bytes):
            x = x.decode("utf-8", errors="replace")
        out.append("" if x is None else str(x).strip())
    return out


def load_asset(asset: dict[str, Any]) -> dict[str, Any]:
    lindi_path = DATA / f"000541-{asset['id']}.nwb.lindi.json"
    if not lindi_path.exists():
        raise RuntimeError(f"missing LINDI index: {lindi_path}")
    f = lindi.LindiH5pyFile.from_lindi_file(str(lindi_path.resolve()))
    base = "/processing/CalciumActivity"
    series = f[f"{base}/SignalRawFluordNMF/dNMFCalciumImResponseSeries"]
    x = np.asarray(series["data"][:], dtype=float)
    rois = np.asarray(series["rois"][:], dtype=int)
    labels = clean_strings(f[f"{base}/NeuronIDs/labels"][()])
    mask_ds = f["/processing/NeuroPAL/NeuroPALSegmentation/NeuroPALNeurons/voxel_mask"]
    mask_rows = np.asarray(list(mask_ds._zarr_array[:]), dtype=float)
    spacing = np.asarray(f["/general/optophysiology/CalciumImVol/grid_spacing"][()], dtype=float)
    start_ds = series["starting_time"]
    start = float(np.asarray(start_ds[()]))
    rate = float(start_ds.attrs["rate"])
    stim = f["/intervals/chemical_stimuli"]
    stim_start = np.asarray(stim["start_time"][:], dtype=float)
    stim_stop = np.asarray(stim["stop_time"][:], dtype=float)
    stim_names = clean_strings(stim["stimulus"][()])
    if x.ndim != 2 or x.shape[1] != len(labels) or x.shape[1] != len(mask_rows):
        raise RuntimeError(f"row join mismatch {asset['id']}: {x.shape}, {len(labels)}, {len(mask_rows)}")
    if not np.array_equal(rois, np.arange(len(labels))):
        raise RuntimeError(f"ROI order mismatch {asset['id']}")
    return {
        "asset_id": asset["id"],
        "lindi_sha256": file_sha256(lindi_path),
        "x": x,
        "labels": labels,
        "coords": mask_rows[:, :3] * spacing[None, :],
        "rate": rate,
        "start": start,
        "stim_start": stim_start,
        "stim_stop": stim_stop,
        "stim_names": stim_names,
    }


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
    construction_idx = np.flatnonzero((np.arange(n_time) < b_end) & allowed)
    finite_fraction = np.mean(np.isfinite(x[construction_idx]), axis=0)
    labels = np.asarray(raw["labels"], dtype=object)
    coords = raw["coords"]
    keep = (
        (labels != "")
        & (finite_fraction >= 0.75)
        & np.isfinite(coords).all(axis=1)
    )
    if keep.sum() < 20 or len(k_idx) < 40:
        raise RuntimeError(f"insufficient eligible data {raw['asset_id']}: neurons={keep.sum()}, K={len(k_idx)}")
    x = x[:, keep]
    coords = coords[keep]
    labels = labels[keep]
    mu = np.nanmedian(x[k_idx], axis=0)
    scale = 1.4826 * np.nanmedian(np.abs(x[k_idx] - mu[None, :]), axis=0)
    good_scale = np.isfinite(mu) & np.isfinite(scale) & (scale > 1.0e-6)
    x, coords, labels = x[:, good_scale], coords[good_scale], labels[good_scale]
    z = (x - mu[good_scale][None, :]) / scale[good_scale][None, :]
    z = np.where(np.isfinite(z), z, 0.0)
    km = z[k_idx].mean(axis=0)
    _, _, vt = np.linalg.svd(z[k_idx] - km, full_matrices=False)
    rank = min(3, vt.shape[0])
    basis = vt[:rank]
    centered = z - km
    residual = centered - (centered @ basis.T) @ basis
    threshold_gate = (z >= 2.5) * np.minimum(1.0, np.maximum(z - 2.5, 0.0) / 2.5)
    event_residual = threshold_gate * residual
    return {
        **raw,
        "z": z,
        "residual": residual,
        "threshold_gate": threshold_gate,
        "event_residual": event_residual,
        "coords": coords,
        "labels": labels.tolist(),
        "allowed": allowed,
        "b_domain": b_domain,
        "test_domain": test_domain,
        "k_count": int(len(k_idx)),
        "eligible_neurons": int(z.shape[1]),
        "excluded_neurons": int(raw["x"].shape[1] - z.shape[1]),
    }


def pair_indices(domain: np.ndarray, lag_before: int = 1, lag_after: int = 1) -> np.ndarray:
    idx = np.arange(len(domain))
    ok = domain.copy()
    for lag in range(1, lag_before + 1):
        ok &= np.roll(domain, lag)
        ok[:lag] = False
    for lag in range(1, lag_after + 1):
        ok &= np.roll(domain, -lag)
        ok[-lag:] = False
    return idx[ok]


def spatial_graph(coords: np.ndarray, k: int = 6) -> tuple[np.ndarray, np.ndarray]:
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    kk = min(k, len(coords) - 1)
    nn = np.argpartition(dist, kth=kk - 1, axis=1)[:, :kk]
    adj = np.zeros_like(dist, dtype=bool)
    for i, js in enumerate(nn):
        adj[i, js] = True
    adj |= adj.T
    np.fill_diagonal(adj, False)
    scale = float(np.median(dist[adj]))
    norm_dist = dist / scale
    weights = np.zeros_like(norm_dist)
    weights[adj] = 1.0 / np.maximum(norm_dist[adj] ** 2, 1.0e-6)
    return adj, weights


def laplacian(weights: np.ndarray) -> np.ndarray:
    sym = (weights + weights.T) / 2.0
    np.fill_diagonal(sym, 0.0)
    return np.diag(sym.sum(axis=1)) - sym


def connection_operators(
    prep: dict[str, Any],
    normalize: bool = False,
    residual_override: np.ndarray | None = None,
    coords_override: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    r = prep["event_residual"] if residual_override is None else residual_override
    b = prep["b_domain"]
    idx = pair_indices(b, 1, 1)
    cplus = r[idx + 1].T @ r[idx] / len(idx)
    cminus = r[idx - 1].T @ r[idx] / len(idx)
    shifted = []
    for tau in SHIFT_LAGS:
        valid = b.copy()
        valid[:tau] = False
        valid &= np.roll(b, tau)
        valid &= np.roll(b, -1)
        valid[-1:] = False
        ti = np.flatnonzero(valid)
        shifted.append(r[ti + 1].T @ r[ti - tau] / len(ti))
    cshift = np.mean(shifted, axis=0)
    c = np.maximum(0.0, (cplus + cplus.T) / 2.0 - (cshift + cshift.T) / 2.0)
    omega = ((cplus - cminus) - (cplus - cminus).T) / 4.0
    adj, spatial_weights = spatial_graph(prep["coords"] if coords_override is None else coords_override)
    c *= adj
    omega *= adj
    lc = laplacian(c)
    lsp = laplacian(spatial_weights)
    c_time = np.maximum(0.0, (cshift + cshift.T) / 2.0) * adj
    omega_time = (cshift - cshift.T) / 2.0 * adj
    ltime = laplacian(c_time)
    if normalize:
        for op in (lc, omega, lsp, ltime, omega_time):
            norm = float(np.linalg.norm(op, 2))
            if norm > 1.0:
                op /= norm
    return {"lc": lc, "omega": omega, "lsp": lsp, "ltime": ltime, "omega_time": omega_time}


def phase_randomized_event_residual(prep: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    """Preserve each neuron's spectrum inside each allowed construction block."""
    source = prep["event_residual"]
    out = np.zeros_like(source)
    domain = prep["b_domain"]
    edges = np.diff(np.r_[False, domain, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    for start, stop in zip(starts, stops):
        block = source[start:stop]
        if len(block) < 4:
            continue
        spectrum = np.fft.rfft(block, axis=0)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=spectrum.shape)
        phase[0] = 0.0
        if len(block) % 2 == 0:
            phase[-1] = 0.0
        randomized = spectrum * np.exp(1j * phase)
        out[start:stop] = np.fft.irfft(randomized, n=len(block), axis=0)
    return out


def feature_blocks(z: np.ndarray, idx: np.ndarray, lmat: np.ndarray, omega: np.ndarray | None) -> list[np.ndarray]:
    blocks = [z[idx], -(z[idx] @ lmat.T)]
    if omega is not None:
        blocks.append(z[idx] @ omega.T)
    return blocks


def fit_scalar_operator(z: np.ndarray, idx: np.ndarray, lmat: np.ndarray, omega: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    blocks = feature_blocks(z, idx, lmat, omega)
    y = z[idx + 1]
    ymean = y.mean(axis=0)
    means = [b.mean(axis=0) for b in blocks]
    design = np.column_stack([(b - m).reshape(-1) for b, m in zip(blocks, means)])
    target = (y - ymean).reshape(-1)
    beta = np.linalg.solve(design.T @ design + RIDGE * np.eye(design.shape[1]), design.T @ target)
    intercept = ymean - sum(beta[j] * means[j] for j in range(len(blocks)))
    return beta, intercept


def predict(z: np.ndarray, idx: np.ndarray, lmat: np.ndarray, omega: np.ndarray | None, beta: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    blocks = feature_blocks(z, idx, lmat, omega)
    return intercept + sum(beta[j] * blocks[j] for j in range(len(blocks)))


def evaluate(prep: dict[str, Any], normalize: bool = False) -> dict[str, Any]:
    z = prep["z"]
    bidx = pair_indices(prep["b_domain"], 0, 1)
    tidx = pair_indices(prep["test_domain"], 0, 1)
    ops = connection_operators(prep, normalize=normalize)
    base_beta, base_b = fit_scalar_operator(z, bidx, ops["lsp"], None)
    base_train_resid = z[bidx + 1] - predict(z, bidx, ops["lsp"], None, base_beta, base_b)
    variance = np.maximum(np.mean(base_train_resid**2, axis=0), 1.0e-8)

    rng_seed = int(hashlib.sha256(prep["asset_id"].encode()).hexdigest()[:8], 16) ^ SEED
    rng = np.random.default_rng(rng_seed)
    perm = rng.permutation(z.shape[1])
    lc_shuffle = ops["lc"][np.ix_(perm, perm)]
    om_shuffle = ops["omega"][np.ix_(perm, perm)]
    identity_ops = connection_operators(prep, normalize=normalize, coords_override=prep["coords"][perm])
    phase_ops = connection_operators(
        prep,
        normalize=normalize,
        residual_override=phase_randomized_event_residual(prep, rng),
    )
    arms = {
        "a3": (ops["lc"], ops["omega"]),
        "no_circulation": (ops["lc"], None),
        "edge_shuffle": (lc_shuffle, om_shuffle),
        "time_shift": (ops["ltime"], ops["omega_time"]),
        "reversal": (ops["lc"], -ops["omega"]),
        "identity_shuffle": (identity_ops["lc"], identity_ops["omega"]),
        "phase_randomized": (phase_ops["lc"], phase_ops["omega"]),
    }
    ytest = z[tidx + 1]
    base_pred = predict(z, tidx, ops["lsp"], None, base_beta, base_b)
    base_err = ytest - base_pred
    arm_rows = {}
    for name, (lmat, omega) in arms.items():
        beta, intercept = fit_scalar_operator(z, bidx, lmat, omega)
        pred = predict(z, tidx, lmat, omega, beta, intercept)
        err = ytest - pred
        delta = float(np.mean(-0.5 * ((err**2 - base_err**2) / variance)))
        if omega is None:
            dynamic = beta[0] * np.eye(z.shape[1]) - beta[1] * lmat
        else:
            dynamic = beta[0] * np.eye(z.shape[1]) - beta[1] * lmat + beta[2] * omega
        arm_rows[name] = {
            "delta_log_score": delta,
            "beta": beta.tolist(),
            "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(dynamic)))),
        }
    return {
        "asset_id": prep["asset_id"],
        "lindi_sha256": prep["lindi_sha256"],
        "trace_shape": list(prep["x"].shape),
        "rate_hz": prep["rate"],
        "labels_sha256": hashlib.sha256(json.dumps(prep["labels"]).encode()).hexdigest(),
        "eligible_neurons": prep["eligible_neurons"],
        "excluded_neurons": prep["excluded_neurons"],
        "calibration_samples": prep["k_count"],
        "threshold": 2.5,
        "threshold_active_fraction": float(np.mean(prep["threshold_gate"][prep["b_domain"]] > 0)),
        "construction_pairs": int(len(bidx)),
        "test_pairs": int(len(tidx)),
        "stimuli": list(zip(prep["stim_start"].tolist(), prep["stim_stop"].tolist(), prep["stim_names"])),
        "operator_receipt": {
            "laplacian_min_eigenvalue": float(np.linalg.eigvalsh(ops["lc"]).min()),
            "omega_skew_error": float(np.max(np.abs(ops["omega"] + ops["omega"].T))),
            "laplacian_rank": int(np.linalg.matrix_rank(ops["lc"], tol=1.0e-8)),
            "omega_rank": int(np.linalg.matrix_rank(ops["omega"], tol=1.0e-8)),
        },
        "arms": arm_rows,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = (
        "a3",
        "no_circulation",
        "edge_shuffle",
        "time_shift",
        "reversal",
        "identity_shuffle",
        "phase_randomized",
    )
    means = {name: float(np.mean([r["arms"][name]["delta_log_score"] for r in rows])) for name in names}
    positive = sum(r["arms"]["a3"]["delta_log_score"] > 0 for r in rows)
    return {"worm_count": len(rows), "positive_a3_count": positive, "mean_delta": means}


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
    normalize = False
    if stage == "confirmation":
        if not DEV_RESULT.exists():
            raise RuntimeError("confirmation requires development result")
        dev = json.loads(DEV_RESULT.read_text(encoding="utf-8"))
        if not dev["decision"]["pass_development"]:
            raise RuntimeError("confirmation sealed because development did not pass")
        normalize = bool(dev["decision"].get("normalized_revision", False))
    rows = [evaluate(split_and_standardize(load_asset(asset)), normalize=normalize) for asset in assets]
    agg = aggregate(rows)
    if stage == "development":
        a3_mean = agg["mean_delta"]["a3"]
        controls = [agg["mean_delta"][n] for n in agg["mean_delta"] if n != "a3"]
        passed = agg["positive_a3_count"] >= 2 and a3_mean > 0 and all(a3_mean > x for x in controls)
        unstable = any(r["arms"]["a3"]["spectral_radius"] >= 1.0 for r in rows)
        normalized_revision = False
        if not passed and unstable:
            rows = [evaluate(split_and_standardize(load_asset(asset)), normalize=True) for asset in assets]
            agg = aggregate(rows)
            a3_mean = agg["mean_delta"]["a3"]
            controls = [agg["mean_delta"][n] for n in agg["mean_delta"] if n != "a3"]
            passed = agg["positive_a3_count"] >= 2 and a3_mean > 0 and all(a3_mean > x for x in controls)
            normalized_revision = True
        decision = {
            "pass_development": passed,
            "instability_trigger": unstable,
            "normalized_revision": normalized_revision,
            "confirmation_open": passed,
        }
    else:
        vals = [r["arms"]["a3"]["delta_log_score"] for r in rows]
        a3_mean = agg["mean_delta"]["a3"]
        controls = [agg["mean_delta"][n] for n in agg["mean_delta"] if n != "a3"]
        p = sign_flip_p(vals)
        decision = {
            "exact_sign_flip_p": p,
            "pass_a3": bool(len(vals) == 5 and all(v > 0 for v in vals) and p < 0.05 and all(a3_mean > x for x in controls)),
        }
    return {
        "status": "COMPLETE",
        "stage": stage,
        "claim_ceiling": "OBSERVATIONAL_CONNECTION_OPERATOR_ONLY",
        "source_manifest_sha256": file_sha256(MANIFEST),
        "script_sha256": file_sha256(Path(__file__)),
        "rows": rows,
        "aggregate": agg,
        "decision": decision,
    }


def self_test() -> None:
    rng = np.random.default_rng(4)
    z = rng.normal(size=(160, 12))
    idx = np.arange(20, 120)
    coords = rng.normal(size=(12, 3))
    _, weights = spatial_graph(coords)
    lmat = laplacian(weights)
    omega = rng.normal(size=(12, 12))
    omega = (omega - omega.T) / 2
    beta, intercept = fit_scalar_operator(z, idx, lmat, omega)
    pred = predict(z, idx, lmat, omega, beta, intercept)
    assert pred.shape == (len(idx), 12)
    assert np.linalg.eigvalsh(lmat).min() > -1.0e-9
    assert np.max(np.abs(omega + omega.T)) < 1.0e-12
    print("PASS_SELF_TEST")


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
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "aggregate": result["aggregate"], "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
