from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import remfile
from dandi.dandiapi import DandiAPIClient
from pynwb import NWBHDF5IO
from scipy.linalg import svd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DANDISET = "001695"
VERSION = "0.260319.2023"
ASSETS = [
    "sub-M02/sub-M02_ses-20240313T100000_behavior+ecephys.nwb",
    "sub-M02/sub-M02_ses-20240314T100000_behavior+ecephys.nwb",
]
DT = 0.02
MAX_DURATION_S = 600.0
MAX_UNITS = 80
N_COMPONENTS = 12
RIDGE_ALPHA = 10.0
RANK = 3
N_FOLDS = 5
GAP_BINS = 15
SEED = 20260819
OUT = Path("dandi_bridge_results")
OUT.mkdir(exist_ok=True)


def decode_strs(x: Any) -> np.ndarray:
    arr = np.asarray(x)
    out = []
    for v in arr:
        if isinstance(v, bytes):
            out.append(v.decode("utf-8", errors="replace"))
        else:
            out.append(str(v))
    return np.asarray(out)


def choose_indices(areas: np.ndarray, types: np.ndarray, area: str) -> np.ndarray:
    if area in {"CA1", "CA3"}:
        mask = (areas == area) & (types == "Pyramidal Cell")
    else:
        mask = (areas == area) & (types != "Narrow Interneuron")
    idx = np.where(mask)[0]
    return idx[:MAX_UNITS]


def bin_spikes(spikes: list[np.ndarray], start: float, stop: float) -> np.ndarray:
    edges = np.arange(start, stop + DT * 0.5, DT)
    mats = []
    for s in spikes:
        s = np.asarray(s, dtype=float)
        s = s[(s >= start) & (s < stop)]
        mats.append(np.histogram(s, bins=edges)[0])
    x = np.stack(mats, axis=1).astype(float)
    # Variance-stabilizing transform for sparse counts.
    return np.sqrt(x)


def load_session(path: str) -> dict[str, Any]:
    with DandiAPIClient() as client:
        ds = client.get_dandiset(DANDISET, VERSION)
        asset = ds.get_asset_by_path(path)
        url = asset.get_content_url(follow_redirects=1, strip_query=True)
        asset_size = int(asset.size)

    r_file = remfile.File(url)
    h5_file = h5py.File(r_file, "r")
    try:
        with NWBHDF5IO(file=h5_file, mode="r", load_namespaces=True) as io:
            nwb = io.read()
            areas = decode_strs(nwb.units["cell_area"][:])
            types = decode_strs(nwb.units["cell_type"][:])
            indices = {a: choose_indices(areas, types, a) for a in ("CA3", "CA1", "RSC")}
            behavior = nwb.processing["behavior"]
            pos_ts = np.asarray(behavior["AnimalPosition"]["Position"].timestamps[:], dtype=float)
            start = float(pos_ts[0])
            stop = min(float(pos_ts[-1]), start + MAX_DURATION_S)
            spikes = {
                a: [np.asarray(nwb.units.get_unit_spike_times(int(i)), dtype=float) for i in idx]
                for a, idx in indices.items()
            }
    finally:
        try:
            h5_file.close()
        except Exception:
            pass

    data = {a: bin_spikes(v, start, stop) for a, v in spikes.items()}
    keep = {}
    for a, x in data.items():
        k = (np.var(x, axis=0) > 0) & (np.sum(x > 0, axis=0) >= 10)
        data[a] = x[:, k]
        keep[a] = int(np.sum(k))
    t = min(x.shape[0] for x in data.values())
    data = {a: x[:t] for a, x in data.items()}
    return {
        "path": path,
        "asset_size": asset_size,
        "start": start,
        "stop": stop,
        "duration": stop - start,
        "unit_counts_selected": {a: int(len(indices[a])) for a in indices},
        "unit_counts_kept": keep,
        "data": data,
    }


def contiguous_folds(n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    edges = np.linspace(0, n, N_FOLDS + 1, dtype=int)
    folds = []
    all_idx = np.arange(n)
    for k in range(N_FOLDS):
        lo, hi = edges[k], edges[k + 1]
        test = np.arange(lo, hi)
        train_mask = np.ones(n, dtype=bool)
        train_mask[max(0, lo - GAP_BINS):min(n, hi + GAP_BINS)] = False
        train = all_idx[train_mask]
        folds.append((train, test))
    return folds


def fit_pca(train: np.ndarray, all_x: np.ndarray) -> tuple[np.ndarray, StandardScaler, PCA]:
    scaler = StandardScaler().fit(train)
    ztr = scaler.transform(train)
    ncomp = min(N_COMPONENTS, train.shape[1], max(1, train.shape[0] - 1))
    pca = PCA(n_components=ncomp, svd_solver="full", random_state=SEED).fit(ztr)
    return pca.transform(scaler.transform(all_x)), scaler, pca


def diag_nlpd(y: np.ndarray, pred: np.ndarray, var: np.ndarray) -> float:
    var = np.maximum(np.asarray(var, dtype=float), 1e-6)
    r = y - pred
    return float(np.mean(0.5 * (np.sum(r * r / var, axis=1) + np.sum(np.log(2 * np.pi * var)))))


def fit_eval_direction(
    source_raw: np.ndarray,
    target_raw: np.ndarray,
    control_raw: np.ndarray,
    lag_bins: int,
    seed_offset: int,
) -> dict[str, Any]:
    n = min(len(source_raw), len(target_raw), len(control_raw)) - lag_bins
    src0 = source_raw[:n]
    tgt0 = target_raw[:n]
    ctl0 = control_raw[:n]
    tgt1 = target_raw[lag_bins:lag_bins + n]
    folds = contiguous_folds(n)
    rng = np.random.default_rng(SEED + seed_offset)
    fold_results = []

    for fold_id, (tr, te) in enumerate(folds):
        src, _, _ = fit_pca(src0[tr], src0)
        tgt_now, tgt_scaler, tgt_pca = fit_pca(tgt0[tr], tgt0)
        # Apply target scaler/PCA fitted on current target to future target.
        tgt_future = tgt_pca.transform(tgt_scaler.transform(tgt1))
        ctl, _, _ = fit_pca(ctl0[tr], ctl0)

        c = np.concatenate([tgt_now, ctl], axis=1)
        y = tgt_future
        base = Ridge(alpha=RIDGE_ALPHA).fit(c[tr], y[tr])
        pred_base_train = base.predict(c[tr])
        pred_base_test = base.predict(c[te])
        y_res = y[tr] - pred_base_train

        src_base = Ridge(alpha=RIDGE_ALPHA).fit(c[tr], src[tr])
        src_res_train = src[tr] - src_base.predict(c[tr])
        src_res_all = src - src_base.predict(c)

        cross = src_res_train.T @ y_res / max(1, len(tr))
        u, singular, vt = svd(cross, full_matrices=False)
        k = min(RANK, u.shape[1], vt.shape[0])
        u_k = u[:, :k]
        z_bridge = src_res_all @ u_k
        feat_bridge = np.concatenate([c, z_bridge], axis=1)
        bridge = Ridge(alpha=RIDGE_ALPHA).fit(feat_bridge[tr], y[tr])
        pred_bridge_train = bridge.predict(feat_bridge[tr])
        pred_bridge_test = bridge.predict(feat_bridge[te])

        # Full source model, more flexible than bridge model.
        feat_full = np.concatenate([c, src], axis=1)
        full = Ridge(alpha=RIDGE_ALPHA).fit(feat_full[tr], y[tr])
        pred_full_train = full.predict(feat_full[tr])
        pred_full_test = full.predict(feat_full[te])

        # Same-rank random source subspace.
        q, _ = np.linalg.qr(rng.normal(size=(src.shape[1], k)))
        z_random = src_res_all @ q[:, :k]
        feat_random = np.concatenate([c, z_random], axis=1)
        random_model = Ridge(alpha=RIDGE_ALPHA).fit(feat_random[tr], y[tr])
        pred_random_train = random_model.predict(feat_random[tr])
        pred_random_test = random_model.predict(feat_random[te])

        # Circularly shifted source control, 5 seconds.
        shift = min(max(1, int(round(5.0 / DT))), n // 3)
        src_shift = np.roll(src, shift, axis=0)
        shift_base = Ridge(alpha=RIDGE_ALPHA).fit(c[tr], src_shift[tr])
        shift_res = src_shift - shift_base.predict(c)
        cross_shift = shift_res[tr].T @ y_res / max(1, len(tr))
        us, _, _ = svd(cross_shift, full_matrices=False)
        z_shift = shift_res @ us[:, :k]
        feat_shift = np.concatenate([c, z_shift], axis=1)
        shifted = Ridge(alpha=RIDGE_ALPHA).fit(feat_shift[tr], y[tr])
        pred_shift_train = shifted.predict(feat_shift[tr])
        pred_shift_test = shifted.predict(feat_shift[te])

        models = {
            "baseline": (pred_base_train, pred_base_test),
            "bridge_rank3": (pred_bridge_train, pred_bridge_test),
            "full_source": (pred_full_train, pred_full_test),
            "random_rank3": (pred_random_train, pred_random_test),
            "shifted_rank3": (pred_shift_train, pred_shift_test),
        }
        metrics = {}
        for name, (ptr, pte) in models.items():
            var = np.var(y[tr] - ptr, axis=0, ddof=1) + 1e-6
            mse = float(np.mean((y[te] - pte) ** 2))
            metrics[name] = {
                "mse": mse,
                "nlpd": diag_nlpd(y[te], pte, var),
            }
        fold_results.append({
            "fold": fold_id,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "singular_values": singular[: min(10, len(singular))].tolist(),
            "metrics": metrics,
        })

    summary = {}
    for name in fold_results[0]["metrics"]:
        summary[name] = {
            "mse_mean": float(np.mean([f["metrics"][name]["mse"] for f in fold_results])),
            "nlpd_mean": float(np.mean([f["metrics"][name]["nlpd"] for f in fold_results])),
        }
    b = summary["baseline"]
    for name, row in summary.items():
        row["mse_improvement_vs_baseline"] = b["mse_mean"] - row["mse_mean"]
        row["nlpd_improvement_vs_baseline"] = b["nlpd_mean"] - row["nlpd_mean"]
    return {"lag_bins": lag_bins, "lag_ms": lag_bins * DT * 1000, "folds": fold_results, "summary": summary}


def analyze_session(session: dict[str, Any]) -> dict[str, Any]:
    d = session["data"]
    directions = [
        ("CA3_to_CA1", "CA3", "CA1", "RSC"),
        ("CA1_to_CA3", "CA1", "CA3", "RSC"),
        ("CA1_to_RSC", "CA1", "RSC", "CA3"),
        ("RSC_to_CA1", "RSC", "CA1", "CA3"),
    ]
    out = {}
    for i, (name, src, tgt, ctl) in enumerate(directions):
        out[name] = {
            "20ms": fit_eval_direction(d[src], d[tgt], d[ctl], 1, i * 100),
            "40ms": fit_eval_direction(d[src], d[tgt], d[ctl], 2, i * 100 + 1),
        }
    return {
        "path": session["path"],
        "asset_size": session["asset_size"],
        "duration": session["duration"],
        "unit_counts_selected": session["unit_counts_selected"],
        "unit_counts_kept": session["unit_counts_kept"],
        "directions": out,
    }


def make_report(results: dict[str, Any]) -> str:
    lines = [
        "# DANDI 001695 direct conditional bridge pilot",
        "",
        f"Dandiset `{DANDISET}` version `{VERSION}`; dt `{DT}` s; max duration `{MAX_DURATION_S}` s; rank `{RANK}`.",
        "",
        "This is a two-session file-level pilot, not an animal-level confirmation.",
        "",
        "| session | direction | lag | bridge ΔNLPD | full-source ΔNLPD | random ΔNLPD | shifted ΔNLPD |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for s in results["sessions"]:
        short = s["path"].split("/")[-1]
        for direction, lags in s["directions"].items():
            for lag_name, r in lags.items():
                sm = r["summary"]
                lines.append(
                    f"| {short} | {direction} | {lag_name} | "
                    f"{sm['bridge_rank3']['nlpd_improvement_vs_baseline']:.5f} | "
                    f"{sm['full_source']['nlpd_improvement_vs_baseline']:.5f} | "
                    f"{sm['random_rank3']['nlpd_improvement_vs_baseline']:.5f} | "
                    f"{sm['shifted_rank3']['nlpd_improvement_vs_baseline']:.5f} |"
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    sessions = []
    for path in ASSETS:
        print(f"stream {path}", flush=True)
        raw = load_session(path)
        print({k: v for k, v in raw.items() if k != "data"}, flush=True)
        sessions.append(analyze_session(raw))
    results = {
        "status": "COMPLETE",
        "seed": SEED,
        "dandiset": DANDISET,
        "version": VERSION,
        "assets": ASSETS,
        "method": {
            "dt": DT,
            "max_duration_s": MAX_DURATION_S,
            "max_units_per_region": MAX_UNITS,
            "pca_components": N_COMPONENTS,
            "ridge_alpha": RIDGE_ALPHA,
            "bridge_rank": RANK,
            "folds": N_FOLDS,
            "gap_bins": GAP_BINS,
        },
        "sessions": sessions,
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (OUT / "report.md").write_text(make_report(results), encoding="utf-8")
    print(make_report(results), flush=True)


if __name__ == "__main__":
    main()
