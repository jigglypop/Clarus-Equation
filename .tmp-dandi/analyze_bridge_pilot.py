from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import requests
from pynwb import NWBHDF5IO

ASSET_ID = "1e4d5403-a8cc-4814-a904-7aff57f8cc4d"
DOWNLOAD = f"https://api.dandiarchive.org/api/assets/{ASSET_ID}/download/"
FILE = Path("/tmp/dandi001695_m02_20240312.nwb")
OUT = Path("dandi_bridge_results")
OUT.mkdir(exist_ok=True)
SEED = 20260819
RNG = np.random.default_rng(SEED)
BIN = 0.05
RANKS = (2, 3, 5, 8)
RIDGES = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
LAGS = (1, 2, 4)
BOOT = 5000


def download() -> None:
    if FILE.exists() and FILE.stat().st_size > 1_000_000:
        return
    with requests.get(DOWNLOAD, stream=True, timeout=120, allow_redirects=True) as r:
        r.raise_for_status()
        with FILE.open("wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)


def behavior_interval(nwb) -> tuple[float, float, dict]:
    meta = {}
    if "behavior" in nwb.processing:
        module = nwb.processing["behavior"]
        meta["interfaces"] = list(module.data_interfaces.keys())
        for name, obj in module.data_interfaces.items():
            if hasattr(obj, "spatial_series"):
                for sname, ss in obj.spatial_series.items():
                    try:
                        if ss.timestamps is not None:
                            ts = np.asarray(ss.timestamps[:], dtype=float)
                        else:
                            n = len(ss.data)
                            ts = float(ss.starting_time) + np.arange(n) / float(ss.rate)
                        ts = ts[np.isfinite(ts)]
                        if len(ts) > 10:
                            meta["selected"] = {"interface": name, "series": sname, "n": len(ts), "start": float(ts.min()), "stop": float(ts.max())}
                            return float(ts.min()), float(ts.max()), meta
                    except Exception as exc:
                        meta.setdefault("errors", []).append(repr(exc))
    # fallback to spike-time support
    df = nwb.units.to_dataframe()
    starts, stops = [], []
    for x in df["spike_times"]:
        a = np.asarray(x, dtype=float)
        if len(a):
            starts.append(float(a.min())); stops.append(float(a.max()))
    return max(starts), min(stops), {**meta, "fallback": "common spike support"}


def bin_region(df, region: str, start: float, stop: float) -> tuple[np.ndarray, list[int]]:
    idx = [i for i, row in df.iterrows() if str(row["cell_area"]).upper() == region.upper()]
    edges = np.arange(start, stop + BIN, BIN)
    counts = np.zeros((len(edges) - 1, len(idx)), dtype=float)
    for j, i in enumerate(idx):
        sp = np.asarray(df.loc[i, "spike_times"], dtype=float)
        counts[:, j] = np.histogram(sp, bins=edges)[0]
    return counts, idx


def fit_pca(train: np.ndarray, all_data: np.ndarray, rank: int) -> tuple[np.ndarray, dict]:
    x = np.sqrt(all_data)
    mu = np.mean(np.sqrt(train), axis=0)
    sd = np.std(np.sqrt(train), axis=0)
    sd[sd < 1e-6] = 1.0
    z = (x - mu) / sd
    ztr = (np.sqrt(train) - mu) / sd
    _, s, vt = np.linalg.svd(ztr, full_matrices=False)
    rank = min(rank, vt.shape[0], vt.shape[1])
    basis = vt[:rank].T
    scores = z @ basis
    return scores, {"mean": mu.tolist(), "sd": sd.tolist(), "singular_values": s[:rank].tolist(), "rank": rank}


def ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float) -> dict:
    xm = x.mean(axis=0); ym = y.mean(axis=0)
    xc = x - xm; yc = y - ym
    gram = xc.T @ xc
    coef = np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), xc.T @ yc)
    pred = xc @ coef + ym
    resid = y - pred
    cov = np.cov(resid, rowvar=False, ddof=1)
    if np.ndim(cov) == 0:
        cov = np.array([[float(cov)]])
    floor = max(1e-6, float(np.trace(cov)) / cov.shape[0] * 1e-4)
    cov = (cov + cov.T) / 2 + floor * np.eye(cov.shape[0])
    return {"coef": coef, "xm": xm, "ym": ym, "cov": cov}


def ridge_predict(model: dict, x: np.ndarray) -> np.ndarray:
    return (x - model["xm"]) @ model["coef"] + model["ym"]


def point_nll(y: np.ndarray, yp: np.ndarray, cov: np.ndarray) -> np.ndarray:
    r = y - yp
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("non-SPD covariance")
    inv = np.linalg.inv(cov)
    q = np.einsum("ni,ij,nj->n", r, inv, r)
    d = y.shape[1]
    return .5 * (q + logdet + d * math.log(2 * math.pi))


def score(model: dict, x: np.ndarray, y: np.ndarray) -> dict:
    yp = ridge_predict(model, x)
    e = y - yp
    nll = point_nll(y, yp, model["cov"])
    var = np.mean((y - y.mean(axis=0)) ** 2)
    return {
        "rmse": float(np.sqrt(np.mean(e * e))),
        "r2": float(1 - np.mean(e * e) / (var + 1e-12)),
        "nlpd": float(np.mean(nll)),
        "point_nll": nll,
    }


def block_bootstrap(delta: np.ndarray, block: int = 100) -> dict:
    delta = np.asarray(delta, dtype=float)
    blocks = [delta[i:i + block] for i in range(0, len(delta), block) if len(delta[i:i + block])]
    means = np.empty(BOOT)
    for b in range(BOOT):
        chosen = RNG.integers(0, len(blocks), size=len(blocks))
        means[b] = float(np.mean(np.concatenate([blocks[i] for i in chosen])))
    return {
        "observed_mean": float(np.mean(delta)),
        "q025": float(np.quantile(means, .025)),
        "q975": float(np.quantile(means, .975)),
        "p_nonpositive": float((np.sum(means <= 0) + 1) / (BOOT + 1)),
    }


def indices(n: int, lag: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    usable = n - lag
    a = int(usable * .60)
    b = int(usable * .80)
    gap = max(20, int(1.0 / BIN))
    train = np.arange(0, max(1, a - gap))
    val = np.arange(min(usable, a + gap), max(min(usable, a + gap), b - gap))
    test = np.arange(min(usable, b + gap), usable)
    return train, val, test


def features(scores: dict[str, np.ndarray], source: str, target: str, third: str, idx: np.ndarray, lag: int, shifted: bool = False) -> tuple[np.ndarray, np.ndarray]:
    src = scores[source]
    if shifted:
        src = np.roll(src, max(100, len(src) // 5), axis=0)
    x = np.column_stack([scores[target][idx], scores[third][idx], src[idx]])
    y = scores[target][idx + lag]
    return x, y


def base_features(scores: dict[str, np.ndarray], target: str, third: str, idx: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.column_stack([scores[target][idx], scores[third][idx]])
    y = scores[target][idx + lag]
    return x, y


def evaluate_path(counts: dict[str, np.ndarray], source: str, target: str, third: str) -> dict:
    candidates = []
    for rank in RANKS:
        max_rank = min(rank, *(counts[r].shape[1] for r in counts))
        if max_rank < 1:
            continue
        for lag in LAGS:
            tr, va, te = indices(len(next(iter(counts.values()))), lag)
            if len(va) < 100 or len(te) < 100:
                continue
            scores = {}
            pca = {}
            for region, c in counts.items():
                scores[region], pca[region] = fit_pca(c[tr], c, max_rank)
            xbtr, ytr = base_features(scores, target, third, tr, lag)
            xbva, yva = base_features(scores, target, third, va, lag)
            xftr, _ = features(scores, source, target, third, tr, lag)
            xfva, _ = features(scores, source, target, third, va, lag)
            for ridge in RIDGES:
                mb = ridge_fit(xbtr, ytr, ridge)
                mf = ridge_fit(xftr, ytr, ridge)
                sb = score(mb, xbva, yva)
                sf = score(mf, xfva, yva)
                candidates.append({"rank": max_rank, "lag": lag, "ridge": ridge, "validation_delta_nlpd": sb["nlpd"] - sf["nlpd"], "scores": scores, "pca": pca, "split": (tr, va, te)})
    best = max(candidates, key=lambda c: c["validation_delta_nlpd"])
    rank, lag, ridge = best["rank"], best["lag"], best["ridge"]
    scores = best["scores"]; tr, va, te = best["split"]
    train = np.concatenate([tr, va])
    xbtr, ytr = base_features(scores, target, third, train, lag)
    xfte, yte = features(scores, source, target, third, te, lag)
    xbte, _ = base_features(scores, target, third, te, lag)
    xftr, _ = features(scores, source, target, third, train, lag)
    xstr, _ = features(scores, source, target, third, train, lag, shifted=True)
    xste, _ = features(scores, source, target, third, te, lag, shifted=True)
    mb = ridge_fit(xbtr, ytr, ridge)
    mf = ridge_fit(xftr, ytr, ridge)
    ms = ridge_fit(xstr, ytr, ridge)
    sb = score(mb, xbte, yte)
    sf = score(mf, xfte, yte)
    ss = score(ms, xste, yte)
    delta = sb["point_nll"] - sf["point_nll"]
    return {
        "source": source,
        "target": target,
        "third_control": third,
        "selected": {"rank": rank, "lag_bins": lag, "lag_ms": lag * BIN * 1000, "ridge": ridge, "validation_delta_nlpd": best["validation_delta_nlpd"]},
        "test": {
            "baseline": {k: v for k, v in sb.items() if k != "point_nll"},
            "bridge": {k: v for k, v in sf.items() if k != "point_nll"},
            "circular_shift": {k: v for k, v in ss.items() if k != "point_nll"},
            "delta_nlpd_base_minus_bridge": float(sb["nlpd"] - sf["nlpd"]),
            "delta_nlpd_shift_minus_bridge": float(ss["nlpd"] - sf["nlpd"]),
            "block_bootstrap_delta_base_minus_bridge": block_bootstrap(delta),
            "n_test": int(len(te)),
        },
    }


def main() -> None:
    download()
    with NWBHDF5IO(str(FILE), mode="r", load_namespaces=True) as io:
        nwb = io.read()
        df = nwb.units.to_dataframe()
        start, stop, behavior_meta = behavior_interval(nwb)
        # Cap excessively long intervals to keep the pilot deterministic and comparable.
        if stop - start > 1800:
            stop = start + 1800
        region_counts = {r: int(np.sum(df["cell_area"].astype(str).str.upper() == r)) for r in ("CA3", "CA1", "RSC")}
        counts = {}
        unit_ids = {}
        for region in ("CA3", "CA1", "RSC"):
            counts[region], unit_ids[region] = bin_region(df, region, start, stop)
        min_bins = min(c.shape[0] for c in counts.values())
        counts = {k: v[:min_bins] for k, v in counts.items()}
        paths = [
            evaluate_path(counts, "CA3", "CA1", "RSC"),
            evaluate_path(counts, "CA1", "RSC", "CA3"),
            evaluate_path(counts, "CA1", "CA3", "RSC"),
            evaluate_path(counts, "RSC", "CA1", "CA3"),
        ]
        result = {
            "status": "COMPLETE",
            "asset_id": ASSET_ID,
            "session": nwb.session_description,
            "behavior_interval": {"start": start, "stop": stop, "duration_s": stop - start, "meta": behavior_meta},
            "bin_s": BIN,
            "n_bins": min_bins,
            "region_unit_counts": region_counts,
            "paths": paths,
        }
    (OUT / "bridge_pilot_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# DANDI 001695 direct bridge pilot", "", f"Units: `{region_counts}`; duration `{stop-start:.1f}s`; bin `{BIN}s`.", "", "| path | rank | lag ms | test ΔNLPD base-bridge | shift-bridge | bootstrap 95% CI |", "|---|---:|---:|---:|---:|---|"]
    for p in paths:
        b = p["test"]["block_bootstrap_delta_base_minus_bridge"]
        lines.append(f"| {p['source']}→{p['target']} | {p['selected']['rank']} | {p['selected']['lag_ms']:.0f} | {p['test']['delta_nlpd_base_minus_bridge']:.5f} | {p['test']['delta_nlpd_shift_minus_bridge']:.5f} | [{b['q025']:.5f}, {b['q975']:.5f}] |")
    (OUT / "bridge_pilot_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
