from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import requests
from pynwb import NWBHDF5IO

OUT = Path("dandi_bridge_results")
OUT.mkdir(exist_ok=True)
CACHE = Path("/tmp/dandi001695")
CACHE.mkdir(exist_ok=True)
SEED = 20260819
RNG = np.random.default_rng(SEED)

# One behavior+ecephys session per animal, fixed before outcome inspection.
ASSETS = {
    "M01": "6d733831-afbf-44c2-8c46-7b3550f5e672",
    "M02": "1e4d5403-a8cc-4814-a904-7aff57f8cc4d",
    "M03": "605ae4d4-454b-435b-97ef-84518ce63932",
    "M05": "091bc936-b149-4598-b89a-e9db45499a69",
}
BIN = 0.05
RANK = 5
HISTORY = 3
HORIZON = 1
RIDGE = 1.0
MAX_DURATION = 1200.0
BLOCK = 100
BOOT = 5000
PATHS = [
    ("CA3", "CA1", "RSC"),
    ("CA1", "RSC", "CA3"),
    ("CA1", "CA3", "RSC"),
    ("RSC", "CA1", "CA3"),
]


def download(asset_id: str, animal: str) -> Path:
    dst = CACHE / f"{animal}.nwb"
    if dst.exists() and dst.stat().st_size > 1_000_000:
        return dst
    url = f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"
    print("download", animal, url, flush=True)
    with requests.get(url, stream=True, timeout=180, allow_redirects=True) as r:
        r.raise_for_status()
        with dst.open("wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    return dst


def behavior_interval(nwb) -> tuple[float, float]:
    if "behavior" in nwb.processing:
        module = nwb.processing["behavior"]
        for obj in module.data_interfaces.values():
            if hasattr(obj, "spatial_series"):
                for ss in obj.spatial_series.values():
                    if ss.timestamps is not None:
                        ts = np.asarray(ss.timestamps[:], dtype=float)
                    else:
                        ts = float(ss.starting_time) + np.arange(len(ss.data)) / float(ss.rate)
                    ts = ts[np.isfinite(ts)]
                    if len(ts) > 100:
                        return float(ts.min()), float(ts.max())
    df = nwb.units.to_dataframe()
    starts, stops = [], []
    for x in df["spike_times"]:
        a = np.asarray(x, dtype=float)
        if len(a):
            starts.append(float(a.min())); stops.append(float(a.max()))
    return max(starts), min(stops)


def bin_regions(df, start: float, stop: float) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    edges = np.arange(start, stop + BIN, BIN)
    out, nums = {}, {}
    areas = df["cell_area"].astype(str).str.upper()
    for region in ("CA3", "CA1", "RSC"):
        rows = list(df.index[areas == region])
        nums[region] = len(rows)
        arr = np.zeros((len(edges) - 1, len(rows)), dtype=float)
        for j, idx in enumerate(rows):
            sp = np.asarray(df.loc[idx, "spike_times"], dtype=float)
            arr[:, j] = np.histogram(sp, bins=edges)[0]
        out[region] = arr
    return out, nums


def split_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    # Guard gaps exceed history+horizon and reduce local leakage.
    usable_start = HISTORY - 1
    usable_stop = n - HORIZON
    cut = int(n * 0.75)
    gap = max(40, int(2.0 / BIN))
    train = np.arange(usable_start, max(usable_start + 1, cut - gap))
    test = np.arange(min(usable_stop, cut + gap), usable_stop)
    return train, test


def fit_region_pca(train_counts: np.ndarray, all_counts: np.ndarray) -> np.ndarray:
    tr = np.sqrt(train_counts)
    al = np.sqrt(all_counts)
    mu = tr.mean(axis=0)
    sd = tr.std(axis=0)
    sd[sd < 1e-6] = 1.0
    ztr = (tr - mu) / sd
    zal = (al - mu) / sd
    _, _, vt = np.linalg.svd(ztr, full_matrices=False)
    rank = min(RANK, vt.shape[0], vt.shape[1])
    return zal @ vt[:rank].T


def design(scores: dict[str, np.ndarray], target: str, third: str, source: str | None, idx: np.ndarray, shifted: bool = False) -> tuple[np.ndarray, np.ndarray]:
    blocks = []
    for lag in range(HISTORY):
        blocks.append(scores[target][idx - lag])
        blocks.append(scores[third][idx - lag])
    if source is not None:
        src = scores[source]
        if shifted:
            src = np.roll(src, max(200, len(src) // 5), axis=0)
        for lag in range(HISTORY):
            blocks.append(src[idx - lag])
    x = np.column_stack(blocks)
    y = scores[target][idx + HORIZON]
    return x, y


def ridge_fit(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - xm, y - ym
    coef = np.linalg.solve(xc.T @ xc + RIDGE * np.eye(x.shape[1]), xc.T @ yc)
    pred = xc @ coef + ym
    resid = y - pred
    cov = np.cov(resid, rowvar=False, ddof=1)
    if np.ndim(cov) == 0:
        cov = np.array([[float(cov)]])
    floor = max(1e-6, float(np.trace(cov)) / cov.shape[0] * 1e-4)
    cov = (cov + cov.T) / 2 + floor * np.eye(cov.shape[0])
    return {"xm": xm, "ym": ym, "coef": coef, "cov": cov}


def nll(model: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    pred = (x - model["xm"]) @ model["coef"] + model["ym"]
    r = y - pred
    inv = np.linalg.inv(model["cov"])
    sign, ld = np.linalg.slogdet(model["cov"])
    if sign <= 0:
        raise RuntimeError("non-SPD residual covariance")
    q = np.einsum("ni,ij,nj->n", r, inv, r)
    return 0.5 * (q + ld + y.shape[1] * math.log(2 * math.pi))


def block_bootstrap(delta: np.ndarray) -> dict[str, float]:
    chunks = [delta[i:i + BLOCK] for i in range(0, len(delta), BLOCK) if len(delta[i:i + BLOCK])]
    means = np.empty(BOOT)
    for b in range(BOOT):
        sel = RNG.integers(0, len(chunks), len(chunks))
        means[b] = np.mean(np.concatenate([chunks[i] for i in sel]))
    return {
        "mean": float(np.mean(delta)),
        "q025": float(np.quantile(means, 0.025)),
        "q975": float(np.quantile(means, 0.975)),
        "p_nonpositive": float((np.sum(means <= 0) + 1) / (BOOT + 1)),
    }


def evaluate(scores: dict[str, np.ndarray], source: str, target: str, third: str) -> dict[str, Any]:
    train, test = split_indices(len(next(iter(scores.values()))))
    xb_tr, y_tr = design(scores, target, third, None, train)
    xf_tr, _ = design(scores, target, third, source, train)
    xs_tr, _ = design(scores, target, third, source, train, shifted=True)
    xb_te, y_te = design(scores, target, third, None, test)
    xf_te, _ = design(scores, target, third, source, test)
    xs_te, _ = design(scores, target, third, source, test, shifted=True)
    mb, mf, ms = ridge_fit(xb_tr, y_tr), ridge_fit(xf_tr, y_tr), ridge_fit(xs_tr, y_tr)
    nb, nf, ns = nll(mb, xb_te, y_te), nll(mf, xf_te, y_te), nll(ms, xs_te, y_te)
    return {
        "source": source, "target": target, "third": third,
        "n_test": int(len(test)),
        "base_nlpd": float(np.mean(nb)),
        "bridge_nlpd": float(np.mean(nf)),
        "shift_nlpd": float(np.mean(ns)),
        "delta_base_minus_bridge": float(np.mean(nb - nf)),
        "delta_shift_minus_bridge": float(np.mean(ns - nf)),
        "bootstrap": block_bootstrap(nb - nf),
    }


def evaluate_session(path: Path, animal: str, reverse_time: bool = False) -> dict[str, Any]:
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        df = nwb.units.to_dataframe()
        start, stop = behavior_interval(nwb)
        stop = min(stop, start + MAX_DURATION)
        counts, nums = bin_regions(df, start, stop)
    if min(nums.values()) < RANK:
        return {"animal": animal, "status": "INELIGIBLE_UNITS", "unit_counts": nums}
    n = min(a.shape[0] for a in counts.values())
    counts = {k: v[:n] for k, v in counts.items()}
    tr, _ = split_indices(n)
    scores = {k: fit_region_pca(v[tr], v) for k, v in counts.items()}
    if reverse_time:
        scores = {k: v[::-1].copy() for k, v in scores.items()}
    return {
        "animal": animal,
        "status": "COMPLETE",
        "reverse_time": reverse_time,
        "unit_counts": nums,
        "duration_s": float(stop - start),
        "paths": [evaluate(scores, *p) for p in PATHS],
    }


def exact_sign_p(values: list[float], alternative: str = "greater") -> float:
    v = np.asarray(values, float)
    obs = float(np.mean(v))
    n = len(v)
    vals = []
    for mask in range(1 << n):
        signs = np.array([1 if mask & (1 << i) else -1 for i in range(n)])
        vals.append(float(np.mean(v * signs)))
    vals = np.asarray(vals)
    if alternative == "greater":
        return float(np.mean(vals >= obs - 1e-15))
    return float(np.mean(np.abs(vals) >= abs(obs) - 1e-15))


def aggregate(sessions: list[dict[str, Any]], reversed_sessions: list[dict[str, Any]]) -> dict[str, Any]:
    path_names = [f"{a}->{b}" for a, b, _ in PATHS]
    out = {}
    for pi, name in enumerate(path_names):
        forward = [s["paths"][pi]["delta_base_minus_bridge"] for s in sessions if s.get("status") == "COMPLETE"]
        reverse = [s["paths"][pi]["delta_base_minus_bridge"] for s in reversed_sessions if s.get("status") == "COMPLETE"]
        contrast = [a - b for a, b in zip(forward, reverse)]
        out[name] = {
            "animal_forward_deltas": forward,
            "animal_reversed_time_deltas": reverse,
            "mean_forward": float(np.mean(forward)),
            "mean_reversed_time": float(np.mean(reverse)),
            "mean_forward_minus_reversed": float(np.mean(contrast)),
            "exact_sign_p_forward_greater_zero": exact_sign_p(forward),
            "exact_sign_p_forward_greater_reversed": exact_sign_p(contrast),
        }
    # Anatomical direction contrasts.
    for pos, neg, label in [("CA3->CA1", "CA1->CA3", "CA3_CA1_direction"), ("CA1->RSC", "RSC->CA1", "CA1_RSC_direction")]:
        vals = np.asarray(out[pos]["animal_forward_deltas"]) - np.asarray(out[neg]["animal_forward_deltas"])
        out[label] = {
            "animal_contrasts": vals.tolist(),
            "mean": float(np.mean(vals)),
            "exact_sign_p_expected_direction": exact_sign_p(vals.tolist()),
        }
    return out


def main() -> None:
    sessions, reverse = [], []
    for animal, aid in ASSETS.items():
        p = download(aid, animal)
        print("analyze", animal, flush=True)
        sessions.append(evaluate_session(p, animal, False))
        reverse.append(evaluate_session(p, animal, True))
        p.unlink(missing_ok=True)
    result = {
        "status": "COMPLETE",
        "locked_parameters": {"bin_s": BIN, "rank": RANK, "history_bins": HISTORY, "horizon_bins": HORIZON, "ridge": RIDGE, "max_duration_s": MAX_DURATION},
        "sessions": sessions,
        "reversed_time_sessions": reverse,
        "aggregate": aggregate(sessions, reverse),
    }
    (OUT / "bridge_strict_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# DANDI 001695 strict multi-animal bridge test", "",
        f"Locked: bin={BIN}s, rank={RANK}, history={HISTORY}, horizon={HORIZON}, ridge={RIDGE}.", "",
        "| path | mean forward ΔNLPD | mean reversed ΔNLPD | forward-reversed | exact p forward>0 | exact p forward>reverse |", "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ["CA3->CA1", "CA1->RSC", "CA1->CA3", "RSC->CA1"]:
        x = result["aggregate"][name]
        lines.append(f"| {name} | {x['mean_forward']:.6f} | {x['mean_reversed_time']:.6f} | {x['mean_forward_minus_reversed']:.6f} | {x['exact_sign_p_forward_greater_zero']:.4f} | {x['exact_sign_p_forward_greater_reversed']:.4f} |")
    for name in ["CA3_CA1_direction", "CA1_RSC_direction"]:
        x = result["aggregate"][name]
        lines.append(f"\n- `{name}` expected-direction contrast: mean `{x['mean']:.6f}`, exact one-sided p `{x['exact_sign_p_expected_direction']:.4f}`")
    (OUT / "bridge_strict_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
