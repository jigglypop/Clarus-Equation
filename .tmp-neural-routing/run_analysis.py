from __future__ import annotations

import json
import math
import os
import pickle
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, stats

SEED = 20260819
RNG = np.random.default_rng(SEED)
DRAWS = 20000
BASE = "https://raw.githubusercontent.com/m-j-wojcik/pfc_learning/main/processed_data/"
OUT = Path("routing_direct_results")
DATA = OUT / "official_cache"
OUT.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

STAGE_COUNTS = (3, 4, 5, 6)
FILES = []
for n in STAGE_COUNTS:
    FILES += [
        f"selectivity_coefficients_exp2_70_100_{n}stages.pickle",
        f"exp2_selectivity_dat_early_50_100_late_100_150_stages_{n}.pickle",
        f"exp2_decoding_time_avg_{n}stages_50_100.pickle",
    ]


def download() -> None:
    for name in FILES:
        dst = DATA / name
        if not dst.exists():
            print(f"download {name}", flush=True)
            urllib.request.urlretrieve(BASE + name, dst)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def sym(a: NDArray[np.float64]) -> NDArray[np.float64]:
    return (a + a.T) / 2.0


def spd(a: NDArray[np.float64], ridge: float = 1e-8) -> NDArray[np.float64]:
    a = sym(np.asarray(a, dtype=float))
    vals, vecs = np.linalg.eigh(a)
    floor = max(ridge, float(np.max(np.abs(vals))) * 1e-10)
    vals = np.maximum(vals, floor)
    return (vecs * vals) @ vecs.T


def inv_spd(a: NDArray[np.float64], ridge: float = 1e-6) -> NDArray[np.float64]:
    return np.linalg.inv(spd(a + ridge * np.eye(a.shape[0])))


def logm_spd(a: NDArray[np.float64]) -> NDArray[np.float64]:
    vals, vecs = np.linalg.eigh(spd(a))
    return (vecs * np.log(vals)) @ vecs.T


def invsqrt_spd(a: NDArray[np.float64]) -> NDArray[np.float64]:
    vals, vecs = np.linalg.eigh(spd(a))
    return (vecs * (1.0 / np.sqrt(vals))) @ vecs.T


def det_normalize(a: NDArray[np.float64]) -> NDArray[np.float64]:
    a = spd(a)
    d = a.shape[0]
    return a / np.linalg.det(a) ** (1.0 / d)


def airm_parts(a: NDArray[np.float64], b: NDArray[np.float64]) -> dict[str, float]:
    m = invsqrt_spd(a) @ b @ invsqrt_spd(a)
    h = logm_spd(m)
    d = h.shape[0]
    tr = float(np.trace(h))
    scale2 = tr * tr / d
    shape = h - np.eye(d) * tr / d
    shape2 = float(np.sum(shape * shape))
    return {
        "airm": math.sqrt(max(0.0, scale2 + shape2)),
        "scale2": scale2,
        "shape2": shape2,
        "shape_fraction": shape2 / (scale2 + shape2 + 1e-15),
    }


def correlation(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return spd(np.corrcoef(x, rowvar=False), ridge=1e-7)


def partial_corr(r: NDArray[np.float64]) -> NDArray[np.float64]:
    omega = inv_spd(r, ridge=1e-6)
    den = np.sqrt(np.outer(np.diag(omega), np.diag(omega)))
    p = -omega / den
    np.fill_diagonal(p, 1.0)
    return p


def coupling_vector(x: NDArray[np.float64]) -> NDArray[np.float64]:
    p = partial_corr(correlation(x))
    return np.array([p[0, 1], p[0, 2], p[1, 2]], dtype=float)


def coupling_energy(x: NDArray[np.float64]) -> float:
    v = coupling_vector(x)
    return float(v @ v)


def cosine(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def stage_vector(arr: Any, n: int) -> NDArray[np.float64] | None:
    x = np.asarray(arr, dtype=float)
    axes = [i for i, size in enumerate(x.shape) if size == n]
    if not axes:
        return None
    axis = axes[0]
    x = np.moveaxis(x, axis, 0)
    return np.nanmean(x.reshape(n, -1), axis=1)


def gaussian_nll(x: NDArray[np.float64], mu: NDArray[np.float64], cov: NDArray[np.float64]) -> float:
    cov = spd(cov, ridge=1e-5)
    r = x - mu
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return float("inf")
    return 0.5 * (len(x) * math.log(2 * math.pi) + logdet + float(r @ np.linalg.solve(cov, r)))


def loo_nlpd(x: NDArray[np.float64], full: bool) -> NDArray[np.float64]:
    out = []
    n, d = x.shape
    for i in range(n):
        tr = np.delete(x, i, axis=0)
        mu = tr.mean(axis=0)
        cov = np.cov(tr, rowvar=False, ddof=1)
        if not full:
            cov = np.diag(np.diag(cov))
        ridge = max(1e-8, float(np.trace(cov)) / d * 1e-4)
        out.append(gaussian_nll(x[i], mu, cov + ridge * np.eye(d)))
    return np.asarray(out)


def sign_flip_p(diff: NDArray[np.float64], draws: int = DRAWS) -> float:
    diff = np.asarray(diff, dtype=float)
    observed = float(np.mean(diff))
    if not np.isfinite(observed):
        return float("nan")
    count = 0
    for _ in range(draws):
        signs = RNG.choice(np.array([-1.0, 1.0]), size=len(diff))
        if float(np.mean(diff * signs)) >= observed:
            count += 1
    return (count + 1) / (draws + 1)


def shuffle_coupling_p(x: NDArray[np.float64], draws: int = DRAWS) -> float:
    obs = coupling_energy(x)
    count = 0
    for _ in range(draws):
        xp = np.column_stack([RNG.permutation(x[:, j]) for j in range(x.shape[1])])
        if coupling_energy(xp) >= obs:
            count += 1
    return (count + 1) / (draws + 1)


def split_half_coupling(x: NDArray[np.float64], draws: int = 5000) -> dict[str, float]:
    vals = []
    n = len(x)
    for _ in range(draws):
        idx = RNG.permutation(n)
        a, b = idx[: n // 2], idx[n // 2 :]
        vals.append(cosine(coupling_vector(x[a]), coupling_vector(x[b])))
    v = np.asarray(vals, dtype=float)
    return {
        "median": float(np.nanmedian(v)),
        "q025": float(np.nanquantile(v, 0.025)),
        "q975": float(np.nanquantile(v, 0.975)),
        "p_nonpositive": float((np.sum(v <= 0) + 1) / (np.sum(np.isfinite(v)) + 1)),
    }


def eigen_report(g0: NDArray[np.float64], g1: NDArray[np.float64]) -> dict[str, Any]:
    w0, u0 = np.linalg.eigh(spd(g0))
    w1, u1 = np.linalg.eigh(spd(g1))
    order0 = np.argsort(w0); order1 = np.argsort(w1)
    w0, u0 = w0[order0], u0[:, order0]
    w1, u1 = w1[order1], u1[:, order1]
    wg, vg = linalg.eigh(spd(g1), spd(g0))
    ang1 = np.degrees(linalg.subspace_angles(u0[:, -1:], u1[:, -1:]))
    ang2 = np.degrees(linalg.subspace_angles(u0[:, -2:], u1[:, -2:]))
    return {
        "eig_initial": w0.tolist(),
        "eig_final": w1.tolist(),
        "condition_initial": float(w0[-1] / w0[0]),
        "condition_final": float(w1[-1] / w1[0]),
        "generalized_eigenvalues": wg.tolist(),
        "generalized_log_stretches": np.log(wg).tolist(),
        "abs_eigenvector_overlap": np.abs(u0.T @ u1).tolist(),
        "top1_principal_angle_deg": float(ang1[0]),
        "top2_principal_angles_deg": ang2.tolist(),
    }


def extract_stage_arrays(obj: dict[str, Any], n: int) -> list[NDArray[np.float64]]:
    raw = obj["selectivity_coefficients"]
    if len(raw) != n:
        raise ValueError(f"expected {n} stages, got {len(raw)}")
    arrays = []
    for a in raw:
        x = np.asarray(a, dtype=float)
        x = np.squeeze(x)
        if x.ndim != 2:
            raise ValueError(f"unexpected coefficient shape {x.shape}")
        if x.shape[1] != 3 and x.shape[0] == 3:
            x = x.T
        if x.shape[1] != 3:
            raise ValueError(f"expected 3 axes, got {x.shape}")
        arrays.append(x)
    return arrays


def canonical_relative_stretches(gs: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    g0 = gs[0]
    out = []
    for g in gs:
        out.append([math.sqrt(float(g[j, j] / g0[j, j])) for j in range(3)])
    return np.asarray(out)


def random_rotation_null(g0: NDArray[np.float64], gf: NDArray[np.float64], observed: float, draws: int = DRAWS) -> float:
    vals, _ = np.linalg.eigh(spd(gf))
    count = 0
    for _ in range(draws):
        q, _ = np.linalg.qr(RNG.normal(size=(3, 3)))
        gr = q @ np.diag(vals) @ q.T
        s = np.log(np.sqrt(np.diag(gr) / np.diag(g0)))
        score = float(np.linalg.norm(s))
        if score >= observed:
            count += 1
    return (count + 1) / (draws + 1)


def load_late_epochs(obj: dict[str, Any], key: str, n: int) -> list[NDArray[np.float64]] | None:
    if key not in obj:
        return None
    raw = obj[key]
    if len(raw) != n:
        return None
    out = []
    for a in raw:
        x = np.asarray(a, dtype=float)
        if x.ndim >= 3:
            x = x[..., 0]
        x = np.squeeze(x)
        if x.ndim != 2:
            return None
        if x.shape[1] < 3 and x.shape[0] >= 3:
            x = x.T
        if x.shape[1] < 3:
            return None
        out.append(x[:, :3])
    return out


def axis_cosines(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array([cosine(a[:, j], b[:, j]) for j in range(min(a.shape[1], b.shape[1]))])


def split_neuron_geometry_alignment(task1: list[NDArray[np.float64]], task2: list[NDArray[np.float64]], draws: int = 5000) -> dict[str, Any]:
    rs_all, rs_shape, rs_xor = [], [], []
    nst = len(task1)
    for _ in range(draws):
        xs, ya, ys, yx = [], [], [], []
        for k in range(nst):
            n = min(len(task1[k]), len(task2[k]))
            idx = RNG.permutation(n)
            ia, ib = idx[: n // 2], idx[n // 2 :]
            c = correlation(task1[k][ia])
            g = inv_spd(c)
            if k == 0:
                gbase = g
            xs.append(airm_parts(det_normalize(gbase), det_normalize(g))["airm"])
            ac = axis_cosines(task1[k][ib], task2[k][ib])
            ya.append(float(np.nanmean(ac)))
            ys.append(float(ac[1]))
            yx.append(float(ac[2]))
        if np.std(xs) > 0 and np.std(ya) > 0:
            rs_all.append(stats.pearsonr(xs, ya).statistic)
        if np.std(xs) > 0 and np.std(ys) > 0:
            rs_shape.append(stats.pearsonr(xs, ys).statistic)
        if np.std(xs) > 0 and np.std(yx) > 0:
            rs_xor.append(stats.pearsonr(xs, yx).statistic)
    def summarize(v: list[float]) -> dict[str, float]:
        a = np.asarray(v, dtype=float)
        return {
            "median": float(np.nanmedian(a)),
            "q025": float(np.nanquantile(a, .025)),
            "q975": float(np.nanquantile(a, .975)),
            "p_nonpositive": float((np.sum(a <= 0) + 1) / (len(a) + 1)),
        }
    return {"all_axes": summarize(rs_all), "shape": summarize(rs_shape), "xor": summarize(rs_xor)}


def analyze_n(n: int) -> dict[str, Any]:
    sel = load_pickle(DATA / f"selectivity_coefficients_exp2_70_100_{n}stages.pickle")
    arrays = extract_stage_arrays(sel, n)
    dec = load_pickle(DATA / f"exp2_decoding_time_avg_{n}stages_50_100.pickle")
    late = load_pickle(DATA / f"exp2_selectivity_dat_early_50_100_late_100_150_stages_{n}.pickle")

    stages = []
    gs = []
    for k, x in enumerate(arrays):
        r = correlation(x)
        g = inv_spd(r)
        gs.append(g)
        pc = coupling_vector(x)
        nll_full = loo_nlpd(x, full=True)
        nll_diag = loo_nlpd(x, full=False)
        benefit = nll_diag - nll_full
        stages.append({
            "stage": k + 1,
            "n_neurons": int(len(x)),
            "covariance": np.cov(x, rowvar=False, ddof=1).tolist(),
            "correlation": r.tolist(),
            "precision": g.tolist(),
            "eigenvalues_precision": np.linalg.eigvalsh(g).tolist(),
            "partial_correlations": pc.tolist(),
            "coupling_energy": float(pc @ pc),
            "coupling_shuffle_p": shuffle_coupling_p(x),
            "coupling_split_half": split_half_coupling(x),
            "loo_nlpd_full_mean": float(np.mean(nll_full)),
            "loo_nlpd_diag_mean": float(np.mean(nll_diag)),
            "full_cov_benefit_mean": float(np.mean(benefit)),
            "full_cov_benefit_signflip_p": sign_flip_p(benefit),
        })

    rel = canonical_relative_stretches(gs)
    decoder = {}
    axis_keys = ["scores_set", "scores_xor2", "scores_context"]
    dec_matrix = []
    for key in axis_keys:
        v = stage_vector(dec.get(key), n) if key in dec else None
        decoder[key] = None if v is None else v.tolist()
        dec_matrix.append(v)
    directional_prediction = None
    if all(v is not None for v in dec_matrix):
        y = np.column_stack(dec_matrix)
        xr = rel[1:].ravel(); yr = y[1:].ravel()
        directional_prediction = {
            "pearson_r": float(stats.pearsonr(xr, yr).statistic),
            "spearman_r": float(stats.spearmanr(xr, yr).statistic),
            "n_points": int(len(xr)),
        }

    er = eigen_report(gs[0], gs[-1])
    ap = airm_parts(gs[0], gs[-1])
    observed_orient = float(np.linalg.norm(np.log(rel[-1])))
    er["canonical_log_stretch_norm"] = observed_orient
    er["eigenvalue_preserving_random_rotation_p"] = random_rotation_null(gs[0], gs[-1], observed_orient)

    task1 = load_late_epochs(late, "epochs_task1_l", n)
    task2 = load_late_epochs(late, "epochs_task2_l", n)
    late_test = None
    if task1 is not None and task2 is not None:
        same_row = [axis_cosines(task1[k], task2[k]).tolist() for k in range(n)]
        late_test = {
            "task1_shapes": [list(x.shape) for x in task1],
            "task2_shapes": [list(x.shape) for x in task2],
            "same_row_axis_cosines": same_row,
            "split_neuron_geometry_alignment": split_neuron_geometry_alignment(task1, task2),
        }

    return {
        "n_stages": n,
        "axis_order": ["stimulus_set", "set_x_context_xor2", "context"],
        "stages": stages,
        "relative_canonical_precision_stretch": rel.tolist(),
        "decoder_stage_vectors": decoder,
        "directional_stretch_vs_decoder": directional_prediction,
        "initial_to_final_eigen": er,
        "initial_to_final_airm": ap,
        "late_task1_task2": late_test,
    }


def markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Official PFC direct routing/subspace audit",
        "",
        f"Seed: `{SEED}`; permutation draws: `{DRAWS}`.",
        "",
        "All inputs are official processed files from `m-j-wojcik/pfc_learning`.",
        "",
        "## Cross-binning summary",
        "",
        "| stages | final coupling | initial coupling | full-cov benefit final | generalized log stretches | top-1 angle | split-neuron median r |",
        "|---:|---:|---:|---:|---|---:|---:|",
    ]
    for n, r in results["by_stage_count"].items():
        st = r["stages"]
        logs = r["initial_to_final_eigen"]["generalized_log_stretches"]
        late = r.get("late_task1_task2")
        split = float("nan") if late is None else late["split_neuron_geometry_alignment"]["all_axes"]["median"]
        lines.append(
            f"| {n} | {st[-1]['coupling_energy']:.4f} | {st[0]['coupling_energy']:.4f} | "
            f"{st[-1]['full_cov_benefit_mean']:.4f} | `{[round(x,3) for x in logs]}` | "
            f"{r['initial_to_final_eigen']['top1_principal_angle_deg']:.2f} | {split:.3f} |"
        )
    lines += ["", "## Machine-readable result", "", "See `results.json` in this artifact."]
    return "\n".join(lines) + "\n"


def main() -> None:
    download()
    by = {}
    for n in STAGE_COUNTS:
        print(f"analyze {n} stages", flush=True)
        by[str(n)] = analyze_n(n)
    results = {
        "status": "COMPLETE",
        "seed": SEED,
        "draws": DRAWS,
        "official_source": "m-j-wojcik/pfc_learning processed_data",
        "by_stage_count": by,
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (OUT / "report.md").write_text(markdown(results), encoding="utf-8")
    print(markdown(results))


if __name__ == "__main__":
    main()
