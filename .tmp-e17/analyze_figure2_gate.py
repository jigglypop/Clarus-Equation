from __future__ import annotations

import json
import math
import re
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import optimize, stats
from scipy.io import loadmat

URL = "https://doi.gin.g-node.org/10.12751/g-node.etlk5k/10.12751_g-node.etlk5k.zip"
ZIP = Path("/tmp/e17-etlk5k.zip")
OUT = Path("e17_direct_results")
OUT.mkdir(exist_ok=True)
SEED = 20260819
RNG = np.random.default_rng(SEED)
DRAWS = 20000
VARIABLES = ("branch_amp", "branch_freq", "spine_amp", "spine_freq")


def ensure_zip() -> None:
    if not ZIP.exists():
        urllib.request.urlretrieve(URL, ZIP)


def vec(x) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    return x[np.isfinite(x)]


def collect() -> tuple[dict[str, list[dict]], list[dict]]:
    ensure_zip()
    root = Path("/tmp/e17-fig2-analysis")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    records: dict[str, list[dict]] = defaultdict(list)
    pairing = []
    with zipfile.ZipFile(ZIP) as z:
        names = sorted(n for n in z.namelist() if n.startswith("Figure2/Data/") and n.endswith("_dff.mat"))
        for name in names:
            dst = root / Path(name).name
            with z.open(name) as src, dst.open("wb") as out:
                shutil.copyfileobj(src, out)
            d = loadmat(dst, simplify_cells=True)
            animal_match = re.search(r"(DCO\d+)", Path(name).name)
            animal = animal_match.group(1) if animal_match else "unknown"
            session = Path(name).stem.replace("_dff", "")
            for variable in VARIABLES:
                obj = d.get(variable)
                if not isinstance(obj, dict) or "Sal" not in obj or "DCZ" not in obj:
                    pairing.append({"session": session, "animal": animal, "variable": variable, "status": "missing"})
                    continue
                x = np.asarray(obj["Sal"], dtype=float).reshape(-1)
                y = np.asarray(obj["DCZ"], dtype=float).reshape(-1)
                equal = len(x) == len(y)
                mask = np.isfinite(x) & np.isfinite(y) if equal else np.zeros(0, dtype=bool)
                pairing.append({
                    "session": session,
                    "animal": animal,
                    "variable": variable,
                    "n_sal": int(len(x)),
                    "n_dcz": int(len(y)),
                    "n_paired_finite": int(np.sum(mask)),
                    "equal_length": bool(equal),
                })
                if not equal:
                    continue
                for idx, (sal, dcz, ok) in enumerate(zip(x, y, mask)):
                    if ok and sal >= 0 and dcz >= 0:
                        records[variable].append({
                            "animal": animal,
                            "session": session,
                            "unit": int(idx),
                            "sal": float(sal),
                            "dcz": float(dcz),
                        })
    return records, pairing


def fit_gain(x: np.ndarray, y: np.ndarray) -> dict:
    denom = float(x @ x)
    a = max(0.0, float(x @ y) / denom) if denom > 0 else 0.0
    return {"name": "gain", "params": [a]}


def predict(model: dict, x: np.ndarray) -> np.ndarray:
    p = model["params"]
    if model["name"] == "gain":
        return p[0] * x
    if model["name"] == "threshold":
        a, tau = p
        return a * np.maximum(x - tau, 0.0)
    if model["name"] == "soft_threshold":
        a, tau, temp = p
        u = (x - tau) / temp
        return a * temp * np.logaddexp(0.0, u)
    raise ValueError(model["name"])


def fit_threshold(x: np.ndarray, y: np.ndarray, smooth: bool) -> dict:
    gain = fit_gain(x, y)["params"][0]
    xmax = max(float(np.max(x)), 1e-8)
    med = max(float(np.median(x[x > 0])) if np.any(x > 0) else xmax / 10, 1e-8)
    if smooth:
        temp = max(med * 0.10, xmax * 1e-4)
        def residual(q):
            a, tau = q
            return a * temp * np.logaddexp(0.0, (x - tau) / temp) - y
        res = optimize.least_squares(residual, x0=[max(gain, 1e-6), med * 0.25], bounds=([0.0, 0.0], [10.0, xmax * 1.5]), max_nfev=5000)
        return {"name": "soft_threshold", "params": [float(res.x[0]), float(res.x[1]), temp], "success": bool(res.success)}
    def residual(q):
        a, tau = q
        return a * np.maximum(x - tau, 0.0) - y
    res = optimize.least_squares(residual, x0=[max(gain, 1e-6), med * 0.25], bounds=([0.0, 0.0], [10.0, xmax * 1.5]), max_nfev=5000)
    return {"name": "threshold", "params": [float(res.x[0]), float(res.x[1])], "success": bool(res.success)}


def metrics(y: np.ndarray, yp: np.ndarray) -> dict:
    err = y - yp
    rmse = float(np.sqrt(np.mean(err * err)))
    baseline = float(np.sqrt(np.mean((y - np.mean(y)) ** 2)))
    mae = float(np.mean(np.abs(err)))
    return {"rmse": rmse, "nrmse_sd": rmse / (baseline + 1e-12), "mae": mae}


def loao_models(rows: list[dict]) -> dict:
    animals = sorted({r["animal"] for r in rows})
    folds = []
    for held in animals:
        tr = [r for r in rows if r["animal"] != held]
        te = [r for r in rows if r["animal"] == held]
        xtr = np.array([r["sal"] for r in tr], dtype=float)
        ytr = np.array([r["dcz"] for r in tr], dtype=float)
        xte = np.array([r["sal"] for r in te], dtype=float)
        yte = np.array([r["dcz"] for r in te], dtype=float)
        models = [fit_gain(xtr, ytr), fit_threshold(xtr, ytr, False), fit_threshold(xtr, ytr, True)]
        folds.append({
            "held_animal": held,
            "n_train": len(tr),
            "n_test": len(te),
            "models": [{**m, "test": metrics(yte, predict(m, xte))} for m in models],
        })
    aggregate = {}
    for name in ("gain", "threshold", "soft_threshold"):
        vals = [next(m for m in f["models"] if m["name"] == name)["test"] for f in folds]
        aggregate[name] = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
    return {"animals": animals, "folds": folds, "animal_mean_metrics": aggregate}


def animal_effects(rows: list[dict]) -> tuple[dict[str, float], dict[str, list[dict]]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[r["animal"]].append(r)
    effects = {}
    detail = {}
    for animal, rr in grouped.items():
        # equal animal weighting; log ratio captures multiplicative suppression
        eps = max(np.median([r["sal"] for r in rr]) * 1e-6, 1e-12)
        logratio = np.array([math.log(r["dcz"] + eps) - math.log(r["sal"] + eps) for r in rr])
        effects[animal] = float(np.mean(logratio))
        sessions = defaultdict(list)
        for r, lr in zip(rr, logratio):
            sessions[r["session"]].append(float(lr))
        detail[animal] = [{"session": s, "n": len(v), "mean_log_ratio": float(np.mean(v))} for s, v in sorted(sessions.items())]
    return effects, detail


def hierarchical_bootstrap(rows: list[dict], draws: int = DRAWS) -> dict:
    tree: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        tree[r["animal"]][r["session"]].append(r)
    animals = sorted(tree)
    values = np.empty(draws)
    for b in range(draws):
        sampled_animals = RNG.choice(animals, size=len(animals), replace=True)
        animal_means = []
        for animal in sampled_animals:
            sessions = sorted(tree[animal])
            sampled_sessions = RNG.choice(sessions, size=len(sessions), replace=True)
            session_means = []
            for session in sampled_sessions:
                units = tree[animal][session]
                idx = RNG.integers(0, len(units), size=len(units))
                sal = np.array([units[i]["sal"] for i in idx], dtype=float)
                dcz = np.array([units[i]["dcz"] for i in idx], dtype=float)
                eps = max(float(np.median(sal)) * 1e-6, 1e-12)
                session_means.append(float(np.mean(np.log(dcz + eps) - np.log(sal + eps))))
            animal_means.append(float(np.mean(session_means)))
        values[b] = float(np.mean(animal_means))
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q025": float(np.quantile(values, .025)),
        "q975": float(np.quantile(values, .975)),
        "p_nonnegative": float((np.sum(values >= 0) + 1) / (draws + 1)),
    }


def baseline_ratio_test(rows: list[dict]) -> dict:
    x = np.array([r["sal"] for r in rows], dtype=float)
    y = np.array([r["dcz"] for r in rows], dtype=float)
    eps = max(float(np.median(x)) * 1e-6, 1e-12)
    lr = np.log(y + eps) - np.log(x + eps)
    rho = stats.spearmanr(x, lr).statistic
    # cluster-aware permutation: shuffle paired DCZ values within each animal, preserving marginals
    animals = sorted({r["animal"] for r in rows})
    obs = float(rho)
    count_ge = 0
    for _ in range(DRAWS):
        yp = y.copy()
        for animal in animals:
            idx = np.array([i for i, r in enumerate(rows) if r["animal"] == animal])
            yp[idx] = RNG.permutation(yp[idx])
        lrp = np.log(yp + eps) - np.log(x + eps)
        rp = float(stats.spearmanr(x, lrp).statistic)
        if abs(rp) >= abs(obs):
            count_ge += 1
    return {"spearman_baseline_vs_log_ratio": obs, "two_sided_within_animal_shuffle_p": (count_ge + 1) / (DRAWS + 1)}


def exact_animal_sign_p(effects: dict[str, float]) -> float:
    # one-sided against negative suppression; exact under independent random signs
    n = len(effects)
    n_negative = sum(v < 0 for v in effects.values())
    return float(sum(math.comb(n, k) for k in range(n_negative, n + 1)) / (2 ** n))


def analyze_variable(rows: list[dict]) -> dict:
    effects, detail = animal_effects(rows)
    return {
        "n_pairs": len(rows),
        "n_sessions": len({r['session'] for r in rows}),
        "animals": sorted({r['animal'] for r in rows}),
        "sal_mean": float(np.mean([r["sal"] for r in rows])),
        "dcz_mean": float(np.mean([r["dcz"] for r in rows])),
        "paired_median_difference_dcz_minus_sal": float(np.median([r["dcz"] - r["sal"] for r in rows])),
        "animal_mean_log_ratios": effects,
        "animal_session_log_ratios": detail,
        "exact_one_sided_animal_sign_p": exact_animal_sign_p(effects),
        "hierarchical_bootstrap": hierarchical_bootstrap(rows),
        "baseline_dependence": baseline_ratio_test(rows),
        "loao_model_comparison": loao_models(rows),
    }


def report(result: dict) -> str:
    lines = [
        "# E17 Figure 2 direct gate-model test",
        "",
        f"Seed `{SEED}`; hierarchical/permutation draws `{DRAWS}`.",
        "",
        "| variable | pairs | sessions | Sal mean | DCZ mean | bootstrap log-ratio 95% CI | gain LOAO nRMSE | threshold LOAO nRMSE | soft-threshold LOAO nRMSE |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for variable, r in result["variables"].items():
        b = r["hierarchical_bootstrap"]
        m = r["loao_model_comparison"]["animal_mean_metrics"]
        lines.append(
            f"| {variable} | {r['n_pairs']} | {r['n_sessions']} | {r['sal_mean']:.5g} | {r['dcz_mean']:.5g} | "
            f"[{b['q025']:.3f}, {b['q975']:.3f}] | {m['gain']['nrmse_sd']:.3f} | {m['threshold']['nrmse_sd']:.3f} | {m['soft_threshold']['nrmse_sd']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    records, pairing = collect()
    result = {
        "status": "COMPLETE",
        "seed": SEED,
        "draws": DRAWS,
        "source": URL,
        "pairing_audit": pairing,
        "variables": {v: analyze_variable(records[v]) for v in VARIABLES if records[v]},
    }
    (OUT / "figure2_gate_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT / "figure2_gate_report.md").write_text(report(result), encoding="utf-8")
    print(report(result))


if __name__ == "__main__":
    main()
