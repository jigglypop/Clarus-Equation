"""a5: are the pre-registered windows (4 x surrogate delta-method se) actually 4-sigma windows?

The card fixes every window as +-4 x the delta-method standard error of a ratio-of-means computed
from the SAME tetrad-free surrogate at N = 8192, and claims "카드가 참일 때 통계당 오발동 6e-5".
The estimators are ratios of means of heavy-tailed quadratic forms with common random numbers.
This script replicates the whole N = 8192 surrogate R times independently (vectorised, so it is
cheap) and measures (a) the true sampling sd of each of the 11 statistics, (b) the mean, and
(c) the actual probability that each pre-registered window fires WHEN THE CARD IS TRUE.

The surrogate is exactly the object the card used to design the windows, so this is a check of the
design calculation, not of the physics.
"""
import json, math, sys
from pathlib import Path
import numpy as np
from scipy.special import ndtr, ndtri

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "verify" / "Q-0012" / "F-01"))
from check_cumulant import (linear_map, quadratic_tensor, caterpillar, ancestor_matrix,
                            KAPPA4, DISTS, WINDOWS, PREREGISTERED, SPIKE_P)

OUT = Path(__file__).parent
N_CELLS = 36
Z_SPIKE = ndtri(1.0 - SPIKE_P / 2.0)


def labels_from_z(z, dist):
    """Same inverse-CDF coupling as the card, written as monotone functions of z (vectorised)."""
    if dist == "gauss":
        return z
    if dist == "rademacher":
        return np.sign(z)
    if dist == "uniform":
        return math.sqrt(3.0) * (2 * ndtr(z) - 1)
    if dist == "laplace":
        u = np.clip(ndtr(z), 1e-15, 1 - 1e-15)
        return -np.sign(u - 0.5) * np.log(1 - 2 * np.abs(u - 0.5)) / math.sqrt(2.0)
    if dist == "spike64":
        return np.sign(z) * (np.abs(z) > Z_SPIKE) / math.sqrt(SPIKE_P)
    raise ValueError(dist)


def batch_values(M, gens, rng, trials, batch=1024):
    out = {mode: {d: np.empty(trials) for d in DISTS} for mode in gens}
    done = 0
    while done < trials:
        b = min(batch, trials - done)
        z = rng.standard_normal((b, N_CELLS, 16))
        for dist in DISTS:
            zeta = labels_from_z(z, dist)
            for mode, HA in gens.items():
                D = np.einsum("vu,tua->tva", HA, zeta, optimize=True)
                G = np.einsum("tva,tvb->tab", D, D, optimize=True)
                phi = np.einsum("tab,abij->tij", G, M, optimize=True)
                out[mode][dist][done:done + b] = np.sum(phi * phi, axis=(1, 2))
        done += b
    return out


def ratio_and_se(a, b):
    n = len(a)
    ma, mb = float(a.mean()), float(b.mean())
    c = np.cov(a, b)
    var = (c[0, 0] / mb ** 2 - 2 * ma * c[0, 1] / mb ** 3 + ma ** 2 * c[1, 1] / mb ** 4) / n
    return ma / mb, math.sqrt(max(var, 0.0))


def slope(rhos):
    num = sum(KAPPA4[d] * (rhos[d] - 1.0) for d in DISTS if d != "gauss")
    den = sum(KAPPA4[d] ** 2 for d in DISTS if d != "gauss")
    return num / den


def one_replicate(M, gens, rng, trials):
    vals = batch_values(M, gens, rng, trials)
    stats, ses = {}, {}
    for mode in gens:
        rhos = {}
        for dist in DISTS:
            if dist == "gauss":
                rhos[dist] = 1.0
                continue
            r, se = ratio_and_se(vals[mode][dist], vals[mode]["gauss"])
            rhos[dist] = r
            stats["rho_%s_%s" % (mode, dist)] = r
            ses["rho_%s_%s" % (mode, dist)] = se
        stats["a_%s" % mode] = slope(rhos)
    stats["slope_ratio"] = stats["a_cat6"] / stats["a_iid36"]
    return stats, ses


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
    M = quadratic_tensor(linear_map())
    H = np.eye(N_CELLS) - np.ones((N_CELLS, N_CELLS)) / N_CELLS
    gens = {"iid36": H, "cat6": H @ ancestor_matrix(caterpillar(6))}
    rng = np.random.default_rng(20260902 + 55555)
    keys = None
    acc, acc_se = [], []
    for r in range(reps):
        st, se = one_replicate(M, gens, rng, trials)
        if keys is None:
            keys = sorted(st)
        acc.append([st[k] for k in keys])
        acc_se.append([se.get(k, float("nan")) for k in keys])
    A = np.array(acc)
    S = np.array(acc_se)
    res = {"replicates": reps, "trials_per_replicate": trials, "stats": {}}
    print("statistic            prereg      mean(rep)    sd(rep)   median deltase   sd/deltase   "
          "window                fire_rate")
    for i, k in enumerate(keys):
        col = A[:, i]
        lo, hi = WINDOWS[k]
        fire = float(np.mean((col < lo) | (col > hi)))
        dse = float(np.nanmedian(S[:, i])) if not np.all(np.isnan(S[:, i])) else float("nan")
        res["stats"][k] = {"prereg": PREREGISTERED[k], "mean": float(col.mean()),
                           "sd": float(col.std(ddof=1)), "median_delta_method_se": dse,
                           "sd_over_delta_se": (col.std(ddof=1) / dse) if dse == dse else None,
                           "window": [lo, hi], "fire_rate": fire,
                           "z_mean_vs_prereg": (col.mean() - PREREGISTERED[k]) / (col.std(ddof=1) / math.sqrt(reps))}
        print("%-20s %9.5f %11.5f %10.5f %9.5f %11.3f   [%8.4f,%8.4f] %8.3f"
              % (k, PREREGISTERED[k], col.mean(), col.std(ddof=1), dse,
                 (col.std(ddof=1) / dse) if dse == dse else float("nan"), lo, hi, fire))
    fires = np.zeros(reps, dtype=bool)
    for i, k in enumerate(keys):
        lo, hi = WINDOWS[k]
        fires |= (A[:, i] < lo) | (A[:, i] > hi)
    res["any_window_fire_rate"] = float(np.mean(fires))
    print("P(at least one of the 11 windows fires | card true) = %.3f" % np.mean(fires))
    (OUT / "a5_window_coverage.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
