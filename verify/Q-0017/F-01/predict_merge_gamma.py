"""Q-0017 F-01 driver (tree-only, no tetrads): merge-averaging DAG kernel kappa_q = A_q A_q^T.

Process rule M(q) (card derivations/Q-0017/F-01.formula.md, pre-registered 2026-09-03):
  substrate : a rooted generation DAG given as a parent array.  Two families:
              (C) uniform rooted Cayley tree of size n (Pruefer, same generator as Q-0008 F-02 K1);
              (L_a) layered random-parent tree with prescribed level widths W_d = (d+1)^a, d = 0..h-1
                    (every cell of depth d >= 1 picks its parent uniformly in level d-1);
                    d_tree = a + 1 (depth ~ n^{1/(a+1)}); a = 0 is the chain, a -> oo the star.
  merge     : every cell v at depth d >= 1 independently, with probability q, becomes a MERGE EVENT --
              it draws a second parent r(v) uniformly among the OTHER cells of depth d-1 (if any),
              and inherits the arithmetic mean of the two parent labels:
                  label_v = xi_v + (label_{p(v)} + label_{r(v)}) / 2      (merge)
                  label_v = xi_v +  label_{p(v)}                          (no merge)
              q = 0 is identical to F-02 heritable mode (kappa = A A^T).
  kernel    : A_q(v,:) = e_v + sum_{parents p} w_{vp} A_q(p,:), w = 1 (single) or 1/2,1/2 (merge);
              (A_q)_{vu} = sum over directed paths u->v of prod_{w on path, w != u} 1/indeg(w).
              kappa_q = A_q A_q^T,  D_q = ||H kappa_q H||_F^2,  eps_bar^2 = eps_star^2 D_q / n^2 (E-018).
  gamma     : gamma = d ln RMS / d ln n = 0.5 d ln(E D_q/n^2) / d ln n  (F-02 K1 convention; > 0 relevant).
  mechanism : generation-mean random walk kernel G_vw = sum_{k <= min(d_v,d_w)} 1/W_k (fully mixed limit),
              computed on the same trees as a diagnostic (ratio D_q / D_G).
  coupling  : common random numbers across q (one threshold u_v, one candidate r_v per vertex).
  variance  : Cayley stages use the exact E[D_0(n)] (F-02 driver cayley_exact) as a control variate.

Seeds: per (family, n) stream default_rng(20260902 + 1000*n + family_offset); bootstrap 20260902.
Usage: python verify/Q-0017/F-01/predict_merge_gamma.py --stage grid|plateau|layered|all [--quick]
Writes verify/Q-0017/F-01/predictions.json (merged by stage).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "verify" / "Q-0008" / "F-02"))

from driver_numbers import cayley_exact, tree_arrays, uniform_rooted_tree  # noqa: E402

SEED = 20260902
BOOT = 400
# stage grid: Cayley on the F-02 K1 grid, fine q
GRID = (8, 16, 32, 64, 128)
Q_FINE = [round(0.05 * k, 2) for k in range(21)]
TRIALS_GRID = {8: 1000, 16: 1000, 32: 1000, 64: 1000, 128: 1000}
# stage plateau: Cayley extended sizes, few q
PLATEAU = (128, 256, 512, 1024, 2048)
Q_PLATEAU = [0.0, 0.25, 0.5, 1.0]
TRIALS_PLATEAU = {128: 600, 256: 400, 512: 250, 1024: 150, 2048: 60}
K2_GRID = (128, 256, 512, 1024)  # physical plateau test grid (Cayley, q = 1)
# stage layered: W_d = (d+1)^a
LAYERED = {
    1: (16, 23, 32, 45, 64),  # n = 136, 276, 528, 1035, 2080
    2: (8, 12, 16, 20),  # n = 204, 650, 1496, 2870
    3: (5, 6, 7, 8, 9),  # n = 225, 441, 784, 1296, 2025
}
Q_LAYERED = [0.0, 1.0]
TRIALS_LAYERED = {1: {16: 300, 23: 300, 32: 200, 45: 120, 64: 60},
                  2: {8: 300, 12: 200, 16: 120, 20: 40},
                  3: {5: 300, 6: 250, 7: 160, 8: 100, 9: 60}}
K3_H = (8, 12, 16)  # physical layered a=2 test grid (q = 1)
# stage k4 (PRE-REGISTERED, NOT RUN at card time): q-independence of the marginal exponent on the cone
K4_H = (45, 64, 90)  # n = 1035, 2080, 4095
K4_Q = [0.25, 0.5, 1.0]
K4_TRIALS = {45: 24, 64: 24, 90: 24}
K4_WINDOW = (-0.15, 0.15)  # each gamma_fit(q) must lie here; alternative (q-dependent sign change) predicts +-0.3
# L1k4 offset 78: a 4-tree code-path smoke with offset 77 (2026-09-03, --quick, no write) returned
# gamma_fit = (-0.007, -0.040, -0.027) for q = (0.25, 0.5, 1.0); the pre-registered K4 run uses a fresh stream.
FAMILY_OFFSET = {"cayley": 0, "L1": 11, "L2": 22, "L3": 33, "L1k4": 78}


# ---------------------------------------------------------------- substrates
def layered_parent(a: int, h: int, rng: np.random.Generator) -> list[int]:
    parent = [-1]
    prev = [0]
    for d in range(1, h):
        w = (d + 1) ** a
        cur = []
        for _ in range(w):
            parent.append(int(prev[int(rng.integers(0, len(prev)))]))
            cur.append(len(parent) - 1)
        prev = cur
    return parent


def layered_n(a: int, h: int) -> int:
    return sum((d + 1) ** a for d in range(h))


# ---------------------------------------------------------------- merge DAG
def merge_draws(parent: list[int], rng: np.random.Generator):
    """Common-random-number draws: threshold u_v ~ U(0,1) and second-parent candidate r_v (-1 if none)."""
    n = len(parent)
    _, depth, _, _ = tree_arrays(parent)
    levels: dict[int, list[int]] = {}
    for v in range(n):
        levels.setdefault(int(depth[v]), []).append(v)
    u = rng.random(n)
    r = np.full(n, -1, dtype=np.int64)
    for v in range(n):
        p = parent[v]
        if p < 0:
            continue
        others = levels[int(depth[v]) - 1]
        if len(others) < 2:
            continue
        # uniform over level(d-1) \ {p}: draw index in [0, W-1), map the parent's slot to the last slot
        j = int(rng.integers(0, len(others) - 1))
        c = others[j]
        if c == p:
            c = others[-1]
        r[v] = c
    level_list = [np.asarray(levels[d], dtype=np.int64) for d in sorted(levels)]
    widths = np.asarray([len(levels[d]) for d in sorted(levels)], dtype=float)
    return level_list, widths, depth, u, r


def kernel_D(parent: list[int], level_list, u: np.ndarray, r: np.ndarray, q: float) -> tuple[float, float]:
    """D_q = ||H A A^T H||_F^2 and tr(H kappa_q H) for merge threshold q (merge iff u_v < q and r_v >= 0)."""
    n = len(parent)
    par = np.asarray(parent, dtype=np.int64)
    A = np.zeros((n, n))
    root = level_list[0]
    A[root, root] = 1.0
    for lv in level_list[1:]:
        merged = (u[lv] < q) & (r[lv] >= 0)
        single = lv[~merged]
        both = lv[merged]
        if single.size:
            A[single] = A[par[single]]
        if both.size:
            A[both] = 0.5 * (A[par[both]] + A[r[both]])
        A[lv, lv] = 1.0
    B = A - A.mean(axis=0, keepdims=True)  # H A
    G = B.T @ B  # ||H A A^T H||_F^2 = ||(HA)^T (HA)||_F^2
    return float(np.sum(G * G)), float(np.trace(G))


def kernel_G(depth: np.ndarray, widths: np.ndarray) -> float:
    """Generation-mean random-walk kernel: G_vw = sum_{k <= min(d_v,d_w)} 1/W_k; returns ||H G H||_F^2."""
    g = np.cumsum(1.0 / widths)
    M = g[np.minimum.outer(depth, depth)]
    M = M - M.mean(axis=0, keepdims=True) - M.mean(axis=1, keepdims=True) + M.mean()
    return float(np.sum(M * M))


def slope(xs, ys) -> float:
    return float(np.polyfit(np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float)), 1)[0])


# ---------------------------------------------------------------- sampling
def run_family(family: str, sizes, trials: dict, qs: list[float], a: int | None = None) -> dict:
    """Per size: D samples (trials x len(qs)) with common random numbers across q; also D_G and depth stats."""
    out: dict = {"family": family, "a": a, "sizes": [], "q": qs, "trials": {}, "D": {}, "trD": {}, "DG": {},
                 "mean_depth": {}, "mean_width": {}, "merge_frac": {}, "wall": {}}
    for key in sizes:
        n = layered_n(a, key) if a is not None else key
        rng = np.random.default_rng(SEED + 1000 * n + FAMILY_OFFSET[family])
        t0 = time.time()
        T = trials[key]
        Ds = np.zeros((T, len(qs)))
        Ts = np.zeros((T, len(qs)))
        DG = np.zeros(T)
        md, mw, mf = [], [], []
        for t in range(T):
            parent = layered_parent(a, key, rng) if a is not None else uniform_rooted_tree(n, rng)
            level_list, widths, depth, u, r = merge_draws(parent, rng)
            for j, q in enumerate(qs):
                Ds[t, j], Ts[t, j] = kernel_D(parent, level_list, u, r, q)
            DG[t] = kernel_G(depth, widths)
            md.append(float(depth.mean()))
            mw.append(float(n / len(widths)))
            mf.append(float(np.mean(r >= 0)))
        out["sizes"].append(n)
        out["trials"][str(n)] = T
        out["D"][str(n)] = Ds
        out["trD"][str(n)] = Ts
        out["DG"][str(n)] = DG
        out["mean_depth"][str(n)] = float(np.mean(md))
        out["mean_width"][str(n)] = float(np.mean(mw))
        out["merge_frac"][str(n)] = float(np.mean(mf))  # fraction of cells with an available partner
        out["wall"][str(n)] = time.time() - t0
        print(f"  {family} n={n:5d} trials={T:4d} wall={out['wall'][str(n)]:.1f}s  E[D]/n^2 q={qs[0]}:{Ds[:,0].mean()/n**2:.4f}"
              f" q={qs[-1]}:{Ds[:,-1].mean()/n**2:.4f}  D_G/n^2:{DG.mean()/n**2:.4f}", flush=True)
    return out


def cv_mean(Y: np.ndarray, X: np.ndarray, mu_x: float) -> tuple[np.ndarray, np.ndarray]:
    """Control-variate mean of columns of Y using X (known mean mu_x). Returns (estimate, se) per column."""
    T = Y.shape[0]
    xc = X - X.mean()
    var_x = float(np.sum(xc * xc)) / (T - 1)
    est = np.zeros(Y.shape[1])
    se = np.zeros(Y.shape[1])
    for j in range(Y.shape[1]):
        y = Y[:, j]
        c = float(np.sum(xc * (y - y.mean()))) / (T - 1) / var_x if var_x > 0 else 0.0
        resid = y - c * X
        est[j] = resid.mean() + c * mu_x
        se[j] = resid.std(ddof=1) / math.sqrt(T)
    return est, se


def summarize(raw: dict, exact_mu: dict[int, float] | None) -> dict:
    """E[D_q]/n^2 per size (control variate if exact_mu given), local gamma, diagnostics."""
    sizes = raw["sizes"]
    qs = raw["q"]
    est, se = {}, {}
    for n in sizes:
        Ds = raw["D"][str(n)] / n**2
        if exact_mu is not None and 0.0 in qs:
            j0 = qs.index(0.0)
            e, s = cv_mean(Ds, Ds[:, j0], exact_mu[n])
        else:
            e, s = Ds.mean(axis=0), Ds.std(axis=0, ddof=1) / math.sqrt(Ds.shape[0])
        est[n], se[n] = e, s
    res = {"family": raw["family"], "a": raw["a"], "sizes": sizes, "q": qs, "trials": raw["trials"],
           "E_D_over_n2": {str(n): [float(x) for x in est[n]] for n in sizes},
           "E_D_over_n2_se": {str(n): [float(x) for x in se[n]] for n in sizes},
           "E_D_over_n2_plain": {str(n): [float(x) for x in raw["D"][str(n)].mean(axis=0) / n**2] for n in sizes},
           "E_trHkH_over_n": {str(n): [float(x) for x in raw["trD"][str(n)].mean(axis=0) / n] for n in sizes},
           "D_G_over_n2": {str(n): float(raw["DG"][str(n)].mean() / n**2) for n in sizes},
           "ratio_D_over_DG": {str(n): [float(np.mean(raw["D"][str(n)][:, j] / raw["DG"][str(n)])) for j in range(len(qs))]
                               for n in sizes},
           "mean_depth": raw["mean_depth"], "mean_width": raw["mean_width"], "partner_frac": raw["merge_frac"],
           "wall": raw["wall"]}
    loc = {}
    for i in range(len(sizes) - 1):
        n1, n2 = sizes[i], sizes[i + 1]
        loc[f"{n1}->{n2}"] = [float(0.5 * math.log(est[n2][j] / est[n1][j]) / math.log(n2 / n1)) for j in range(len(qs))]
    res["gamma_local"] = loc
    res["gamma_fit_all_sizes"] = [slope(sizes, [math.sqrt(est[n][j]) for n in sizes]) for j in range(len(qs))]
    return res, est


def bootstrap_gamma(raw: dict, sizes, exact_mu, qs_idx=None) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap (over trees, independent per size) of the least-squares gamma over `sizes`, per q."""
    qs = raw["q"]
    rng = np.random.default_rng(SEED)
    cols = list(range(len(qs))) if qs_idx is None else qs_idx
    gb = np.zeros((BOOT, len(cols)))
    for b in range(BOOT):
        ys = []
        for n in sizes:
            Ds = raw["D"][str(n)] / n**2
            idx = rng.integers(0, Ds.shape[0], size=Ds.shape[0])
            Db = Ds[idx]
            if exact_mu is not None and 0.0 in qs:
                e, _ = cv_mean(Db, Db[:, qs.index(0.0)], exact_mu[n])
            else:
                e = Db.mean(axis=0)
            ys.append(np.sqrt(e[cols]))
        ys = np.stack(ys)
        logn = np.log(np.asarray(sizes, float))
        X = np.vstack([logn, np.ones_like(logn)]).T
        gb[b] = np.linalg.lstsq(X, np.log(ys), rcond=None)[0][0]
    return gb.mean(axis=0), gb.std(axis=0, ddof=1)


def root_of(qs, g) -> float | None:
    for j in range(len(qs) - 1):
        if g[j] > 0 >= g[j + 1]:
            return float(qs[j] + (qs[j + 1] - qs[j]) * g[j] / (g[j] - g[j + 1]))
    return None


# ---------------------------------------------------------------- mechanism recursion (analytic)
def spread_recursion(q: float, depth: int) -> list[float]:
    """S_d = 1 + (1 - q/2) S_{d-1}, S_1 = 1 (wide-population idealisation). S_inf = 2/q; q = 0 gives S_d = d."""
    S = [0.0, 1.0]
    for _ in range(2, depth + 1):
        S.append(1.0 + (1.0 - q / 2.0) * S[-1])
    return S


def exponent_law(d_tree: float) -> float:
    return max(2.0 / d_tree - 1.0, -0.5)


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=("grid", "plateau", "layered", "all", "k4"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out_path = HERE / "predictions.json"
    result: dict = {}
    if out_path.is_file():
        try:
            result = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            result = {}
    result.update({"card": "F-01", "question": "Q-0017", "seed": SEED,
                   "convention": "gamma = d ln RMS / d ln n = 0.5 d ln(E D/n^2)/d ln n (F-02 K1 convention; >0 relevant)",
                   "exponent_law": {"formula": "gamma_merge(d_tree) = max(2/d_tree - 1, -1/2) for every q > 0",
                                    "values": {str(d): exponent_law(d) for d in (1, 2, 3, 4, 6)}},
                   "spread_recursion": {str(q): spread_recursion(q, 12) for q in (0.0, 0.1, 0.25, 0.5, 1.0)}})
    quick = args.quick

    if args.stage in ("grid", "all"):
        print("[grid] Cayley F-02 grid", GRID, "q fine", flush=True)
        trials = {n: (40 if quick else TRIALS_GRID[n]) for n in GRID}
        raw = run_family("cayley", GRID, trials, Q_FINE)
        mu = {n: cayley_exact(n)["E_D"] / n**2 for n in GRID}
        summ, est = summarize(raw, mu)
        gm, gse = bootstrap_gamma(raw, GRID, mu)
        summ["gamma_grid"] = [float(slope(GRID, [math.sqrt(est[n][j]) for n in GRID])) for j in range(len(Q_FINE))]
        summ["gamma_grid_boot_mean"] = [float(x) for x in gm]
        summ["gamma_grid_se_boot"] = [float(x) for x in gse]
        summ["q_star_on_grid"] = root_of(Q_FINE, summ["gamma_grid"])
        summ["gamma_grid_min"] = float(min(summ["gamma_grid"]))
        summ["exact_E_D_over_n2_q0"] = {str(n): mu[n] for n in GRID}
        summ["q0_plain_mc_vs_exact_rel"] = {str(n): float(raw["D"][str(n)][:, 0].mean() / n**2 / mu[n] - 1) for n in GRID}
        summ["F02_exact_slope_q0"] = float(slope(GRID, [math.sqrt(mu[n]) for n in GRID]))
        summ["ratio_to_iid_128"] = [float(math.sqrt(est[128][j] * 128**2 / 127)) for j in range(len(Q_FINE))]
        result["grid_stage"] = summ
        print(json.dumps({"gamma_grid": dict(zip([str(q) for q in Q_FINE], [round(x, 4) for x in summ["gamma_grid"]])),
                          "se": [round(x, 4) for x in gse], "q_star": summ["q_star_on_grid"]}), flush=True)

    if args.stage in ("plateau", "all"):
        print("[plateau] Cayley", PLATEAU, "q", Q_PLATEAU, flush=True)
        trials = {n: (6 if quick else TRIALS_PLATEAU[n]) for n in PLATEAU}
        raw = run_family("cayley", PLATEAU, trials, Q_PLATEAU)
        mu = {n: cayley_exact(n)["E_D"] / n**2 for n in PLATEAU}
        summ, est = summarize(raw, mu)
        summ["exact_E_D_over_n2_q0"] = {str(n): mu[n] for n in PLATEAU}
        k2 = list(K2_GRID)
        gm, gse = bootstrap_gamma(raw, k2, mu)
        summ["K2_grid"] = k2
        summ["gamma_K2grid"] = [float(slope(k2, [math.sqrt(est[n][j]) for n in k2])) for j in range(len(Q_PLATEAU))]
        summ["gamma_K2grid_se_boot"] = [float(x) for x in gse]
        summ["ratio_to_iid_1024"] = [float(math.sqrt(est[1024][j] * 1024**2 / 1023)) for j in range(len(Q_PLATEAU))]
        summ["ratio_to_iid_1024_se"] = [float(0.5 * summ["ratio_to_iid_1024"][j] * summ["E_D_over_n2_se"]["1024"][j] / est[1024][j])
                                        for j in range(len(Q_PLATEAU))]
        result["plateau_stage"] = summ
        print(json.dumps({"gamma_K2grid": summ["gamma_K2grid"], "se": summ["gamma_K2grid_se_boot"],
                          "E_D_over_n2": summ["E_D_over_n2"], "ratio_1024": summ["ratio_to_iid_1024"]}), flush=True)

    if args.stage in ("layered", "all"):
        result.setdefault("layered_stage", {})
        for a, hs in LAYERED.items():
            fam = f"L{a}"
            print(f"[layered] a={a} d_tree={a+1} h={hs}", flush=True)
            trials = {h: (6 if quick else TRIALS_LAYERED[a][h]) for h in hs}
            raw = run_family(fam, hs, trials, Q_LAYERED, a=a)
            summ, est = summarize(raw, None)
            summ["h"] = list(hs)
            summ["d_tree"] = a + 1
            summ["exponent_law_asymptote"] = exponent_law(a + 1)
            gm, gse = bootstrap_gamma(raw, raw["sizes"], None)
            summ["gamma_fit_se_boot"] = [float(x) for x in gse]
            if a == 2:
                k3 = [layered_n(2, h) for h in K3_H]
                gm3, gse3 = bootstrap_gamma(raw, k3, None)
                summ["K3_grid_n"] = k3
                summ["gamma_K3grid"] = [float(slope(k3, [math.sqrt(est[n][j]) for n in k3])) for j in range(len(Q_LAYERED))]
                summ["gamma_K3grid_se_boot"] = [float(x) for x in gse3]
            result["layered_stage"][fam] = summ
            print(json.dumps({"family": fam, "gamma_fit": summ["gamma_fit_all_sizes"], "se": summ["gamma_fit_se_boot"],
                              "gamma_local": summ["gamma_local"], "K3": summ.get("gamma_K3grid")}), flush=True)

    if args.stage == "k4":
        # K4 (tree-only kill, pre-registered 2026-09-03, not run at card time): cone a = 1 (d_tree = 2),
        # q in K4_Q, h in K4_H, fresh seed stream (FAMILY_OFFSET L1k4); each gamma_fit(q) must lie in K4_WINDOW.
        print(f"[k4] cone a=1 h={K4_H} q={K4_Q}", flush=True)
        trials = {h: (4 if quick else K4_TRIALS[h]) for h in K4_H}
        raw = run_family("L1k4", K4_H, trials, K4_Q, a=1)
        summ, est = summarize(raw, None)
        gm, gse = bootstrap_gamma(raw, raw["sizes"], None)
        summ["gamma_fit_se_boot"] = [float(x) for x in gse]
        summ["window"] = list(K4_WINDOW)
        summ["verdict"] = {str(q): ("KILL" if not (K4_WINDOW[0] <= g <= K4_WINDOW[1]) else "survive")
                           for q, g in zip(K4_Q, summ["gamma_fit_all_sizes"])}
        result["k4_stage"] = summ
        print(json.dumps({"gamma_fit": summ["gamma_fit_all_sizes"], "se": summ["gamma_fit_se_boot"],
                          "verdict": summ["verdict"]}), flush=True)
    if not quick:
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        print("wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
