"""a8: ladder step 3 says the exponent is carried by the generation-mean walk kernel G
(a smooth function of the depth pair).  Split E[D] exactly:

    E ||H kappa H||^2 = ||H E[kappa] H||^2  +  E ||H (kappa - E kappa) H||^2
                          smooth/mean part        ancestry fluctuation part

For the layered family the depths are deterministic, so E[kappa] is a genuine depth-profile kernel and
the first term is the term the card mechanism computes (hub^2 * ||H G H||^2).  Unbiased split from T
trials:  A = (T*||H kbar H||^2 - mean_t ||H k_t H||^2)/(T-1),  F = mean_t - A.
If F dominates and its exponent differs from gamma_G, the mechanism does not control the observable.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "verify" / "Q-0017" / "F-01"))
import predict_merge_gamma as P  # noqa

OUT = {}


def kappa_of(parent, level_list, u, r, q):
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
    return A @ A.T


def cen(M):
    M = M - M.mean(axis=0, keepdims=True)
    M = M - M.mean(axis=1, keepdims=True)
    return M


def fit(ns, ys):
    return float(np.polyfit(np.log(np.asarray(ns, float)), 0.5 * np.log(np.asarray(ys, float)), 1)[0])


def main():
    plan = {1: ((16, 23, 32, 45), (300, 300, 250, 150)), 2: ((8, 12, 16), (600, 400, 250))}
    for a, (hs, Ts) in plan.items():
        rows = []
        for h, T in zip(hs, Ts):
            rng = np.random.default_rng(20260903 + h)
            kbar = None
            s2 = 0.0
            dg = 0.0
            for t in range(T):
                parent = P.layered_parent(a, h, rng)
                level_list, widths, depth, u, r = P.merge_draws(parent, rng)
                kap = kappa_of(parent, level_list, u, r, 1.0)
                C = cen(kap)
                s2 += float(np.sum(C * C))
                kbar = kap if kbar is None else kbar + kap
                dg += P.kernel_G(depth, widths)
            n = len(parent)
            kbar /= T
            Cb = cen(kbar)
            m2 = float(np.sum(Cb * Cb))
            s2 /= T
            dgm = dg / T
            A = (T * m2 - s2) / (T - 1)
            F = s2 - A
            rows.append({"h": h, "n": n, "T": T, "E_D": s2, "mean_part": A, "fluct_part": F,
                         "D_G": dgm, "mean_over_D_G": A / dgm, "mean_frac": A / s2,
                         "E_D_over_n2": s2 / n ** 2, "mean_over_n2": A / n ** 2,
                         "fluct_over_n2": F / n ** 2, "D_G_over_n2": dgm / n ** 2})
            print(f"a={a} n={n:5d} T={T:4d} E_D/n2={s2/n**2:.6f} mean/n2={A/n**2:.6f} "
                  f"fluct/n2={F/n**2:.6f} D_G/n2={dgm/n**2:.6f} mean_frac={A/s2:.4f} "
                  f"hub2=A/D_G={A/dgm:.3f}", flush=True)
        ns = [r["n"] for r in rows]
        OUT[f"a={a}"] = {
            "rows": rows, "d_tree": a + 1, "law": max(2.0 / (a + 1.0) - 1.0, -0.5),
            "gamma_total": fit(ns, [r["E_D_over_n2"] for r in rows]),
            "gamma_mean_part": fit(ns, [r["mean_over_n2"] for r in rows]),
            "gamma_fluct_part": fit(ns, [r["fluct_over_n2"] for r in rows]),
            "gamma_G": fit(ns, [r["D_G_over_n2"] for r in rows])}
        print(json.dumps({k: v for k, v in OUT[f"a={a}"].items() if k != "rows"}, indent=1))
    (HERE / "a8_mean_vs_fluct.json").write_text(json.dumps(OUT, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
