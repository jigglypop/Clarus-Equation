"""a10: EXACT identity for the mean merge kernel (strengthens / tests ladder step 3).

Claim derived here for the layered family (uniform parent in level d-1, uniform second parent in
level d-1 minus {p}, weights 1/2):
    E[kappa_vw] = sum_{k <= min(d_v,d_w)} S_k / W_k      for v != w,
    E[kappa_vv] = E[kappa_vw]|_{min=d_v} + S_{d_v},
with the EXACT finite-width spread recursion  S_d = 1 + (1 - q/2 - 1/W_{d-1}) S_{d-1}.
Consequences: the card hub factor h_q is exactly S_star (not just ">= 1"), and
    ||H E[kappa] H||_F^2 = S_star^2 ||H G H||_F^2 + (diagonal ~ n S_bar^2) + O(exp small),
i.e. gamma = max(gamma_G, -1/2) for the MEAN kernel.  Tested by direct MC.
"""
from __future__ import annotations
import json, sys
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


def predicted(a, h, q):
    W = np.array([(d + 1) ** a for d in range(h)], dtype=float)
    S = np.zeros(h)
    S[0] = 1.0
    for d in range(1, h):
        Wprev = W[d - 1]
        qe = q if Wprev >= 2 else 0.0          # no partner exists when the previous level has 1 cell
        eff = 1.0 - qe / 2.0 - 1.0 / Wprev
        S[d] = 1.0 + eff * S[d - 1]
    c = np.cumsum(S / W)          # c[m] = sum_{k<=m} S_k / W_k
    return S, W, c


def main():
    for a, h, q, T in ((1, 20, 1.0, 400), (1, 26, 0.5, 300), (2, 9, 1.0, 300), (2, 9, 0.5, 250), (3, 6, 1.0, 250)):
        rng = np.random.default_rng(20260903 + 13 * h + a)
        kbar = None
        for _ in range(T):
            parent = P.layered_parent(a, h, rng)
            level_list, widths, depth, u, r = P.merge_draws(parent, rng)
            kap = kappa_of(parent, level_list, u, r, q)
            kbar = kap if kbar is None else kbar + kap
        kbar /= T
        n = len(parent)
        S, W, c = predicted(a, h, q)
        rows = []
        for m in range(h):
            # off-diagonal pairs with min depth exactly m: take v in level m, w in the deepest level
            vs = [v for v in range(n) if depth[v] == m]
            ws = [v for v in range(n) if depth[v] == h - 1]
            vals = [kbar[v, w] for v in vs for w in ws if v != w]
            if not vals:
                continue
            # pair (depth m, depth h-1):  min = m.  If m < h-1 the deeper cell recurses down to
            # level m+1, giving C_{m+1} = c[m]; if m == h-1 both are in the same level, giving C_m = c[m-1].
            pred = float(c[m]) if m < h - 1 else float(c[m - 1])
            rows.append({"m": m, "mc_mean_kappa": float(np.mean(vals)), "predicted": pred,
                         "rel": float(np.mean(vals) / pred - 1), "G_m_card": float(np.cumsum(1.0 / W)[m]),
                         "S_m": float(S[m])})
        diag = [{"d": d, "mc": float(np.mean([kbar[v, v] for v in range(n) if depth[v] == d])),
                 "predicted": float(c[d - 1] + S[d])} for d in range(1, h)]
        OUT[f"a{a}_h{h}_q{q}"] = {
            "n": n, "T": T,
            "max_abs_rel_offdiag": max(abs(r["rel"]) for r in rows[1:]),
            "offdiag_rows": rows,
            "diag_max_rel": max(abs(d["mc"] / d["predicted"] - 1) for d in diag),
            "diag_rows": diag,
            "S_star_exact_finiteW_last": float(S[-1]),
            "S_star_card_2overq": 2.0 / q,
            "hub_ratio_c_over_G_last": float(c[-1] / np.cumsum(1.0 / W)[-1])}
        print(f"a={a} h={h} q={q}: max|rel| offdiag={OUT[f'a{a}_h{h}_q{q}']['max_abs_rel_offdiag']:.4f} "
              f"diag={OUT[f'a{a}_h{h}_q{q}']['diag_max_rel']:.4f} "
              f"S_last={S[-1]:.4f} (card 2/q={2.0/q:.3f}) hub=c/G={OUT[f'a{a}_h{h}_q{q}']['hub_ratio_c_over_G_last']:.4f}",
              flush=True)
    (HERE / "a10_meankernel_identity.json").write_text(json.dumps(OUT, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
