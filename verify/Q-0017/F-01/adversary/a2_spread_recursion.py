"""a2: independent derivation + MC check of the spread recursion S_d = 1 + (1-q/2) S_{d-1}.

Independent derivation (finite level width W, labels scalar, unit noise):
  V_d = Var(L_v),  C_d = Cov(L_v, L_w), v != w in the same level, S_d = V_d - C_d.
  non-merge child: L = xi + L_p ;  merge child: L = xi + (L_p + L_r)/2, p != r uniform.
  V_d = 1 + (1-q) V_{d-1} + q (V_{d-1} + C_{d-1})/2 = 1 + V_{d-1} - (q/2) S_{d-1}
  For two distinct children the parent draws are independent, P[share a parent slot] = 1/W in
  every merge/no-merge combination, hence  C_d = C_{d-1} + S_{d-1}/W  (EXACT for constant W).
  =>  S_d = 1 + (1 - q/2 - 1/W) S_{d-1},   S_star = 2 / (q + 2/W).
  The card's S_d = 1 + (1-q/2) S_{d-1}, S_star = 2/q is the W -> infinity limit only.
Checked by direct label MC on a constant-width layered DAG.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "verify" / "Q-0008" / "F-02"))
sys.path.insert(0, str(ROOT / "verify" / "Q-0017" / "F-01"))
from predict_merge_gamma import kernel_D, merge_draws, layered_parent, layered_n, spread_recursion  # noqa

OUT = {}


def mc_spread(W, h, q, trials, seed):
    """Constant-width layered DAG, scalar labels: empirical V_d, C_d over trials."""
    rng = np.random.default_rng(seed)
    V = np.zeros(h)
    C = np.zeros(h)
    L = np.zeros((trials, W))
    L[:] = rng.normal(size=(trials, W))  # level 0: independent roots (V=1, C=0)
    V[0], C[0] = 1.0, 0.0
    for d in range(1, h):
        p = rng.integers(0, W, size=(trials, W))
        r = rng.integers(0, W - 1, size=(trials, W))
        r = np.where(r == p, W - 1, r)
        merged = rng.random(size=(trials, W)) < q
        Lp = np.take_along_axis(L, p, axis=1)
        Lr = np.take_along_axis(L, r, axis=1)
        new = rng.normal(size=(trials, W)) + np.where(merged, 0.5 * (Lp + Lr), Lp)
        L = new
        # ensemble variance / covariance across the level (average over trials)
        V[d] = float(np.mean(L * L))
        cov = (L.sum(axis=1) ** 2 - (L * L).sum(axis=1)) / (W * (W - 1))
        C[d] = float(np.mean(cov))
    return V, C


def exact_rec(W, h, q):
    S = np.zeros(h)
    S[0] = 1.0
    for d in range(1, h):
        S[d] = 1.0 + (1.0 - q / 2.0 - 1.0 / W) * S[d - 1]
    return S


def main():
    res = {}
    for W, q in ((8, 1.0), (8, 0.5), (32, 1.0), (32, 0.25), (128, 1.0), (128, 0.5), (512, 1.0)):
        h = 40
        V, C = mc_spread(W, h, q, 4000, 20260902 + W * 7 + int(100 * q))
        S = V - C
        Sx = exact_rec(W, h, q)
        card = spread_recursion(q, h)[1:h + 1]  # card recursion S_1 = 1, ...
        res[f"W{W}_q{q}"] = {
            "S_mc_tail": [round(float(x), 4) for x in S[-4:]],
            "S_exact_finiteW_tail": [round(float(x), 4) for x in Sx[-4:]],
            "S_card_idealised_tail": [round(float(x), 4) for x in card[-4:]],
            "S_star_exact_finiteW": 2.0 / (q + 2.0 / W),
            "S_star_card_2overq": 2.0 / q,
            "mc_vs_exact_rel": float(S[-1] / Sx[-1] - 1.0),
            "card_vs_exact_rel": float(card[-1] / Sx[-1] - 1.0),
        }
    OUT["spread_recursion"] = res

    # ---- tr(H kappa H)/n on the real substrates vs S_star = 2/q (card claims saturation)
    sb = {}
    for a, hs in ((1, (16, 32, 64)), (2, (8, 12, 16)), (3, (5, 7, 9))):
        for h in hs:
            n = layered_n(a, h)
            rng = np.random.default_rng(999 + n)
            vals = []
            for _ in range(30 if n < 1200 else 10):
                parent = layered_parent(a, h, rng)
                level_list, widths, depth, u, r = merge_draws(parent, rng)
                vals.append(kernel_D(parent, level_list, u, r, 1.0)[1] / n)
            sb[f"a{a}_n{n}"] = {"S_bar": float(np.mean(vals)), "S_star_2overq": 2.0,
                                "sum_1_over_W": float(np.sum(1.0 / np.asarray([(d + 1) ** a for d in range(h)])))}
    OUT["S_bar_vs_S_star"] = sb
    print(json.dumps(OUT, indent=1, ensure_ascii=False))
    (HERE / "a2_spread_recursion.json").write_text(json.dumps(OUT, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
