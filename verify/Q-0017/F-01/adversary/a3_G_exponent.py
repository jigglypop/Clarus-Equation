"""a3: is gamma_G = max(2/d_tree - 1, -1/2) a first-principles law for the generation-mean kernel G,
or a two-point interpolation fitted to the observed a=1 / a=2 families?

G_vw = g(min(d_v,d_w)), g(d) = sum_{k<=d} 1/W_k, W_d = (d+1)^a, d = 0..h-1, n = sum W_d, d_tree = a+1.
G depends on depth only => ||H G H||_F^2 collapses to an O(h) formula (derived here, validated against
brute force O(h^2) and against the card driver's own kernel_G).  Numerically stable version: replace g
by phi = g - g(h-1) - weighted mean (centering is invariant), computed from the reversed tail sum.
This reads the TRUE asymptotic exponent at n up to ~1e12 instead of the simulated n <= 2870.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "verify" / "Q-0017" / "F-01"))
from predict_merge_gamma import kernel_G, layered_parent, layered_n, merge_draws, exponent_law  # noqa

OUT = {}


def _phi(a: float, h: int):
    d = np.arange(h, dtype=float)
    w = (d + 1.0) ** a
    n = float(math.fsum(w))
    inv = 1.0 / w
    tail = np.concatenate((np.cumsum(inv[::-1])[::-1][1:], [0.0]))  # tail_d = sum_{k>d} 1/W_k
    f = -tail                                                       # g(d) - g(h-1)
    f = f - float(math.fsum(w * f)) / n                             # weighted-centre (invariant)
    return w, n, f


def DG_fast(a: float, h: int):
    w, n, f = _phi(a, h)
    cw = np.cumsum(w)
    P = np.cumsum(w * f)
    P2 = np.cumsum(w * f * f)
    T = n - cw
    R = P + f * T
    m_d = R / n
    m = float(math.fsum(w * R)) / n ** 2
    P2m = np.concatenate(([0.0], P2[:-1]))
    sq = float(math.fsum(w * (P2m + f * f * (w + T))))
    D = sq - 2.0 * n * float(math.fsum(w * m_d ** 2)) + n ** 2 * m ** 2
    return float(D), n


def DG_brute(a: float, h: int):
    w, n, f = _phi(a, h)
    M = f[np.minimum.outer(np.arange(h), np.arange(h))]
    mrow = (M * w).sum(axis=1) / n
    tot = float((w[:, None] * w[None, :] * M).sum()) / n ** 2
    C = M - mrow[:, None] - mrow[None, :] + tot
    return float(((np.sqrt(np.outer(w, w)) * C) ** 2).sum()), n


def main():
    val = {}
    for a in (0.0, 1.0, 1.5, 2.0, 3.0, 4.0):
        for h in (17, 200, 1500):
            fst, _ = DG_fast(a, h)
            bru, _ = DG_brute(a, h)
            val[f"a{a}_h{h}"] = {"fast": fst, "brute": bru, "rel": fst / bru - 1 if bru else 0.0}
    OUT["formula_validation_vs_brute"] = val
    print("max |rel| fast-vs-brute:", max(abs(v["rel"]) for v in val.values()))

    drv = {}
    for a, h in ((1, 16), (1, 32), (2, 8), (2, 12), (3, 5), (3, 7)):
        rng = np.random.default_rng(5)
        parent = layered_parent(a, h, rng)
        _, widths, depth, _, _ = merge_draws(parent, rng)
        d_card = kernel_G(depth, widths)
        d_mine, n = DG_fast(float(a), h)
        drv[f"a{a}_h{h}"] = {"card_kernel_G": d_card, "mine": d_mine, "rel": d_mine / d_card - 1}
    OUT["vs_card_kernel_G"] = drv
    print("max |rel| vs card kernel_G:", max(abs(v["rel"]) for v in drv.values()))

    hs = [1000, 4000, 16000, 64000, 256000, 1024000]
    for a in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 2.9, 3.0, 3.1, 3.5, 4.0, 6.0):
        seq, prev = [], None
        for h in hs:
            D, n = DG_fast(a, h)
            if prev is not None and D > 0 and prev[0] > 0:
                seq.append({"h": h, "n": n, "gamma_local": 0.5 * math.log((D / n**2) / (prev[0] / prev[1]**2)) / math.log(n / prev[1])})
            prev = (D, n)
        law = exponent_law(a + 1.0)
        OUT.setdefault("gamma_G_asymptotic", {})[f"a={a}"] = {
            "d_tree": a + 1.0, "law": law, "middle_branch_2_over_dtree_minus1": 2.0 / (a + 1.0) - 1.0,
            "gamma_G_last": seq[-1]["gamma_local"], "n_last": seq[-1]["n"],
            "seq": [round(s["gamma_local"], 4) for s in seq], "err_vs_law": seq[-1]["gamma_local"] - law}
    for k, v in OUT["gamma_G_asymptotic"].items():
        print(f"{k:9s} d_tree={v['d_tree']:.2f} law={v['law']:+.4f} mid={v['middle_branch_2_over_dtree_minus1']:+.4f} "
              f"gammaG(n={v['n_last']:.2e})={v['gamma_G_last']:+.4f} err={v['err_vs_law']:+.4f} seq={v['seq']}")

    small = [{"h": h, "n": DG_fast(1.5, h)[1], "D_over_n2": DG_fast(1.5, h)[0] / DG_fast(1.5, h)[1] ** 2}
             for h in (16, 23, 32, 45, 64, 90, 128)]
    xs = np.log([s["n"] for s in small]); ys = 0.5 * np.log([s["D_over_n2"] for s in small])
    OUT["a1p5_small_h_fit"] = {"gamma_G_fit_small_h": float(np.polyfit(xs, ys, 1)[0]), "card_claim": -0.17,
                               "law": -0.2, "gamma_G_asymptotic": OUT["gamma_G_asymptotic"]["a=1.5"]["gamma_G_last"]}
    print(json.dumps(OUT["a1p5_small_h_fit"], indent=1))

    fin = {}
    for a, hs2 in ((1, (16, 23, 32, 45, 64)), (2, (8, 12, 16, 20)), (3, (5, 6, 7, 8, 9))):
        tab = []
        for h in hs2:
            D, n = DG_fast(float(a), h)
            tab.append({"h": h, "n_cont": n, "n_int": layered_n(a, h), "D_over_n2": D / n ** 2})
        xs = np.log([t["n_cont"] for t in tab]); ys = 0.5 * np.log([t["D_over_n2"] for t in tab])
        fin[f"a={a}"] = {"gamma_G_fit_on_simulated_grid": float(np.polyfit(xs, ys, 1)[0]),
                         "law": exponent_law(a + 1.0), "table": tab}
    OUT["gamma_G_on_simulated_grids"] = fin
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "table"} for k, v in fin.items()}, indent=1))
    (HERE / "a3_G_exponent.json").write_text(json.dumps(OUT, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
