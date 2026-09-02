"""a6: (A) direct measurement of the hub factor kappa_vw / G_vw (ladder step 3 assumes it is O(1));
       (B) crossover size where the G component overtakes the diagonal floor (the card says n >~ 1e5);
       (C) an INDEPENDENT REPLICATE of kill K4 on a fresh seed stream (offset 179, NOT the
           pre-registered 78) to measure how much kill power K4 has.
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "verify" / "Q-0008" / "F-02"))
sys.path.insert(0, str(ROOT / "verify" / "Q-0017" / "F-01"))
import predict_merge_gamma as P  # noqa

OUT = {}


def kappa_and_G(parent, level_list, widths, depth, u, r, q):
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
    kap = A @ A.T
    g = np.cumsum(1.0 / widths)
    G = g[np.minimum.outer(depth, depth)]
    return kap, G


def main():
    # ---------- (A) hub factor on the deepest level, cone (a=1) and a=2
    hub = {}
    for a, hs, T in ((1, (16, 23, 32, 45, 64), 12), (2, (8, 12, 16, 20), 8)):
        for h in hs:
            rng = np.random.default_rng(4321 + h)
            rs = []
            for _ in range(T):
                parent = P.layered_parent(a, h, rng)
                level_list, widths, depth, u, r = P.merge_draws(parent, rng)
                kap, G = kappa_and_G(parent, level_list, widths, depth, u, r, 1.0)
                deep = level_list[-1]
                sub = np.ix_(deep, deep)
                m = ~np.eye(len(deep), dtype=bool)
                rs.append(float(np.mean(kap[sub][m]) / np.mean(G[sub][m])))
            hub[f"a{a}_h{h}_n{len(parent)}"] = {"n": len(parent), "hub_deepest_pairs": float(np.mean(rs)),
                                                "sd": float(np.std(rs))}
    OUT["hub_factor_direct"] = hub
    for a in (1, 2):
        ks = [(v["n"], v["hub_deepest_pairs"]) for k, v in hub.items() if k.startswith(f"a{a}_")]
        xs = np.log([k[0] for k in ks]); ys = np.log([k[1] for k in ks])
        OUT.setdefault("hub_growth_exponent", {})[f"a={a}"] = {
            "points": ks, "dlnhub_dlnn": float(np.polyfit(xs, ys, 1)[0])}
    print(json.dumps(OUT["hub_growth_exponent"], indent=1, default=float))

    # ---------- (B) crossover: fit total = cG n^{2 gamma_G} + c1/n on the card tables
    pj = json.loads((ROOT / "verify/Q-0017/F-01/predictions.json").read_text(encoding="utf-8"))
    cross = {}
    for fam, a in (("L1", 1), ("L2", 2), ("L3", 3)):
        st = pj["layered_stage"][fam]
        j = st["q"].index(1.0)
        ns = np.array(st["sizes"], float)
        tot = np.array([st["E_D_over_n2"][str(int(n))][j] for n in ns])
        gG = 2.0 / (a + 1.0) - 1.0
        M = np.vstack([ns ** (2 * gG), 1.0 / ns]).T
        sol, *_ = np.linalg.lstsq(M, tot, rcond=None)
        ncr = float("inf")
        if sol[0] > 0 and sol[1] > 0:
            ncr = (sol[1] / sol[0]) ** (1.0 / (1.0 + 2 * gG)) if abs(1 + 2 * gG) > 1e-9 else float("inf")
        cross[fam] = {"d_tree": a + 1, "gamma_G": gG, "c_G": float(sol[0]), "c_diag": float(sol[1]),
                      "crossover_n": ncr, "card_claim": "beyond n ~ 1e5",
                      "fit_resid_rel": [float(x) for x in (M @ sol - tot) / tot]}
    OUT["crossover"] = cross
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "fit_resid_rel"} for k, v in cross.items()},
                     indent=1, default=float))

    # ---------- (C) K4 replicate on a fresh stream
    P.FAMILY_OFFSET["L1adv"] = 179
    t0 = time.time()
    raw = P.run_family("L1adv", P.K4_H, P.K4_TRIALS, P.K4_Q, a=1)
    summ, est = P.summarize(raw, None)
    gm, gse = P.bootstrap_gamma(raw, raw["sizes"], None)
    OUT["k4_replicate_offset179"] = {
        "sizes": raw["sizes"], "q": P.K4_Q, "trials": P.K4_TRIALS,
        "gamma_fit_all_sizes": summ["gamma_fit_all_sizes"], "boot_se": [float(x) for x in gse],
        "window": list(P.K4_WINDOW),
        "verdict": {str(q): ("KILL" if not (P.K4_WINDOW[0] <= g <= P.K4_WINDOW[1]) else "survive")
                    for q, g in zip(P.K4_Q, summ["gamma_fit_all_sizes"])},
        "card_smoke_offset77": [-0.007, -0.040, -0.027],
        "E_D_over_n2": summ["E_D_over_n2"], "gamma_local": summ["gamma_local"],
        "wall_s": time.time() - t0}
    print(json.dumps(OUT["k4_replicate_offset179"], indent=1, default=float))
    (HERE / "a6_hub_and_k4.json").write_text(json.dumps(OUT, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
