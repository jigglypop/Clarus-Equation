"""a4: independent tree-only recomputation of the card numbers + Cayley marginality test.

(1) Fully independent pipeline: own Pruefer -> uniform rooted labelled tree, own merge draw,
    own A_q = inv(I-M), own D = ||H A A^T H||_F^2.  Fresh seed stream (20260903, not 20260902).
(2) cone a=1 q=1 and layered a=2 q=1 recomputation.
(3) Model comparison on the 9-point Cayley q=1 table: constant positive exponent vs marginal-with-log.
(4) Hub factor and G/diagonal crossover size.
"""
from __future__ import annotations
import heapq, json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "verify" / "Q-0008" / "F-02"))
from driver_numbers import cayley_exact  # noqa

OUT = {}
SEED = 20260903


def my_pruefer_rooted(n, rng):
    if n == 1:
        return [-1]
    if n == 2:
        return [-1, 0] if rng.integers(0, 2) == 0 else [1, -1]
    seq = rng.integers(0, n, size=n - 2)
    deg = np.ones(n, dtype=int)
    for s in seq:
        deg[s] += 1
    adj = [[] for _ in range(n)]
    leaves = [i for i in range(n) if deg[i] == 1]
    heapq.heapify(leaves)
    for s in seq:
        leaf = heapq.heappop(leaves)
        adj[leaf].append(int(s))
        adj[int(s)].append(leaf)
        deg[s] -= 1
        if deg[s] == 1:
            heapq.heappush(leaves, int(s))
    u = heapq.heappop(leaves)
    v = heapq.heappop(leaves)
    adj[u].append(v)
    adj[v].append(u)
    root = int(rng.integers(0, n))
    par = [-2] * n
    par[root] = -1
    st = [root]
    while st:
        x = st.pop()
        for y in adj[x]:
            if par[y] == -2:
                par[y] = x
                st.append(y)
    return par


def my_layered(a, h, rng):
    par = [-1]
    prev = [0]
    for d in range(1, h):
        cur = []
        for _ in range((d + 1) ** a):
            par.append(int(prev[int(rng.integers(0, len(prev)))]))
            cur.append(len(par) - 1)
        prev = cur
    return par


def my_depths(par):
    n = len(par)
    depth = np.full(n, -1, dtype=int)
    ch = [[] for _ in range(n)]
    root = -1
    for v, p in enumerate(par):
        if p >= 0:
            ch[p].append(v)
        else:
            root = v
    depth[root] = 0
    st = [root]
    while st:
        x = st.pop()
        for y in ch[x]:
            depth[y] = depth[x] + 1
            st.append(y)
    return depth


def my_D(par, q, rng):
    n = len(par)
    depth = my_depths(par)
    levels = {}
    for v in range(n):
        levels.setdefault(int(depth[v]), []).append(v)
    M = np.zeros((n, n))
    for v in range(n):
        p = par[v]
        if p < 0:
            continue
        others = levels[int(depth[v]) - 1]
        if len(others) >= 2 and rng.random() < q:
            c = p
            while c == p:
                c = others[int(rng.integers(0, len(others)))]
            M[v, p] += 0.5
            M[v, c] += 0.5
        else:
            M[v, p] += 1.0
    A = np.linalg.inv(np.eye(n) - M)
    B = A - A.mean(axis=0, keepdims=True)
    K = B.T @ B
    return float(np.sum(K * K))


def gamma_fit(ns, vals):
    return float(np.polyfit(np.log(np.asarray(ns, float)), 0.5 * np.log(np.asarray(vals, float)), 1)[0])


def main():
    rng = np.random.default_rng(SEED)
    T = 200
    tab = {}
    for n in (8, 16, 32, 64, 128):
        vals = [my_D(my_pruefer_rooted(n, rng), 1.0, rng) / n ** 2 for _ in range(T)]
        tab[n] = (float(np.mean(vals)), float(np.std(vals, ddof=1) / math.sqrt(T)))
    g = gamma_fit(list(tab), [tab[n][0] for n in tab])
    coef = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) / (10 * math.log(2))
    se = math.sqrt(sum((0.5 * tab[n][1] / tab[n][0] * c) ** 2 for n, c in zip(tab, coef)))
    OUT["indep_cayley_q1_grid"] = {"trials": T, "seed": SEED,
                                   "E_D_over_n2": {str(n): tab[n][0] for n in tab},
                                   "se_per_size": {str(n): tab[n][1] for n in tab},
                                   "gamma_fit": g, "gamma_se_approx": se,
                                   "card_value": 0.1973, "card_se": 0.006,
                                   "z_vs_card": (g - 0.1973) / max(se, 1e-9)}
    rng0 = np.random.default_rng(SEED + 5)
    tab0 = {}
    for n in (8, 16, 32, 64, 128):
        vals = [my_D(my_pruefer_rooted(n, rng0), 0.0, rng0) / n ** 2 for _ in range(T)]
        tab0[n] = float(np.mean(vals))
    OUT["indep_cayley_q0_vs_F02_exact"] = {
        "mc": {str(n): tab0[n] for n in tab0},
        "exact": {str(n): cayley_exact(n)["E_D"] / n ** 2 for n in tab0},
        "rel": {str(n): tab0[n] / (cayley_exact(n)["E_D"] / n ** 2) - 1 for n in tab0},
        "gamma_mc": gamma_fit(list(tab0), [tab0[n] for n in tab0]),
        "gamma_exact": gamma_fit(list(tab0), [cayley_exact(n)["E_D"] / n ** 2 for n in tab0])}

    for a, hs, card in ((1, (16, 23, 32, 45), -0.0388), (2, (8, 12, 16), -0.3789)):
        rng = np.random.default_rng(SEED + a)
        t = {}
        Tt = 120 if a == 1 else 60
        for h in hs:
            vals = [my_D(my_layered(a, h, rng), 1.0, rng) for _ in range(Tt)]
            n = len(my_layered(a, h, np.random.default_rng(0)))
            t[n] = float(np.mean(vals)) / n ** 2
        OUT["indep_layered_a%d_q1" % a] = {"trials": Tt,
                                           "E_D_over_n2": {str(k): v for k, v in t.items()},
                                           "gamma_fit": gamma_fit(list(t), list(t.values())),
                                           "card_value": card}

    pj = json.loads((ROOT / "verify/Q-0017/F-01/predictions.json").read_text(encoding="utf-8"))
    gs, ps = pj["grid_stage"], pj["plateau_stage"]
    j1g, j1p = gs["q"].index(1.0), ps["q"].index(1.0)
    pts = []
    for n in gs["sizes"]:
        pts.append((n, gs["E_D_over_n2"][str(n)][j1g], gs["E_D_over_n2_se"][str(n)][j1g]))
    for n in ps["sizes"]:
        if n != 128:
            pts.append((n, ps["E_D_over_n2"][str(n)][j1p], ps["E_D_over_n2_se"][str(n)][j1p]))
    ns = np.array([p[0] for p in pts], float)
    ys = np.array([p[1] for p in pts], float)
    ses = np.array([p[2] for p in pts], float)
    wl = (ys / ses) ** 2
    lnn, lny = np.log(ns), np.log(ys)
    Xp = np.vstack([lnn, np.ones_like(lnn)]).T
    Wm = np.diag(wl)
    bp = np.linalg.solve(Xp.T @ Wm @ Xp, Xp.T @ Wm @ lny)
    chi_p = float(np.sum(wl * (lny - Xp @ bp) ** 2))
    best = None
    for b in np.linspace(-1.9, 30.0, 6382):
        z = 2 * np.log(lnn + b)
        c = float(np.sum(wl * (lny - z)) / np.sum(wl))
        chi = float(np.sum(wl * (lny - z - c) ** 2))
        if best is None or chi < best[0]:
            best = (chi, float(b), c)
    OUT["cayley_q1_model_comparison"] = {
        "sizes": ns.tolist(), "E_D_over_n2": ys.tolist(), "se": ses.tolist(),
        "power_law": {"gamma": float(bp[0] / 2), "chi2": chi_p, "npar": 2},
        "marginal_log_D_prop_lnn_sq": {"b": best[1], "chi2": best[0], "npar": 2,
                                       "implied_local_gamma": {str(int(n)): 1.0 / (math.log(n) + best[1]) for n in ns}},
        "verdict": "power" if chi_p < best[0] else "marginal"}

    cross = {}
    for fam, a in (("L1", 1), ("L2", 2), ("L3", 3)):
        st = pj["layered_stage"][fam]
        j = st["q"].index(1.0)
        rows = []
        for n in st["sizes"]:
            tot = st["E_D_over_n2"][str(n)][j]
            dg = st["D_G_over_n2"][str(n)]
            rows.append({"n": n, "D_over_n2": tot, "D_G_over_n2": dg, "ratio_tot_over_G": tot / dg})
        Amat = np.array([[r["D_G_over_n2"], 1.0 / r["n"]] for r in rows])
        yv = np.array([r["D_over_n2"] for r in rows])
        sol, *_ = np.linalg.lstsq(Amat, yv, rcond=None)
        gG = 2.0 / (a + 1.0) - 1.0
        n0 = rows[0]["n"]
        cG = sol[0] * rows[0]["D_G_over_n2"] * n0 ** (-2 * gG)
        n_cross = float("inf")
        if sol[0] > 0 and sol[1] > 0 and abs(1 + 2 * gG) > 1e-9:
            n_cross = (sol[1] / cG) ** (1.0 / (1.0 + 2 * gG))
        cross[fam] = {"d_tree": a + 1, "rows": rows, "hub2_fit": float(sol[0]), "c_diag_fit": float(sol[1]),
                      "gamma_G_law": gG, "crossover_n_G_beats_diagonal": n_cross}
    OUT["hub_and_crossover"] = cross
    print(json.dumps({k: v for k, v in OUT.items() if k != "hub_and_crossover"}, indent=1, default=float))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in cross.items()}, indent=1))
    (HERE / "a4_indep_gamma.json").write_text(json.dumps(OUT, indent=1, ensure_ascii=False, default=float),
                                              encoding="utf-8")


if __name__ == "__main__":
    main()
