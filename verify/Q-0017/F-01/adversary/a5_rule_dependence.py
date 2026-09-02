"""a5: is gamma = max(2/d_tree-1,-1/2) a property of d_tree, or of the NON-LOCAL second-parent rule?

The card draws the second parent uniformly over the whole previous level.  That is a mean-field
(non-local) merge: ancestral mass spreads over the entire level, which is exactly what produces the
generation-mean walk kernel G.  A 2-complex merge is presumably LOCAL.  Variants tested on the same
substrates, same q, same sizes:
  uniform  : card rule, r uniform in level(d-1) minus {p}
  ring1    : r = an index-neighbour of p inside level(d-1) (maximally local)
  ringK    : r uniform within +-K index positions of p
  auntie   : r uniform among the other children of the grandparent (tree-local merge)
  samegen  : r uniform among level(d) cells with a smaller index (same-generation merge, still a DAG)
  skew     : card rule but weights (0.75, 0.25) instead of (1/2, 1/2)
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "verify" / "Q-0008" / "F-02"))
from driver_numbers import uniform_rooted_tree  # noqa

OUT = {}


def layered(a, h, rng):
    par = [-1]
    prev = [0]
    for d in range(1, h):
        cur = []
        for _ in range((d + 1) ** a):
            par.append(int(prev[int(rng.integers(0, len(prev)))]))
            cur.append(len(par) - 1)
        prev = cur
    return par


def depths_levels(par):
    n = len(par)
    ch = [[] for _ in range(n)]
    root = -1
    for v, p in enumerate(par):
        if p >= 0:
            ch[p].append(v)
        else:
            root = v
    depth = np.zeros(n, dtype=int)
    st = [root]
    while st:
        x = st.pop()
        for y in ch[x]:
            depth[y] = depth[x] + 1
            st.append(y)
    levels = {}
    for v in range(n):
        levels.setdefault(int(depth[v]), []).append(v)
    return depth, levels, ch


def build_M(par, q, rule, rng, K=1, w1=0.5):
    n = len(par)
    depth, levels, ch = depths_levels(par)
    pos = {}
    for d, lst in levels.items():
        for i, v in enumerate(lst):
            pos[v] = i
    M = np.zeros((n, n))
    for v in range(n):
        p = par[v]
        if p < 0:
            continue
        d = int(depth[v])
        prev = levels[d - 1]
        W = len(prev)
        c = -1
        if rng.random() < q:
            if rule in ("uniform", "skew"):
                if W >= 2:
                    c = p
                    while c == p:
                        c = prev[int(rng.integers(0, W))]
            elif rule in ("ring1", "ringK"):
                kk = 1 if rule == "ring1" else K
                if W >= 2:
                    off = 0
                    while off == 0:
                        off = int(rng.integers(-kk, kk + 1))
                    c = prev[(pos[p] + off) % W]
                    if c == p:
                        c = -1
            elif rule == "auntie":
                gp = par[p]
                if gp >= 0 and len(ch[gp]) >= 2:
                    c = p
                    while c == p:
                        c = ch[gp][int(rng.integers(0, len(ch[gp])))]
            elif rule == "samegen":
                same = [u for u in levels[d] if u < v]
                if same:
                    c = same[int(rng.integers(0, len(same)))]
        if c >= 0:
            a1 = w1 if rule == "skew" else 0.5
            M[v, p] += a1
            M[v, c] += 1.0 - a1
        else:
            M[v, p] += 1.0
    return M


def D_of(M):
    n = M.shape[0]
    A = np.linalg.inv(np.eye(n) - M)
    B = A - A.mean(axis=0, keepdims=True)
    K = B.T @ B
    return float(np.sum(K * K)) / n ** 2


def gamma(ns, ys):
    return float(np.polyfit(np.log(np.asarray(ns, float)), 0.5 * np.log(np.asarray(ys, float)), 1)[0])


def main():
    rules = ("uniform", "ring1", "ringK", "auntie", "samegen", "skew")
    # cone a = 1 (d_tree = 2), q = 1
    res = {}
    for rule in rules:
        rng = np.random.default_rng(20260903 + hash(rule) % 1000)
        tab = {}
        for h in (16, 23, 32, 45):
            vals = []
            for _ in range(60):
                par = layered(1, h, rng)
                vals.append(D_of(build_M(par, 1.0, rule, rng, K=3)))
            tab[len(par)] = float(np.mean(vals))
        res[rule] = {"E_D_over_n2": tab, "gamma": gamma(list(tab), list(tab.values()))}
    OUT["cone_a1_q1_rules"] = res

    # layered a = 2 (d_tree = 3), q = 1
    res2 = {}
    for rule in ("uniform", "ring1", "auntie"):
        rng = np.random.default_rng(777 + hash(rule) % 1000)
        tab = {}
        for h in (8, 12, 16):
            vals = []
            for _ in range(40):
                par = layered(2, h, rng)
                vals.append(D_of(build_M(par, 1.0, rule, rng)))
            tab[len(par)] = float(np.mean(vals))
        res2[rule] = {"E_D_over_n2": tab, "gamma": gamma(list(tab), list(tab.values()))}
    OUT["layered_a2_q1_rules"] = res2

    # Cayley F-02 grid, q = 1
    res3 = {}
    for rule in ("uniform", "ring1", "auntie"):
        rng = np.random.default_rng(31337 + hash(rule) % 1000)
        tab = {}
        for n in (8, 16, 32, 64, 128):
            vals = []
            for _ in range(150):
                par = uniform_rooted_tree(n, rng)
                vals.append(D_of(build_M(par, 1.0, rule, rng)))
            tab[n] = float(np.mean(vals))
        res3[rule] = {"E_D_over_n2": tab, "gamma": gamma(list(tab), list(tab.values()))}
    OUT["cayley_q1_rules"] = res3
    for k, v in OUT.items():
        print(k)
        for rule, r in v.items():
            print(f"   {rule:9s} gamma={r['gamma']:+.4f}  table={[round(x,5) for x in r['E_D_over_n2'].values()]}")
    (HERE / "a5_rule_dependence.json").write_text(json.dumps(OUT, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
