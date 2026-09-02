"""Q-0016 F-01 adversary a1: independent re-derivation of kappa_split, closed forms, sign claim, bound.

Nothing is imported from the card scripts.  Every object is rebuilt from the card WORDS:
  C_uu = 1 ; C_uu' = -s/(k_z-1) for u != u' children of the same z with k_z >= 2 ; else 0
  A_vu = [u is ancestor-or-self of v]
  kappa_split = A C A^T ;  D_split = ||H kappa_split H||_F^2 , H = I - J/n
  claimed:   kappa_split = A A^T - B ,  B_vw = [v,w incomparable] / (k_lca(v,w) - 1)
"""
from __future__ import annotations
import itertools, json, math
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent / "a1_algebra.json"
R: dict = {}


def anc_sets(parent):
    n = len(parent)
    anc = [None] * n

    def get(v):
        if anc[v] is None:
            anc[v] = {v} | (get(parent[v]) if parent[v] >= 0 else set())
        return anc[v]

    for v in range(n):
        get(v)
    return anc


def A_matrix(parent):
    n = len(parent)
    anc = anc_sets(parent)
    A = np.zeros((n, n))
    for v in range(n):
        for u in anc[v]:
            A[v, u] = 1.0
    return A


def C_matrix(parent, s=1.0):
    n = len(parent)
    ch = [[] for _ in range(n)]
    for v, p in enumerate(parent):
        if p >= 0:
            ch[p].append(v)
    C = np.eye(n)
    for kids in ch:
        k = len(kids)
        if k >= 2:
            for a in kids:
                for b in kids:
                    if a != b:
                        C[a, b] = -s / (k - 1.0)
    return C


def B_matrix(parent):
    n = len(parent)
    anc = anc_sets(parent)
    depth = np.zeros(n, dtype=int)
    for v in sorted(range(n), key=lambda x: len(anc[x])):
        depth[v] = 0 if parent[v] < 0 else depth[parent[v]] + 1
    kof = [0] * n
    for v, p in enumerate(parent):
        if p >= 0:
            kof[p] += 1
    B = np.zeros((n, n))
    for v in range(n):
        for w in range(n):
            if v in anc[w] or w in anc[v]:
                continue
            lca = max(anc[v] & anc[w], key=lambda u: depth[u])
            B[v, w] = 1.0 / (kof[lca] - 1.0)
    return B


def D_of(kappa):
    n = kappa.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    M = H @ kappa @ H
    return float(np.sum(M * M))


def D_split(parent, s=1.0):
    A = A_matrix(parent)
    return D_of(A @ C_matrix(parent, s) @ A.T)


def D_f02(parent):
    A = A_matrix(parent)
    return D_of(A @ A.T)


def chain(n):
    return [-1] + list(range(n - 1))


def star(n):
    return [-1] + [0] * (n - 1)


def cbin(d):
    n = 2 ** (d + 1) - 1
    return [-1] + [(i - 1) // 2 for i in range(1, n)]


_MEMO: dict = {}


def _gen(m):
    if m in _MEMO:
        return _MEMO[m]
    if m == 1:
        _MEMO[1] = [()]
        return _MEMO[1]
    res = []

    def size(t):
        return 1 + sum(size(c) for c in t)

    def key(t):
        return (size(t), len(t), tuple(sorted(key(c) for c in t)))

    def parts(rem, maxsize, acc):
        if rem == 0:
            res.append(tuple(acc))
            return
        for sz in range(min(rem, maxsize), 0, -1):
            for t in _gen(sz):
                if acc and size(acc[-1]) == sz and key(acc[-1]) < key(t):
                    continue
                parts(rem - sz, sz, acc + [t])

    parts(m - 1, m - 1, [])
    _MEMO[m] = res
    return res


def shapes(n):
    out = []
    for t in _gen(n):
        parent = []

        def build(node, p):
            me = len(parent)
            parent.append(p)
            for c in node:
                build(c, me)

        build(t, -1)
        out.append(parent)
    return out


def all_rooted_labelled(n):
    import heapq
    if n == 1:
        yield [-1]
        return
    if n == 2:
        yield [-1, 0]
        yield [1, -1]
        return
    for seq in itertools.product(range(n), repeat=n - 2):
        deg = [1] * n
        for x in seq:
            deg[x] += 1
        adj = [[] for _ in range(n)]
        leaves = sorted(i for i in range(n) if deg[i] == 1)
        heapq.heapify(leaves)
        for x in seq:
            lf = heapq.heappop(leaves)
            adj[lf].append(x)
            adj[x].append(lf)
            deg[x] -= 1
            if deg[x] == 1:
                heapq.heappush(leaves, x)
        u = heapq.heappop(leaves)
        v = heapq.heappop(leaves)
        adj[u].append(v)
        adj[v].append(u)
        for root in range(n):
            par = [-2] * n
            par[root] = -1
            st = [root]
            while st:
                x = st.pop()
                for y in adj[x]:
                    if par[y] == -2:
                        par[y] = x
                        st.append(y)
            yield par


def random_tree(n, rng):
    return [-1] + [int(rng.integers(0, i)) for i in range(1, n)]


rng = np.random.default_rng(20260902)
worst = 0.0
mineig = math.inf
worst_tree = None
for n in (2, 3, 4, 5, 8, 13, 21):
    for _ in range(40):
        p = random_tree(n, rng)
        A = A_matrix(p)
        e = float(np.max(np.abs(A @ C_matrix(p) @ A.T - (A @ A.T - B_matrix(p)))))
        if e > worst:
            worst, worst_tree = e, list(p)
        mineig = min(mineig, float(np.linalg.eigvalsh(C_matrix(p)).min()))
for p in shapes(8) + shapes(9):
    A = A_matrix(p)
    worst = max(worst, float(np.max(np.abs(A @ C_matrix(p) @ A.T - (A @ A.T - B_matrix(p))))))
    mineig = min(mineig, float(np.linalg.eigvalsh(C_matrix(p)).min()))
R["ACAt_eq_AAt_minus_B_max_abs"] = worst
R["ACAt_eq_AAt_minus_B_worst_tree"] = worst_tree
R["C_min_eigenvalue"] = mineig

p2 = [-1, 0, 0]
C2 = C_matrix(p2)
R["k2_degeneracy"] = {
    "parent": p2, "C": C2.tolist(), "eigs": np.linalg.eigvalsh(C2).tolist(),
    "rank": int(np.linalg.matrix_rank(C2, tol=1e-10)),
    "note": "sibling block [[1,-1],[-1,1]] is rank 1: eta_2 = -eta_1 almost surely (degenerate label law)",
}
defs = []
for p in shapes(9):
    C = C_matrix(p)
    nint = sum(1 for v in range(len(p)) if sum(1 for w in p if w == v) >= 2)
    defs.append((len(p) - int(np.linalg.matrix_rank(C, tol=1e-9))) - nint)
R["rank_deficit_equals_num_internal_ge2_all_shapes_n9"] = bool(max(defs) == 0 and min(defs) == 0)
R["kappa_split_min_eig_all_shapes_n9"] = min(
    float(np.linalg.eigvalsh(A_matrix(p) @ C_matrix(p) @ A_matrix(p).T).min()) for p in shapes(9)
)

R["chain_equals_F02"] = {}
for n in (2, 3, 5, 8, 16, 36):
    R["chain_equals_F02"][str(n)] = {
        "D_split": D_split(chain(n)),
        "closed_F02": (n * n - 1) * (2 * n * n + 7) / 180.0,
        "D_f02_matrix": D_f02(chain(n)),
    }
R["star"] = {}
for n in (3, 4, 5, 8, 16, 36, 128):
    R["star"][str(n)] = {
        "D_split_matrix": D_split(star(n)), "closed_split": (n - 1) ** 2 / (n - 2),
        "D_f02_matrix": D_f02(star(n)), "closed_f02": n - 2 + 1 / n ** 2,
    }
R["complete_binary"] = {}
for d in range(1, 8):
    p = cbin(d)
    n = len(p)
    sz = [1] * n
    for v in reversed(range(n)):
        if p[v] >= 0:
            sz[p[v]] += sz[v]
    kof = [0] * n
    for v, q in enumerate(p):
        if q >= 0:
            kof[q] += 1
    R["complete_binary"][str(n)] = {
        "depth": d, "D_split_matrix": D_split(p),
        "closed_2n2p6nm4log": 2.0 * n * n + 6.0 * n - 4.0 * (n + 1) * math.log2(n + 1),
        "sum_T_z_sq_internal": sum((sz[v] - 1) ** 2 for v in range(n) if kof[v] >= 2),
        "D_f02_matrix": D_f02(p),
        "sqrtD_over_n_split": math.sqrt(D_split(p)) / n,
        "sqrtD_over_n_f02": math.sqrt(D_f02(p)) / n,
        "ratio_split_over_f02": D_split(p) / D_f02(p),
    }

sres = []
for n in (3, 4, 8, 16):
    for s in (0.0, 0.25, 0.5, 0.75, 1.0):
        m = D_split(star(n), s)
        c = ((1 - s) * (n - 2 + 1 / n ** 2) + s * (n - 1) ** 2 / (n - 2)
             - s * (1 - s) * (n - 1) * (n + 2) / (n ** 2 * (n - 2)))
        sres.append({"n": n, "s": s, "matrix": m, "closed": c, "abs_err": abs(m - c)})
R["star_of_s"] = {"max_abs_err": max(r["abs_err"] for r in sres), "rows": sres}

R["cayley_exhaustive"] = {}
for n in (2, 3, 4, 5, 6):
    tot_s = tot_f = 0.0
    cnt = 0
    for p in all_rooted_labelled(n):
        tot_s += D_split(p)
        tot_f += D_f02(p)
        cnt += 1
    R["cayley_exhaustive"][str(n)] = {"count": cnt, "E_D_split": tot_s / cnt, "E_D_f02": tot_f / cnt}
R["n3_exact_56_over_27"] = {
    "computed": R["cayley_exhaustive"]["3"]["E_D_split"], "target": 56 / 27,
    "abs_err": abs(R["cayley_exhaustive"]["3"]["E_D_split"] - 56 / 27),
}

sign_viol = []
bound_viol = []
cnt = 0
lo = (math.inf, None)
hi = (-math.inf, None)
for n in range(2, 13):
    for p in shapes(n):
        ds = D_split(p)
        df = D_f02(p)
        cnt += 1
        if df > 0:
            r = ds / df
            if r < lo[0]:
                lo = (r, list(p))
            if r > hi[0]:
                hi = (r, list(p))
            bnd = 2 * n / math.sqrt(df) + n * n / df
            if abs(r - 1) > bnd + 1e-9:
                bound_viol.append({"n": n, "parent": list(p), "ratio": r, "bound": bnd})
        if ds < df - 1e-9:
            sign_viol.append({"n": n, "parent": list(p), "D_split": ds, "D_f02": df, "diff": ds - df})
R["shapes_scanned_n2_12"] = cnt
R["sign_violations_count"] = len(sign_viol)
R["sign_violations"] = sign_viol[:20]
R["bound_violations_count"] = len(bound_viol)
R["bound_violations"] = bound_viol[:10]
R["min_ratio_split_over_f02"] = {"ratio": lo[0], "parent": lo[1]}
R["max_ratio_split_over_f02"] = {"ratio": hi[0], "parent": hi[1]}

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print(json.dumps({k: v for k, v in R.items() if not isinstance(v, (dict, list))}, indent=2))
print("n3 exact:", R["n3_exact_56_over_27"])
print("cayley exh:", {k: (round(v["E_D_split"], 6), round(v["E_D_f02"], 6)) for k, v in R["cayley_exhaustive"].items()})
print("sign_viol:", R["sign_violations_count"], "bound_viol:", R["bound_violations_count"])
print("min ratio:", R["min_ratio_split_over_f02"])
print("star_of_s max err:", R["star_of_s"]["max_abs_err"])
print("cbin:", {k: (round(v["D_split_matrix"], 4), round(v["closed_2n2p6nm4log"], 4), v["sum_T_z_sq_internal"],
                   round(v["D_f02_matrix"], 4), round(v["ratio_split_over_f02"], 4)) for k, v in R["complete_binary"].items()})
print("star:", {k: (round(v["D_split_matrix"], 6), round(v["closed_split"], 6)) for k, v in R["star"].items()})
