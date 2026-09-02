"""Adversary b2 (content): the card's stated law  gamma_her = 1/d_tree,  with d_tree DEFINED by
'depth ~ n^{1/d_tree}' (card symbols line), is FALSE as a general law.

Counterexample family 'star of chains': root + k disjoint chains of length k.  n = k^2+1,
depth = k = (n-1)^{1/2}  =>  d_tree = 2  =>  card predicts gamma_her = 1/2.
Exact driver:  D = W2' - (2/n) S_row + W2^2/n^2  with  W2' = sum_u (2 depth_u + 1) s_u^2 ~ k^5/6,
so sqrt(D)/n ~ n^{1/4}/sqrt(6):  gamma_her = 1/4, not 1/2.

Reason: the card's premise is 'kappa entries ~ depth', but kappa_{vw} = #common ancestors, and in
this family a TYPICAL pair sits on two different chains and shares only the root (kappa = 1).
Depth alone does not fix gamma; the typical common-ancestor count does.  Both a 'd_tree = 2, gamma
= 1/2' family (uniform Cayley, caterpillar) and a 'd_tree = 2, gamma = 1/4' family exist.
"""
import math
import numpy as np

def driver_matrix(parent):
    n = len(parent)
    A = np.zeros((n, n))
    for v in range(n):
        u = v
        while u >= 0:
            A[v, u] = 1.0
            u = parent[u]
    k = A @ A.T
    H = np.eye(n) - np.ones((n, n)) / n
    K = H @ k @ H
    return float(np.sum(K * K))

def driver_fast(parent):
    n = len(parent)
    ch = [[] for _ in range(n)]
    root = -1
    for v, p in enumerate(parent):
        (ch[p].append(v) if p >= 0 else None)
        if p < 0:
            root = v
    order = [root]
    i = 0
    while i < len(order):
        order.extend(ch[order[i]]); i += 1
    depth = np.zeros(n, dtype=np.int64); sub = np.ones(n, dtype=np.int64); pre = np.zeros(n, dtype=np.int64)
    for v in order[1:]:
        depth[v] = depth[parent[v]] + 1
    for v in reversed(order):
        if parent[v] >= 0:
            sub[parent[v]] += sub[v]
    for v in order:
        pre[v] = sub[v] + (pre[parent[v]] if parent[v] >= 0 else 0)
    s = sub.astype(float)
    w2 = float(np.sum(s * s))
    w2p = float(np.sum((2 * depth + 1) * s * s))
    srow = float(np.sum(pre.astype(float) ** 2))
    return w2p - 2 * srow / n + w2 * w2 / (n * n), int(depth.max())

def star_of_chains(k):
    """root 0, then k chains of length k."""
    parent = [-1]
    for c in range(k):
        prev = 0
        for j in range(k):
            parent.append(prev)
            prev = len(parent) - 1
    return parent

def caterpillar(k):
    """spine of length k, each spine vertex carries (k-1) leaves.  n = k^2, depth = k."""
    parent = [-1]
    spine = [0]
    for j in range(1, k):
        parent.append(spine[-1]); spine.append(len(parent) - 1)
    for v in spine:
        for _ in range(k - 1):
            parent.append(v)
    return parent

def chain(n): return [-1] + list(range(n - 1))
def binary(n): return [-1] + [(i - 1) // 2 for i in range(1, n)]

def uniform_cayley(n, rng):
    import heapq
    if n <= 2:
        return [-1] + [0] * (n - 1)
    seq = rng.integers(0, n, size=n - 2)
    deg = np.ones(n, dtype=int)
    for s in seq: deg[s] += 1
    adj = [[] for _ in range(n)]
    leaves = [i for i in range(n) if deg[i] == 1]; heapq.heapify(leaves)
    for s in seq:
        lf = heapq.heappop(leaves); adj[lf].append(int(s)); adj[int(s)].append(lf)
        deg[s] -= 1
        if deg[s] == 1: heapq.heappush(leaves, int(s))
    u = heapq.heappop(leaves); v = heapq.heappop(leaves); adj[u].append(v); adj[v].append(u)
    root = int(rng.integers(0, n)); parent = [-2] * n; parent[root] = -1; st = [root]
    while st:
        x = st.pop()
        for y in adj[x]:
            if parent[y] == -2: parent[y] = x; st.append(y)
    return parent

def slope(xs, ys):
    return float(np.polyfit(np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float)), 1)[0])

print("== consistency of the O(n) driver with the dense matrix (independent re-derivation) ==")
worst = 0.0
for k in (2, 3, 4):
    for fam in (star_of_chains(k), caterpillar(k)):
        d1, _ = driver_fast(fam); d2 = driver_matrix(fam)
        worst = max(worst, abs(d1 - d2) / (1 + d2))
print("   max rel err =", worst)

print("\n== family table:  gamma_her measured  vs  card law 1/d_tree with depth ~ n^{1/d_tree} ==")
print(f"{'family':22s} {'sizes n':28s} {'depth exponent 1/d_tree':24s} {'gamma_her':>10s}  {'card 1/d_tree':>13s}")

rows = []
ks = (6, 9, 13, 19, 28, 40)
ns, ds, dep = [], [], []
for k in ks:
    p = star_of_chains(k); d, dm = driver_fast(p)
    ns.append(len(p)); ds.append(d); dep.append(dm)
g = slope(ns, [math.sqrt(d) / n for d, n in zip(ds, ns)])
dexp = slope(ns, dep)
rows.append(("star of chains", ns, dexp, g, 1.0 / (1.0 / dexp)))
print(f"{'star of chains':22s} {str(ns):28s} {dexp:24.4f} {g:10.4f}  {dexp:13.4f}")

ns, ds, dep = [], [], []
for k in ks:
    p = caterpillar(k); d, dm = driver_fast(p)
    ns.append(len(p)); ds.append(d); dep.append(dm)
g = slope(ns, [math.sqrt(d) / n for d, n in zip(ds, ns)])
dexp = slope(ns, dep)
print(f"{'caterpillar':22s} {str(ns):28s} {dexp:24.4f} {g:10.4f}  {dexp:13.4f}")

S = (16, 32, 64, 128, 256, 512, 1024)
vals = [driver_fast(chain(n))[0] for n in S]
print(f"{'chain':22s} {str(S):28s} {1.0:24.4f} {slope(S,[math.sqrt(v)/n for v,n in zip(vals,S)]):10.4f}  {1.0:13.4f}")
vals = [driver_fast(binary(n))[0] for n in S]
dexp = slope(S, [driver_fast(binary(n))[1] for n in S])
print(f"{'balanced binary':22s} {str(S):28s} {dexp:24.4f} {slope(S,[math.sqrt(v)/n for v,n in zip(vals,S)]):10.4f}  {dexp:13.4f} (log depth: 1/d_tree->0)")
rng = np.random.default_rng(20260902)
M = 400
vals = [float(np.mean([driver_fast(uniform_cayley(n, rng))[0] for _ in range(M)])) for n in S]
deps = []
rng = np.random.default_rng(20260902)
for n in S:
    deps.append(float(np.mean([driver_fast(uniform_cayley(n, rng))[1] for _ in range(60)])))
print(f"{'uniform Cayley (MC)':22s} {str(S):28s} {slope(S,deps):24.4f} {slope(S,[math.sqrt(v)/n for v,n in zip(vals,S)]):10.4f}  {slope(S,deps):13.4f}")

print("\n== asymptotics of the counterexample (exact driver, large k) ==")
for k in (40, 60, 90, 135, 200):
    p = star_of_chains(k); d, dm = driver_fast(p); n = len(p)
    print(f"   k={k:<4} n={n:<6} depth={dm:<5} sqrt(D)/n = {math.sqrt(d)/n:10.5f}   n^0.25/sqrt(6) = {n**0.25/math.sqrt(6):10.5f}")
kk = (60, 90, 135, 200)
vv = [math.sqrt(driver_fast(star_of_chains(k))[0]) / (k * k + 1) for k in kk]
nn = [k * k + 1 for k in kk]
print("   local gamma over the last decade =", round(slope(nn, vv), 4), " (card law would require 0.5)")
