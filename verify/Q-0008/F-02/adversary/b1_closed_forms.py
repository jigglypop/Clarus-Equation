"""Adversary b1 (dimension): independent recomputation of F-02's four closed forms.

kappa_iid = I;  kappa_coh = sum_s 1_s 1_s^T;  kappa_chain = A A^T (path);  kappa_star = A A^T (star).
D = ||H kappa H||_F^2, H = I - J/n.  No code is reused from driver_numbers.py: kernels are built
from their definitions and D is computed by dense linear algebra.  Cayley E[D] is checked by an
INDEPENDENT exhaustive enumeration of all rooted labelled trees (all parent maps, tree filter).
"""
import itertools, math
import numpy as np

def Dof(kappa):
    n = kappa.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K = H @ kappa @ H
    return float(np.sum(K * K))

def anc_kernel(parent):
    n = len(parent)
    A = np.zeros((n, n))
    for v in range(n):
        u = v
        while u >= 0:
            A[v, u] = 1.0
            u = parent[u]
    return A @ A.T

print("== 1. ||H I H||_F^2 = n-1 ==")
for n in range(1, 10):
    print(f"   n={n}: direct={Dof(np.eye(n)):.10f}  closed={n-1}")

print("\n== 2. coherent two species: D = 4 n^2 p^2 (1-p)^2,  p = n_B/n ==")
worst = 0.0
for n in range(1, 10):
    for nB in range(0, n + 1):
        k = np.zeros((n, n))
        idx = np.arange(n)
        sB = idx < nB
        k[np.ix_(sB, sB)] = 1.0
        k[np.ix_(~sB, ~sB)] = 1.0
        p = nB / n
        closed = 4 * n**2 * p**2 * (1 - p) ** 2
        worst = max(worst, abs(Dof(k) - closed))
print(f"   max |direct - closed| over n<=9, all n_B = {worst:.3e}")
print("   (single species n_B in {0,n} -> D = 0, matches 13.3)")

print("\n== 3. chain: D = (n^2-1)(2n^2+7)/180 ==")
for n in range(1, 12):
    parent = [-1] + list(range(n - 1))
    closed = (n**2 - 1) * (2 * n**2 + 7) / 180
    print(f"   n={n:<3} direct={Dof(anc_kernel(parent)):16.9f}  closed={closed:16.9f}  diff={Dof(anc_kernel(parent))-closed:.2e}")

print("\n== 4. star: D = n - 2 + 1/n^2 ==")
for n in range(1, 12):
    parent = [-1] + [0] * (n - 1)
    closed = n - 2 + 1 / n**2
    print(f"   n={n:<3} direct={Dof(anc_kernel(parent)):16.9f}  closed={closed:16.9f}  diff={Dof(anc_kernel(parent))-closed:.2e}")
print("   NOTE n=1: closed form gives 1-2+1 = 0 OK; n=2 gives 0.25 (star=chain at n=2) OK")

print("\n== 5. n=1 sanity: H = 0 so every kernel gives D = 0 ==")
print("   H(1) =", np.eye(1) - np.ones((1, 1)), " D(any kappa) =", Dof(np.array([[7.3]])))

print("\n== 6. INDEPENDENT exhaustive Cayley E[D], E[tr(H kappa)] (all rooted labelled trees) ==")
for n in range(2, 8):
    tot_d = tot_tr = 0.0
    cnt = 0
    verts = list(range(n))
    for root in verts:
        others = [v for v in verts if v != root]
        for assign in itertools.product(verts, repeat=n - 1):
            parent = [-1] * n
            ok = True
            for v, p in zip(others, assign):
                if p == v:
                    ok = False
                    break
                parent[v] = p
            if not ok:
                continue
            # tree test: every vertex reaches root
            good = True
            for v in others:
                u, steps = v, 0
                while u != root and steps <= n:
                    u = parent[u]
                    steps += 1
                if u != root:
                    good = False
                    break
            if not good:
                continue
            k = anc_kernel(parent)
            H = np.eye(n) - np.ones((n, n)) / n
            tot_d += Dof(k)
            tot_tr += float(np.trace(H @ k))
            cnt += 1
    print(f"   n={n}: #rooted labelled trees={cnt} (n^(n-1)={n**(n-1)})  E[D]={tot_d/cnt:.10f}  E[tr(H k)]={tot_tr/cnt:.10f}")
