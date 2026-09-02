"""A2: composition-face kernel ||H_3 kappa_f H_3||_F^2 = 10/9.

  (a) exact recomputation for the grandparent-parent-child chain, all depths d
  (b) is "every composition face is a 3-chain" a THEOREM or the DEFINITION of a composable pair?
      -> enumerate composition faces of random uniform rooted trees under the 12.1 rule
  (c) counterexample: attachment-rule variants (11.9 item 1) that pair different branches
  (d) chain closed form (n^2-1)(2n^2+7)/180 vs ||H kappa H||^2
"""
from __future__ import annotations
import json, sys
from fractions import Fraction
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "verify" / "Q-0008" / "F-02"))
from driver_numbers import uniform_rooted_tree, tree_arrays, chain_closed_form  # noqa: E402

out = {}

def D(kappa):
    k = np.asarray(kappa, float); m = k.shape[0]
    H = np.eye(m) - np.ones((m, m)) / m
    K = H @ k @ H
    return float(np.sum(K * K))

def kappa_from_paths(paths):
    """paths[i] = set of ancestors-or-self of vertex i; kappa_ij = |path_i & path_j|."""
    m = len(paths)
    return np.array([[len(paths[i] & paths[j]) for j in range(m)] for i in range(m)], float)

# (a) grandparent-parent-child chain at depth d (grandparent depth d)
chain_by_depth = {}
for d in range(0, 13):
    anc = set(range(d + 1))                     # root path of grandparent, |.| = d+1
    mid = anc | {100}
    kid = mid | {101}
    chain_by_depth[d] = D(kappa_from_paths([anc, mid, kid]))
out["a_chain_face_D_by_depth"] = chain_by_depth
out["a_all_equal_10_over_9"] = max(abs(v - 10 / 9) for v in chain_by_depth.values())

# (b) enumerate composition faces (u->m, m->v composable pairs) of random trees; are they all chains?
rng = np.random.default_rng(4242)
non_chain = 0; total = 0; depths = set()
for _ in range(60):
    n = int(rng.integers(6, 40))
    parent = uniform_rooted_tree(n, rng)
    _, depth, _, _ = tree_arrays(parent)
    for v in range(n):
        m = parent[v]
        if m is None or m < 0:
            continue
        u = parent[m]
        if u < 0:
            continue
        total += 1
        depths.add(int(depth[u]))
        # is (u,m,v) a grandparent-parent-child chain by construction?
        if not (parent[v] == m and parent[m] == u):
            non_chain += 1
out["b_composition_faces_enumerated"] = total
out["b_non_chain_faces"] = non_chain
out["b_grandparent_depths_seen"] = sorted(depths)
out["b_note"] = ("12.1 attaches a face to each composable pair (u->m, m->v).  In a tree the fine "
                 "edges ARE parent->child, so a composable pair IS a grandparent-parent-child chain "
                 "by definition; the enumeration finds no other kind because no other kind exists.")

# (c) attachment-rule variants (11.9 item 1): what if the merged face is NOT a chain?
variants = {}
anc = set(range(1))                     # d = 0 grandparent
variants["chain (u<-m<-v)  [card]"] = kappa_from_paths([anc, anc | {10}, anc | {10, 11}])
variants["siblings (m,v children of u)"] = kappa_from_paths([anc, anc | {10}, anc | {11}])
variants["cousins (m,v in different branches, depth 2)"] = kappa_from_paths(
    [anc, anc | {10, 12}, anc | {11, 13}])
variants["u,m siblings; v child of m"] = kappa_from_paths(
    [anc | {10}, anc | {11}, anc | {11, 12}])
variants["v is child of u (skip), m child of v"] = kappa_from_paths(
    [anc, anc | {10}, anc | {10, 11}])   # same as chain
variants["disjoint roots (iid)"] = np.eye(3)
res_c = {}
for name, k in variants.items():
    d_ = D(k)
    res_c[name] = {"kappa": k.tolist(), "D": d_, "rho_vs_iid": (d_ / 2.0) ** 0.5}
out["c_attachment_variants"] = res_c

# (c2) depth-dependence of a NON-chain (sibling) face -- does universality survive?
sib_by_depth = {}
for d in range(0, 13):
    anc = set(range(d + 1))
    sib_by_depth[d] = D(kappa_from_paths([anc, anc | {100}, anc | {101}]))
out["c2_sibling_face_D_by_depth"] = sib_by_depth

# (d) chain closed form check
cf = {}
for n in (3, 4, 5, 8, 16):
    parent = [-1] + list(range(n - 1))
    A = np.zeros((n, n))
    order, _, _, _ = tree_arrays(parent)
    for v in order:
        if parent[v] >= 0:
            A[v] = A[parent[v]]
        A[v, v] = 1.0
    cf[n] = {"matrix_D": D(A @ A.T),
             "formula_(n2-1)(2n2+7)/180": (n * n - 1) * (2 * n * n + 7) / 180,
             "F02_chain_closed_form": float(chain_closed_form(n))}
out["d_chain_closed_form"] = cf
out["d_n3_value"] = {"10/9": 10 / 9, "formula": (9 - 1) * (18 + 7) / 180}

print(json.dumps(out, indent=2, ensure_ascii=False))
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
