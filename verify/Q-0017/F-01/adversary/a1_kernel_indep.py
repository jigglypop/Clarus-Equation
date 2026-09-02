"""a1: independent re-implementation of kappa_q = A_q A_q^T and cross-check vs the card driver.

Independent route: A_q = (I - M)^{-1} where M[v,p] = 1/indeg(v) for each parent p of v.
The Neumann series (I-M)^{-1} = sum_k M^k is exactly the path-sum definition
(A_q)_{vu} = sum_{paths u->v} prod_{w != u} 1/indeg(w) (nilpotent on a DAG), so this is a
definitionally distinct computation from the card's level-sweep recursion.
Also: chain / star / n=1 / q->0 / q->1 limits and comparison to F-02 exact Cayley table.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "verify" / "Q-0008" / "F-02"))
sys.path.insert(0, str(ROOT / "verify" / "Q-0017" / "F-01"))
from driver_numbers import cayley_exact, tree_arrays, uniform_rooted_tree  # noqa
from predict_merge_gamma import kernel_D, merge_draws, layered_parent, layered_n  # noqa

OUT = {}


def A_indep(parent, second, merged):
    """A = (I - M)^{-1}; merged[v] True -> two parents with weight 1/2 each."""
    n = len(parent)
    M = np.zeros((n, n))
    for v in range(n):
        p = parent[v]
        if p < 0:
            continue
        if merged[v]:
            M[v, p] += 0.5
            M[v, second[v]] += 0.5
        else:
            M[v, p] += 1.0
    return np.linalg.inv(np.eye(n) - M)


def D_indep(A):
    n = A.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K = H @ (A @ A.T) @ H
    return float(np.sum(K * K)), float(np.trace(K))


def A_pathsum(parent, second, merged, max_len=None):
    """Brute-force path enumeration (third route) for tiny n."""
    n = len(parent)
    M = np.zeros((n, n))
    for v in range(n):
        p = parent[v]
        if p < 0:
            continue
        if merged[v]:
            M[v, p] += 0.5
            M[v, second[v]] += 0.5
        else:
            M[v, p] += 1.0
    A = np.eye(n)
    P = np.eye(n)
    for _ in range(n + 2):
        P = P @ M
        A = A + P
    return A


def main():
    rng = np.random.default_rng(20260902)
    # ---- (a) small-DAG cross check against the card's kernel_D
    diffs = []
    rowsum_err = []
    for trial in range(60):
        n = int(rng.integers(3, 9))
        parent = uniform_rooted_tree(n, rng)
        level_list, widths, depth, u, r = merge_draws(parent, rng)
        for q in (0.0, 0.25, 0.5, 1.0):
            merged = [(u[v] < q) and (r[v] >= 0) for v in range(n)]
            A = A_indep(parent, r, merged)
            Ap = A_pathsum(parent, r, merged)
            D1, T1 = D_indep(A)
            D2, T2 = kernel_D(parent, level_list, u, r, q)
            diffs.append((abs(D1 - D2) / max(D2, 1e-30), abs(T1 - T2) / max(abs(T2), 1e-30),
                          float(np.max(np.abs(A - Ap)))))
            # mass conservation: row sum of A must be depth_v + 1
            rowsum_err.append(float(np.max(np.abs(A.sum(axis=1) - (depth + 1)))))
    diffs = np.array(diffs)
    OUT["small_dag_cross_check"] = {
        "n_cases": int(diffs.shape[0]),
        "max_rel_err_D": float(diffs[:, 0].max()),
        "max_rel_err_trHkH": float(diffs[:, 1].max()),
        "max_abs_err_A_vs_pathsum": float(diffs[:, 2].max()),
        "max_abs_err_rowsum_minus_depth_plus_1": float(max(rowsum_err)),
    }

    # ---- (b) q = 0 exact Cayley table (F-02) reproduced by the independent kernel
    q0 = {}
    for n in (4, 5, 6, 7):
        # exhaustive over all rooted labelled trees is expensive; use big MC instead for n<=7
        tot = 0.0
        T = 40000
        g = np.random.default_rng(4242 + n)
        for _ in range(T):
            parent = uniform_rooted_tree(n, g)
            merged = [False] * n
            A = A_indep(parent, [-1] * n, merged)
            tot += D_indep(A)[0]
        q0[str(n)] = {"mc_E_D": tot / T, "exact_E_D": cayley_exact(n)["E_D"],
                      "rel": tot / T / cayley_exact(n)["E_D"] - 1.0}
    OUT["q0_vs_F02_exact"] = q0

    # ---- (c) chain and star closed forms (merge must be inert)
    chain = {}
    for n in (4, 8, 12):
        parent = [-1] + list(range(n - 1))
        level_list, widths, depth, u, r = merge_draws(parent, rng)
        vals = {str(q): kernel_D(parent, level_list, u, r, q)[0] for q in (0.0, 0.5, 1.0)}
        chain[str(n)] = {"driver_D": vals, "closed_form": (n ** 2 - 1) * (2 * n ** 2 + 7) / 180,
                         "n_partners_available": int(np.sum(r >= 0))}
    OUT["chain_merge_inert"] = chain
    star = {}
    for n in (4, 8, 16):
        parent = [-1] + [0] * (n - 1)
        level_list, widths, depth, u, r = merge_draws(parent, rng)
        vals = {str(q): kernel_D(parent, level_list, u, r, q)[0] for q in (0.0, 0.5, 1.0)}
        star[str(n)] = {"driver_D": vals, "closed_form": n - 2 + 1.0 / n ** 2,
                        "n_partners_available": int(np.sum(r >= 0))}
    OUT["star_merge_inert"] = star

    # ---- (d) n = 1
    parent = [-1]
    level_list, widths, depth, u, r = merge_draws(parent, rng)
    OUT["n_eq_1"] = {"D": kernel_D(parent, level_list, u, r, 1.0)[0]}

    # ---- (e) two-level "wide first generation" star-like: W_0=1 so depth-1 cells never merge,
    #          but depth-2 cells do.  Check the driver honours that.
    par = [-1, 0, 0, 0, 1, 1, 2, 2, 3, 3]
    level_list, widths, depth, u, r = merge_draws(par, np.random.default_rng(7))
    OUT["W0_is_1_blocks_depth1_merge"] = {
        "depth": depth.tolist(),
        "second_parent": r.tolist(),
        "depth1_all_minus1": bool(all(r[v] == -1 for v in range(len(par)) if depth[v] == 1)),
    }
    print(json.dumps(OUT, indent=1, ensure_ascii=False))
    (HERE / "a1_kernel_indep.json").write_text(json.dumps(OUT, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
