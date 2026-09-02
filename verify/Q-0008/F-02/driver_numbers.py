"""Q-0008 F-02: prediction numbers for the centered-kernel driver D(kappa) = ||H kappa H||_F^2.

This script produces the PRE-REGISTERED numbers of card derivations/Q-0008/F-02.formula.md.
It touches trees only (no tetrads, no simplicity residuals) -- it is the formula's side of
the ledger, not the kill.  The kill is verify/Q-0008/F-02/check_modes.py.

Exact combinatorics (uniform rooted labelled = Cayley trees, n vertices):
  N_k        = C(n,k) k^{k-1} (n-k)^{n-k} / n^{n-1}         (k<n),  N_n = 1
             = expected number of vertices with subtree size k
  N_{a,c}    = N_a^{(n)} N_c^{(a)}                            (strict ancestor pairs, c<a)
  N^inc_{a,c}= C(n,a) C(n-a,c) a^{a-1} c^{c-1} m^{m+1} / n^{n-1},  m=n-a-c>=1 (ordered incomparable)
  D = sum_u (s_u - s_u^2/n)^2 + 2 sum_{u'<u} (s_{u'} - s_u s_{u'}/n)^2 + sum_{inc} s_u^2 s_{u'}^2/n^2
  tr(H kappa) = sum_u s_u (1 - s_u/n)
Validated below against brute-force enumeration of all n^{n-1} rooted labelled trees (n<=7).

Q-spine depth-b block (11.6): spine s_0..s_{b-1}; each spine vertex has Poisson(1) side children;
each side child roots a Poisson(1) GW tree; vertices at depth > b-1 are cut.  E[n_b]=b(b+1)/2.
Its driver expectation E[D/n^2] has no closed form here; it is a tree-only Monte Carlo with
stated standard error (seed 20261902 = 20260902+1000, distinct from the kill seed).
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TREE_SEED = 20260902 + 1000
QSPINE_TRIALS = 200_000
CAYLEY_GRID = (8, 16, 32, 64, 128)
QSPINE_DEPTHS = (1, 2, 3, 4, 5, 6, 7, 8)


# ---------------------------------------------------------------- generic driver from parent array
def tree_arrays(parent: list[int]):
    n = len(parent)
    children: list[list[int]] = [[] for _ in range(n)]
    root = -1
    for v, p in enumerate(parent):
        if p >= 0:
            children[p].append(v)
        else:
            root = v
    order = [root]
    i = 0
    while i < len(order):
        order.extend(children[order[i]])
        i += 1
    depth = np.zeros(n, dtype=np.int64)
    sub = np.ones(n, dtype=np.int64)
    for v in order[1:]:
        depth[v] = depth[parent[v]] + 1
    for v in reversed(order):
        if parent[v] >= 0:
            sub[parent[v]] += sub[v]
    prefix = np.zeros(n, dtype=np.int64)  # sum of subtree sizes along root path (inclusive)
    for v in order:
        prefix[v] = sub[v] + (prefix[parent[v]] if parent[v] >= 0 else 0)
    return order, depth, sub, prefix


def driver_fast(parent: list[int]) -> tuple[float, float, float]:
    """(D, tr(H kappa), W2) in O(n) from subtree sizes: D = W2' - (2/n) S_row + W2^2/n^2."""
    n = len(parent)
    _, depth, sub, prefix = tree_arrays(parent)
    s = sub.astype(np.float64)
    w2 = float(np.sum(s * s))
    w2p = float(np.sum((2.0 * depth + 1.0) * s * s))
    s_row = float(np.sum(prefix.astype(np.float64) ** 2))
    d = w2p - 2.0 * s_row / n + w2 * w2 / (n * n)
    tr_hk = float(np.sum(s * (1.0 - s / n)))
    return d, tr_hk, w2


def driver_matrix(parent: list[int]) -> float:
    """Direct definition: kappa = A A^T, A[v,u]=1 iff u ancestor-or-self of v; D = ||H kappa H||_F^2."""
    n = len(parent)
    order, _, _, _ = tree_arrays(parent)
    A = np.zeros((n, n))
    for v in order:
        if parent[v] >= 0:
            A[v] = A[parent[v]]
        A[v, v] = 1.0
    kappa = A @ A.T
    H = np.eye(n) - np.ones((n, n)) / n
    K = H @ kappa @ H
    return float(np.sum(K * K))


# ---------------------------------------------------------------- exact Cayley combinatorics
def _lg(x: int) -> float:
    return math.lgamma(x + 1)


def log_choose(n: int, k: int) -> float:
    return _lg(n) - _lg(k) - _lg(n - k)


def n_k_log(n: int, k: int) -> float:
    """log N_k^{(n)}."""
    if k == n:
        return 0.0
    return log_choose(n, k) + (k - 1) * math.log(k) + (n - k) * math.log(n - k) - (n - 1) * math.log(n)


def n_inc_log(n: int, a: int, c: int) -> float:
    m = n - a - c
    assert m >= 1
    return (
        log_choose(n, a)
        + log_choose(n - a, c)
        + (a - 1) * math.log(a)
        + (c - 1) * math.log(c)
        + (m + 1) * math.log(m)
        - (n - 1) * math.log(n)
    )


def cayley_exact(n: int) -> dict[str, float]:
    """E[D], E[tr(H kappa)], E[W2] for uniform rooted labelled trees on n vertices (log-space floats)."""
    ED = 0.0
    ETr = 0.0
    EW2 = 0.0
    Nk = [0.0] * (n + 1)
    for a in range(1, n + 1):
        Nk[a] = math.exp(n_k_log(n, a))
        ED += Nk[a] * (a - a * a / n) ** 2
        ETr += Nk[a] * a * (1.0 - a / n)
        EW2 += Nk[a] * a * a
    # strict ancestor pairs: N_a^{(n)} N_c^{(a)}, weight 2 c^2 (1 - a/n)^2
    for a in range(2, n + 1):
        fac = (1.0 - a / n) ** 2
        if fac == 0.0:
            continue
        inner = 0.0
        for c in range(1, a):
            inner += math.exp(n_k_log(a, c)) * c * c
        ED += 2.0 * Nk[a] * inner * fac
    # ordered incomparable pairs: weight a^2 c^2 / n^2
    for a in range(1, n - 1):
        for c in range(1, n - a):
            ED += math.exp(n_inc_log(n, a, c)) * (a * a * c * c) / (n * n)
    return {"E_D": ED, "E_trHk": ETr, "E_W2": EW2}


def cayley_exact_fraction(n: int) -> Fraction:
    """Same E[D] with exact rational arithmetic (validation for small n)."""
    def N(nn: int, k: int) -> Fraction:
        if k == nn:
            return Fraction(1)
        return Fraction(math.comb(nn, k) * k ** (k - 1) * (nn - k) ** (nn - k), nn ** (nn - 1))

    ED = Fraction(0)
    for a in range(1, n + 1):
        ED += N(n, a) * Fraction(a * n - a * a, n) ** 2
    for a in range(2, n + 1):
        fac = Fraction(n - a, n) ** 2
        for c in range(1, a):
            ED += 2 * N(n, a) * N(a, c) * c * c * fac
    for a in range(1, n - 1):
        for c in range(1, n - a):
            m = n - a - c
            cnt = Fraction(math.comb(n, a) * math.comb(n - a, c) * a ** (a - 1) * c ** (c - 1) * m ** (m + 1), n ** (n - 1))
            ED += cnt * Fraction(a * a * c * c, n * n)
    return ED


def brute_force_cayley(n: int) -> tuple[float, float]:
    """Average D over ALL n^{n-1} rooted labelled trees (Pruefer x root)."""
    if n == 1:
        return 0.0, 0.0
    total_d = 0.0
    total_tr = 0.0
    count = 0
    for seq in itertools.product(range(n), repeat=n - 2):
        degree = [1] * n
        for s in seq:
            degree[s] += 1
        adjacency: list[list[int]] = [[] for _ in range(n)]
        leaves = sorted(i for i in range(n) if degree[i] == 1)
        import heapq

        heapq.heapify(leaves)
        for s in seq:
            leaf = heapq.heappop(leaves)
            adjacency[leaf].append(s)
            adjacency[s].append(leaf)
            degree[s] -= 1
            if degree[s] == 1:
                heapq.heappush(leaves, s)
        u = heapq.heappop(leaves)
        v = heapq.heappop(leaves)
        adjacency[u].append(v)
        adjacency[v].append(u)
        for root in range(n):
            parent = [-2] * n
            parent[root] = -1
            stack = [root]
            while stack:
                x = stack.pop()
                for y in adjacency[x]:
                    if parent[y] == -2:
                        parent[y] = x
                        stack.append(y)
            d, tr, _ = driver_fast(parent)
            total_d += d
            total_tr += tr
            count += 1
    return total_d / count, total_tr / count


# ---------------------------------------------------------------- deterministic families
def chain_parent(n: int) -> list[int]:
    return [-1] + list(range(n - 1))


def star_parent(n: int) -> list[int]:
    return [-1] + [0] * (n - 1)


def binary_parent(n: int) -> list[int]:
    return [-1] + [(i - 1) // 2 for i in range(1, n)]


def chain_closed_form(n: int) -> Fraction:
    """D_chain(n) from kappa_ij = min(i,j): sum min^2 - (2/n) sum_i row_i^2 + W2^2/n^2 (all closed sums)."""
    n = Fraction(n)
    S2 = n * (n + 1) * (2 * n + 1) / 6
    S3 = n * n * (n + 1) ** 2 / 4
    S4 = n * (n + 1) * (2 * n + 1) * (3 * n * n + 3 * n - 1) / 30
    sum_min_sq = n * (n + 1) * (n * n + n + 1) / 6
    m = 2 * n + 1
    sum_row_sq = (m * m * S2 - 2 * m * S3 + S4) / 4
    W2 = S2
    return sum_min_sq - 2 * sum_row_sq / n + W2 * W2 / (n * n)


# ---------------------------------------------------------------- Q-spine block sampler (declared model)
def qspine_block(b: int, rng: np.random.Generator) -> list[int]:
    """Depth-b Q-spine block: spine of b vertices (depths 0..b-1), Poisson(1) side children per spine
    vertex, each side child a Poisson(1) GW tree, everything at depth > b-1 cut.  Returns parent array."""
    parent: list[int] = []
    depth: list[int] = []
    # spine
    for k in range(b):
        parent.append(k - 1 if k > 0 else -1)
        depth.append(k)
    # side branches: BFS over frontier of (vertex, depth); spine vertices spawn Poisson(1) side children
    frontier: list[int] = []
    for k in range(b):
        if k + 1 <= b - 1:
            m = int(rng.poisson(1.0))
            for _ in range(m):
                parent.append(k)
                depth.append(k + 1)
                frontier.append(len(parent) - 1)
    i = 0
    while i < len(frontier):
        v = frontier[i]
        i += 1
        if depth[v] + 1 <= b - 1:
            m = int(rng.poisson(1.0))
            for _ in range(m):
                parent.append(v)
                depth.append(depth[v] + 1)
                frontier.append(len(parent) - 1)
    return parent


def uniform_rooted_tree(n: int, rng: np.random.Generator) -> list[int]:
    """Uniform rooted labelled tree via Pruefer (same construction as the kill script)."""
    import heapq

    if n == 1:
        return [-1]
    if n == 2:
        root = int(rng.integers(0, 2))
        return [-1, 0] if root == 0 else [1, -1]
    seq = rng.integers(0, n, size=n - 2)
    degree = np.ones(n, dtype=int)
    for s in seq:
        degree[s] += 1
    adjacency: list[list[int]] = [[] for _ in range(n)]
    leaves = [i for i in range(n) if degree[i] == 1]
    heapq.heapify(leaves)
    for s in seq:
        leaf = heapq.heappop(leaves)
        adjacency[leaf].append(int(s))
        adjacency[int(s)].append(leaf)
        degree[s] -= 1
        if degree[s] == 1:
            heapq.heappush(leaves, int(s))
    u = heapq.heappop(leaves)
    v = heapq.heappop(leaves)
    adjacency[u].append(v)
    adjacency[v].append(u)
    root = int(rng.integers(0, n))
    parent = [-2] * n
    parent[root] = -1
    stack = [root]
    while stack:
        x = stack.pop()
        for y in adjacency[x]:
            if parent[y] == -2:
                parent[y] = x
                stack.append(y)
    return parent


def slope(xs, ys) -> float:
    return float(np.polyfit(np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float)), 1)[0])


def main() -> int:
    out: dict = {"card": "F-02", "question": "Q-0008", "tree_seed": TREE_SEED}

    # --- (0) validations: fast driver == matrix driver; exact == brute force; log == Fraction
    rng = np.random.default_rng(1)
    worst = 0.0
    for n in (2, 3, 5, 9, 17):
        for _ in range(20):
            p = uniform_rooted_tree(n, rng)
            worst = max(worst, abs(driver_fast(p)[0] - driver_matrix(p)) / (1 + driver_matrix(p)))
    for b in (2, 4, 6):
        for _ in range(20):
            p = qspine_block(b, rng)
            worst = max(worst, abs(driver_fast(p)[0] - driver_matrix(p)) / (1 + driver_matrix(p)))
    out["check_fast_vs_matrix_max_rel_err"] = worst
    bf = {}
    for n in (2, 3, 4, 5, 6, 7):
        d_bf, tr_bf = brute_force_cayley(n)
        ex = cayley_exact(n)
        fr = float(cayley_exact_fraction(n))
        bf[n] = {"brute_D": d_bf, "exact_D_log": ex["E_D"], "exact_D_frac": fr, "brute_tr": tr_bf, "exact_tr": ex["E_trHk"]}
    out["check_cayley_exact_vs_bruteforce"] = bf
    out["check_cayley_bruteforce_max_rel_err"] = max(
        max(abs(v["brute_D"] - v["exact_D_frac"]) / v["exact_D_frac"], abs(v["brute_tr"] - v["exact_tr"]) / v["exact_tr"])
        for v in bf.values()
    )
    out["check_chain_closed_form_vs_matrix"] = {
        n: {"closed": float(chain_closed_form(n)), "matrix": driver_matrix(chain_parent(n))} for n in (2, 3, 8, 16)
    }
    out["check_star_closed_form_vs_matrix"] = {
        n: {"closed": n - 2 + 1 / n**2, "matrix": driver_matrix(star_parent(n))} for n in (2, 3, 8, 16)
    }

    # --- (1) Cayley grid: exact E[D], gamma_her, ratios
    cay = {}
    for n in CAYLEY_GRID + (36, 256):
        ex = cayley_exact(n)
        cay[n] = {
            **ex,
            "rms_her_over_iid": math.sqrt(ex["E_D"] / (n - 1)),
            "sqrtD_over_n": math.sqrt(ex["E_D"]) / n,
            "mix_excess_X": 2.0 * ex["E_trHk"] / math.sqrt((n - 1) * ex["E_D"]),
        }
    out["cayley"] = {str(k): v for k, v in cay.items()}
    out["gamma_her_cayley_grid"] = slope(CAYLEY_GRID, [cay[n]["sqrtD_over_n"] for n in CAYLEY_GRID])
    out["gamma_her_cayley_local"] = {
        f"{a}->{b}": math.log(cay[b]["sqrtD_over_n"] / cay[a]["sqrtD_over_n"]) / math.log(b / a)
        for a, b in zip(CAYLEY_GRID[:-1], CAYLEY_GRID[1:])
    }
    out["gamma_iid_grid_exact"] = slope(CAYLEY_GRID, [math.sqrt(n - 1) / n for n in CAYLEY_GRID])

    # --- (2) deterministic families on the same grid (discriminating baselines)
    fam = {}
    for name, fn in (("chain", chain_parent), ("star", star_parent), ("balanced_binary", binary_parent)):
        vals = {n: driver_matrix(fn(n)) for n in CAYLEY_GRID + (36,)}
        fam[name] = {
            "D": {str(n): v for n, v in vals.items()},
            "gamma": slope(CAYLEY_GRID, [math.sqrt(vals[n]) / n for n in CAYLEY_GRID]),
            "rms_over_iid_128": math.sqrt(vals[128] / 127),
            "rms_over_iid_36": math.sqrt(vals[36] / 35),
        }
    out["families"] = fam

    # --- (3) Q-spine depth-b block: tree-only Monte Carlo
    rng = np.random.default_rng(TREE_SEED)
    qs = {}
    for b in QSPINE_DEPTHS:
        dn2 = np.empty(QSPINE_TRIALS)
        dd = np.empty(QSPINE_TRIALS)
        nn = np.empty(QSPINE_TRIALS)
        for t in range(QSPINE_TRIALS):
            p = qspine_block(b, rng)
            n = len(p)
            d, _, _ = driver_fast(p)
            dn2[t] = d / (n * n)
            dd[t] = d
            nn[t] = n
        E_dn2 = float(dn2.mean())
        se = float(dn2.std(ddof=1) / math.sqrt(QSPINE_TRIALS))
        n_star = b * (b + 1) // 2
        qs[b] = {
            "E_n": float(nn.mean()),
            "E_n_exact": n_star,
            "E_D_over_n2": E_dn2,
            "se_E_D_over_n2": se,
            "cv_D_over_n2": float(dn2.std(ddof=1) / E_dn2) if E_dn2 > 0 else None,
            "rms_pred_over_eps_star": math.sqrt(E_dn2),
            "alt_meanfield_sqrt(ED)/En": math.sqrt(float(dd.mean())) / float(nn.mean()),
            "ratio_to_iid_at_nstar": math.sqrt(E_dn2) * n_star / math.sqrt(n_star - 1) if n_star > 1 else None,
            "max_n": int(nn.max()),
        }
    out["qspine"] = {str(k): v for k, v in qs.items()}
    bs = [b for b in QSPINE_DEPTHS if b >= 2]
    out["qspine_slope_vs_b"] = slope(bs, [qs[b]["rms_pred_over_eps_star"] for b in bs])
    out["qspine_slope_vs_En"] = slope([qs[b]["E_n_exact"] for b in bs], [qs[b]["rms_pred_over_eps_star"] for b in bs])
    out["qspine_slope_vs_En_meanfield_alt"] = slope(
        [qs[b]["E_n_exact"] for b in bs], [qs[b]["alt_meanfield_sqrt(ED)/En"] for b in bs]
    )
    out["qspine_shape_factor_b8_vs_cayley36"] = qs[8]["ratio_to_iid_at_nstar"] / cay[36]["rms_her_over_iid"]
    out["qspine_b8_vs_chain36"] = qs[8]["ratio_to_iid_at_nstar"] / fam["chain"]["rms_over_iid_36"]

    # --- (4) defect dilution (coherent kernel with p = 1/n): driver 4 n^2 p^2 (1-p)^2
    grid = (4, 8, 16, 32, 64)
    defect = {n: (n - 1) / n**2 for n in grid}
    out["defect"] = {
        "eps_over_const": {str(n): v for n, v in defect.items()},
        "slope_grid": slope(grid, [defect[n] for n in grid]),
        "ratio_64_over_8": defect[64] / defect[8],
        "alt_clt_ratio_64_over_8": (math.sqrt(63) / 64) / (math.sqrt(7) / 8),
        "alt_13_5_constant": 1.0,
    }

    (HERE / "predictions.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}, indent=2))
    print("cayley:", {n: round(cay[n]["rms_her_over_iid"], 4) for n in cay})
    print("X(32) =", round(cay[32]["mix_excess_X"], 4), " E_trHk(32)=", round(cay[32]["E_trHk"], 3), " E_D(32)=", round(cay[32]["E_D"], 2))
    print("families:", {k: round(v["gamma"], 4) for k, v in fam.items()})
    print("qspine:", {b: (round(qs[b]["E_n"], 3), round(qs[b]["rms_pred_over_eps_star"], 5), round(qs[b]["ratio_to_iid_at_nstar"] or 0, 4)) for b in qs})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
