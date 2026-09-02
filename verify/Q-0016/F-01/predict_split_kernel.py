"""Q-0016 F-01: prediction numbers for the split-conserving inheritance kernel (competitor of Q-0008 F-02).

Hypothesis A (card F-01, s = 1 = complete conservation, declared, not fitted):
  when a parent z splits into k >= 2 children, the children's label increments eta_c satisfy
      sum_{c in ch(z)} eta_c = 0      (children's labels average to the parent's label: tree martingale)
  with UNCHANGED unit marginal variance, i.e. eta_c = sqrt(k/(k-1)) (xi_c - mean_{ch(z)} xi),  xi i.i.d. N(0,1).
  Only children (k = 1) and the root are unconstrained (eta = xi), exactly as in F-02.
  Increment covariance C (n x n): C_uu = 1; C_uu' = -1/(k_z - 1) for u != u' children of the same z; else 0.
  Label covariance kernel:  kappa_split = A C A^T,  A_vu = [u <= v]  (ancestor-or-self indicator).
  Equivalent closed form:   kappa_split = A A^T - B,   B_vw = [v, w incomparable] / (k_{lca(v,w)} - 1).
  Driver:                   D_split = || H kappa_split H ||_F^2,  H = I - J/n.
  Law (shared eps_star = sqrt(10) delta^2, E-20260902-018 is kappa-independent):
      eps_block^2 = eps_star^2 D_split / n^2.

This script touches trees only (no tetrads).  It is the formula's side of the ledger, not the kill.
Q-spine depth-b blocks reuse EXACTLY the F-02 tree stream (seed 20261902, depths 1..8, 200000 trees per
depth, one rng), so the F-02 table E[D/n^2] is reproduced bit-for-bit as a protocol check and the two
kernels are compared on the same trees (paired).  The rng state is checkpointed per depth so the run can
be split into several invocations (--depths).

Usage:
  python predict_split_kernel.py --part fast            # validations, closed forms, families, Cayley MC
  python predict_split_kernel.py --part qspine --depths 1,2,3,4,5   # then --depths 6,7 and --depths 8
  python predict_split_kernel.py --part assemble        # merge partial json into predictions.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "verify" / "Q-0008" / "F-02"))
from driver_numbers import (  # noqa: E402
    binary_parent,
    cayley_exact,
    chain_closed_form,
    chain_parent,
    driver_fast,
    driver_matrix,
    qspine_block,
    star_parent,
    tree_arrays,
    uniform_rooted_tree,
)

TREE_SEED = 20260902 + 1000          # same tree-only seed as F-02 driver_numbers.py
S_CONS = 1.0                          # complete conservation (declared; s is NOT a free parameter)
QSPINE_TRIALS = 200_000               # per depth, same as F-02
QSPINE_DEPTHS = (1, 2, 3, 4, 5, 6, 7, 8)
CAYLEY_GRID = (8, 16, 32, 64, 128)
CAYLEY_MC = {8: 40000, 16: 40000, 32: 40000, 64: 20000, 128: 10000, 36: 20000}
STATE_FILE = HERE / "qspine_rng_state.json"
PARTIAL_FILE = HERE / "predictions_qspine_partial.json"
FAST_FILE = HERE / "predictions_fast.json"
OUT_FILE = HERE / "predictions.json"


# ---------------------------------------------------------------- split kernel (machine-readable definition)
def children_of(parent: list[int]) -> list[list[int]]:
    ch: list[list[int]] = [[] for _ in parent]
    for v, p in enumerate(parent):
        if p >= 0:
            ch[p].append(v)
    return ch


def split_C(parent: list[int], s: float = S_CONS) -> np.ndarray:
    """Increment covariance: unit diagonal, -s/(k-1) between the k children of one parent (k >= 2)."""
    n = len(parent)
    C = np.eye(n)
    for ch in children_of(parent):
        k = len(ch)
        if k >= 2:
            idx = np.asarray(ch)
            C[np.ix_(idx, idx)] -= s / (k - 1)
            C[idx, idx] += s / (k - 1)
    return C


def ancestor_matrix(parent: list[int]) -> np.ndarray:
    n = len(parent)
    order, _, _, _ = tree_arrays(parent)
    A = np.zeros((n, n))
    for v in order:
        if parent[v] >= 0:
            A[v] = A[parent[v]]
        A[v, v] = 1.0
    return A


def kappa_split(parent: list[int], s: float = S_CONS) -> np.ndarray:
    A = ancestor_matrix(parent)
    return A @ split_C(parent, s) @ A.T


def driver_split(parent: list[int], s: float = S_CONS) -> float:
    """D_split = ||H A C A^T H||_F^2 via centered columns HA (column-mean subtraction)."""
    A = ancestor_matrix(parent)
    HA = A - A.mean(axis=0, keepdims=True)
    K = HA @ split_C(parent, s) @ HA.T
    return float(np.sum(K * K))


def driver_split_from_A(A: np.ndarray, C: np.ndarray) -> float:
    HA = A - A.mean(axis=0, keepdims=True)
    K = HA @ C @ HA.T
    return float(np.sum(K * K))


def kappa_split_via_B(parent: list[int]) -> np.ndarray:
    """Independent construction: kappa - B, B_vw = [incomparable]/(k_lca - 1)."""
    n = len(parent)
    A = ancestor_matrix(parent)
    kappa = A @ A.T
    ch = children_of(parent)
    k_of = np.array([len(c) for c in ch])
    comparable = (A + A.T) > 0
    depth = tree_arrays(parent)[1]
    # lca via ancestor sets: lca(v,w) = deepest u with A[v,u] = A[w,u] = 1
    B = np.zeros((n, n))
    for v in range(n):
        for w in range(n):
            if comparable[v, w]:
                continue
            common = np.where((A[v] > 0) & (A[w] > 0))[0]
            lca = common[np.argmax(depth[common])]
            B[v, w] = 1.0 / (k_of[lca] - 1)
    return kappa - B


def split_labels(parent: list[int], xi: np.ndarray, s: float = S_CONS) -> np.ndarray:
    """Sampler for the physical MC: labels with covariance kappa_split (x) I (per label component)."""
    order, _, _, _ = tree_arrays(parent)
    eta = np.array(xi, dtype=float, copy=True)
    for ch in children_of(parent):
        k = len(ch)
        if k >= 2:
            idx = np.asarray(ch)
            mean = xi[idx].mean(axis=0)
            # correlation -s/(k-1), unit marginal variance:  eta = a*(xi - mean) + b*mean with
            # a^2 (1-1/k) + b^2/k = 1  and  a^2(-1/k) + b^2/k = -s/(k-1)  =>  a^2 = 1 + s/(k-1) ... solved below
            a2 = (1.0 + s / (k - 1.0))
            b2 = 1.0 - s
            eta[idx] = math.sqrt(a2) * (xi[idx] - mean) + math.sqrt(b2) * mean
    labels = np.zeros_like(eta)
    for v in order:
        p = parent[v]
        labels[v] = eta[v] + (labels[p] if p >= 0 else 0.0)
    return labels


# ---------------------------------------------------------------- deterministic families
def complete_binary_parent(depth: int) -> list[int]:
    n = 2 ** (depth + 1) - 1
    return [-1] + [(i - 1) // 2 for i in range(1, n)]


def complete_binary_closed(n: int) -> float:
    """D_split for the complete binary tree, n = 2^{d+1}-1:  2n^2 + 6n - 4(n+1) log2(n+1)."""
    return 2.0 * n * n + 6.0 * n - 4.0 * (n + 1) * math.log2(n + 1)


def star_closed(n: int) -> float:
    return (n - 1.0) ** 2 / (n - 2.0)


def star_of_chains_parent(k: int) -> list[int]:
    """root + k chains of length k (n = k^2 + 1)."""
    parent = [-1]
    for c in range(k):
        for i in range(k):
            parent.append(0 if i == 0 else len(parent) - 1)
    return parent


def caterpillar_parent(k: int) -> list[int]:
    """spine of k vertices, each with k-1 extra leaves (n = k^2)."""
    parent = [-1] + list(range(k - 1))
    for v in range(k):
        for _ in range(k - 1):
            parent.append(v)
    return parent


def slope(xs, ys) -> float:
    return float(np.polyfit(np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float)), 1)[0])


# ---------------------------------------------------------------- exhaustive Cayley (n <= 7)
def all_rooted_trees(n: int):
    import heapq

    if n == 1:
        yield [-1]
        return
    for seq in itertools.product(range(n), repeat=n - 2):
        degree = [1] * n
        for s_ in seq:
            degree[s_] += 1
        adjacency: list[list[int]] = [[] for _ in range(n)]
        leaves = sorted(i for i in range(n) if degree[i] == 1)
        heapq.heapify(leaves)
        for s_ in seq:
            leaf = heapq.heappop(leaves)
            adjacency[leaf].append(s_)
            adjacency[s_].append(leaf)
            degree[s_] -= 1
            if degree[s_] == 1:
                heapq.heappush(leaves, s_)
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
            yield parent


# ---------------------------------------------------------------- parts
def part_fast() -> dict:
    t0 = time.time()
    out: dict = {"card": "F-01", "question": "Q-0016", "tree_seed": TREE_SEED, "s_conservation": S_CONS}

    # (0) definition cross-checks: A C A^T == kappa - B ; PSD ; sampler covariance ; chain C == I
    rng = np.random.default_rng(1)
    worst_def = 0.0
    min_eig = math.inf
    for n in (2, 3, 5, 9, 17):
        for _ in range(10):
            p = uniform_rooted_tree(n, rng)
            k1 = kappa_split(p)
            k2 = kappa_split_via_B(p)
            worst_def = max(worst_def, float(np.max(np.abs(k1 - k2))))
            min_eig = min(min_eig, float(np.linalg.eigvalsh(split_C(p)).min()))
    for b in (2, 4, 6):
        for _ in range(10):
            p = qspine_block(b, rng)
            worst_def = max(worst_def, float(np.max(np.abs(kappa_split(p) - kappa_split_via_B(p)))))
            min_eig = min(min_eig, float(np.linalg.eigvalsh(split_C(p)).min()))
    out["check_ACAt_equals_kappa_minus_B_max_abs"] = worst_def
    out["check_C_min_eigenvalue"] = min_eig
    # sampler: empirical covariance of split_labels vs kappa_split (one tree, many draws)
    p = qspine_block(5, np.random.default_rng(7))
    n = len(p)
    draws = 200_000
    X = np.stack([split_labels(p, rng.normal(size=n)) for _ in range(draws)])
    emp = X.T @ X / draws
    out["check_sampler_cov_vs_kappa_split"] = {
        "n": n,
        "max_abs_err": float(np.max(np.abs(emp - kappa_split(p)))),
        "max_abs_entry": float(np.max(np.abs(kappa_split(p)))),
        "draws": draws,
    }
    # conservation identity on the sampler: children mean == parent label (exact)
    ch = children_of(p)
    lab = split_labels(p, rng.normal(size=n))
    cons = max(abs(lab[c].mean() - lab[z]) for z, c in enumerate(ch) if len(c) >= 2)
    out["check_sampler_children_mean_equals_parent_max_abs"] = float(cons)
    # (1) closed forms
    out["closed_chain_equals_F02"] = {
        str(n): {"split": driver_split(chain_parent(n)), "f02_closed": float(chain_closed_form(n))} for n in (2, 3, 8, 16, 36)
    }
    out["closed_star"] = {
        str(n): {"split_matrix": driver_split(star_parent(n)), "closed_(n-1)^2/(n-2)": star_closed(n),
                 "f02_star": n - 2 + 1 / n**2} for n in (3, 4, 8, 16, 36, 128)
    }
    out["closed_complete_binary"] = {}
    for d in range(1, 8):
        p = complete_binary_parent(d)
        n = len(p)
        out["closed_complete_binary"][str(n)] = {
            "depth": d,
            "split_matrix": driver_split(p),
            "closed_2n2+6n-4(n+1)log2(n+1)": complete_binary_closed(n),
            "f02_matrix": driver_matrix(p),
            "sqrtD_over_n_split": math.sqrt(complete_binary_closed(n)) / n,
            "sqrtD_over_n_f02": math.sqrt(driver_matrix(p)) / n,
            "ratio_to_iid_split": math.sqrt(complete_binary_closed(n) / (n - 1)),
            "ratio_to_iid_f02": math.sqrt(driver_matrix(p) / (n - 1)),
        }
    # (2) deterministic families on the K1 grid
    fam = {}
    for name, fn in (("chain", chain_parent), ("star", star_parent), ("balanced_binary", binary_parent)):
        vals = {n: driver_split(fn(n)) for n in CAYLEY_GRID + (36,)}
        f02 = {n: driver_matrix(fn(n)) for n in CAYLEY_GRID + (36,)}
        fam[name] = {
            "D_split": {str(n): v for n, v in vals.items()},
            "D_f02": {str(n): v for n, v in f02.items()},
            "gamma_split": slope(CAYLEY_GRID, [math.sqrt(vals[n]) / n for n in CAYLEY_GRID]),
            "gamma_f02": slope(CAYLEY_GRID, [math.sqrt(f02[n]) / n for n in CAYLEY_GRID]),
            "rms_over_iid_128_split": math.sqrt(vals[128] / 127),
            "rms_over_iid_128_f02": math.sqrt(f02[128] / 127),
            "rms_over_iid_36_split": math.sqrt(vals[36] / 35),
            "rms_over_iid_36_f02": math.sqrt(f02[36] / 35),
        }
    soc = {}
    for k in (3, 5, 8, 11, 16):
        p = star_of_chains_parent(k)
        n = len(p)
        soc[str(n)] = {"k": k, "D_split": driver_split(p), "D_f02": driver_matrix(p)}
    fam["star_of_chains"] = soc
    cat = {}
    for k in (3, 5, 8, 11):
        p = caterpillar_parent(k)
        n = len(p)
        cat[str(n)] = {"k": k, "D_split": driver_split(p), "D_f02": driver_matrix(p)}
    fam["caterpillar"] = cat
    out["families"] = fam

    # (3) exhaustive Cayley n <= 6 (exact expectations over all n^{n-1} rooted labelled trees)
    ex = {}
    for n in (2, 3, 4, 5, 6):
        tot_s = 0.0
        tot_f = 0.0
        cnt = 0
        for p in all_rooted_trees(n):
            tot_s += driver_split(p)
            tot_f += driver_fast(p)[0]
            cnt += 1
        ex[str(n)] = {"count": cnt, "E_D_split": tot_s / cnt, "E_D_f02": tot_f / cnt,
                      "E_D_f02_exact": cayley_exact(n)["E_D"]}
    out["cayley_exhaustive"] = ex

    # (4) Cayley grid Monte Carlo (Pruefer sampler of the kill), paired with F-02 exact
    rng = np.random.default_rng(TREE_SEED)
    cay = {}
    for n in CAYLEY_GRID + (36,):
        m = CAYLEY_MC[n]
        ds = np.empty(m)
        df = np.empty(m)
        for t in range(m):
            p = uniform_rooted_tree(n, rng)
            ds[t] = driver_split(p)
            df[t] = driver_fast(p)[0]
        e_s = float(ds.mean())
        se_s = float(ds.std(ddof=1) / math.sqrt(m))
        ratio = ds / df
        cay[str(n)] = {
            "trials": m,
            "E_D_split": e_s,
            "se_E_D_split": se_s,
            "E_D_f02_mc": float(df.mean()),
            "E_D_f02_exact": cayley_exact(n)["E_D"],
            "E_ratio_split_over_f02_paired": float(ratio.mean()),
            "se_ratio_paired": float(ratio.std(ddof=1) / math.sqrt(m)),
            "rms_split_over_iid": math.sqrt(e_s / (n - 1)),
            "rms_f02_over_iid_exact": math.sqrt(cayley_exact(n)["E_D"] / (n - 1)),
            "sqrtD_over_n_split": math.sqrt(e_s) / n,
        }
    out["cayley_mc"] = cay
    xs = list(CAYLEY_GRID)
    out["gamma_split_cayley_grid"] = slope(xs, [cay[str(n)]["sqrtD_over_n_split"] for n in xs])
    out["gamma_split_cayley_local"] = {
        f"{a}->{b}": math.log(cay[str(b)]["sqrtD_over_n_split"] / cay[str(a)]["sqrtD_over_n_split"]) / math.log(b / a)
        for a, b in zip(xs[:-1], xs[1:])
    }
    out["her_ratio_128_split"] = cay["128"]["rms_split_over_iid"]
    out["her_ratio_128_f02_exact"] = cay["128"]["rms_f02_over_iid_exact"]
    out["wall_seconds_fast"] = time.time() - t0
    FAST_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def part_qspine(depths: tuple[int, ...], trials: int) -> dict:
    """Replay the F-02 tree stream (seed 20261902, depths 1..8 in order) with rng-state checkpoints."""
    if STATE_FILE.exists() and PARTIAL_FILE.exists():
        partial = json.loads(PARTIAL_FILE.read_text(encoding="utf-8"))
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        rng = np.random.default_rng()
        rng.bit_generator.state = state
        done = int(partial["_next_depth"])
    else:
        partial = {"_trials": trials, "_seed": TREE_SEED}
        rng = np.random.default_rng(TREE_SEED)
        done = 1
    for b in depths:
        if b != done:
            raise SystemExit(f"depth order violated: next expected depth {done}, got {b}")
        t0 = time.time()
        dn2_s = np.empty(trials)
        dn2_f = np.empty(trials)
        nn = np.empty(trials)
        for t in range(trials):
            p = qspine_block(b, rng)
            n = len(p)
            if n == 1:
                dn2_s[t] = 0.0
                dn2_f[t] = 0.0
            else:
                dn2_s[t] = driver_split(p) / (n * n)
                dn2_f[t] = driver_fast(p)[0] / (n * n)
            nn[t] = n
        e_s = float(dn2_s.mean())
        e_f = float(dn2_f.mean())
        diff = dn2_s - dn2_f
        n_star = b * (b + 1) // 2
        partial[str(b)] = {
            "E_n": float(nn.mean()),
            "E_n_exact": n_star,
            "E_D_over_n2_split": e_s,
            "se_E_D_over_n2_split": float(dn2_s.std(ddof=1) / math.sqrt(trials)),
            "E_D_over_n2_f02_replay": e_f,
            "se_E_D_over_n2_f02_replay": float(dn2_f.std(ddof=1) / math.sqrt(trials)),
            "E_diff_split_minus_f02": float(diff.mean()),
            "se_diff_paired": float(diff.std(ddof=1) / math.sqrt(trials)),
            "ratio_split_over_f02": (e_s / e_f) if e_f > 0 else None,
            "rms_pred_over_eps_star_split": math.sqrt(e_s),
            "ratio_to_iid_at_nstar_split": math.sqrt(e_s) * n_star / math.sqrt(n_star - 1) if n_star > 1 else None,
            "cv_D_over_n2_split": float(dn2_s.std(ddof=1) / e_s) if e_s > 0 else None,
            "max_n": int(nn.max()),
            "wall_seconds": time.time() - t0,
        }
        done = b + 1
        partial["_next_depth"] = done
        PARTIAL_FILE.write_text(json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8")
        STATE_FILE.write_text(json.dumps(rng.bit_generator.state), encoding="utf-8")
        print(f"depth {b}: n={partial[str(b)]['E_n']:.4f} split={e_s:.5f} f02={e_f:.5f} "
              f"ratio={partial[str(b)]['ratio_split_over_f02']} wall={partial[str(b)]['wall_seconds']:.1f}s", flush=True)
    return partial


def part_assemble() -> dict:
    fast = json.loads(FAST_FILE.read_text(encoding="utf-8"))
    partial = json.loads(PARTIAL_FILE.read_text(encoding="utf-8"))
    f02 = json.loads((ROOT / "verify" / "Q-0008" / "F-02" / "predictions.json").read_text(encoding="utf-8"))
    qs = {b: partial[str(b)] for b in QSPINE_DEPTHS if str(b) in partial}
    out = dict(fast)
    out["qspine"] = {str(b): v for b, v in qs.items()}
    out["qspine_trials_per_depth"] = partial.get("_trials")
    # protocol check: replayed F-02 table must equal F-02 predictions.json (same trees)
    out["check_f02_replay_max_abs"] = max(
        abs(qs[b]["E_D_over_n2_f02_replay"] - f02["qspine"][str(b)]["E_D_over_n2"]) for b in qs
    )
    bs = [b for b in qs if b >= 2]
    if bs:
        out["qspine_slope_vs_En_split"] = slope([qs[b]["E_n_exact"] for b in bs], [qs[b]["rms_pred_over_eps_star_split"] for b in bs])
        out["qspine_slope_vs_b_split"] = slope(bs, [qs[b]["rms_pred_over_eps_star_split"] for b in bs])
        out["qspine_slope_vs_En_f02"] = f02["qspine_slope_vs_En"]
    if 8 in qs:
        out["qspine_ratio_b8_over_iid36_split"] = qs[8]["ratio_to_iid_at_nstar_split"]
        out["qspine_ratio_b8_over_iid36_f02"] = f02["qspine"]["8"]["ratio_to_iid_at_nstar"]
        out["qspine_b8_amplitude_ratio_split_over_f02"] = qs[8]["ratio_to_iid_at_nstar_split"] / f02["qspine"]["8"]["ratio_to_iid_at_nstar"]
        out["qspine_b8_shape_factor_vs_cayley36_split"] = qs[8]["ratio_to_iid_at_nstar_split"] / f02["cayley"]["36"]["rms_her_over_iid"]
    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("fast", "qspine", "assemble"), required=True)
    ap.add_argument("--depths", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--trials", type=int, default=QSPINE_TRIALS)
    args = ap.parse_args()
    if args.part == "fast":
        out = part_fast()
        print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}, indent=2))
        print("star:", {n: (round(v["split_matrix"], 4), round(v["f02_star"], 4)) for n, v in out["closed_star"].items()})
        print("binary:", {n: (round(v["split_matrix"], 3), round(v["closed_2n2+6n-4(n+1)log2(n+1)"], 3), round(v["f02_matrix"], 3)) for n, v in out["closed_complete_binary"].items()})
        print("cayley_mc:", {n: (round(v["E_D_split"], 2), round(v["E_D_f02_exact"], 2), round(v["E_ratio_split_over_f02_paired"], 4)) for n, v in out["cayley_mc"].items()})
        print("families gamma:", {k: (round(v["gamma_split"], 4), round(v["gamma_f02"], 4)) for k, v in out["families"].items() if "gamma_split" in v})
    elif args.part == "qspine":
        depths = tuple(int(x) for x in args.depths.split(","))
        part_qspine(depths, args.trials)
    else:
        out = part_assemble()
        print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}, indent=2))
        print("qspine:", {b: (round(v["E_D_over_n2_split"], 5), round(v["E_D_over_n2_f02_replay"], 5), round(v["ratio_split_over_f02"] or 0, 4)) for b, v in out["qspine"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
