"""Q-0011 F-01: effective-ancestor law for the heritable block exponent.

Card: derivations/Q-0011/F-01.formula.md (2026-09-02).  Seeds, grids, delta, trial counts and
windows below are PRE-REGISTERED; do not edit after seeing a physics result.

The card claims (axiom candidate) that the centered driver of F-02,

    D(T) = || H kappa H ||_F^2,   kappa_vw = |path(v) cap path(w)|,   H = I - J/n,

is fixed, up to a bounded amplitude, by a purely LOCAL statistic of the tree: each shared
ancestor u is weighted by the fraction of the block outside its own subtree,

    kappa_eff(v,w) = sum_{u <= v, u <= w} (1 - s_u/n),      s_u = |sub(u)|
    mu2_eff        = E_{v,w}[ kappa_eff(v,w)^2 ]            (uniform ORDERED pairs, v=w included)
    c(T)           = D / (n^2 mu2_eff)     with   1/4 <= c(T) <= 2   [axiom candidate]

so that the heritable RMS residual exponent is  gamma_her = (1/2) exp_n(mu2_eff).

Exact positive decomposition used throughout (no cancellation; kappa = sum_u 1_sub(u) 1_sub(u)^T
and the subtrees form a laminar family):

    D = sum_u s_u^2 (1-s_u/n)^2 + 2 sum_{u' < u} s_u^2 (1-s_u'/n)^2 + (1/n^2) sum_{u || u'} s_u^2 s_u'^2
      = A (nested)                                                  + B (disjoint)
    n^2 mu2_eff = sum_u s_u^2 (1-s_u/n)^2 + 2 sum_{u' < u} s_u^2 (1-s_u/n)(1-s_u'/n) = A_tilde >= A

Modes
  theory     tree-only combinatorics -> predictions.json  (the formula side; NOT a kill)
  battery    universality/amplitude table over the disclosed families (consistency, NOT a kill)
  soc|broom|tls|rrt|amp|all   physics kill runs -> result.json

Physics convention (identical to verify/Q-0008/F-02/check_modes.py): cells are polar-aligned
self-dual triples of tetrads I + delta*label, labels are heritable root-path sums of iid
N(0,1)^{4x4} increments, the statistic is the trial RMS of the 12.4 normalized simplicity residual.

Usage: python verify/Q-0011/F-01/check_families.py --mode {theory,battery,soc,broom,tls,rrt,amp,all} [--smoke]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
F02 = ROOT / "verify" / "Q-0008" / "F-02"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(F02))

from driver_numbers import (  # noqa: E402
    cayley_exact,
    chain_parent,
    driver_fast,
    driver_matrix,
    star_parent,
    tree_arrays,
    uniform_rooted_tree,
)

SEED = 20260902           # physics seed (heritable arms)
SEED_IID = 20260903       # physics seed (iid control arm)
TREE_SEED = 20261902      # tree-only Monte Carlo seed (F-02 convention: kill seed + 1000)
DELTA = 0.005
MIN_DET = 0.05
RRT_MC = 20_000           # tree-only trials per size for E[D] of the random recursive tree
RRT_MC_ALT = 4_000        # tree-only trials per size for the competing laws (Var, depth)

SOC_KS = (4, 5, 6, 7, 8, 9, 10, 11, 12)      # star-of-chains, n = k^2 + 1
TLS_KS = (4, 5, 6, 7, 8, 9, 10, 11, 12)      # two-level star, n = k^2 + k + 1
BROOM_SIZES = (8, 16, 32, 64, 128)           # broom, spine m = ceil(sqrt(n))
RRT_SIZES = (8, 16, 32, 64, 128)
SLOPE_TRIALS = 128
RRT_TRIALS = 256
AMP_TRIALS = 256
AMP_SOC_K = 8                                 # n = 65
AMP_TLS_K = 8                                 # n = 73

PREREGISTERED = {
    "soc_slope": 0.2466,
    "broom_slope": -0.0349,
    "tls_slope": -0.2594,
    "rrt_slope": 0.1882,
    "soc_over_broom_65": 2.3408,
    "tls_over_iid_73": 3.2103,
}
WINDOWS = {
    "soc_slope": (0.187, 0.307),
    "broom_slope": (-0.095, 0.025),
    "tls_slope": (-0.319, -0.199),
    "rrt_slope": (0.128, 0.248),
    "soc_over_broom_65": (2.06, 2.62),
    "tls_over_iid_73": (2.82, 3.60),
}
AMPLITUDE_WINDOW = (0.25, 2.0)   # card: 1/4 <= c(T) <= 2 for every rooted tree, every n >= 2


# ---------------------------------------------------------------- tree families
def soc_parent(k: int) -> list[int]:
    """star-of-chains: root + k disjoint chains of length k;  n = k^2 + 1."""
    parent = [-1]
    for _ in range(k):
        prev = 0
        for _ in range(k):
            parent.append(prev)
            prev = len(parent) - 1
    return parent


def tls_parent(k: int) -> list[int]:
    """two-level star: root + k branches, each carrying k leaves;  n = k^2 + k + 1."""
    parent = [-1]
    branches = []
    for _ in range(k):
        parent.append(0)
        branches.append(len(parent) - 1)
    for b in branches:
        for _ in range(k):
            parent.append(b)
    return parent


def broom_parent(n: int, exponent: float = 0.5) -> list[int]:
    """broom: spine of m = ceil(n^exponent) from the root, all remaining vertices are leaves at the tip."""
    m = min(math.ceil(n**exponent), n - 1)
    parent = [-1]
    prev = 0
    for _ in range(m):
        parent.append(prev)
        prev = len(parent) - 1
    tip = prev
    while len(parent) < n:
        parent.append(tip)
    return parent


def lollipop_parent(n: int, exponent: float = 0.75) -> list[int]:
    """lollipop: spine of m = ceil(n^exponent) from the root, remaining vertices are leaves AT THE ROOT."""
    m = min(math.ceil(n**exponent), n - 1)
    parent = [-1]
    prev = 0
    for _ in range(m):
        parent.append(prev)
        prev = len(parent) - 1
    while len(parent) < n:
        parent.append(0)
    return parent


def caterpillar_parent(k: int) -> list[int]:
    """spine of k vertices, each carrying k-1 leaves;  n = k^2."""
    parent = [-1]
    spine = [0]
    prev = 0
    for _ in range(k - 1):
        parent.append(prev)
        prev = len(parent) - 1
        spine.append(prev)
    for s in spine:
        for _ in range(k - 1):
            parent.append(s)
    return parent


def kary_parent(k: int, depth: int) -> list[int]:
    parent = [-1]
    frontier = [0]
    for _ in range(depth):
        nxt = []
        for f in frontier:
            for _ in range(k):
                parent.append(f)
                nxt.append(len(parent) - 1)
        frontier = nxt
    return parent


def split_parent(n: int, k: int) -> list[int]:
    """root + k branches, remaining vertices spread as leaves over the branches (k=2 is the 2-split)."""
    parent = [-1]
    branches = []
    for _ in range(k):
        parent.append(0)
        branches.append(len(parent) - 1)
    i = 0
    while len(parent) < n:
        parent.append(branches[i % k])
        i += 1
    return parent


def power_profile_parent(n: int, m: int, p: float) -> list[int]:
    """spine of m vertices whose subtree sizes follow s_j = n(1-(j/m)^p); leaves hung to match."""
    sizes = [max(1, int(n * (1.0 - (j / m) ** p))) for j in range(m)]
    parent = [-1]
    spine = [0]
    prev = 0
    for _ in range(m - 1):
        parent.append(prev)
        prev = len(parent) - 1
        spine.append(prev)
    hang = [max(0, int(sizes[j] - sizes[j + 1] - 1)) for j in range(m - 1)] + [max(0, int(sizes[-1]) - 1)]
    for j, h in enumerate(hang):
        for _ in range(h):
            if len(parent) < n:
                parent.append(spine[j])
    while len(parent) < n:
        parent.append(spine[0])
    return parent


def rrt_parent(n: int, rng: np.random.Generator) -> list[int]:
    """random recursive tree: vertex i attaches to a uniform parent in {0,...,i-1}."""
    return [-1] + [int(rng.integers(0, i)) for i in range(1, n)]


def binary_parent_complete(h: int) -> list[int]:
    return kary_parent(2, h)


# ---------------------------------------------------------------- tree statistics (O(n))
def tree_stats(parent: list[int]) -> dict[str, float]:
    """D, A (nested part), B (disjoint part), n^2 mu2_eff, raw pair variance, max depth."""
    n = len(parent)
    order, depth, sub, _prefix = tree_arrays(parent)
    d_driver, _tr, _w2 = driver_fast(parent)
    s = sub.astype(np.float64)
    w = 1.0 - s / n
    p_sum = np.zeros(n)   # sum of weights of strict ancestors
    q_sum = np.zeros(n)   # sum of squared weights of strict ancestors
    for v in order:
        par = parent[v]
        if par >= 0:
            p_sum[v] = p_sum[par] + w[par]
            q_sum[v] = q_sum[par] + w[par] ** 2
    diag = float(np.sum(s * s * w * w))
    a_nested = diag + 2.0 * float(np.sum(s * s * q_sum))
    a_tilde = diag + 2.0 * float(np.sum(s * s * w * p_sum))
    e_k2 = float(np.sum((2 * depth + 1) * s * s)) / n**2
    e_k = float(np.sum(s * s)) / n**2
    return {
        "n": float(n),
        "D": float(d_driver),
        "A": a_nested,
        "B": float(d_driver) - a_nested,
        "n2_mu2_eff": a_tilde,
        "mu2_eff": a_tilde / n**2,
        "c": float(d_driver) / a_tilde if a_tilde > 0 else float("nan"),
        "pair_var": e_k2 - e_k * e_k,
        "max_depth": float(depth.max()),
    }


def slope(xs, ys) -> float:
    return float(np.polyfit(np.log(np.asarray(xs, dtype=float)), np.log(np.asarray(ys, dtype=float)), 1)[0])


def exponent_pair(family, sizes) -> dict[str, float]:
    """(exp_n(D/n^2), exp_n(mu2_eff), max |c-1| info) for a family callable size -> parent list."""
    rows = [tree_stats(family(x)) for x in sizes]
    ns = [r["n"] for r in rows]
    return {
        "sizes": ns,
        "exp_driver": slope(ns, [r["D"] / r["n"] ** 2 for r in rows]),
        "exp_mu2_eff": slope(ns, [r["mu2_eff"] for r in rows]),
        "exp_pair_var": slope(ns, [r["pair_var"] for r in rows]),
        "exp_depth": slope(ns, [max(r["max_depth"], 1.0) for r in rows]),
        "c_min": min(r["c"] for r in rows),
        "c_max": max(r["c"] for r in rows),
        "B_over_D_last": rows[-1]["B"] / rows[-1]["D"],
    }


# ---------------------------------------------------------------- theory mode (tree-only)
def rrt_expected_driver(sizes, trials: int) -> dict[str, list[float]]:
    e_d, se = [], []
    for n in sizes:
        rng = np.random.default_rng(TREE_SEED + n)
        vals = np.array([driver_fast(rrt_parent(n, rng))[0] for _ in range(trials)])
        e_d.append(float(vals.mean()))
        se.append(float(vals.std(ddof=1) / math.sqrt(trials)))
    return {"E_D": e_d, "se": se}


def rrt_alternatives(sizes, trials: int) -> dict[str, float]:
    var_mean, depth_mean, mu_mean = [], [], []
    for n in sizes:
        rng = np.random.default_rng(TREE_SEED + n)
        vs, ds, ms = [], [], []
        for _ in range(trials):
            st = tree_stats(rrt_parent(n, rng))
            vs.append(st["pair_var"])
            ds.append(max(st["max_depth"], 1.0))
            ms.append(st["mu2_eff"])
        var_mean.append(float(np.mean(vs)))
        depth_mean.append(float(np.mean(ds)))
        mu_mean.append(float(np.mean(ms)))
    return {
        "pair_var_law_slope": slope(sizes, var_mean) / 2.0,
        "depth_law_slope": slope(sizes, depth_mean),
        "mu2_eff_law_slope": slope(sizes, mu_mean) / 2.0,
    }


def theory() -> dict:
    out: dict = {"seed_tree": TREE_SEED, "delta": DELTA}

    # --- exact closed forms reproduced by the positive decomposition
    checks = []
    for n in (2, 3, 5, 8, 13, 21):
        st_star = tree_stats(star_parent(n))
        st_chain = tree_stats(chain_parent(n))
        checks.append(
            {
                "n": n,
                "star_D": st_star["D"],
                "star_closed": n - 2 + 1 / n**2,
                "star_A_plus_B_closed": ((n - 1) ** 3 + (n - 1) * (n - 2)) / n**2,
                "star_n2mu2_closed": (n - 1) ** 3 / n**2,
                "chain_D": st_chain["D"],
                "chain_closed": (n**2 - 1) * (2 * n**2 + 7) / 180,
                "chain_A": st_chain["A"],
                "chain_B": st_chain["B"],
                "chain_n2mu2": st_chain["n2_mu2_eff"],
                "chain_n2mu2_closed": (n**2 - 1) * (n**2 + 1) / 60,
            }
        )
    out["closed_form_checks"] = checks
    out["closed_form_max_rel_err"] = max(
        max(
            abs(c["star_D"] - c["star_closed"]) / (1 + abs(c["star_closed"])),
            abs(c["star_D"] - c["star_A_plus_B_closed"]) / (1 + abs(c["star_closed"])),
            abs(c["chain_D"] - c["chain_closed"]) / (1 + abs(c["chain_closed"])),
            abs(c["chain_A"] - c["chain_closed"]) / (1 + abs(c["chain_closed"])),
            abs(c["chain_n2mu2"] - c["chain_n2mu2_closed"]) / (1 + abs(c["chain_n2mu2_closed"])),
        )
        for c in checks
    )
    # driver_fast vs the direct matrix definition on small trees of every predicted family
    direct = []
    for parent in (soc_parent(3), tls_parent(3), broom_parent(13), lollipop_parent(13), caterpillar_parent(4)):
        direct.append(abs(driver_fast(parent)[0] - driver_matrix(parent)))
    out["driver_fast_vs_matrix_max_abs"] = max(direct)

    # --- star-of-chains (exact)
    soc_rows = [tree_stats(soc_parent(k)) for k in SOC_KS]
    soc_ns = [r["n"] for r in soc_rows]
    out["soc"] = {
        "ks": list(SOC_KS),
        "n": soc_ns,
        "D": [r["D"] for r in soc_rows],
        "rms_over_eps_star": [math.sqrt(r["D"]) / r["n"] for r in soc_rows],
        "slope": slope(soc_ns, [math.sqrt(r["D"]) / r["n"] for r in soc_rows]),
        "alt_pair_var_law": slope(soc_ns, [r["pair_var"] for r in soc_rows]) / 2.0,
        "alt_depth_law": slope(soc_ns, [r["max_depth"] for r in soc_rows]),
        "law_exponent_mu2_eff": slope(soc_ns, [r["mu2_eff"] for r in soc_rows]) / 2.0,
        "c": [r["c"] for r in soc_rows],
    }
    # --- broom (exact)
    broom_rows = [tree_stats(broom_parent(n)) for n in BROOM_SIZES]
    out["broom"] = {
        "n": list(BROOM_SIZES),
        "m": [math.ceil(n**0.5) for n in BROOM_SIZES],
        "D": [r["D"] for r in broom_rows],
        "slope": slope(BROOM_SIZES, [math.sqrt(r["D"]) / r["n"] for r in broom_rows]),
        "alt_pair_var_law": slope(BROOM_SIZES, [r["pair_var"] for r in broom_rows]) / 2.0,
        "alt_depth_law": slope(BROOM_SIZES, [r["max_depth"] for r in broom_rows]),
        "law_exponent_mu2_eff": slope(BROOM_SIZES, [r["mu2_eff"] for r in broom_rows]) / 2.0,
        "c": [r["c"] for r in broom_rows],
    }
    # --- two-level star (exact)
    tls_rows = [tree_stats(tls_parent(k)) for k in TLS_KS]
    tls_ns = [r["n"] for r in tls_rows]
    out["tls"] = {
        "ks": list(TLS_KS),
        "n": tls_ns,
        "D": [r["D"] for r in tls_rows],
        "D_closed_form": [
            (k**4 * (k + 1) ** 3 + 2 * k**6 + k * (k - 1) * (k + 1) ** 4 + 2 * k**2 * (k - 1) * (k + 1) ** 2 + k**2 * (k**2 - 1))
            / (k**2 + k + 1) ** 2
            for k in TLS_KS
        ],
        "slope": slope(tls_ns, [math.sqrt(r["D"]) / r["n"] for r in tls_rows]),
        "alt_pair_var_law": slope(tls_ns, [r["pair_var"] for r in tls_rows]) / 2.0,
        "alt_depth_law": 0.0,
        "alt_iid": slope(tls_ns, [math.sqrt(n - 1) / n for n in tls_ns]),
        "law_exponent_mu2_eff": slope(tls_ns, [r["mu2_eff"] for r in tls_rows]) / 2.0,
        "c": [r["c"] for r in tls_rows],
    }
    out["tls"]["closed_form_max_abs_err"] = max(
        abs(a - b) for a, b in zip(out["tls"]["D"], out["tls"]["D_closed_form"])
    )
    # --- random recursive tree (tree-only Monte Carlo)
    rrt = rrt_expected_driver(RRT_SIZES, RRT_MC)
    out["rrt"] = {
        "n": list(RRT_SIZES),
        "E_D": rrt["E_D"],
        "se": rrt["se"],
        "trials": RRT_MC,
        "slope": slope(RRT_SIZES, [math.sqrt(d) / n for d, n in zip(rrt["E_D"], RRT_SIZES)]),
    }
    out["rrt"].update({f"alt_{k}": v for k, v in rrt_alternatives(RRT_SIZES, RRT_MC_ALT).items()})

    # --- amplitude / cross-family ratios
    d_soc65 = tree_stats(soc_parent(AMP_SOC_K))["D"]
    d_broom65 = tree_stats(broom_parent(AMP_SOC_K**2 + 1))["D"]
    d_tls73 = tree_stats(tls_parent(AMP_TLS_K))["D"]
    n_tls73 = AMP_TLS_K**2 + AMP_TLS_K + 1
    out["ratios"] = {
        "D_soc_65": d_soc65,
        "D_broom_65": d_broom65,
        "soc_over_broom_65": math.sqrt(d_soc65 / d_broom65),
        "D_tls_73": d_tls73,
        "n_tls": n_tls73,
        "tls_over_iid_73": math.sqrt(d_tls73 / (n_tls73 - 1)),
        "cayley_65_exact_E_D": cayley_exact(65)["E_D"],
    }

    out["prereg"] = {
        "soc_slope": out["soc"]["slope"],
        "broom_slope": out["broom"]["slope"],
        "tls_slope": out["tls"]["slope"],
        "rrt_slope": out["rrt"]["slope"],
        "soc_over_broom_65": out["ratios"]["soc_over_broom_65"],
        "tls_over_iid_73": out["ratios"]["tls_over_iid_73"],
    }
    out["prereg_card"] = PREREGISTERED
    out["prereg_max_abs_diff"] = max(
        abs(out["prereg"][k] - PREREGISTERED[k]) for k in PREREGISTERED
    )
    out["battery"] = battery()
    return out


def battery() -> dict:
    """Disclosed consistency battery (NOT a kill): universality of the exponent and the amplitude window."""
    rng = np.random.default_rng(TREE_SEED)
    fams = {
        "chain": (chain_parent, (512, 2048, 8192, 32768)),
        "star": (star_parent, (512, 2048, 8192, 32768)),
        "binary_complete": (binary_parent_complete, (9, 11, 13, 15)),
        "4ary_complete": (lambda d: kary_parent(4, d), (4, 5, 6, 7)),
        "star_of_chains": (soc_parent, (16, 32, 64, 128)),
        "caterpillar": (caterpillar_parent, (16, 32, 64, 128)),
        "lollipop_075": (lambda n: lollipop_parent(n, 0.75), (512, 2048, 8192, 32768)),
        "lollipop_06": (lambda n: lollipop_parent(n, 0.6), (512, 2048, 8192, 32768)),
        "broom_05": (lambda n: broom_parent(n, 0.5), (512, 2048, 8192, 32768)),
        "broom_075": (lambda n: broom_parent(n, 0.75), (512, 2048, 8192, 32768)),
        "broom_09": (lambda n: broom_parent(n, 0.9), (512, 2048, 8192, 32768)),
        "two_level_star": (tls_parent, (16, 32, 64, 128)),
        "split2": (lambda n: split_parent(n, 2), (512, 2048, 8192, 32768)),
        "split3": (lambda n: split_parent(n, 3), (512, 2048, 8192, 32768)),
        "power_profile_p16": (lambda n: power_profile_parent(n, 300, 16.0), (4096, 16384, 65536)),
        "power_profile_p128": (lambda n: power_profile_parent(n, 300, 128.0), (4096, 16384, 65536)),
        "cayley": (lambda n: uniform_rooted_tree(n, rng), (512, 2048, 8192, 32768)),
        "rrt": (lambda n: rrt_parent(n, rng), (512, 2048, 8192, 32768)),
    }
    table = {}
    worst_exp, worst_c_lo, worst_c_hi = 0.0, 10.0, 0.0
    for name, (fn, sizes) in fams.items():
        row = exponent_pair(fn, sizes)
        row["exp_gap"] = row["exp_driver"] - row["exp_mu2_eff"]
        row["pair_var_law_gap"] = row["exp_pair_var"] - row["exp_driver"]
        table[name] = row
        # the power-profile entries keep the spine length fixed while n grows, so they are NOT a
        # scaling family and their "exponent" is undefined; their amplitude still has to obey the window.
        if not name.startswith("power_profile"):
            worst_exp = max(worst_exp, abs(row["exp_gap"]))
        worst_c_lo = min(worst_c_lo, row["c_min"])
        worst_c_hi = max(worst_c_hi, row["c_max"])

    # exhaustive over every rooted shape with n <= 9 (parent[v] < v enumerates all shapes)
    import itertools  # noqa: PLC0415

    exhaustive = {}
    for n in range(2, 10):
        lo, hi = 10.0, 0.0
        for tail in itertools.product(*[range(i) for i in range(1, n)]):
            c = tree_stats([-1] + list(tail))["c"]
            lo, hi = min(lo, c), max(hi, c)
        exhaustive[n] = [lo, hi]
        worst_c_lo, worst_c_hi = min(worst_c_lo, lo), max(worst_c_hi, hi)

    # random shapes and structured extremes
    rng2 = np.random.default_rng(TREE_SEED)
    r_lo, r_hi = 10.0, 0.0
    for _ in range(20_000):
        n = int(rng2.integers(5, 60))
        c = tree_stats([-1] + [int(rng2.integers(0, i)) for i in range(1, n)])["c"]
        r_lo, r_hi = min(r_lo, c), max(r_hi, c)
    s_lo, s_hi = 10.0, 0.0
    for n, m, p in ((20000, 300, 128.0), (20000, 1000, 128.0), (200000, 1000, 128.0), (20000, 300, 16.0)):
        c = tree_stats(power_profile_parent(n, m, p))["c"]
        s_lo, s_hi = min(s_lo, c), max(s_hi, c)
    for n in (2000, 20000):
        c = tree_stats(split_parent(n, 2))["c"]
        s_lo, s_hi = min(s_lo, c), max(s_hi, c)
    worst_c_lo = min(worst_c_lo, r_lo, s_lo)
    worst_c_hi = max(worst_c_hi, r_hi, s_hi)

    return {
        "families": table,
        "max_abs_exponent_gap_scaling_families": worst_exp,
        "exhaustive_small_trees": exhaustive,
        "random_shapes": {"trials": 20000, "c_min": r_lo, "c_max": r_hi},
        "structured_extremes": {"c_min": s_lo, "c_max": s_hi},
        "amplitude_min": worst_c_lo,
        "amplitude_max": worst_c_hi,
        "amplitude_window": list(AMPLITUDE_WINDOW),
        "amplitude_inside": bool(AMPLITUDE_WINDOW[0] <= worst_c_lo and worst_c_hi <= AMPLITUDE_WINDOW[1]),
        "note": "disclosed before pre-registration; consistency only, not a kill. K6 applies to trees NOT covered here.",
    }


# ---------------------------------------------------------------- physics (kill side)
def _physics():
    from examples.physics.causal_face_simplicity import (  # noqa: PLC0415
        geometric_self_dual_triple,
        simplicity_residual,
    )
    from examples.physics.urbantke_shape_matching_rg import optimal_internal_alignment  # noqa: PLC0415

    return geometric_self_dual_triple, simplicity_residual, optimal_internal_alignment


def block_residual(labels: np.ndarray, delta: float) -> float:
    triple, residual, align = _physics()
    reference = triple(np.eye(4))
    blocked = np.zeros_like(reference)
    for lab in labels:
        tetrad = np.eye(4) + delta * lab
        if float(np.linalg.det(tetrad)) <= MIN_DET:
            return math.nan
        blocked += align(reference, triple(tetrad)).aligned_candidate
    return residual(blocked)


def heritable_labels(parent: list[int], xi: np.ndarray) -> np.ndarray:
    order, _, _, _ = tree_arrays(parent)
    labels = np.zeros_like(xi)
    for v in order:
        p = parent[v]
        labels[v] = xi[v] + (labels[p] if p >= 0 else 0.0)
    return labels


def rms_heritable(parent_fn, n: int, trials: int, seed: int) -> float:
    rng = np.random.default_rng(seed + n)
    vals = []
    while len(vals) < trials:
        parent = parent_fn(n, rng)
        value = block_residual(heritable_labels(parent, rng.normal(size=(n, 4, 4))), DELTA)
        if math.isfinite(value):
            vals.append(value)
    arr = np.asarray(vals)
    return float(np.sqrt(np.mean(arr * arr)))


def rms_iid(n: int, trials: int, seed: int) -> float:
    rng = np.random.default_rng(seed + n)
    vals = []
    while len(vals) < trials:
        value = block_residual(rng.normal(size=(n, 4, 4)), DELTA)
        if math.isfinite(value):
            vals.append(value)
    arr = np.asarray(vals)
    return float(np.sqrt(np.mean(arr * arr)))


def run_slope(name: str, parent_fn, sizes, trials: int, seed: int) -> dict:
    rms = [rms_heritable(parent_fn, n, trials, seed) for n in sizes]
    return {"mode": name, "sizes": list(sizes), "rms": rms, "trials": trials, "seed": seed,
            "slope": slope(sizes, rms)}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="theory",
                    choices=["theory", "battery", "soc", "broom", "tls", "rrt", "amp", "all"])
    ap.add_argument("--smoke", action="store_true", help="tiny run; never writes result.json")
    args = ap.parse_args(argv)

    if args.mode == "theory":
        out = theory()
        (HERE / "predictions.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: out[k] for k in ("prereg", "prereg_card", "prereg_max_abs_diff",
                                              "closed_form_max_rel_err", "driver_fast_vs_matrix_max_abs")},
                         ensure_ascii=False, indent=2))
        print(json.dumps({"battery_exponent_gap": out["battery"]["max_abs_exponent_gap_scaling_families"],
                          "battery_amplitude_min": out["battery"]["amplitude_min"],
                          "battery_amplitude_max": out["battery"]["amplitude_max"],
                          "battery_amplitude_inside": out["battery"]["amplitude_inside"]}, ensure_ascii=False))
        return 0
    if args.mode == "battery":
        print(json.dumps(battery(), ensure_ascii=False, indent=2))
        return 0

    trials = 4 if args.smoke else None
    stats: dict[str, float] = {}
    detail: dict[str, dict] = {}

    if args.mode in ("soc", "all"):
        ks = SOC_KS[:2] if args.smoke else SOC_KS
        d = run_slope("soc", lambda n, _rng: soc_parent(int(round(math.sqrt(n - 1)))),
                      [k * k + 1 for k in ks], trials or SLOPE_TRIALS, SEED)
        detail["soc"] = d
        stats["soc_slope"] = d["slope"]
    if args.mode in ("broom", "all"):
        sizes = BROOM_SIZES[:2] if args.smoke else BROOM_SIZES
        d = run_slope("broom", lambda n, _rng: broom_parent(n), sizes, trials or SLOPE_TRIALS, SEED)
        detail["broom"] = d
        stats["broom_slope"] = d["slope"]
    if args.mode in ("tls", "all"):
        ks = TLS_KS[:2] if args.smoke else TLS_KS
        d = run_slope("tls", lambda n, _rng: tls_parent(int(round((math.sqrt(4 * n - 3) - 1) / 2))),
                      [k * k + k + 1 for k in ks], trials or SLOPE_TRIALS, SEED)
        detail["tls"] = d
        stats["tls_slope"] = d["slope"]
    if args.mode in ("rrt", "all"):
        sizes = RRT_SIZES[:2] if args.smoke else RRT_SIZES
        d = run_slope("rrt", rrt_parent, sizes, trials or RRT_TRIALS, SEED)
        detail["rrt"] = d
        stats["rrt_slope"] = d["slope"]
    if args.mode in ("amp", "all"):
        t = trials or AMP_TRIALS
        n_soc = AMP_SOC_K**2 + 1
        n_tls = AMP_TLS_K**2 + AMP_TLS_K + 1
        r_soc = rms_heritable(lambda n, _rng: soc_parent(AMP_SOC_K), n_soc, t, SEED)
        r_broom = rms_heritable(lambda n, _rng: broom_parent(n), n_soc, t, SEED)
        r_tls = rms_heritable(lambda n, _rng: tls_parent(AMP_TLS_K), n_tls, t, SEED)
        r_iid = rms_iid(n_tls, t, SEED_IID)
        detail["amp"] = {"rms_soc_65": r_soc, "rms_broom_65": r_broom, "rms_tls_73": r_tls,
                         "rms_iid_73": r_iid, "trials": t}
        stats["soc_over_broom_65"] = r_soc / r_broom
        stats["tls_over_iid_73"] = r_tls / r_iid

    verdict = {
        key: {
            "value": value,
            "preregistered": PREREGISTERED[key],
            "window": list(WINDOWS[key]),
            "inside": bool(WINDOWS[key][0] <= value <= WINDOWS[key][1]),
        }
        for key, value in stats.items()
    }
    result = {"card": "derivations/Q-0011/F-01.formula.md", "mode": args.mode, "smoke": args.smoke,
              "delta": DELTA, "seed": SEED, "stats": stats, "verdict": verdict, "detail": detail}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not args.smoke:
        (HERE / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
