"""Q-0016 F-01 adversary a2: the sign claim (conservation ENLARGES the centred residual),
the two-species limit, and the incomparable-pair mechanism.  Reuses a1 primitives only."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from a1_algebra import (A_matrix, B_matrix, C_matrix, D_of, D_f02, D_split, chain, cbin,  # noqa: E402
                        shapes, star, random_tree)

OUT = HERE / "a2_sign_and_limits.json"
R: dict = {}

# ---------------------------------------------------------------- (1) full census of sign violations
viol = []
tot = 0
by_n = {}
for n in range(2, 14):
    cnt_n = 0
    bad_n = 0
    for p in shapes(n):
        ds, df = D_split(p), D_f02(p)
        cnt_n += 1
        tot += 1
        if ds < df - 1e-9:
            bad_n += 1
            viol.append({"n": n, "parent": list(p), "D_split": ds, "D_f02": df, "ratio": ds / df})
    by_n[str(n)] = {"shapes": cnt_n, "violations": bad_n}
R["census_by_n"] = by_n
R["total_shapes"] = tot
R["total_violations"] = len(viol)
viol.sort(key=lambda r: r["ratio"])
R["worst_10"] = viol[:10]
R["smallest_n_with_violation"] = min((v["n"] for v in viol), default=None)


def describe(parent):
    n = len(parent)
    ch = [[] for _ in range(n)]
    for v, p in enumerate(parent):
        if p >= 0:
            ch[p].append(v)
    depth = [0] * n
    for v in range(1, n):
        depth[v] = depth[parent[v]] + 1
    ks = sorted(len(c) for c in ch if len(c) >= 2)
    return {"n": n, "max_depth": max(depth), "branch_degrees": ks,
            "leaves": sum(1 for c in ch if not c)}


R["worst_10_shape"] = [describe(v["parent"]) for v in viol[:10]]
sm = [v for v in viol if v["n"] == R["smallest_n_with_violation"]]
R["smallest_n_examples"] = [{"parent": v["parent"], "ratio": v["ratio"], **describe(v["parent"])} for v in sm[:5]]

# ---------------------------------------------------------------- (2) mechanism: deep chain + late branch family
fam = []
for L in range(2, 26):
    # chain of length L, then the last vertex gets 2 children each carrying 1 extra leaf
    parent = [-1] + list(range(L - 1))
    tip = L - 1
    parent += [tip, tip]
    a, b = L, L + 1
    parent += [a, b]
    ds, df = D_split(parent), D_f02(parent)
    fam.append({"L": L, "n": len(parent), "D_split": ds, "D_f02": df, "ratio": ds / df})
R["deep_chain_with_late_fork"] = fam
R["deep_chain_ratio_min"] = min(f["ratio"] for f in fam)
R["deep_chain_ratio_at_L25"] = fam[-1]["ratio"]

# pure: chain of length L then a fork of 2 leaves (T-shape)
fam2 = []
for L in range(1, 40):
    parent = [-1] + list(range(L - 1)) if L >= 2 else [-1]
    tip = L - 1
    parent = parent + [tip, tip]
    ds, df = D_split(parent), D_f02(parent)
    fam2.append({"L": L, "n": len(parent), "ratio": ds / df, "D_split": ds, "D_f02": df})
R["chain_with_terminal_fork"] = {"min_ratio": min(f["ratio"] for f in fam2),
                                 "argmin_L": min(fam2, key=lambda f: f["ratio"])["L"],
                                 "rows": fam2[:40]}

# ---------------------------------------------------------------- (3) two-species limit: is the card's
# "tree-free, therefore invariant" claim consistent with its own axiom if the two species are siblings?
two = {}
for n in (8, 16, 36, 64):
    for p_frac in (0.25, 0.5, 0.75):
        a = int(round(p_frac * n))
        b = n - a
        if a < 1 or b < 1:
            continue
        u = np.concatenate([np.ones(a), np.ones(b)])
        # F-02 / card recovers[3]: kappa = 1_a 1_a^T (+) 1_b 1_b^T  (independent species labels)
        k_ind = np.zeros((n, n))
        k_ind[:a, :a] = 1.0
        k_ind[a:, a:] = 1.0
        # split-conserving version if the two species are the k=2 children of one split (corr -1)
        v = np.concatenate([np.ones(a), -np.ones(b)])
        k_cons = np.outer(v, v)
        two[f"n{n}_p{p_frac}"] = {
            "n": n, "p": a / n,
            "D_independent": D_of(k_ind), "card_closed_4n2p2(1-p)2": 4 * n * n * (a / n) ** 2 * (1 - a / n) ** 2,
            "D_conserved_if_species_are_siblings": D_of(k_cons),
            "ratio_conserved_over_independent": D_of(k_cons) / D_of(k_ind) if D_of(k_ind) > 0 else None,
        }
R["two_species"] = two

# ---------------------------------------------------------------- (4) n=1 and chain / s->0 limits, exactly
R["n1"] = {"D_split": D_split([-1]), "D_f02": D_f02([-1])}
R["chain_limit_max_abs_err"] = max(
    abs(D_split(chain(n)) - (n * n - 1) * (2 * n * n + 7) / 180.0) for n in range(2, 40))
rng = np.random.default_rng(20260902)
s0 = 0.0
R["s_to_0_max_abs_err_random_trees"] = max(
    abs(D_split(p, s0) - D_f02(p)) for p in [random_tree(n, rng) for n in (3, 5, 8, 13, 21) for _ in range(30)])

# ---------------------------------------------------------------- (5) relative-correction bound, stress
worst = {"slack": math.inf}
for n in range(2, 14):
    for p in shapes(n):
        df = D_f02(p)
        if df <= 0:
            continue
        ds = D_split(p)
        bnd = 2 * n / math.sqrt(df) + n * n / df
        slack = bnd - abs(ds / df - 1)
        if slack < worst["slack"]:
            worst = {"slack": slack, "n": n, "parent": list(p), "bound": bnd, "lhs": abs(ds / df - 1)}
R["bound_min_slack"] = worst

# is the bound O(1/depth) for complete binary?  (depth -> inf but D ~ 2 n^2, not n^2 depth^2)
cb = {}
for d in range(1, 11):
    p = cbin(d)
    n = len(p)
    df = D_f02(p)
    cb[str(n)] = {"depth": d, "D_f02_over_n2": df / n ** 2, "D_f02_over_n2_depth2": df / (n ** 2 * d ** 2),
                  "bound": 2 * n / math.sqrt(df) + n * n / df,
                  "ratio_minus_1": D_split(p) / df - 1}
R["complete_binary_condition_C"] = cb

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print("census:", json.dumps(by_n))
print("total shapes", tot, "violations", len(viol), "smallest n", R["smallest_n_with_violation"])
print("worst:", json.dumps(R["worst_10"][:3], default=float))
print("worst shape:", json.dumps(R["worst_10_shape"][:3]))
print("chain+terminal fork min ratio:", R["chain_with_terminal_fork"]["min_ratio"], "at L=", R["chain_with_terminal_fork"]["argmin_L"])
print("deep chain late fork min ratio:", R["deep_chain_ratio_min"], "L25:", R["deep_chain_ratio_at_L25"])
print("two species:", json.dumps(R["two_species"], default=float, indent=1)[:800])
print("n1:", R["n1"], "chain err:", R["chain_limit_max_abs_err"], "s0 err:", R["s_to_0_max_abs_err_random_trees"])
print("bound min slack:", json.dumps(R["bound_min_slack"], default=float))
print("cbin cond C:", json.dumps(R["complete_binary_condition_C"], default=float))
