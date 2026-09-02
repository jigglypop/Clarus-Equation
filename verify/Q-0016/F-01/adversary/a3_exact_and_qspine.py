"""Q-0016 F-01 adversary a3:
(1) EXACT rational counterexample to the card sign claim ("conservation enlarges the centred residual"),
(2) bit-level replay of the pre-registered Q-spine table for b = 1..4 using the card's own tree stream,
(3) fraction of Q-spine trees on which the sign claim fails."""
from __future__ import annotations
import json, math, sys, time
from fractions import Fraction
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "verify" / "Q-0008" / "F-02"))
from a1_algebra import A_matrix, C_matrix, D_f02, D_split  # noqa: E402
from driver_numbers import qspine_block  # noqa: E402

OUT = HERE / "a3_exact_and_qspine.json"
R: dict = {}


def exact_D(parent, s=Fraction(1)):
    """D = ||H kappa H||_F^2 in exact rationals."""
    n = len(parent)
    ch = [[] for _ in range(n)]
    for v, p in enumerate(parent):
        if p >= 0:
            ch[p].append(v)
    anc = [None] * n

    def get(v):
        if anc[v] is None:
            anc[v] = {v} | (get(parent[v]) if parent[v] >= 0 else set())
        return anc[v]

    for v in range(n):
        get(v)
    A = [[Fraction(1) if u in anc[v] else Fraction(0) for u in range(n)] for v in range(n)]
    C = [[Fraction(0)] * n for _ in range(n)]
    for v in range(n):
        C[v][v] = Fraction(1)
    for kids in ch:
        k = len(kids)
        if k >= 2:
            for a in kids:
                for b in kids:
                    if a != b:
                        C[a][b] = -s / (k - 1)
    K = [[sum(A[v][i] * sum(C[i][j] * A[w][j] for j in range(n)) for i in range(n)) for w in range(n)]
         for v in range(n)]
    col = [sum(K[v][w] for v in range(n)) / n for w in range(n)]
    tot = sum(col) / n
    M = [[K[v][w] - col[w] - col[v] + tot for w in range(n)] for v in range(n)]
    return sum(M[v][w] * M[v][w] for v in range(n) for w in range(n))


def exact_D_f02(parent):
    return exact_D(parent, Fraction(0))


CEX = [
    ("chain_L6_terminal_fork_n8", [-1, 0, 1, 2, 3, 4, 5, 5]),
    ("chain_L5_terminal_trident_n8", [-1, 0, 1, 2, 3, 4, 4, 4]),
    ("chain_L7_terminal_fork_n9", [-1, 0, 1, 2, 3, 4, 5, 6, 6]),
    ("worst_n12", [-1, 0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 6]),
    ("worst_n13", [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 7]),
]
R["exact_counterexamples"] = []
for name, p in CEX:
    ds, df = exact_D(p), exact_D_f02(p)
    R["exact_counterexamples"].append({
        "name": name, "parent": p, "n": len(p),
        "D_split_exact": str(ds), "D_f02_exact": str(df),
        "D_split_float": float(ds), "D_f02_float": float(df),
        "ratio": float(ds / df), "difference_split_minus_f02": str(ds - df),
        "sign_claim_violated": bool(ds < df),
        "float_check_split": D_split(p), "float_check_f02": D_f02(p),
    })

# ---------------------------------------------------------------- (2) Q-spine replay b=1..4
TREE_SEED = 20260902 + 1000
TRIALS = 200_000
t0 = time.time()
rng = np.random.default_rng(TREE_SEED)
rep = {}
frac_below = {}
for b in (1, 2, 3, 4):
    dn2_s = np.empty(TRIALS)
    dn2_f = np.empty(TRIALS)
    nn = np.empty(TRIALS)
    below = 0
    tb = time.time()
    for t in range(TRIALS):
        p = qspine_block(b, rng)
        n = len(p)
        nn[t] = n
        if n == 1:
            dn2_s[t] = dn2_f[t] = 0.0
            continue
        A = A_matrix(p)
        HA = A - A.mean(axis=0, keepdims=True)
        Ks = HA @ C_matrix(p) @ HA.T
        Kf = HA @ HA.T
        ds = float(np.sum(Ks * Ks))
        df = float(np.sum(Kf * Kf))
        dn2_s[t] = ds / (n * n)
        dn2_f[t] = df / (n * n)
        if df > 0 and ds < df - 1e-12:
            below += 1
    rep[str(b)] = {
        "E_n": float(nn.mean()),
        "E_D_over_n2_split": float(dn2_s.mean()),
        "se_split": float(dn2_s.std(ddof=1) / math.sqrt(TRIALS)),
        "E_D_over_n2_f02_replay": float(dn2_f.mean()),
        "se_f02": float(dn2_f.std(ddof=1) / math.sqrt(TRIALS)),
        "ratio": float(dn2_s.mean() / dn2_f.mean()) if dn2_f.mean() > 0 else None,
        "wall_s": time.time() - tb,
    }
    frac_below[str(b)] = below / TRIALS
    print("depth", b, rep[str(b)]["E_D_over_n2_split"], rep[str(b)]["E_D_over_n2_f02_replay"],
          "frac_sign_violating", frac_below[str(b)], "wall", round(rep[str(b)]["wall_s"], 1), flush=True)
R["qspine_replay"] = rep
R["qspine_fraction_D_split_lt_D_f02"] = frac_below
R["qspine_replay_wall_s"] = time.time() - t0

card = json.loads((HERE.parent / "predictions.json").read_text(encoding="utf-8"))
f02 = json.loads((ROOT / "verify" / "Q-0008" / "F-02" / "predictions.json").read_text(encoding="utf-8"))
cmp_rows = []
for b in ("1", "2", "3", "4"):
    cmp_rows.append({
        "b": b,
        "card_split": card["qspine"][b]["E_D_over_n2_split"],
        "adversary_split": rep[b]["E_D_over_n2_split"],
        "abs_diff_split": abs(card["qspine"][b]["E_D_over_n2_split"] - rep[b]["E_D_over_n2_split"]),
        "card_f02_replay": card["qspine"][b]["E_D_over_n2_f02_replay"],
        "f02_official": f02["qspine"][b]["E_D_over_n2"],
        "adversary_f02": rep[b]["E_D_over_n2_f02_replay"],
        "abs_diff_f02_vs_official": abs(rep[b]["E_D_over_n2_f02_replay"] - f02["qspine"][b]["E_D_over_n2"]),
    })
R["replay_comparison"] = cmp_rows
R["max_abs_diff_split"] = max(r["abs_diff_split"] for r in cmp_rows)
R["max_abs_diff_f02_vs_official"] = max(r["abs_diff_f02_vs_official"] for r in cmp_rows)

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
for r in R["exact_counterexamples"]:
    print(r["name"], r["parent"], "D_split=", r["D_split_exact"], "D_f02=", r["D_f02_exact"],
          "ratio=%.6f" % r["ratio"], "VIOLATED" if r["sign_claim_violated"] else "ok")
print("replay max|diff| split:", R["max_abs_diff_split"], " f02 vs official:", R["max_abs_diff_f02_vs_official"])
print(json.dumps(R["replay_comparison"], indent=1, default=float))
print("qspine frac sign-violating:", json.dumps(frac_below))
