"""Q-0016 F-01 adversary a8b: every NEW numeric claim introduced by card revision 2."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "verify" / "Q-0008" / "F-02"))
from a1_algebra import D_f02, D_split, cbin  # noqa: E402

OUT = HERE / "a8b_new_claims.json"
R: dict = {}

# claim scope[2]: "K_A2 window allows s in [0.71,1.00]; amplitude ratio 4.082@0.70 -> 4.504@1.00;
#                  slope window requires s >~ 0.80"
sizes = (7, 15, 31, 63)
rows = []
for s in np.linspace(0.0, 1.0, 101):
    grid = [math.sqrt(D_split(cbin(int(round(math.log2(m + 1))) - 1), float(s))) / m for m in sizes]
    r15 = math.sqrt(D_split(cbin(3), float(s)) / 14.0)
    sl = float(np.polyfit(np.log(sizes), np.log(grid), 1)[0])
    rows.append({"s": round(float(s), 2), "ratio_15": r15, "slope": sl,
                 "pass_ratio": 4.031 <= r15 <= 4.977, "pass_slope": 0.111 <= sl <= 0.180})
pr = [r["s"] for r in rows if r["pass_ratio"]]
ps = [r["s"] for r in rows if r["pass_slope"]]
both = [r["s"] for r in rows if r["pass_ratio"] and r["pass_slope"]]
R["s_band"] = {
    "ratio_window_only": [min(pr), max(pr)], "slope_window_only": [min(ps), max(ps)],
    "both": [min(both), max(both)], "n_both": len(both),
    "ratio_15_at_0.70": next(r["ratio_15"] for r in rows if r["s"] == 0.70),
    "ratio_15_at_0.71": next(r["ratio_15"] for r in rows if r["s"] == 0.71),
    "ratio_15_at_1.00": next(r["ratio_15"] for r in rows if r["s"] == 1.00),
    "slope_at_0.71": next(r["slope"] for r in rows if r["s"] == 0.71),
    "slope_at_0.80": next(r["slope"] for r in rows if r["s"] == 0.80),
    "card_says_ratio_4.082_at_0.70": 4.082, "card_says_band_0.71_to_1.00": True,
    "card_says_slope_needs_s_ge_0.80": True,
}

# claim scope[6]: "chain of length L with a terminal 2-fork: ratio<1 for L>=7 (n>=11), min 0.964 at L=10-11"
famA = []                                   # simple terminal fork: chain(L) + 2 leaves, n = L+2
for L in range(1, 26):
    par = [-1] + list(range(L - 1))
    tip = L - 1
    par = par + [tip, tip]
    famA.append({"L": L, "n": len(par), "ratio": D_split(par) / D_f02(par)})
famB = []                                   # chain(L) + 2 children, each carrying one extra leaf, n = L+4
for L in range(2, 26):
    par = [-1] + list(range(L - 1))
    tip = L - 1
    par += [tip, tip]
    a, b = L, L + 1
    par += [a, b]
    famB.append({"L": L, "n": len(par), "ratio": D_split(par) / D_f02(par)})
R["family_A_terminal_fork"] = {
    "first_L_with_ratio_below_1": next(r["L"] for r in famA if r["ratio"] < 1),
    "first_n_with_ratio_below_1": next(r["n"] for r in famA if r["ratio"] < 1),
    "min_ratio": min(r["ratio"] for r in famA),
    "argmin_L": min(famA, key=lambda r: r["ratio"])["L"], "rows": famA[:14]}
R["family_B_fork_with_extra_leaves"] = {
    "first_L_with_ratio_below_1": next(r["L"] for r in famB if r["ratio"] < 1),
    "first_n_with_ratio_below_1": next(r["n"] for r in famB if r["ratio"] < 1),
    "min_ratio": min(r["ratio"] for r in famB),
    "argmin_L": min(famB, key=lambda r: r["ratio"])["L"], "rows": famB[:14]}
R["scope6_family_sentence"] = {
    "card_text": "length-L chain + terminal two-fork: ratio<1 for L>=7 (n>=11), min 0.964 at L=10-11",
    "matches_family_A": False, "matches_family_B": True,
    "note": ("The numbers 'L>=7 (n>=11)' and 'min 0.964 at L=10-11' are family B (each fork branch also "
             "carries one extra leaf, n = L+4). The words describe family A (a bare terminal 2-fork, "
             "n = L+2), for which the first violation is L=6 (n=8) with ratio 0.97469 -- the card's own "
             "headline counterexample. Numbers correct, family label wrong.")}

# claim scope[7]: complete-binary bound O(1), values 2.63/2.35/2.18 at n=63/127/255, limit sqrt2+1/2
cb = {}
for d in (5, 6, 7):
    p = cbin(d)
    n = len(p)
    df = D_f02(p)
    cb[str(n)] = {"bound": 2 * n / math.sqrt(df) + n * n / df, "ratio": D_split(p) / df}
R["complete_binary_bound"] = {"values": cb, "limit_sqrt2_plus_half": math.sqrt(2) + 0.5}

# claim recovers[3]: two-species read as siblings gives exactly 4x  (n = 8, 10, 36 matrix check)
tw = {}
for n in (8, 10, 36):
    for a in (n // 4, n // 2):
        b = n - a
        H = np.eye(n) - np.ones((n, n)) / n
        kin = np.zeros((n, n))
        kin[:a, :a] = 1.0
        kin[a:, a:] = 1.0
        u = np.concatenate([np.ones(a), -np.ones(b)])
        kco = np.outer(u, u)
        di = float(np.sum((H @ kin @ H) ** 2))
        dc = float(np.sum((H @ kco @ H) ** 2))
        tw["n%d_a%d" % (n, a)] = {"D_ind": di, "D_sib": dc, "ratio": dc / di,
                                  "closed_4n2p2": 4 * n * n * (a / n) ** 2 * (1 - a / n) ** 2}
R["two_species_4x"] = tw

# claim verify[22]/[23]/[24]
R["verify_new_items"] = {
    "v22_Rsq": {"lhs_at_k3_s0.4": ((3 + 1) + (3 + 1) * 3 * (-0.4 / 3)) / (3 + 1), "rhs": 1 - 0.4},
    "v23_two_species": {"lhs": (4 * 7 * 0.3 * 0.7) ** 2, "rhs": 4 * (4 * 49 * 0.09 * 0.49)},
    "v24_counterexample": 2465 / 64 - 2529 / 64 + 1,
}

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print("s band:", json.dumps(R["s_band"], indent=1, default=float))
print("famA:", json.dumps({k: v for k, v in R["family_A_terminal_fork"].items() if k != "rows"}, default=float))
print("famB:", json.dumps({k: v for k, v in R["family_B_fork_with_extra_leaves"].items() if k != "rows"}, default=float))
print("cb bound:", json.dumps(R["complete_binary_bound"], default=float))
print("two species:", json.dumps(R["two_species_4x"], default=float))
print("verify new:", json.dumps(R["verify_new_items"], default=float))
