"""Q-0016 F-01 adversary a6: reverse consistency gate (symmetry of the circular exclusion),
s-tolerance of the pre-registered kill windows, and the family ratios quoted in the card."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "verify" / "Q-0008" / "F-02"))
from a1_algebra import A_matrix, C_matrix, D_f02, D_split, cbin  # noqa: E402
from driver_numbers import qspine_block, uniform_rooted_tree  # noqa: E402

OUT = HERE / "a6_reverse_gate_and_s.json"
R: dict = {}

a4 = json.loads((HERE / "a4_kill_audit.json").read_text(encoding="utf-8"))
rev = {}
for nkey, blk in a4["physics_mc_mini_binary"].items():
    nn = int(nkey)
    pb = cbin(int(round(math.log2(nn + 1))) - 1)
    pf = math.sqrt(D_f02(pb) / (nn - 1))
    ps = math.sqrt(D_split(pb) / (nn - 1))
    rev[nkey] = {"sampler_used": "SPLIT (the F-01 P_micro)", "trials": blk["trials"],
                 "observed_ratio_over_iid": blk["ratio_obs"],
                 "F01_prediction": ps, "F02_prediction": pf,
                 "rel_dev_of_F01": blk["ratio_obs"] / ps - 1,
                 "rel_dev_of_F02": blk["ratio_obs"] / pf - 1,
                 "F02_outside_a_5pct_window": abs(blk["ratio_obs"] / pf - 1) > 0.05}
R["reverse_consistency_gate"] = rev
R["reverse_gate_note"] = ("The card records F-02 K1/K3 numbers as a measurement that puts F-01 outside "
                          "the window. The mirror experiment (same physics renderer, F-01 sampler) puts "
                          "F-02 outside by a larger margin. The gate is symmetric and therefore carries "
                          "no evidence about which P_micro is physical: it reports which sampler was coded.")

sizes = (7, 15, 31, 63)
rows = []
for s in np.linspace(0.0, 1.0, 101):
    grid = [math.sqrt(D_split(cbin(int(round(math.log2(m + 1))) - 1), float(s))) / m for m in sizes]
    r15 = math.sqrt(D_split(cbin(3), float(s)) / 14.0)
    sl = float(np.polyfit(np.log(sizes), np.log(grid), 1)[0])
    rows.append({"s": float(s), "ratio_15": r15, "slope": sl,
                 "pass_ratio": 4.031 <= r15 <= 4.977, "pass_slope": 0.111 <= sl <= 0.180})
ok = [r for r in rows if r["pass_ratio"] and r["pass_slope"]]
R["s_tolerance_K_A2"] = {"s_min_passing": min((r["s"] for r in ok), default=None),
                         "s_max_passing": max((r["s"] for r in ok), default=None),
                         "n_passing_of_101": len(ok), "rows_every_10": rows[::10]}


def star_of_chains(k):
    par = [-1]
    for _ in range(k):
        for i in range(k):
            par.append(0 if i == 0 else len(par) - 1)
    return par


def caterpillar(k):
    par = [-1] + list(range(k - 1))
    for v in range(k):
        for _ in range(k - 1):
            par.append(v)
    return par


fam = {"star_of_chains": {}, "caterpillar": {}}
for k in (3, 5, 8, 11, 16):
    q = star_of_chains(k)
    fam["star_of_chains"][str(len(q))] = {"k": k, "ratio": D_split(q) / D_f02(q)}
for k in (3, 5, 8, 11):
    q = caterpillar(k)
    fam["caterpillar"][str(len(q))] = {"k": k, "ratio": D_split(q) / D_f02(q)}
R["families"] = fam

rng2 = np.random.default_rng(20261902)
cay = {}
for m in (8, 16, 32, 64, 128):
    trials = 3000 if m <= 32 else 1000
    rs = [D_split(q) / D_f02(q) for q in (uniform_rooted_tree(m, rng2) for _ in range(trials))]
    cay[str(m)] = {"trials": trials, "E_paired_ratio": float(np.mean(rs)),
                   "se": float(np.std(rs, ddof=1) / math.sqrt(trials))}
R["cayley_paired_ratio"] = cay

rng3 = np.random.default_rng(424242)
below = 0
tot = 0
ratios = []
for _ in range(20000):
    q = qspine_block(8, rng3)
    m = len(q)
    if m == 1:
        continue
    A = A_matrix(q)
    HA = A - A.mean(axis=0, keepdims=True)
    Ks = HA @ C_matrix(q) @ HA.T
    Kf = HA @ HA.T
    ds = float(np.sum(Ks * Ks))
    df = float(np.sum(Kf * Kf))
    if df <= 0:
        continue
    tot += 1
    ratios.append(ds / df)
    if ds < df - 1e-12:
        below += 1
R["qspine_b8_sign"] = {"trees": tot, "fraction_D_split_lt_D_f02": below / tot,
                       "min_paired_ratio": float(np.min(ratios)), "mean_paired_ratio": float(np.mean(ratios)),
                       "seed": 424242}

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print("reverse gate:", json.dumps(R["reverse_consistency_gate"], indent=1, default=float))
print("s tolerance:", R["s_tolerance_K_A2"]["s_min_passing"], "-", R["s_tolerance_K_A2"]["s_max_passing"],
      "(%d/101)" % R["s_tolerance_K_A2"]["n_passing_of_101"])
print("families:", json.dumps(R["families"], default=float))
print("cayley:", json.dumps({k: round(v["E_paired_ratio"], 4) for k, v in cay.items()}))
print("qspine b8 sign:", json.dumps(R["qspine_b8_sign"], default=float))
