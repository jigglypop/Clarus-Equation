"""Audit 2 (dimension / decomposition): D == A + B and c == D/(n^2 mu2_eff), three ways.

  (i)  literal matrix definition   ||H kappa H||_F^2 and ||A W A^T||_F^2
  (ii) adversary O(n) formula      (independent re-derivation)
  (iii) card's check_families.tree_stats
  (iv) exact rational arithmetic
Exhaustive over every rooted shape with n <= 7, plus random shapes at n = 50.
"""
from __future__ import annotations

import itertools
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[3] / "verify" / "Q-0008" / "F-02"))

from a_core import stats_exact, stats_fast, stats_matrix  # noqa: E402
import a_fam as F  # noqa: E402
from check_families import tree_stats as card_stats  # noqa: E402

out = {"check": "decomposition_and_c_agreement"}
worst = {"fast_vs_matrix_D": 0.0, "fast_vs_matrix_At": 0.0, "fast_vs_card_D": 0.0,
         "fast_vs_card_At": 0.0, "fast_vs_card_c": 0.0, "exact_vs_fast_c": 0.0,
         "AplusB_vs_D": 0.0, "A_le_Atilde_violations": 0}
count = 0
for n in range(2, 8):
    for tail in itertools.product(*[range(i) for i in range(1, n)]):
        p = [-1] + list(tail)
        f = stats_fast(p)
        m = stats_matrix(p)
        cd = card_stats(p)
        ex = stats_exact(p)
        count += 1
        worst["fast_vs_matrix_D"] = max(worst["fast_vs_matrix_D"], abs(f["D"] - m["D"]) / (1 + m["D"]))
        worst["fast_vs_matrix_At"] = max(worst["fast_vs_matrix_At"], abs(f["n2_mu2_eff"] - m["n2_mu2_eff"]) / (1 + m["n2_mu2_eff"]))
        worst["fast_vs_card_D"] = max(worst["fast_vs_card_D"], abs(f["D"] - cd["D"]) / (1 + cd["D"]))
        worst["fast_vs_card_At"] = max(worst["fast_vs_card_At"], abs(f["n2_mu2_eff"] - cd["n2_mu2_eff"]) / (1 + cd["n2_mu2_eff"]))
        worst["fast_vs_card_c"] = max(worst["fast_vs_card_c"], abs(f["c"] - cd["c"]))
        worst["exact_vs_fast_c"] = max(worst["exact_vs_fast_c"], abs(float(ex["c"]) - f["c"]))
        worst["AplusB_vs_D"] = max(worst["AplusB_vs_D"], abs(float(ex["A"] + ex["B"] - ex["D"])))
        if f["A"] > f["n2_mu2_eff"] * (1 + 1e-12):
            worst["A_le_Atilde_violations"] += 1
out["exhaustive_n_le_7_shapes"] = count
out["worst_small"] = dict(worst)

rng = np.random.default_rng(20260902)
w2 = {"fast_vs_matrix_D": 0.0, "fast_vs_matrix_At": 0.0, "fast_vs_card_c": 0.0, "AplusB_vs_D": 0.0}
for _ in range(60):
    p = F.uniform_shape(50, rng)
    f = stats_fast(p)
    m = stats_matrix(p)
    cd = card_stats(p)
    ex = stats_exact(p)
    w2["fast_vs_matrix_D"] = max(w2["fast_vs_matrix_D"], abs(f["D"] - m["D"]) / (1 + m["D"]))
    w2["fast_vs_matrix_At"] = max(w2["fast_vs_matrix_At"], abs(f["n2_mu2_eff"] - m["n2_mu2_eff"]) / (1 + m["n2_mu2_eff"]))
    w2["fast_vs_card_c"] = max(w2["fast_vs_card_c"], abs(f["c"] - cd["c"]))
    w2["AplusB_vs_D"] = max(w2["AplusB_vs_D"], abs(float(ex["A"] + ex["B"] - ex["D"])))
out["random_n50"] = w2

# structured families: matrix vs fast at moderate n
w3 = 0.0
fams = {
    "hub_at_depth1_n60": F.hub_at_depth(60, 1),
    "spindle_n60": F.spindle(60, 10, 30),
    "comb_n60": F.comb(60, 3),
    "double_broom_n60": F.double_broom(60, 20),
    "kary3_d3": F.kary(3, 3),
    "split_stars_60_2": F.split_stars(60, [1, 1]),
    "split_chains_60_2": F.split_chains(60, 2),
    "power_profile_60": F.power_profile(60, 12, 8.0),
}
for name, p in fams.items():
    f, m = stats_fast(p), stats_matrix(p)
    w3 = max(w3, abs(f["D"] - m["D"]) / (1 + m["D"]), abs(f["n2_mu2_eff"] - m["n2_mu2_eff"]) / (1 + m["n2_mu2_eff"]))
out["structured_matrix_vs_fast_max_rel"] = w3
out["structured_c"] = {k: stats_fast(v)["c"] for k, v in fams.items()}

# closed forms claimed by the card
cf = []
for n in (2, 3, 5, 8, 13, 21, 64):
    st_c = stats_fast(F.chain(n))
    st_s = stats_fast(F.star(n))
    cf.append({
        "n": n,
        "chain_A_minus_F02closed": st_c["A"] - (n ** 2 - 1) * (2 * n ** 2 + 7) / 180,
        "chain_At_minus_closed": st_c["n2_mu2_eff"] - (n ** 2 - 1) * (n ** 2 + 1) / 60,
        "chain_B": st_c["B"],
        "chain_c": st_c["c"],
        "star_D_minus_closed": st_s["D"] - (n - 2 + 1 / n ** 2),
        "star_At_minus_closed": st_s["n2_mu2_eff"] - (n - 1) ** 3 / n ** 2,
        "star_c": st_s["c"],
    })
out["closed_forms"] = cf
out["closed_form_max_abs_err"] = max(max(abs(r[k]) for k in
                                         ("chain_A_minus_F02closed", "chain_At_minus_closed",
                                          "star_D_minus_closed", "star_At_minus_closed")) for r in cf)
print(json.dumps(out, ensure_ascii=False, indent=2))
(HERE / "a1_validate.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
