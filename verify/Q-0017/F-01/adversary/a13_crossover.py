"""a13: at which n does the middle branch 2/d_tree-1 become visible for d_tree=3 (a=2)?
Components (exact where possible):
  G part  = S_star^2 * ||H G H||_F^2 / n^2   (a3 exact O(h) formula + a10 exact spread recursion)
  rest    = measured total - G part (diagonal + ancestry fluctuation), fitted as c * n^p
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[3]
from a3_G_exponent import DG_fast  # noqa


def S_of(a, h, q=1.0):
    W = np.array([(d + 1) ** a for d in range(h)], dtype=float)
    S = np.zeros(h)
    S[0] = 1.0
    for d in range(1, h):
        Wp = W[d - 1]
        qe = q if Wp >= 2 else 0.0
        S[d] = 1.0 + (1.0 - qe / 2.0 - 1.0 / Wp) * S[d - 1]
    return S


def main():
    pj = json.loads((ROOT / "verify/Q-0017/F-01/predictions.json").read_text(encoding="utf-8"))
    out = {}
    for a, fam, hs in ((1, "L1", (16, 23, 32, 45, 64)), (2, "L2", (8, 12, 16, 20)), (3, "L3", (5, 6, 7, 8, 9))):
        st = pj["layered_stage"][fam]
        j = st["q"].index(1.0)
        rows = []
        for h, n in zip(hs, st["sizes"]):
            DG, ncont = DG_fast(float(a), h)
            S = S_of(a, h)
            Gpart = S[-1] ** 2 * DG / ncont ** 2
            tot = st["E_D_over_n2"][str(n)][j]
            rows.append({"h": h, "n": n, "total": tot, "G_part_Sstar2_DG": Gpart,
                         "rest": tot - Gpart, "G_frac": Gpart / tot, "S_star": float(S[-1])})
        lg = np.polyfit(np.log([r["n"] for r in rows]), np.log([r["rest"] for r in rows]), 1)
        gG = 2.0 / (a + 1.0) - 1.0
        cG = rows[0]["G_part_Sstar2_DG"] * rows[0]["n"] ** (-2 * gG)
        p, lc = float(lg[0]), float(lg[1])
        ncr = math.exp((lc - math.log(cG)) / (2 * gG - p)) if abs(2 * gG - p) > 1e-9 else float("inf")
        out[fam] = {"d_tree": a + 1, "rows": rows, "rest_exponent_p": p, "gamma_G_law": gG,
                    "n_where_G_part_equals_rest": ncr,
                    "card_claim_crossover": "n ~ 1e5 (card scope)"}
        print(f"{fam} d_tree={a+1}: G_frac={[round(r['G_frac'],4) for r in rows]} "
              f"rest ~ n^{p:.3f}  G ~ n^{2*gG:.3f}  crossover n={ncr:.3e}")
    (HERE / "a13_crossover.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
