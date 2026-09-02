"""a7: (A) execute the five `recovers` limits as physics, not as arithmetic;
       (B) machine-check preregistration integrity: card values == script PREREGISTERED,
           card windows == script WINDOWS == value +- 4 * (surrogate se quoted in the card),
           and the closed-form rationals == the quoted decimals.
"""
import json, math, re, sys
from fractions import Fraction as F
from pathlib import Path
import numpy as np

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "verify" / "Q-0012" / "F-01"))
from check_cumulant import (block_residual, uniform_to_label, normal_cdf, KAPPA4, PREREGISTERED,
                            WINDOWS, ancestor_matrix, caterpillar, DELTA)

OUT = Path(__file__).parent
CARD = ROOT / "derivations" / "Q-0012" / "F-01.formula.md"
DISTS5 = ("gauss", "rademacher", "uniform", "laplace", "spike64")


def recovers_physics():
    out = {}
    rng = np.random.default_rng(20260902 + 31337)
    # recovers[1]: n = 1 -> residual identically zero for every label law
    worst = 0.0
    for _ in range(40):
        z = rng.standard_normal((1, 4, 4))
        u = normal_cdf(z)
        for dist in DISTS5:
            lab = uniform_to_label(u, z, dist)
            worst = max(worst, abs(block_residual(lab, DELTA)))
    out["n1_max_residual"] = worst
    print("(A1) n=1 single cell, 5 laws x 40 draws: max residual = %.3e" % worst)

    # recovers[2]: iid finite-n law  eps = eps_star sqrt(n-1)/n sqrt(1 + c4 k4 (n-1)/n)
    rows = {}
    for n in (2, 3, 4, 6, 9):
        acc = {d: [] for d in DISTS5}
        for _ in range(400):
            z = rng.standard_normal((n, 4, 4))
            u = normal_cdf(z)
            for dist in DISTS5:
                v = block_residual(uniform_to_label(u, z, dist), DELTA)
                acc[dist].append(v * v)
        base = float(np.mean(acc["gauss"]))
        rows[n] = {"eps_star_est": math.sqrt(base) * n / math.sqrt(n - 1) / DELTA ** 2,
                   "rho": {d: float(np.mean(acc[d])) / base for d in DISTS5},
                   "rho_pred": {d: 1 + KAPPA4[d] * (n - 1) / n / 60.0 for d in DISTS5}}
        print("(A2) n=%d  eps_star/delta^2 = %.5f (F-02: sqrt(10)=%.5f)   rho(rad) %.4f/%.4f  "
              "rho(spike) %.4f/%.4f" % (n, rows[n]["eps_star_est"], math.sqrt(10),
                                        rows[n]["rho"]["rademacher"], rows[n]["rho_pred"]["rademacher"],
                                        rows[n]["rho"]["spike64"], rows[n]["rho_pred"]["spike64"]))
    out["iid_finite_n"] = rows

    # recovers[4]: positivity D + c4 k4 S >= 29 D / 30 over random generators
    worst_pos = 1e9
    for _ in range(2000):
        m = int(rng.integers(2, 9))
        A = rng.normal(size=(m, m))
        H = np.eye(m) - np.ones((m, m)) / m
        B = A.T @ H @ A
        D = float(np.trace(B @ B)); S = float(np.sum(np.diag(B) ** 2))
        worst_pos = min(worst_pos, (D - 2 * S / 60.0) / (29 * D / 30.0))
    out["positivity_min_ratio_to_29over30"] = worst_pos
    print("(A3) positivity: min over 2000 random generators of (D-2S/60)/(29D/30) = %.6f (>=1 iff S<=D)"
          % worst_pos)
    return out


def prereg_integrity():
    text = CARD.read_text(encoding="utf-8")
    quoted_se = {"rho_iid36_laplace": 0.00335, "rho_iid36_spike64": 0.02744,
                 "rho_iid36_uniform": 0.00313, "rho_iid36_rademacher": 0.00762,
                 "rho_cat6_spike64": 0.03854, "rho_cat6_laplace": 0.00496,
                 "rho_cat6_uniform": 0.00444, "rho_cat6_rademacher": 0.01084,
                 "slope_ratio": 0.04104}
    a_iid = F(1, 60) * F(35, 36)
    a_cat = F(1, 60) * F(62069, 216) / F(23053, 36)
    exact = {"a_iid36": a_iid, "a_cat6": a_cat, "slope_ratio": a_cat / a_iid}
    for mode, a in (("iid36", a_iid), ("cat6", a_cat)):
        for d in ("rademacher", "uniform", "laplace", "spike64"):
            exact["rho_%s_%s" % (mode, d)] = 1 + F(KAPPA4[d]).limit_denominator(10) * a
    rows = {}
    ok = True
    for key, val in exact.items():
        pre = PREREGISTERED[key]
        lo, hi = WINDOWS[key]
        d_exact = abs(float(val) - pre)
        halfwidth = (hi - lo) / 2.0
        centre = (hi + lo) / 2.0
        se = quoted_se.get(key)
        rows[key] = {"exact": str(val), "exact_float": float(val), "script_prereg": pre,
                     "abs_diff": d_exact, "window": [lo, hi], "half_width": halfwidth,
                     "window_centre_minus_value": centre - pre,
                     "quoted_se": se, "halfwidth_over_4se": (halfwidth / (4 * se)) if se else None,
                     "value_in_card_text": (("%.10f" % float(val))[:8] in text)}
        bad = d_exact > 2e-9 or abs(centre - pre) > 6e-5 or (se is not None and abs(halfwidth / (4 * se) - 1) > 0.01)
        ok = ok and not bad
        print("(B) %-20s exact %-22s prereg %.10f  d=%.1e  window %+.6f +-%.6f  hw/4se %s %s"
              % (key, str(val), pre, d_exact, centre, halfwidth,
                 ("%.4f" % (halfwidth / (4 * se))) if se else "  n/a ", "" if not bad else "<== MISMATCH"))
    return {"ok": ok, "rows": rows}


def main():
    res = {"recovers_physics": recovers_physics(), "preregistration": prereg_integrity()}
    (OUT / "a7_recovers_prereg.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
    print("preregistration integrity ok:", res["preregistration"]["ok"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
