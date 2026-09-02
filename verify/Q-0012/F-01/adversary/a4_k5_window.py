"""a4: is the K5 window a valid test, and does it hold for the laws that carry the signal?

(A) Seed sensitivity of the card's own K5 battery (run_form with sizes 3,5,8,12 and the three
    label configurations the card fixed).  The card froze the window max_rel_err <= 0.02 from an
    exploratory maximum of 7e-3.  If a fair share of seeds exceeds 0.02 then K5 can fire on a TRUE
    card (it only measures how small delta is, not whether the quadratic form is the right object).
(B) The same deterministic identity for spike64 / laplace / uniform labels at sizes where the
    spike law is not degenerate (n >= 24), iid and heritable(caterpillar k=6).
"""
import json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "verify" / "Q-0012" / "F-01"))
from check_cumulant import (linear_map, quadratic_tensor, gram_form, REFERENCE, run_form,
                            caterpillar, heritable, block_residual, quadratic_residual,
                            uniform_to_label, normal_cdf, DELTA, DELTA_FINE)

OUT = Path(__file__).parent
DISTS5 = ("gauss", "rademacher", "uniform", "laplace", "spike64")


def main():
    M = quadratic_tensor(linear_map())
    g0 = float(np.linalg.norm(gram_form(REFERENCE, REFERENCE)))
    res = {}

    maxima, ratios = [], []
    for k in range(40):
        block = run_form(M, g0, sizes=(3, 5, 8, 12), seed=90000 + 137 * k)
        maxima.append(block["max_rel_err_delta0005"])
        ratios.append(block["ratio_delta_scaling"])
    maxima = np.array(maxima); ratios = np.array(ratios)
    res["k5_seed_sensitivity"] = {
        "seeds": 40, "max_rel_err": {"min": float(maxima.min()), "median": float(np.median(maxima)),
                                     "max": float(maxima.max()), "p90": float(np.quantile(maxima, 0.9))},
        "frac_over_window_0.02": float(np.mean(maxima > 0.02)),
        "ratio": {"min": float(ratios.min()), "max": float(ratios.max()), "median": float(np.median(ratios))},
        "frac_ratio_outside_3_7": float(np.mean((ratios < 3.0) | (ratios > 7.0))),
    }
    print("(A) K5 battery over 40 seeds: max_rel_err median %.3e  p90 %.3e  max %.3e   P(>0.02)=%.3f"
          % (np.median(maxima), np.quantile(maxima, 0.9), maxima.max(), np.mean(maxima > 0.02)))
    print("    delta-scaling ratio range [%.2f, %.2f]  P(outside [3,7]) = %.3f"
          % (ratios.min(), ratios.max(), np.mean((ratios < 3.0) | (ratios > 7.0))))

    rng = np.random.default_rng(20260902 + 31337)
    par6 = caterpillar(6)
    rows = []
    for n, parent in ((24, None), (36, None), (36, par6), (64, None)):
        for rep in range(8):
            z = rng.standard_normal((n, 4, 4))
            u = normal_cdf(z)
            for dist in DISTS5:
                zeta = uniform_to_label(u, z, dist)
                labels = zeta if parent is None else heritable(parent, zeta)
                if float(np.max(np.abs(labels))) == 0.0:
                    continue
                row = {"n": n, "mode": "iid" if parent is None else "cat6", "dist": dist, "rep": rep,
                       "max_abs_label": float(np.max(np.abs(labels)))}
                ok = True
                for tag, d in (("d0005", DELTA), ("d0001", DELTA_FINE)):
                    act = block_residual(labels, d)
                    frm = quadratic_residual(labels, d, M, g0)
                    if not (math.isfinite(act) and frm > 0):
                        ok = False
                        break
                    row[tag] = {"actual": act, "form": frm, "rel_err": (act - frm) / frm}
                if ok:
                    rows.append(row)
    agg = {}
    for mode in ("iid", "cat6"):
        for dist in DISTS5:
            sel = [r for r in rows if r["mode"] == mode and r["dist"] == dist]
            if not sel:
                continue
            c = np.array([r["d0005"]["rel_err"] for r in sel])
            f = np.array([r["d0001"]["rel_err"] for r in sel])
            agg["%s_%s" % (mode, dist)] = {
                "count": len(sel), "max_abs_d0005": float(np.max(np.abs(c))),
                "mean_d0005": float(c.mean()), "rms_d0005": float(np.sqrt(np.mean(c ** 2))),
                "rms_d0001": float(np.sqrt(np.mean(f ** 2))),
                "ratio": float(np.sqrt(np.mean(c ** 2)) / np.sqrt(np.mean(f ** 2))),
                "max_abs_label": float(max(r["max_abs_label"] for r in sel)),
            }
            a = agg["%s_%s" % (mode, dist)]
            print("(B) %-5s %-11s N=%2d  max|rel| %8.2e  mean %+8.2e  rms %8.2e  ratio %5.2f  max|label| %5.2f"
                  % (mode, dist, a["count"], a["max_abs_d0005"], a["mean_d0005"], a["rms_d0005"],
                     a["ratio"], a["max_abs_label"]))
    res["heavy_tail_form"] = agg
    res["rows"] = rows
    (OUT / "a4_k5_window.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
