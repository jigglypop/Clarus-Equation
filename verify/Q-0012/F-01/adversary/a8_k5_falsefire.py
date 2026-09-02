"""a8: pin down the K5 false-fire rate and locate which configuration drives it.

K5 kills the card if run_form(sizes=(3,5,8,12), 3 label configurations)'s MAXIMUM relative error
at delta = 0.005 exceeds 0.02.  The battery is a random draw (12 configurations from one seed), so
the max-statistic has a sampling distribution.  The card calls it "표본오차가 없는 결정론 항등식".
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "verify" / "Q-0012" / "F-01"))
from check_cumulant import linear_map, quadratic_tensor, gram_form, REFERENCE, run_form

OUT = Path(__file__).parent


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    M = quadratic_tensor(linear_map())
    g0 = float(np.linalg.norm(gram_form(REFERENCE, REFERENCE)))
    maxima, ratios = [], []
    per_cell = defaultdict(list)
    argmax = defaultdict(int)
    for k in range(reps):
        b = run_form(M, g0, sizes=(3, 5, 8, 12), seed=500000 + 911 * k)
        maxima.append(b["max_rel_err_delta0005"])
        ratios.append(b["ratio_delta_scaling"])
        best, tag = -1.0, None
        for r in b["configurations"]:
            e = abs(r["delta0005"]["rel_err"])
            key = "n%d_%s" % (r["n"], r["labels"])
            per_cell[key].append(e)
            if e > best:
                best, tag = e, key
        argmax[tag] += 1
    maxima = np.array(maxima); ratios = np.array(ratios)
    res = {"reps": reps,
           "max_rel_err": {"min": float(maxima.min()), "q25": float(np.quantile(maxima, .25)),
                           "median": float(np.median(maxima)), "q75": float(np.quantile(maxima, .75)),
                           "p90": float(np.quantile(maxima, .9)), "max": float(maxima.max())},
           "false_fire_rate_window_0.02": float(np.mean(maxima > 0.02)),
           "false_fire_rate_if_window_0.05": float(np.mean(maxima > 0.05)),
           "false_fire_rate_if_window_0.10": float(np.mean(maxima > 0.10)),
           "ratio_range": [float(ratios.min()), float(ratios.max())],
           "ratio_outside_3_7": float(np.mean((ratios < 3) | (ratios > 7))),
           "per_configuration_median": {k: float(np.median(v)) for k, v in per_cell.items()},
           "per_configuration_p95": {k: float(np.quantile(v, .95)) for k, v in per_cell.items()},
           "argmax_counts": dict(argmax)}
    print("K5 max_rel_err over %d seeds: median %.4f  p90 %.4f  max %.4f" %
          (reps, np.median(maxima), np.quantile(maxima, .9), maxima.max()))
    print("false-fire P(max > 0.02) = %.3f   (0.05 -> %.3f, 0.10 -> %.3f)" %
          (np.mean(maxima > 0.02), np.mean(maxima > 0.05), np.mean(maxima > 0.10)))
    print("delta-scaling ratio in [%.2f, %.2f], P(outside [3,7]) = %.3f" %
          (ratios.min(), ratios.max(), np.mean((ratios < 3) | (ratios > 7))))
    for k in sorted(per_cell, key=lambda x: -np.median(per_cell[x])):
        print("   %-20s median %8.2e  p95 %8.2e   argmax %3d/%d"
              % (k, np.median(per_cell[k]), np.quantile(per_cell[k], .95), argmax.get(k, 0), reps))
    (OUT / "a8_k5_falsefire.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
