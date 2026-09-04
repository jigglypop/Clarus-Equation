"""a12: does the E-018 kernel law eps_bar^2 = eps_star^2 ||H kappa H||_F^2 / n^2 (eps_star^2 = 10 delta^4)
hold for the DAG merge kernel?  Ladder step 1a asserts it; K1 is supposed to test it.

Run OFF the pre-registered grid and OFF the pre-registered seed so that the card kill statistic
(k1_slope_q1_f02grid, Cayley n in {8,16,32,64,128}, seed 20260902, 256 trials) stays unobserved:
sizes n in {12,24,48,96}, seed 424242, 200 trials.  For each configuration compute BOTH the physical
block residual and the exact D of that same configuration (paired), then test
    mean(residual^2) / (eps_star^2 mean(D)/n^2)  ==  1.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "verify" / "Q-0008" / "F-02"))
sys.path.insert(0, str(ROOT / "verify" / "Q-0017" / "F-01"))
from check_modes import DELTA, block_residual  # noqa
from driver_numbers import uniform_rooted_tree  # noqa
import predict_merge_gamma as P  # noqa
import check_merge as CM  # noqa

SIZES = (12, 24, 48, 96)
TRIALS = 200
SEED = 424242
EPS_STAR2 = 10.0 * DELTA ** 4


def main():
    out = {"delta": DELTA, "eps_star2": EPS_STAR2, "sizes": list(SIZES), "trials": TRIALS,
           "seed": SEED, "note": "off the pre-registered K1 grid and seed on purpose", "rows": {}}
    for q in (0.0, 1.0):
        rows = []
        rng = np.random.default_rng(SEED + int(10 * q))
        for n in SIZES:
            res2, Ds = [], []
            for _ in range(TRIALS):
                parent = uniform_rooted_tree(n, rng)
                level_list, widths, depth, u, r = P.merge_draws(parent, rng)
                lab = CM.merge_labels(parent, level_list, u, r, q, rng.normal(size=(n, 4, 4)))
                v = block_residual(lab, DELTA)
                if not math.isfinite(v):
                    continue
                res2.append(v * v)
                Ds.append(P.kernel_D(parent, level_list, u, r, q)[0])
            m_res2 = float(np.mean(res2))
            m_D = float(np.mean(Ds))
            pred = EPS_STAR2 * m_D / n ** 2
            rows.append({"n": n, "rms_measured": math.sqrt(m_res2), "rms_predicted": math.sqrt(pred),
                         "ratio_meas_over_pred": math.sqrt(m_res2 / pred),
                         "E_D_over_n2": m_D / n ** 2,
                         "se_rms": float(np.std(res2, ddof=1) / math.sqrt(len(res2)) / (2 * math.sqrt(m_res2)))})
        sl_m = float(np.polyfit(np.log([r["n"] for r in rows]), np.log([r["rms_measured"] for r in rows]), 1)[0])
        sl_p = float(np.polyfit(np.log([r["n"] for r in rows]), np.log([r["rms_predicted"] for r in rows]), 1)[0])
        out["rows"][str(q)] = {"rows": rows, "slope_measured": sl_m, "slope_kernel_predicted": sl_p,
                              "slope_gap": sl_m - sl_p}
        print(f"q={q}: ratio meas/pred = {[round(r['ratio_meas_over_pred'], 4) for r in rows]}  "
              f"slope_meas={sl_m:+.4f} slope_kernel={sl_p:+.4f} gap={sl_m-sl_p:+.4f}", flush=True)
    (HERE / "a12_kernel_law_dag.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
