"""a7: kill-window audit.

(1) Do the card predicts values equal predictions.json exactly (no post-hoc edit)?
(2) Does each kill window contain the value the FORMULA itself predicts, and does it exclude the
    discriminating alternative?  A window that excludes its own theory prediction tests the machine,
    not the theory.
(3) Wall-clock cost of each check_merge.py mode (kill executability, 5 min budget).
"""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "verify" / "Q-0008" / "F-02"))
sys.path.insert(0, str(ROOT / "verify" / "Q-0017" / "F-01"))
import check_merge as CM  # noqa
import predict_merge_gamma as P  # noqa
from driver_numbers import uniform_rooted_tree  # noqa

OUT = {}
pj = json.loads((ROOT / "verify/Q-0017/F-01/predictions.json").read_text(encoding="utf-8"))
gs, ps, ls = pj["grid_stage"], pj["plateau_stage"], pj["layered_stage"]


def fit(ns, ys):
    return float(np.polyfit(np.log(np.asarray(ns, float)), 0.5 * np.log(np.asarray(ys, float)), 1)[0])


def main():
    j1g = gs["q"].index(1.0)
    j1p = ps["q"].index(1.0)
    src = {
        "k1_slope_q1_f02grid": gs["gamma_grid"][j1g],
        "k2_slope_q1_plateaugrid": ps["gamma_K2grid"][j1p],
        "k2_ratio_1024_q1_over_iid": ps["ratio_to_iid_1024"][j1p],
        "k2_ratio_1024_q05_over_q1": math.sqrt(ps["E_D_over_n2"]["1024"][ps["q"].index(0.5)]
                                               / ps["E_D_over_n2"]["1024"][j1p]),
        "k3_slope_L2_q1": ls["L2"]["gamma_K3grid"][ls["L2"]["q"].index(1.0)],
        "k5_slope_L1_q1": fit([136, 276, 528, 1035],
                              [ls["L1"]["E_D_over_n2"][str(n)][ls["L1"]["q"].index(1.0)]
                               for n in (136, 276, 528, 1035)]),
    }
    OUT["prereg_vs_predictions_json"] = {
        k: {"card_script": CM.PREREGISTERED[k], "predictions_json": v, "abs_diff": abs(CM.PREREGISTERED[k] - v)}
        for k, v in src.items()}

    # formula's own asymptotic prediction for each statistic, and the discriminating alternative
    law = {"k1_slope_q1_f02grid": ("Cayley d_tree=2 -> 0", 0.0, "q=0 exact 0.5302"),
           "k2_slope_q1_plateaugrid": ("Cayley d_tree=2 -> 0", 0.0, "q=0 exact 0.5061 / iid -0.50"),
           "k2_ratio_1024_q1_over_iid": ("no closed prediction (amplitude)", None, "q=0 262.86 / aligned 1.0"),
           "k2_ratio_1024_q05_over_q1": ("S_star = 2/q mechanism -> 2.0", 2.0, "no mechanism 1.0"),
           "k3_slope_L2_q1": ("d_tree=3 -> -1/3", -1.0 / 3.0, "q=0 +0.2695 / floor -0.50"),
           "k5_slope_L1_q1": ("cone d_tree=2 -> 0", 0.0, "q=0 0.504")}
    rows = {}
    for k, (desc, val, alt) in law.items():
        lo, hi = CM.WINDOWS[k]
        rows[k] = {"window": [lo, hi], "prereg_finite_size_value": CM.PREREGISTERED[k],
                   "formula_asymptotic": desc, "formula_value": val,
                   "formula_value_inside_window": (None if val is None else bool(lo <= val <= hi)),
                   "alternative": alt}
    # K4 (tree-only)
    rows["k4_gamma_cone_q_independence"] = {"window": list(P.K4_WINDOW), "prereg_finite_size_value": 0.0,
                                            "formula_asymptotic": "0 for every q>0", "formula_value": 0.0,
                                            "formula_value_inside_window": True,
                                            "alternative": "q-dependent sign change +-0.3",
                                            "already_seen_smoke_offset77": [-0.007, -0.040, -0.027]}
    OUT["window_vs_formula"] = rows

    # (3) timing of the physical MC kill modes
    tm = {}
    for n, T in ((128, 6), (512, 4), (1024, 3), (1496, 2)):
        rng = np.random.default_rng(11)
        t0 = time.time()
        for _ in range(T):
            CM.sample_merge(lambda g, n=n: uniform_rooted_tree(n, g), 1.0, rng, CM.DELTA)
        tm[str(n)] = (time.time() - t0) / T
    OUT["per_trial_seconds_cayley"] = tm
    est = {"k1": sum(tm["128"] * (n / 128) ** 1.0 for n in CM.K1_SIZES) * CM.K1_TRIALS / 60.0}
    est["k2_min"] = (sum(tm.get(str(n), tm["1024"] * n / 1024) for n in CM.K2_SIZES) * CM.K2_TRIALS
                     + 2 * tm["1024"] * CM.K2_TRIALS) / 60.0
    est["k3_min"] = sum(tm["1496"] * (P.layered_n(2, h) / 1496) for h in CM.K3_H) * CM.K3_TRIALS / 60.0
    est["k5_min"] = sum(tm["1024"] * (P.layered_n(1, h) / 1024) for h in CM.K5_H) * CM.K5_TRIALS / 60.0
    OUT["estimated_minutes"] = est
    print(json.dumps(OUT, indent=1, default=float))
    (HERE / "a7_kill_audit.json").write_text(json.dumps(OUT, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
