"""a14: are the K2 windows wider than the statistical error of the physical MC at 128 trials?
Only the DISPERSION of the RMS estimator is recorded (central values are deliberately not reported),
at n=512 with seed 424242 (not the pre-registered K2 seed), so the kill statistics stay unobserved.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "verify" / "Q-0008" / "F-02"))
sys.path.insert(0, str(ROOT / "verify" / "Q-0017" / "F-01"))
from check_modes import DELTA, sample_iid  # noqa
from driver_numbers import uniform_rooted_tree  # noqa
import check_merge as CM  # noqa

N, T, SEED = 256, 64, 424242


def main():
    out = {"n": N, "trials_used": T, "seed": SEED, "target_trials": 128}
    for tag, q in (("merge_q1", 1.0), ("merge_q05", 0.5)):
        rng = np.random.default_rng(SEED + int(10 * q))
        vals = np.array([CM.sample_merge(lambda g: uniform_rooted_tree(N, g), q, rng, DELTA) for _ in range(T)])
        v2 = vals ** 2
        rel = float(np.std(v2, ddof=1) / math.sqrt(T) / (2 * v2.mean()))
        out[tag] = {"rel_se_of_RMS_at_%d" % T: rel,
                    "rel_se_of_RMS_at_128": rel * math.sqrt(T / 128.0),
                    "cv_of_squared_residual": float(np.std(v2, ddof=1) / v2.mean())}
    rng = np.random.default_rng(SEED + 77)
    vi = np.array([sample_iid(N, rng, DELTA) for _ in range(T)]) ** 2
    reli = float(np.std(vi, ddof=1) / math.sqrt(T) / (2 * vi.mean()))
    out["iid"] = {"rel_se_of_RMS_at_128": reli * math.sqrt(T / 128.0)}
    s1 = out["merge_q1"]["rel_se_of_RMS_at_128"]
    s05 = out["merge_q05"]["rel_se_of_RMS_at_128"]
    si = out["iid"]["rel_se_of_RMS_at_128"]
    out["k2b_ratio_rel_se"] = math.sqrt(s1 ** 2 + si ** 2)
    out["k2c_ratio_rel_se"] = math.sqrt(s1 ** 2 + s05 ** 2)
    out["k2b_window_halfwidth_rel"] = 6.9 / 34.516
    out["k2c_window_halfwidth_rel"] = 0.25 / 1.667
    out["k2b_window_in_sigma"] = out["k2b_window_halfwidth_rel"] / out["k2b_ratio_rel_se"]
    out["k2c_window_in_sigma"] = out["k2c_window_halfwidth_rel"] / out["k2c_ratio_rel_se"]
    from math import erf
    out["k2b_false_kill_prob"] = 1 - erf(out["k2b_window_in_sigma"] / math.sqrt(2))
    out["k2c_false_kill_prob"] = 1 - erf(out["k2c_window_in_sigma"] / math.sqrt(2))
    print(json.dumps({k: v for k, v in out.items() if k not in ("merge_q1", "merge_q05", "iid")}, indent=1))
    print(json.dumps({k: out[k] for k in ("merge_q1", "merge_q05", "iid")}, indent=1))
    (HERE / "a14_k2_noise.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
