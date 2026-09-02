"""adversary attempt-04 audit: is the +1.5 sigma her_slope excess an O(delta^4) truncation effect?

Common random numbers: for each trial the same heritable labels are evaluated at delta and delta/2.
If eps is a pure delta^2 quadratic form, RMS(delta)/RMS(delta/2) = 4 exactly; the deviation bounds the
quartic contamination at the pre-registered delta = 0.005.  Seed 555013 (declared before running).
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
F02 = ROOT / "verify" / "Q-0008" / "F-02"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(F02))
import check_modes as cm  # noqa: E402
from driver_numbers import uniform_rooted_tree  # noqa: E402

out = {"seed": 555013, "trials": 128}
rng = np.random.default_rng(555013)
for n, T in ((8, 128), (128, 128)):
    a, b = [], []
    for _ in range(T):
        parent = uniform_rooted_tree(n, rng)
        lab = cm.heritable_labels(parent, rng.normal(size=(n, 4, 4)))
        a.append(cm.block_residual(lab, cm.DELTA))
        b.append(cm.block_residual(lab, cm.DELTA / 2))
    ra, rb = cm.rms(a), cm.rms(b)
    per = np.array(a) / np.array(b)
    out[str(n)] = {"rms_delta": ra, "rms_half_delta": rb, "ratio": ra / rb, "ratio_minus_4": ra / rb - 4.0,
                   "rel_quartic_contamination": ra / rb / 4.0 - 1.0,
                   "per_trial_ratio_mean": float(per.mean()), "per_trial_ratio_sd": float(per.std(ddof=1)),
                   "per_trial_ratio_max_abs_dev_from_4": float(np.max(np.abs(per - 4.0)))}
json.dump(out, open(HERE / "delta4_check.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
