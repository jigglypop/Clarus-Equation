"""Audit 5 (kill_executable): tiny physics smoke on a DIFFERENT seed and a small grid,
plus an empirical check of the +-0.06 slope window from the trial-to-trial spread."""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[3]))
import check_families as CF  # noqa: E402

ADV_SEED = 20260902 + 4242          # NOT the pre-registered seed
out = {"adversary_seed": ADV_SEED, "note": "smoke only: n<=16, trials<=16, different seed; "
                                           "never touches result.json or the pre-registered grid"}
t0 = time.time()
# 1. the kill code path runs at all
try:
    d = CF.run_slope("broom_smoke", lambda n, _rng: CF.broom_parent(n), (8, 16), 8, ADV_SEED)
    out["broom_smoke"] = d
    d2 = CF.run_slope("tls_smoke", lambda n, _rng: CF.tls_parent(int(round((math.sqrt(4 * n - 3) - 1) / 2))),
                      (7, 13), 8, ADV_SEED)
    out["tls_smoke"] = d2
    d3 = CF.run_slope("soc_smoke", lambda n, _rng: CF.soc_parent(int(round(math.sqrt(n - 1)))),
                      (5, 10), 8, ADV_SEED)
    out["soc_smoke"] = d3
    d4 = CF.run_slope("rrt_smoke", CF.rrt_parent, (8, 16), 8, ADV_SEED)
    out["rrt_smoke"] = d4
    r_iid = CF.rms_iid(13, 8, ADV_SEED + 1)
    out["iid_smoke_rms_13"] = r_iid
    out["kill_path_executable"] = True
except Exception as exc:  # noqa: BLE001
    out["kill_path_executable"] = False
    out["error"] = repr(exc)

# 2. spread of the RMS estimator at 8 trials -> implied slope s.e. at 128 trials
reps = []
for r in range(10):
    reps.append(CF.rms_heritable(lambda n, _rng: CF.broom_parent(n), 16, 8, ADV_SEED + 100 * r))
reps = np.array(reps)
rel = float(reps.std(ddof=1) / reps.mean())
sigma_ln_8 = rel                       # relative s.d. of the RMS estimate at 8 trials
sigma_ln_128 = sigma_ln_8 * math.sqrt(8 / 128)
xs = np.log([8, 16, 32, 64, 128])
sxx = float(((xs - xs.mean()) ** 2).sum())
out["rms_spread"] = {"reps": reps.tolist(), "rel_sd_at_8_trials": rel,
                     "implied_rel_sd_at_128": sigma_ln_128,
                     "implied_slope_se_5pt_grid": sigma_ln_128 / math.sqrt(sxx),
                     "card_window_halfwidth": 0.06,
                     "window_in_sigma": 0.06 / (sigma_ln_128 / math.sqrt(sxx))}
out["seconds"] = time.time() - t0
print(json.dumps(out, indent=2))
(HERE / "a11_smoke.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
