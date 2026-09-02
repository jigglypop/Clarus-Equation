"""Adversary a8 (dimension/content): the multiplicity claim  ln Omega_align - ln Omega_mis
= 9 ln(1/eps_res) per cell assumes 'each folded visible direction is resolved at eps_res'.
But in the SAME sentence (ladder step 7) eps_res is F-02's threshold on the block RESIDUAL, and
the residual is QUADRATIC in the label.  Then the microstate volume at threshold eps_res scales
like eps_res^{9/2} per cell, not eps_res^{9}.

Decisive measurement: take n = 2 (centered folded space = 9 real dimensions).  Measure the small-x
exponent of  P[eps_block <= x] ~ x^k.  Card's counting => k = 9.  Quadratic-form counting => k = 4.5.
"""
import math
import sys
from pathlib import Path
import numpy as np

ROOT = Path(r"c:/dev/ce/Clarus-Equation")
F02 = ROOT / "verify" / "Q-0008" / "F-02"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(F02))
from check_modes import block_residual  # noqa: E402

import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("q0010_driver", ROOT / "verify/Q-0010/F-01/driver_numbers.py")
drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv)
BASIS, GROUPS = drv.orthonormal_label_basis()
FLAT = BASIS.reshape(16, 16)
FOLD = np.eye(16)
for i in GROUPS["scale"] + GROUPS["sd"]:
    FOLD[i, i] = 0.0        # (1 - P_align): the folded channel the card counts

DELTA = 0.005
for n, trials in ((2, 40000), (3, 20000)):
    rng = np.random.default_rng(13579)
    vals = []
    while len(vals) < trials:
        c = rng.normal(size=(n, 16)) @ FOLD.T
        v = block_residual((c @ FLAT).reshape(-1, 4, 4), DELTA)
        if math.isfinite(v):
            vals.append(v)
    x = np.sort(np.asarray(vals))
    p = np.arange(1, len(x) + 1) / len(x)
    lo, hi = int(0.002 * len(x)), int(0.05 * len(x))
    k = np.polyfit(np.log(x[lo:hi]), np.log(p[lo:hi]), 1)[0]
    lo2, hi2 = int(0.0005 * len(x)), int(0.01 * len(x))
    k2 = np.polyfit(np.log(x[lo2:hi2]), np.log(p[lo2:hi2]), 1)[0]
    dim_centered_folded = 9 * (n - 1)
    print("n=%d  trials=%d  lower-tail exponent k = %.3f (0.2-5%%) / %.3f (0.05-1%%)"
          % (n, trials, k, k2))
    print("     card counting would give k = %d (9 per cell x %d independent cells?);"
          " quadratic-form counting gives k = %.1f = dim(centered folded)/2 = %d/2"
          % (9 * n, n, dim_centered_folded / 2, dim_centered_folded))
