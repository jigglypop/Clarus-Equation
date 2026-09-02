"""b6 (re-audit): is the pre-registered labels path still the object a5/a7 audited at revision 1?

The card is untracked, so there is no diff.  What can be checked is that check_cumulant's CURRENT
label machinery still agrees, sample by sample, with the INDEPENDENT re-implementation written for
the revision-1 audit (a5.labels_from_z + ancestor_matrix), and that the pre-registered kappa4 values
are the true excess kurtoses of those laws.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "verify" / "Q-0012" / "F-01"))
sys.path.insert(0, str(ROOT / "verify" / "Q-0012" / "F-01" / "adversary"))
from check_cumulant import (uniform_to_label, normal_cdf, heritable, caterpillar, ancestor_matrix,
                            KAPPA4, DISTS, DELTA, TRIALS, N_CELLS, CAT_K, SEED, MIN_DET, EXACT,
                            lattice_constants)  # noqa
from a5_window_coverage import labels_from_z  # noqa

OUT = Path(__file__).parent
rng = np.random.default_rng(31415926)
z = rng.standard_normal((36, 16))
u = normal_cdf(z)
res = {"frozen_constants": {"SEED": SEED, "DELTA": DELTA, "TRIALS": TRIALS, "N_CELLS": N_CELLS,
                            "CAT_K": CAT_K, "MIN_DET": MIN_DET, "KAPPA4": KAPPA4,
                            "EXACT": {k: float(v) for k, v in EXACT.items()}}}
res["label_map_max_abs_diff_vs_independent"] = {
    d: float(np.abs(uniform_to_label(u, z, d) - labels_from_z(z, d)).max()) for d in DISTS}
par = caterpillar(CAT_K)
A = ancestor_matrix(par)
zeta = uniform_to_label(u, z, "laplace")
res["heritable_vs_ancestor_matrix_max_abs_diff"] = float(np.abs(heritable(par, zeta) - A @ zeta).max())
big = rng.standard_normal((400000,))
ub = normal_cdf(big)
res["empirical_kappa4"] = {}
for d in DISTS:
    x = uniform_to_label(ub, big, d)
    res["empirical_kappa4"][d] = {"declared": KAPPA4[d], "var": float(x.var()),
                                  "excess_kurtosis": float((x ** 4).mean() / (x ** 2).mean() ** 2 - 3)}
lat = lattice_constants(par)
res["lattice_cat6"] = {"S_gen": lat["S_gen"], "exact_S_gen": float(EXACT["cat6_S_gen"]),
                       "S_ker": lat["S_ker"], "exact_S_ker": float(EXACT["cat6_S_ker"]),
                       "D": lat["D"], "exact_D": float(EXACT["cat6_D"]), "n": lat["n"]}
print(json.dumps(res, indent=1))
(OUT / "b6_labels_path_identity.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
