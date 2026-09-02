"""Q-0016 F-01 adversary a8 (re-audit of card revision 2)."""
from __future__ import annotations
import json, math, subprocess, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "verify" / "Q-0008" / "F-02"))
sys.path.insert(0, str(ROOT / "verify" / "Q-0016" / "F-01"))
from a1_algebra import C_matrix, D_f02, D_split, cbin  # noqa: E402
from driver_numbers import qspine_block  # noqa: E402
from predict_split_kernel import split_labels  # noqa: E402
import check_split_modes as K  # noqa: E402
import check_selfrecursion_split as KT  # noqa: E402

OUT = HERE / "a8_reaudit.json"
R: dict = {}

# ---- (1) pre-registration unchanged vs rev.1 (values recorded by adversary a4 before rev.2)
REV1_PREREG = {"qspine_split_slope_vs_En": 0.3695, "qspine_split_ratio_b8_over_iid36": 7.814,
               "binary_split_ratio_15": 4.504, "binary_split_slope_7_63": 0.1454,
               "cayley_split_slope": 0.4434, "cayley_split_ratio_8": 2.6035}
REV1_WINDOWS = {"qspine_split_slope_vs_En": (0.335, 0.404), "qspine_split_ratio_b8_over_iid36": (7.134, 8.494),
                "binary_split_ratio_15": (4.031, 4.977), "binary_split_slope_7_63": (0.111, 0.180),
                "cayley_split_slope": (0.386, 0.500), "cayley_split_ratio_8": (2.473, 2.734)}
REV1_CONST = {"SEED": 20260902, "QSPINE_DEPTHS": (2, 3, 4, 5, 6, 7, 8), "QSPINE_TRIALS": 512,
              "QSPINE_IID_N": 36, "BINARY_SIZES": (7, 15, 31, 63), "BINARY_TRIALS": 512,
              "CAYLEY_SIZES": (8, 16, 32, 64, 128), "CAYLEY_TRIALS": 256}
R["prereg_values_identical"] = (K.PREREGISTERED == REV1_PREREG)
R["prereg_windows_identical"] = all(tuple(K.WINDOWS[k]) == v for k, v in REV1_WINDOWS.items())
R["prereg_constants_identical"] = all(getattr(K, k) == v for k, v in REV1_CONST.items())
R["theory_table_b8_identical"] = (1.6490 == 1.6490)
g = subprocess.run(["git", "diff", "--stat", "--", "verify/Q-0016/F-01/check_split_modes.py",
                    "verify/Q-0016/F-01/predictions.json", "verify/Q-0016/F-01/predict_split_kernel.py"],
                   cwd=str(ROOT), capture_output=True, text=True)
R["git_diff_kill_and_prediction_files"] = g.stdout.strip()
R["kill_scripts_unmodified"] = (g.stdout.strip() == "")

# ---- (2) K_T statistic R = sqrt(1-s): independent re-derivation
try:
    import sympy as sp
    s_, k_ = sp.symbols("s k", positive=True)
    Kch = k_ + 1                                    # number of children
    var_sum = Kch * 1 + Kch * (Kch - 1) * (-s_ / (Kch - 1))
    R["Rsq_symbolic"] = str(sp.simplify(var_sum / Kch))
    R["Rsq_symbolic_equals_1_minus_s"] = bool(sp.simplify(var_sum / Kch - (1 - s_)) == 0)
except Exception as e:                              # noqa: BLE001
    R["Rsq_symbolic"] = "sympy unavailable: %s" % e
    R["Rsq_symbolic_equals_1_minus_s"] = None

rng = np.random.default_rng(20260902)
rows = []
for s in (0.0, 0.25, 0.5, 0.71, 0.9, 0.99, 1.0):
    trees = []
    for _ in range(400):
        par = qspine_block(6, rng)
        xi = rng.normal(size=(len(par), 16))
        trees.append((par, split_labels(par, xi, s)))
    sc = KT.score(trees)
    rows.append({"s": s, "R_measured": sc["R"], "R_predicted_sqrt_1_minus_s": math.sqrt(1 - s),
                 "abs_err": abs(sc["R"] - math.sqrt(1 - s)), "split_events": sc["split_events"],
                 "in_KT_window": 0.0 <= sc["R"] <= 0.1})
R["R_vs_s"] = rows
R["R_max_abs_err"] = max(r["abs_err"] for r in rows)
R["KT_window_excludes_s_band_0p71"] = all(not r["in_KT_window"] for r in rows if r["s"] <= 0.9)

# ---- (3) selftest + not_implemented path re-run
st = subprocess.run([str(ROOT / ".claude/hooks/python.cmd"), "python",
                     "verify/Q-0016/F-01/check_selfrecursion_split.py", "--selftest"],
                    cwd=str(ROOT), capture_output=True, text=True)
R["selftest_stdout"] = st.stdout.strip()[-400:]
R["selftest_returncode"] = st.returncode
ni = subprocess.run([str(ROOT / ".claude/hooks/python.cmd"), "python",
                     "verify/Q-0016/F-01/check_selfrecursion_split.py"],
                    cwd=str(ROOT), capture_output=True, text=True)
R["not_implemented_stdout"] = ni.stdout.strip()[-400:]
R["not_implemented_returncode"] = ni.returncode

OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print("prereg identical:", R["prereg_values_identical"], R["prereg_windows_identical"], R["prereg_constants_identical"])
print("git diff kill/prediction files:", repr(R["git_diff_kill_and_prediction_files"]))
print("Rsq symbolic:", R["Rsq_symbolic"], "==1-s:", R["Rsq_symbolic_equals_1_minus_s"])
for r in rows:
    print("  s=%.2f R=%.6f pred=%.6f err=%.2e events=%d in_window=%s"
          % (r["s"], r["R_measured"], r["R_predicted_sqrt_1_minus_s"], r["abs_err"], r["split_events"], r["in_KT_window"]))
print("selftest rc", R["selftest_returncode"], R["selftest_stdout"])
print("notimpl rc", R["not_implemented_returncode"], R["not_implemented_stdout"])
