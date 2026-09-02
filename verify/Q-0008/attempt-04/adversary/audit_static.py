"""adversary attempt-04 audit: hashes, card/script constant cross-check, trial-level statistics."""
from __future__ import annotations
import hashlib, json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
A04 = HERE.parent
ROOT = HERE.parents[3]
F02 = ROOT / "verify" / "Q-0008" / "F-02"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(F02))
import check_modes as cm  # noqa: E402
import driver_numbers as dn  # noqa: E402

out = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


files = {"check_modes.py": F02 / "check_modes.py", "driver_numbers.py": F02 / "driver_numbers.py",
         "card": ROOT / "derivations" / "Q-0008" / "F-02.formula.md",
         "check_se.py": A04 / "check_se.py", "assemble_result.py": A04 / "assemble_result.py"}
res04 = json.loads((A04 / "result.json").read_text(encoding="utf-8"))
out["sha256_now"] = {k: sha(v) for k, v in files.items()}
out["sha256_recorded"] = res04["integrity"]
out["sha256_match"] = {
    "check_modes.py": out["sha256_now"]["check_modes.py"] == res04["integrity"]["check_modes.py_sha256"],
    "driver_numbers.py": out["sha256_now"]["driver_numbers.py"] == res04["integrity"]["driver_numbers.py_sha256"],
    "card": out["sha256_now"]["card"] == res04["integrity"]["card_sha256"],
}

card_txt = (ROOT / "derivations" / "Q-0008" / "F-02.formula.md").read_text(encoding="utf-8")
window_strings = {
    "her_slope": "[0.43, 0.63]", "her_ratio_128": "[26.0, 39.1]", "mix_X_32": "[0.49, 0.99]",
    "iid_slope": "[-0.58, -0.38]", "defect_ratio_64_over_8": "[0.124, 0.158]", "defect_slope": "[-0.96, -0.86]",
}
out["card_window_text_present"] = {k: (v in card_txt) or (v.replace(", ", ",") in card_txt)
                                   for k, v in window_strings.items()}
out["script_windows"] = {k: list(v) for k, v in cm.WINDOWS.items()}
out["script_prereg"] = dict(cm.PREREGISTERED)
CARD_U = {"her_slope": 0.10, "her_ratio_128": 6.5, "mix_X_32": 0.25, "iid_slope": 0.10,
          "defect_ratio_64_over_8": 0.017, "defect_slope": 0.05}
out["window_vs_value_pm_u"] = {}
for k, u in CARD_U.items():
    v = cm.PREREGISTERED[k]
    lo, hi = cm.WINDOWS[k]
    out["window_vs_value_pm_u"][k] = {"value_pm_u": [v - u, v + u], "window": [lo, hi],
                                      "low_slack": (v - u) - lo, "high_slack": hi - (v + u)}
out["constants"] = {"SEED": cm.SEED, "DELTA": cm.DELTA, "MIN_DET": cm.MIN_DET, "SIZES": list(cm.SIZES),
                    "TRIALS": cm.TRIALS, "MIX_N": cm.MIX_N, "MIX_TRIALS": cm.MIX_TRIALS,
                    "DEFECT_GRID": list(cm.DEFECT_GRID), "DEFECT_PERTURBATION": cm.DEFECT_PERTURBATION,
                    "DEFECT_MIN_DET": cm.DEFECT_MIN_DET}
src = (F02 / "check_modes.py").read_text(encoding="utf-8")
out["rejections_field_is_dead_constant"] = (src.count("rejections") == 1)

ex = {n: dn.cayley_exact(n) for n in (8, 16, 32, 64, 128, 36)}
out["cayley_exact"] = {str(n): {"E_D": ex[n]["E_D"], "E_trHk": ex[n]["E_trHk"],
                                "ratio_pred": math.sqrt(ex[n]["E_D"] / (n - 1)),
                                "sqrtD_over_n": math.sqrt(ex[n]["E_D"]) / n} for n in ex}
grid = (8, 16, 32, 64, 128)
out["gamma_her_exact_grid"] = dn.slope(grid, [math.sqrt(ex[n]["E_D"]) / n for n in grid])
out["gamma_iid_exact_grid"] = dn.slope(grid, [math.sqrt(n - 1) / n for n in grid])
out["X32_exact"] = 2 * ex[32]["E_trHk"] / math.sqrt(31 * ex[32]["E_D"])
out["ratio128_exact"] = math.sqrt(ex[128]["E_D"] / 127)
json.dump(out, open(HERE / "_part1.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("part1 ok")
