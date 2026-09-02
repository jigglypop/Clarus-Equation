"""Q-0016 F-01 adversary a7: random_sample_20 (seed 20260902) — the card CODE vs the card WORDS,
plus the two-species alternative and a provenance/mtime check."""
from __future__ import annotations
import json, math, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "verify" / "Q-0008" / "F-02"))
sys.path.insert(0, str(ROOT / "verify" / "Q-0016" / "F-01"))
from a1_algebra import D_f02, D_split  # noqa: E402
from driver_numbers import driver_fast, driver_matrix, qspine_block, uniform_rooted_tree  # noqa: E402
from predict_split_kernel import driver_split, kappa_split, kappa_split_via_B  # noqa: E402

OUT = HERE / "a7_sample20.json"
R: dict = {}

rng = np.random.default_rng(20260902)
rows = []
for i in range(20):
    n = int(rng.integers(2, 26))
    p = uniform_rooted_tree(n, rng) if i % 2 == 0 else qspine_block(int(rng.integers(2, 9)), rng)
    mine_s, mine_f = D_split(p), D_f02(p)
    card_s = driver_split(p)
    card_f = driver_matrix(p)
    fast_f = driver_fast(p)[0]
    rows.append({"i": i, "n": len(p), "mine_split": mine_s, "card_split": card_s,
                 "abs_err_split": abs(mine_s - card_s),
                 "mine_f02": mine_f, "card_f02_matrix": card_f, "card_f02_fast": fast_f,
                 "abs_err_f02": max(abs(mine_f - card_f), abs(mine_f - fast_f)),
                 "ratio": (mine_s / mine_f) if mine_f > 0 else None})
R["sample20"] = rows
R["sample20_max_abs_err_split"] = max(r["abs_err_split"] for r in rows)
R["sample20_max_abs_err_f02"] = max(r["abs_err_f02"] for r in rows)
R["sample20_min_ratio"] = min((r["ratio"] for r in rows if r["ratio"] is not None))

files = ["derivations/Q-0016/F-01.formula.md", "verify/Q-0016/F-01/predict_split_kernel.py",
         "verify/Q-0016/F-01/predictions.json", "verify/Q-0016/F-01/check_split_modes.py",
         "verify/Q-0016/F-01/hook_result.json"]
R["mtimes"] = {f: os.path.getmtime(ROOT / f) for f in files}
R["mtime_order_human"] = sorted(((os.path.getmtime(ROOT / f), f) for f in files))
R["predictions_json_predates_card"] = (os.path.getmtime(ROOT / "verify/Q-0016/F-01/predictions.json")
                                       < os.path.getmtime(ROOT / "derivations/Q-0016/F-01.formula.md"))
R["kill_result_json_exists"] = (ROOT / "verify/Q-0016/F-01/result.json").exists()

pred = json.loads((ROOT / "verify/Q-0016/F-01/predictions.json").read_text(encoding="utf-8"))
R["card_predictions_selfcheck"] = {
    "check_ACAt_equals_kappa_minus_B_max_abs": pred.get("check_ACAt_equals_kappa_minus_B_max_abs"),
    "check_C_min_eigenvalue": pred.get("check_C_min_eigenvalue"),
    "check_f02_replay_max_abs": pred.get("check_f02_replay_max_abs"),
    "qspine_b8_split": pred["qspine"]["8"]["E_D_over_n2_split"],
    "qspine_slope_vs_En_split": pred.get("qspine_slope_vs_En_split"),
    "qspine_ratio_b8_over_iid36_split": pred.get("qspine_ratio_b8_over_iid36_split"),
}
OUT.write_text(json.dumps(R, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print("sample20 max err split:", R["sample20_max_abs_err_split"], "f02:", R["sample20_max_abs_err_f02"])
print("sample20 min ratio:", R["sample20_min_ratio"])
print("mtime order:", json.dumps(R["mtime_order_human"], default=str))
print("predictions.json predates card:", R["predictions_json_predates_card"],
      "kill result.json exists:", R["kill_result_json_exists"])
print("card selfcheck:", json.dumps(R["card_predictions_selfcheck"], default=float))
