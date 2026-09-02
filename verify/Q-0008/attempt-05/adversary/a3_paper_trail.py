"""Adversary a3: paper-trail arithmetic (no sampling).

Checks, independently of the prover's own scripts:
 - card K3 numbers  ==  predictions.json  ==  check_modes.PREREGISTERED  ==  derivation text
 - windows == value +- declared uncertainty
 - S3.1 table in the derivation vs result.json (rms/delta^2, ratios, obs/card)
 - sigma distances quoted in (S5)/(S6)
 - slope recomputed by closed-form OLS from the recorded RMS, on the exact grid and on the
   observed mean-n grid (sensitivity)
 - E[n_b] identity and the step-4 bound sqrt(D/n^2) <= b at the observed amplitude
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
F02 = ROOT / "verify" / "Q-0008" / "F-02"
A5 = ROOT / "verify" / "Q-0008" / "attempt-05"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(F02))
import check_modes as cm

off = json.loads((F02 / "result.json").read_text(encoding="utf-8"))
att = json.loads((A5 / "result.json").read_text(encoding="utf-8"))
offc = json.loads((A5 / "result_official_F-02.json").read_text(encoding="utf-8"))
pj = json.loads((F02 / "predictions.json").read_text(encoding="utf-8"))
q = off["qspine"]
out = {}

out["official_copy_identical"] = off == offc
out["prereg_chain"] = {
    "slope": {"card": 0.5047, "script": cm.PREREGISTERED["qspine_slope_vs_En"],
              "predictions_json": pj["qspine_slope_vs_En"]},
    "ratio": {"card": 6.832, "script": cm.PREREGISTERED["qspine_ratio_b8_over_iid36"],
              "predictions_json": pj["qspine"]["8"]["ratio_to_iid_at_nstar"]},
    "slope_match": abs(cm.PREREGISTERED["qspine_slope_vs_En"] - pj["qspine_slope_vs_En"]) < 5e-5,
    "ratio_match": abs(cm.PREREGISTERED["qspine_ratio_b8_over_iid36"] - pj["qspine"]["8"]["ratio_to_iid_at_nstar"]) < 5e-4,
}
out["windows"] = {
    "slope_declared": list(cm.WINDOWS["qspine_slope_vs_En"]),
    "slope_from_value_unc": [0.5047 - 0.085, 0.5047 + 0.085],
    "ratio_declared": list(cm.WINDOWS["qspine_ratio_b8_over_iid36"]),
    "ratio_from_value_unc": [6.832 * 0.88, 6.832 * 1.12],
}
out["windows"]["slope_edge_diff"] = [cm.WINDOWS["qspine_slope_vs_En"][0] - (0.5047 - 0.085),
                                     cm.WINDOWS["qspine_slope_vs_En"][1] - (0.5047 + 0.085)]
out["windows"]["ratio_edge_diff"] = [cm.WINDOWS["qspine_ratio_b8_over_iid36"][0] - 6.832 * 0.88,
                                     cm.WINDOWS["qspine_ratio_b8_over_iid36"][1] - 6.832 * 1.12]

E_N = [3, 6, 10, 15, 21, 28, 36]
rms = q["rms"]
rms_i = q["rms_iid_36"]
x = np.log(np.array(E_N, float))
y = np.log(np.array(rms))
slope_exact = float(np.sum((x - x.mean()) * y) / np.sum((x - x.mean()) ** 2))
xm = np.log(np.array(q["mean_n"], float))
slope_meann = float(np.sum((xm - xm.mean()) * y) / np.sum((xm - xm.mean()) ** 2))
ratio = rms[-1] / rms_i
out["recomputed"] = {
    "slope_exact_grid": slope_exact, "reported_slope": off["stats"]["qspine_slope_vs_En"],
    "slope_exact_matches": abs(slope_exact - off["stats"]["qspine_slope_vs_En"]) < 1e-12,
    "slope_observed_mean_n_grid_SENSITIVITY": slope_meann,
    "slope_meann_in_window": bool(0.42 <= slope_meann <= 0.59),
    "ratio": ratio, "reported_ratio": off["stats"]["qspine_ratio_b8_over_iid36"],
    "ratio_matches": abs(ratio - off["stats"]["qspine_ratio_b8_over_iid36"]) < 1e-12,
    "derivation_S4_slope": 0.4998573796, "derivation_S4_ratio": 6.6713918570,
}
se_s = att["stats"]["qspine_slope_vs_En"]["se"]
se_r = att["stats"]["qspine_ratio_b8_over_iid36"]["se"]
out["sigma"] = {
    "slope_to_lower": (slope_exact - 0.42) / se_s, "slope_to_upper": (0.59 - slope_exact) / se_s,
    "slope_vs_prereg": (slope_exact - 0.5047) / se_s,
    "ratio_to_lower": (ratio - 6.01) / se_r, "ratio_to_upper": (7.65 - ratio) / se_r,
    "ratio_vs_prereg": (ratio - 6.832) / se_r,
    "alt_chain_slope": (1.0 - slope_exact) / se_s, "alt_mf_slope": (0.533 - slope_exact) / se_s,
    "alt_chain_ratio": (23.11 - ratio) / se_r, "alt_mf_ratio": (8.245 - ratio) / se_r,
    "alt_cayley_ratio": (9.064 - ratio) / se_r,
}
out["alt_vs_window_edge_in_SE"] = {
    "mean_field_ratio_above_upper_edge_in_SE": (8.245 - 7.65) / se_r,
    "cayley_ratio_above_upper_edge_in_SE": (9.064 - 7.65) / se_r,
    "mean_field_slope_inside_slope_window": bool(0.42 <= 0.533 <= 0.59),
    "window_halfwidth_over_SE_slope": 0.085 / se_s,
    "window_halfwidth_over_SE_ratio": 0.82 / se_r,
}
CARD = [0.1017, 0.2126, 0.3558, 0.5327, 0.7411, 0.9842, 1.2607]
tab = []
for k, b in enumerate(range(2, 9)):
    pred = math.sqrt(CARD[k]) * 36 / math.sqrt(35)
    obs = rms[k] / rms_i
    tab.append({"b": b, "rms_over_delta2": rms[k] / cm.DELTA ** 2, "obs_ratio": obs,
                "card_ratio": pred, "obs_over_card": obs / pred,
                "mean_n": q["mean_n"][k], "E_n": E_N[k]})
out["per_depth"] = tab
out["S3_1_text_check"] = {
    "rms_over_delta2_text": [0.978, 1.502, 1.873, 2.304, 2.731, 3.041, 3.470],
    "rms_over_delta2_computed": [round(t["rms_over_delta2"], 3) for t in tab],
    "obs_over_card_text": [0.969, 1.029, 0.992, 0.997, 1.002, 0.968, 0.976],
    "obs_over_card_computed": [round(t["obs_over_card"], 3) for t in tab],
    "rms_iid_over_delta2": rms_i / cm.DELTA ** 2,
}
out["step4_bound"] = {"sqrt_D_over_n2_obs_from_ratio": ratio * math.sqrt(35) / 36, "b": 8,
                      "bound_ok": ratio * math.sqrt(35) / 36 <= 8}
out["E_n_identity"] = {b: (sum(b - k for k in range(b)), b * (b + 1) // 2) for b in range(1, 9)}
(HERE / "a3_paper_trail.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
