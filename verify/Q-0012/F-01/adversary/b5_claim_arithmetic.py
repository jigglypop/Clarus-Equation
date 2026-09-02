"""b5 (re-audit): recompute every sigma / percentage claim written in F-01 revision 2."""
import json, math
from pathlib import Path
from fractions import Fraction as F

OUT = Path(__file__).parent
c4 = F(1, 60)
S_gen_cat, S_ker_cat, D_cat = F(62069, 216), F(54023, 432), F(23053, 36)
S_iid, D_iid = F(35 ** 2, 36), F(35)
a_cat, a_iid = c4 * S_gen_cat / D_cat, c4 * S_iid / D_iid
a_cat_ker = c4 * S_ker_cat / D_cat
se = {"P1": 0.00335, "P2": 0.02744, "P3": 0.00313, "P4": 0.00762,
      "P5": 0.03854, "P6": 0.00496, "P7": 0.00444, "P8": 0.01084, "P9": 0.04104}
rows = {}

rho_P2 = 1 + 61 * a_iid
hw_P2 = 0.1098
rows["P2_c4_sharpness"] = {
    "card_says": "c4 tested to +-5.5%",
    "rho": float(rho_P2), "half_width": hw_P2,
    "half_width_over_rho": hw_P2 / float(rho_P2),
    "CORRECT_half_width_over_(rho-1)": hw_P2 / float(rho_P2 - 1),
    "c4_window": [float(c4) * (1 - hw_P2 / float(rho_P2 - 1)), float(c4) * (1 + hw_P2 / float(rho_P2 - 1))]}

rho_P5, rho_P5_ker = 1 + 61 * a_cat, 1 + 61 * a_cat_ker
rows["P5_kernel_alt"] = {"rho_gen": float(rho_P5), "rho_ker": float(rho_P5_ker),
                         "sigma_gen_minus_ker": (float(rho_P5) - float(rho_P5_ker)) / se["P5"],
                         "card_says_6.7_sigma": 6.7,
                         "sigma_ker_below_window_low": (1.3020 - float(rho_P5_ker)) / se["P5"],
                         "card_says_2.7": 2.7}
ratio_gen = float((S_gen_cat / D_cat) / (S_iid / D_iid))
ratio_ker = float((S_ker_cat / D_cat) / (S_iid / D_iid))
rows["P9_kernel_alt"] = {"gen": ratio_gen, "ker": ratio_ker,
                         "sigma_ker_below_window_low": (0.2974 - ratio_ker) / se["P9"], "card_says_2.4": 2.4}
bias = 0.013922
rows["truncation_bias_share"] = {
    "card_says": "1.4% is 9% of the P5 half-width 0.1542",
    "relative_bias": bias, "half_width": 0.1542,
    "naive_relative_over_halfwidth": bias / 0.1542,
    "CORRECT_absolute_shift": bias * float(rho_P5),
    "CORRECT_share_of_halfwidth": bias * float(rho_P5) / 0.1542,
    "shifted_centre": float(rho_P5) * (1 + bias),
    "sigma_from_upper_edge": (1.6104 - float(rho_P5) * (1 + bias)) / se["P5"]}
rows["misc"] = {
    "S_gen_over_S_ker_cat6": float(S_gen_cat / S_ker_cat), "card_says_2.30": 2.30,
    "a_cat6": float(a_cat), "a_cat6_kernel_alt": float(a_cat_ker), "card_says_0.003255": 0.003255,
    "P1_gauss_sigma": (float(1 + 3 * a_iid) - 1.0) / se["P1"], "card_says_14sigma": 14,
    "b4_cayley_recover_pct": 100 * (math.sqrt((1 - 2 * 0.221359066 / 60) / (1 - 2 * (31 / 32) / 60)) - 1),
    "F02_gamma_shift_over_K5_width": abs(0.5 * (-2 / 60) * (1 / 32) / (1 - 2 * (31 / 32) / 60)) / 0.20,
    "dof_spike_weight": 61 ** 2 / (61 ** 2 + 3 ** 2 + 1.2 ** 2 + 2 ** 2)}
rows["K3_windows_that_contain_the_null"] = {
    "rho_cat6_uniform": [0.9733, 1.0088], "rho_cat6_rademacher": [0.9417, 1.0284],
    "note": "these two cannot discriminate rho = 1; the card calls them linearity controls"}
print(json.dumps(rows, indent=1))
(OUT / "b5_claim_arithmetic.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
