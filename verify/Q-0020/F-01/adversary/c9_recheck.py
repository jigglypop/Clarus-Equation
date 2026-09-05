# -*- coding: utf-8 -*-
import json, re, os
ROOT = r"C:\dev\ce\Clarus-Equation"
CARD = os.path.join(ROOT, "derivations", "Q-0020", "F-01.formula.md")
ADV  = os.path.join(ROOT, "verify", "Q-0020", "F-01", "adversary")
txt = open(CARD, encoding="utf-8").read()
out = {}
frozen_v = [39, 0.7448007, -1.3273034, 7.4820420, 62.0688, 0.0, 0.0, 1.0, 0.0, 1.0]
frozen_u = [0, 1.0e-6, 1.0e-6, 1.0e-5, 1.0e-3, 1.0e-5, 1.0e-2, 0.99, 0.5, 0.25]
pred_block = txt.split("predicts:")[1].split("recovers:")[0]
vals = [float(x) for x in re.findall(r"^\s*value:\s*(\S+)", pred_block, re.M)]
uncs = [float(x) for x in re.findall(r"^\s*uncertainty:\s*(\S+)", pred_block, re.M)]
out["prereg_values"] = {"observed": vals, "expected_rev1": frozen_v, "match": vals == frozen_v}
out["prereg_uncertainties"] = {"observed": uncs, "expected_rev1": frozen_u, "match": uncs == frozen_u}
kb = txt.split("kill:")[1].split("consistency_checks:")[0]
kill_nums = {
  "K1_tol_1e-2": ("1e-2" in kb) or (u"1e\u22122" in kb),
  "K1_m_234": kb.count("234") >= 2,
  "K2_m_39": (u"\u2260 39" in kb),
  "K2_nneg_31": "31" in kb,
  "K2_spec_0.5": "0.5" in kb,
  "K3_window": ("[0.8, 1.25]" in kb) or ("[0.8,1.25]" in kb),
  "K4_present": "additivity_residual.d_W" in kb,
}
out["kill_numbers_intact"] = kill_nums
out["kill_all_intact"] = all(kill_nums.values())
hr = json.load(open(os.path.join(ROOT,"verify","Q-0020","F-01","hook_result.json"), encoding="utf-8"))
vb = txt.split("verify:")[1].split("\n---")[0]
rev1_ok = []
for d in hr["details"]:
    e = d.get("expr") or d.get("lhs")
    rev1_ok.append(e in vb)
out["verify_block"] = {
  "n_details_rev1": len(hr["details"]),
  "n_entries_in_card": len(re.findall(r"^\s*-\s*type:", vb, re.M)),
  "all_rev1_expr_present": all(rev1_ok),
  "missing": [hr["details"][i].get("expr") for i,k in enumerate(rev1_ok) if not k],
  "tols": [float(t) for t in re.findall(r"tol:\s*(\S+)", vb)],
}
out["verify_block"]["tols_expected"] = [1e-12,1e-12,1e-6,1e-3,1e-3,1e-4,1e-6,2e-6,1e-5,1e-12,1e-3]
out["verify_block"]["tols_match"] = out["verify_block"]["tols"] == out["verify_block"]["tols_expected"]
c8 = json.load(open(os.path.join(ADV,"c8_spectrum.json"), encoding="utf-8"))
tri_sig = [c for c in c8["sigma2_W_clusters"] if abs(c["value"])<1e-10]
tri_rho = [c for c in c8["rho_R_clusters"] if abs(c["value"])<1e-10]
s32 = c8["three_two_sector"]
out["claim_two_conv_free_numbers"] = {
  "card_value": 0.3208245916, "sigma2_W": s32["sigma2_W"], "rho_R": s32["rho_R"],
  "abs_diff": s32["abs_diff"],
  "card_value_ok_1e-9": abs(s32["sigma2_W"] - 0.3208245916) < 1e-9,
  "trivial_mult_W": tri_sig[0]["multiplicity"] if tri_sig else None,
  "trivial_mult_R": tri_rho[0]["multiplicity"] if tri_rho else None,
  "three_two_mult_W": [c["multiplicity"] for c in c8["sigma2_W_clusters"] if abs(c["value"]-0.3208246)<1e-6],
  "three_two_mult_R": [c["multiplicity"] for c in c8["rho_R_clusters"] if abs(c["value"]-0.3208246)<1e-6],
  "std_convention_dependent": c8["std_sector"]["convention_dependent"],
  "card_claims_R_W_agree_2.2e-16": ("2.2e-16" in txt) or (u"2.2e\u221216" in txt),
}
c5 = json.load(open(os.path.join(ADV,"c5_kill_power.json"), encoding="utf-8"))
c6 = json.load(open(os.path.join(ADV,"c6_k3_power_k1_structure.json"), encoding="utf-8"))
r20 = c5["k3_random20"]
eff = max(abs(r20["max"]-1.0), abs(r20["min"]-1.0))
out["claim_k3_power"] = {
  "card_range": [1.0025, 1.0093], "audit_range": [r20["min"], r20["max"]],
  "match": abs(r20["min"]-1.0025)<1e-3 and abs(r20["max"]-1.0093)<1e-3,
  "n_seeds": r20["n"], "effect": eff,
  "window_upper_halfwidth": 0.25, "ratio": 0.25/eff,
  "card_says_30x": (u"\uc57d 30\ubc30" in txt),
  "window_over_effect_using_lower_0.2": 0.2/eff,
}
c4 = json.load(open(os.path.join(ADV,"c4_content_sign.json"), encoding="utf-8"))
out["claim_Sc_axiom"] = {
  "sign_flip_kills_lstar": c4["sign_flip_kills_lstar"],
  "GHP_monotone": c4["stationary"]["S_c<0 (GHP S_E=-S_geo/8piG)"]["gamma_monotone_decreasing"],
  "card_has_axiom_line": (u"S_c \uc758 \ubd80\ud638" in txt),
  "card_counts_three_axioms": txt.count(u"[\uacf5\ub9ac: \ud6c4\ubcf4]"),
}
c7 = json.load(open(os.path.join(ADV,"c7_m39_gauge.json"), encoding="utf-8"))
out["claim_gauge_conditional"] = {
  "lstar2_m39": c7["m_by_gauge_convention"]["gauge_in_numerator_only"]["lstar2"],
  "lstar2_m35": c7["m_by_gauge_convention"]["gauge_in_both (cancels)"]["lstar2"],
  "spread": c7["lstar2_relative_spread"],
  "card_says_10.3pct": "10.3%" in txt,
  "spread_is_10.26": abs(c7["lstar2_relative_spread"]-0.103) < 0.002,
  "card_says_55.7028": "55.7028" in txt,
}
lad = txt.split("ladder:")[1].split("novelty:")[0]
step3 = [l for l in lad.splitlines() if "step: 3" in l]
cc = txt.split("consistency_checks:")[1].split("ladder:")[0]
out["claim_step3_tuple"] = {
  "step3_2_5_4": any("(2,5,4)" in l for l in step3),
  "step3_2_4_5": any("(2,4,5)" in l for l in step3),
  "consistency_2_5_4": "(2,5,4)" in cc,
  "observed_W": c8["multiplicities_W"], "observed_R": c8["multiplicities_R"],
}
out["claim_gamma_min"] = {
  "gamma_min_R": c4["competition"]["gamma_min_over_hbar_R"],
  "card_has_30.23_33.51": ("30.23" in txt and "33.51" in txt),
  "positive": c4["competition"]["gamma_min_positive"],
  "lnOmega_at_lstar": c4["competition"]["ln_Omega_at_lstar_R"],
  "card_has_10.73": ("10.73" in txt),
}
c3 = json.load(open(os.path.join(ADV,"c3_glued_schur.json"), encoding="utf-8"))
c2 = json.load(open(os.path.join(ADV,"c2_convention.json"), encoding="utf-8"))
out["claim_glued_recover"] = {
  "card_has_2.4e-8": ("2.4e-8" in txt) or (u"2.4e\u22128" in txt),
  "c3_direct_max_abs_diff": c3["direct_fine_glued"]["max_abs_diff"],
  "c3_pass": c3["direct_fine_glued"]["pass"],
  "c2_naive_route_pass": c2["recover_glued_schur"]["pass"],
}
out["claim_k1_exact"] = {
  "card_says_exact": (u"\uc815\ud655 \ud56d\ub4f1\uc2dd \uc608\uce21" in txt),
  "toy_max_residual": c6["k1_toy_verdict"]["max_abs_residual"],
  "generic_theorem": c6["k1_toy_verdict"]["exact"],
}
out["dof"] = {
  "free_parameters_empty": "free_parameters: []" in txt,
  "n_predicts": len(vals),
  "observed": len([m for m in re.findall(r"already_observed:\s*(\w+)", pred_block) if m=="true"]),
  "unobserved": len([m for m in re.findall(r"already_observed:\s*(\w+)", pred_block) if m=="false"]),
}
out["comparison_frozen"] = re.findall(r"comparison_frozen:\s*(\w+)", pred_block)
out["revision_field"] = re.search(r"^revision:\s*(\d+)", txt, re.M).group(1)
dst = os.path.join(ADV, "c9_recheck.json")
open(dst, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=2))
print(json.dumps(out, ensure_ascii=False, indent=2))
