import json
from pathlib import Path
R = Path(r"C:/dev/ce/Clarus-Equation")
sm = json.loads((R / "verify/Q-0013/F-02/smoke.json").read_text(encoding="utf-8"))
st, pr = sm["stats"], sm["predicted_at_smoke_sizes"]
PRED = {"ker": 0.07733980, "diag2": 0.57735027, "off2": 1.08012345, "iso2": 1.58113883,
        "cross": 1.0, "floor_hat": 0.11785113}
out = {}
for tag, key in (("kernel", "smoke_ker_curve_over_delta2"), ("diag4", "smoke_diag_curve_over_delta2"),
                 ("univ_o", "smoke_univ_o_curve_over_delta2"), ("univ_d", "smoke_univ_d_curve_over_delta2")):
    for n, v in st[key].items():
        out["%s_n%s" % (tag, n)] = v / pr[tag][n]
for tag, key, pkey in (("diag2", "smoke_diag_eps2_over_delta2", "diag4"),
                       ("off2", "smoke_off_eps2_over_delta2", "off12"),
                       ("iso2", "smoke_iso_eps2_over_delta2", "iso16")):
    out[tag] = st[key] / pr[pkey]["2"]
out["cross"] = st["smoke_cross_eps2_sq_over_delta4"] / PRED["cross"]
out["floor_hat"] = st["smoke_univ_floor_hat_over_delta2"] / PRED["floor_hat"]
for tag in ("ce_i", "ce_ii", "ce_iii"):
    d = st["smoke_" + tag]
    for n in d["observed_over_delta2"]:
        out["%s_n%s" % (tag, n)] = d["observed_over_delta2"][n] / d["master"][n]
res = {"ratios_observed_over_predicted": out,
       "min": min(out.values()), "max": max(out.values()),
       "card_prose_claim": "smoke 0.9~1.14",
       "outside_card_prose": {k: v for k, v in out.items() if v < 0.9 or v > 1.14}}
(R / "verify/Q-0013/F-02/adversary/b11_report.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=2))
