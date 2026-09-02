import json, pathlib
p = json.loads((pathlib.Path(__file__).parents[1] / "pilot.json").read_text(encoding="utf-8"))
out = {}
for mode in ("her", "iid"):
    for n, row in p["chains"][mode]["per_size"].items():
        out[f"{mode}_{n}"] = {
            "numeric_over_analytic_rms": row["theta_rms_numeric"] / row["theta_rms_analytic"],
            "numeric_over_isserlis_pred": row["theta_rms_numeric"] / row["theta_rms_pred_isserlis"],
            "eps_numeric_over_F02": row["eps_rms_numeric"] / row["eps_rms_pred_F02"],
            "se_rel": row["theta_rms_numeric_se"] / row["theta_rms_numeric"],
        }
vals = [v["numeric_over_analytic_rms"] for v in out.values()]
out["_range_numeric_over_analytic"] = [min(vals), max(vals)]
out["_card_scope_claim"] = "0.998~1.001"
out["_pilot_iid_slope_n_le_8"] = p["chains"]["iid"]["theta_slope_numeric"]
out["_card_P5_prediction"] = 0.5
out["_card_P5_already_observed_flag"] = False
out["_pilot_her_slope_n_le_8"] = p["chains"]["her"]["theta_slope_numeric"]
print(json.dumps(out, indent=2))
pathlib.Path(__file__).with_name("a7_ratio_table.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
