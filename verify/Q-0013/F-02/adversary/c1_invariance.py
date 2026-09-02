"""adversary c1 (re-audit): invariance of the preregistered numbers between card rev1 and rev2.

rev1 record = b3_report.json card_pred (parsed by me from rev1) + b6_report.json prereg
(PRED/WINDOW imported verbatim from check_floor.py at 00:25).
rev2 = current derivations/Q-0013/F-02.formula.md + current check_floor.py.
No preregistered-size run is performed.
"""
from __future__ import annotations
import hashlib, importlib.util, json, re, sys
from pathlib import Path

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
OUT = ROOT / "verify" / "Q-0013" / "F-02" / "adversary"
CARD = ROOT / "derivations" / "Q-0013" / "F-02.formula.md"
SCRIPT = ROOT / "verify" / "Q-0013" / "F-02" / "check_floor.py"

spec = importlib.util.spec_from_file_location("cf", SCRIPT)
cf = importlib.util.module_from_spec(spec)
sys.modules["cf"] = cf
spec.loader.exec_module(cf)

b3 = json.loads((OUT / "b3_report.json").read_text(encoding="utf-8"))
b6 = json.loads((OUT / "b6_report.json").read_text(encoding="utf-8"))

txt = CARD.read_text(encoding="utf-8")

# --- 1. card predicts: value / uncertainty pairs (rev2)
vals = [float(m) for m in re.findall(r"^    value: ([-0-9.e+]+)$", txt, re.M)]
uncs = [float(m) for m in re.findall(r"^    uncertainty: ([-0-9.e+]+)$", txt, re.M)]
obs = re.findall(r"^  - observable: \"?(.{0,60})", txt, re.M)
ao = re.findall(r"^    already_observed: (\w+)$", txt, re.M)
cf_frozen = re.findall(r"^    comparison_frozen: (\w+)$", txt, re.M)

# --- 2. windows quoted in the card body (kill + predicts notes), as [lo, hi] pairs
wins = [(float(a), float(b)) for a, b in
        re.findall(r"\[\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\]", txt)]

# --- 3. rev1 vs rev2 for the 16 card_pred keys (b3 parsed rev1 values)
rev1 = b3["card_pred"]
# map card rev2 predicts order -> keys (same order as rev1 parse)
KEYS = ["ker_eps64_over_delta2", "ker_slope", "diag_eps2_over_delta2", "diag_eps64_over_delta2",
        "diag_slope", "off_eps2_over_delta2", "iso_eps2_over_delta2", "cross_eps2_sq_over_delta4",
        "univ_o_eps64_over_delta2", "univ_o_eps4_over_delta2", "univ_floor_hat_over_delta2",
        "univ_d_eps64_over_delta2", "zero_max_residual", "ce_i_eps64", "ce_ii_eps64"]
rev2_vals = dict(zip(KEYS, vals))
diff_rev1_rev2 = {}
for k in KEYS:
    if k in rev1:
        diff_rev1_rev2[k] = {"rev1": rev1[k], "rev2": rev2_vals[k],
                             "equal": abs(rev1[k] - rev2_vals[k]) == 0.0}
    else:
        diff_rev1_rev2[k] = {"rev1": None, "rev2": rev2_vals[k], "equal": None}

# --- 4. script PRED/WINDOW now vs b6 record (rev1-era, imported verbatim)
pred_now_vs_b6 = {k: {"now": cf.PRED[k], "b6": b6["prereg"]["PRED_vs_exact"][k]["card_script_PRED"],
                      "equal": cf.PRED[k] == b6["prereg"]["PRED_vs_exact"][k]["card_script_PRED"]}
                  for k in cf.PRED}
win_now_vs_b6 = {}
for k, (lo, hi) in cf.WINDOW.items():
    rec = b6["prereg"]["WINDOW_vs_model"].get(k)
    ref = rec["script"] if isinstance(rec, dict) and "script" in rec else None
    win_now_vs_b6[k] = {"now": [lo, hi], "b6": ref,
                        "equal": (ref is not None and [lo, hi] == list(ref))}

# --- 5. card values vs live script PRED (the card must not have drifted from the script)
card_vs_script = {}
for k in cf.PRED:
    if k in rev2_vals:
        card_vs_script[k] = {"card": rev2_vals[k], "script": cf.PRED[k],
                             "equal": rev2_vals[k] == cf.PRED[k]}

# --- 6. card kill windows vs script WINDOW (kill lines only)
kill_block = txt.split("kill:")[1].split("ladder:")[0]
kill_wins = [(float(a), float(b)) for a, b in
             re.findall(r"\[(-?\d+\.\d+), (-?\d+\.\d+)\]", kill_block)]

# --- 7. script constants
consts = {k: getattr(cf, k) for k in ("DELTA", "ZERO_DELTAS", "SIZES", "TRIALS", "N2_TRIALS",
                                      "SEED", "MIN_DET", "MODEL_REPLICATES")}

# --- 8. which stats the SCRIPT's verdict() treats as kill vs what the CARD says
card_kill_stats = sorted(set(re.findall(r"`stats\.(\w+)`", kill_block)))
script_verdict_stats = sorted(cf.WINDOW)
consistency_block = txt.split("consistency_checks:")[1].split("ladder:")[0]
card_consistency_stats = sorted(set(re.findall(r"`stats\.(\w+)`|`(\w+)`", consistency_block)[0:0]) |
                                set(re.findall(r"stats\.(\w+)", consistency_block)))

rep = {
    "hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
               for p in (SCRIPT, CARD, ROOT / "verify/Q-0013/F-02/structure_constants.json")},
    "script_mtime": SCRIPT.stat().st_mtime, "card_mtime": CARD.stat().st_mtime,
    "card_rev2_values_count": len(vals), "uncertainties_count": len(uncs),
    "already_observed_flags": ao, "comparison_frozen_all_true": all(v == "true" for v in cf_frozen),
    "rev1_vs_rev2_values": diff_rev1_rev2,
    "script_PRED_now_vs_b6": pred_now_vs_b6,
    "script_WINDOW_now_vs_b6": win_now_vs_b6,
    "card_values_vs_script_PRED": card_vs_script,
    "card_kill_windows": kill_wins,
    "script_constants": {k: (list(v) if isinstance(v, tuple) else v) for k, v in consts.items()},
    "card_kill_stats": card_kill_stats,
    "script_verdict_checks": script_verdict_stats,
    "card_consistency_stats": card_consistency_stats,
    "script_verdict_minus_card_kill": sorted(set(script_verdict_stats) - set(card_kill_stats)),
    "all_values_unchanged": all(v["equal"] for v in diff_rev1_rev2.values() if v["equal"] is not None),
    "all_script_pred_unchanged": all(v["equal"] for v in pred_now_vs_b6.values()),
    "all_script_window_unchanged": all(v["equal"] for v in win_now_vs_b6.values()),
    "card_matches_script": all(v["equal"] for v in card_vs_script.values()),
}
(OUT / "c1_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
for k in ("all_values_unchanged", "all_script_pred_unchanged", "all_script_window_unchanged",
          "card_matches_script", "card_kill_stats", "script_verdict_minus_card_kill",
          "card_consistency_stats", "script_constants", "already_observed_flags"):
    print(k, "=", json.dumps(rep[k], ensure_ascii=False))
print("card_kill_windows:", rep["card_kill_windows"])
bad = {k: v for k, v in diff_rev1_rev2.items() if v["equal"] is False}
print("CHANGED VALUES:", json.dumps(bad, ensure_ascii=False))
