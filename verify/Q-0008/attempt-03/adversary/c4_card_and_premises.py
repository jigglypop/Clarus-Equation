"""adversary c4 (Q-0008 attempt-03): (i) is the card unchanged except for one scope sentence?
(ii) are the physical premises of the step-2 lemma automatically satisfied in the card's model?

(i) compares formula / kill / ladder strings in derivations/Q-0008/F-02.formula.md against the copy
    frozen in ledger/questions.yaml at card adoption, and the pre-registered numbers and windows
    against verify/Q-0008/F-02/check_modes.py.
(ii) checks that Sigma(e) is exactly simple for every nondegenerate tetrad (so 'each cell exactly
     simple' is not an extra assumption inside the model) and that the negative branch is refused.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from examples.physics.gravity.causal_face_simplicity import (  # noqa: E402
    geometric_self_dual_triple,
    plebanski_gram,
    simplicity_residual,
)
from examples.physics.gravity.urbantke_shape_matching_rg import (  # noqa: E402
    normalized_urbantke_metric,
    optimal_internal_alignment,
)

CARD = ROOT / "derivations" / "Q-0008" / "F-02.formula.md"
LEDGER = ROOT / "ledger" / "questions.yaml"
MODES = ROOT / "verify" / "Q-0008" / "F-02" / "check_modes.py"


def norm(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8')
        except Exception:
            pass
    res: dict = {}
    raw = CARD.read_text(encoding="utf-8")
    front = yaml.safe_load(raw.split("---")[1])
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    q = [item for item in ledger["questions"] if item["id"] == "Q-0008"][0] \
        if isinstance(ledger, dict) and "questions" in ledger else \
        [item for item in ledger if item.get("id") == "Q-0008"][0]

    res["formula_matches_ledger"] = norm(front["formula"]) == norm(q["formula"])
    res["kill_matches_ledger"] = [
        {"i": i, "same": norm(a) == norm(b)}
        for i, (a, b) in enumerate(zip(front["kill"], q["kill"]))
    ]
    res["kill_count"] = {"card": len(front["kill"]), "ledger": len(q["kill"])}
    res["ladder_matches_ledger"] = [
        {"step": a["step"], "same_claim": norm(a["claim"]) == norm(b["claim"]),
         "same_kind": a["kind"] == b["kind"]}
        for a, b in zip(front["ladder"], q["ladder"])
    ]

    modes = MODES.read_text(encoding="utf-8")
    prereg = dict(re.findall(r'"([A-Za-z0-9_]+)": (-?[\d.]+),', modes.split("PREREGISTERED = {")[1]
                             .split("}")[0]))
    windows = dict(re.findall(r'"([A-Za-z0-9_]+)": \((-?[\d.]+, -?[\d.]+)\),',
                              modes.split("WINDOWS = {")[1].split("}")[0]))
    card_pred = [{"observable": p["observable"][:40], "value": p["value"],
                  "uncertainty": p["uncertainty"]} for p in front["predicts"]]
    res["preregistered_in_script"] = prereg
    res["windows_in_script"] = windows
    res["card_predicts"] = card_pred
    res["value_pm_uncertainty_matches_window"] = []
    keys = ["her_slope", "her_ratio_128", "mix_X_32", "qspine_slope_vs_En",
            "qspine_ratio_b8_over_iid36", "defect_ratio_64_over_8", "defect_slope", "iid_slope"]
    for key, p in zip(keys, front["predicts"]):
        lo, hi = (float(x) for x in windows[key].split(","))
        v, u = float(p["value"]), float(p["uncertainty"])
        res["value_pm_uncertainty_matches_window"].append({
            "key": key, "card_value": v, "script_value": float(prereg[key]),
            "value_agrees": abs(v - float(prereg[key])) <= 1e-9,
            "window": [lo, hi], "value_pm_u": [v - u, v + u],
            "window_agrees": abs((v - u) - lo) <= 0.011 and abs((v + u) - hi) <= 0.011,
        })

    res["caterpillar_mentions"] = [
        norm(line)[:150] for line in raw.splitlines() if "caterpillar" in line.lower()
    ]

    # (ii) is Sigma(e) exactly simple for ANY nondegenerate tetrad?
    rng = np.random.default_rng(20260902)
    worst, worst_small_det = 0.0, (None, 0.0)
    for scale in (0.05, 0.3, 1.0, 3.0):
        for _ in range(50):
            e = np.eye(4) + scale * rng.normal(size=(4, 4))
            det = float(np.linalg.det(e))
            if abs(det) < 1e-6:
                continue
            r = simplicity_residual(geometric_self_dual_triple(e))
            worst = max(worst, r)
            if abs(det) < 0.05 and r > worst_small_det[1]:
                worst_small_det = (det, r)
    res["sigma_of_any_tetrad_is_exactly_simple"] = {
        "max_residual": worst, "worst_near_degenerate": worst_small_det,
        "note": "gram(Sigma(e)) = 2 det(e) I, so the cell-simplicity premise is automatic in the model",
    }
    e = np.eye(4) + 0.3 * rng.normal(size=(4, 4))
    g = plebanski_gram(geometric_self_dual_triple(e))
    res["gram_equals_2det_identity"] = {
        "gram_diag": [float(x) for x in np.diag(g)], "two_det": 2.0 * float(np.linalg.det(e)),
    }

    ref = geometric_self_dual_triple(np.eye(4))
    try:
        normalized_urbantke_metric(-ref)
        refused = False
    except ValueError:
        refused = True
    res["negative_of_reference"] = {
        "simplicity_residual": simplicity_residual(-ref),
        "urbantke_refuses_minus_reference": refused,
        "note": "-Sigma_0 is exactly simple; the block {Sigma_0,-Sigma_0} gives Y=0 (corollary 0/0), "
                "excluded only by the positive-branch scope, not by the derivation assumptions",
    }
    try:
        optimal_internal_alignment(ref, -ref)
        align_ok = True
    except ValueError:
        align_ok = False
    res["negative_of_reference"]["polar_alignment_accepts"] = align_ok

    out = Path(__file__).resolve().parent / "c4_result.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
