"""C8: 세 번째 '규약 무관' 수 - [3,2] 섹터 sigma^2 = rho = 0.3208246 가 R 과 W 에서 같은가.
그리고 사다리 3단 예측 '군집 (2,4,5)' 가 관측 (2,5,4) 와 맞는지(카드 본문과 사다리 3단이 다르게 적혔다).
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
sys.path.insert(0, str(SRC))
import predict_fold_budget as P

out = {}
reg = P.regular_level1(2.0)
s2 = np.array(sorted(reg["sigma2_W"]))
rho = np.array(sorted(np.real(reg["rho_R"])))
out["sigma2_W_sorted"] = s2.tolist()
out["rho_R_sorted"] = rho.tolist()
out["sigma2_W_clusters"] = reg["sigma2_W_clusters"]
out["rho_R_clusters"] = reg["rho_R_clusters"]

# [3,2] 값 일치
v_W = [c for c in reg["sigma2_W_clusters"] if c["multiplicity"] == 5][0]["value"]
v_R = [c for c in reg["rho_R_clusters"] if c["multiplicity"] == 5][0]["value"]
out["three_two_sector"] = {"sigma2_W": v_W, "rho_R": v_R, "abs_diff": abs(v_W-v_R),
                           "identical": bool(abs(v_W-v_R) < 1e-9)}

# 군집 다중도
out["multiplicities_W"] = [c["multiplicity"] for c in reg["sigma2_W_clusters"]]
out["multiplicities_R"] = [c["multiplicity"] for c in reg["rho_R_clusters"]]
out["ladder_step3_says"] = "(2,4,5)"
out["observed"] = "W: (2,5,4) by increasing value; R: (4,2,5) by increasing value"
out["ladder_step3_mismatch"] = ("Ladder step 3 claims clusters are (2,4,5) and 'kills the card if clusters are not (2,4,5)'. "
                                "consistency_checks and the card body say (2,5,4). As multiset {2,4,5} both agree; "
                                "as an ordered tuple they do not. P2 notation ambiguity in the kill condition of step 3.")

# trivial sigma^2 = 0 정확성
triv_W = [c for c in reg["sigma2_W_clusters"] if c["multiplicity"] == 2][0]["value"]
out["trivial_sigma2_W"] = triv_W
out["trivial_is_zero"] = bool(abs(triv_W) < 1e-12)

# std 섹터: R 은 -1.327 (음), W 는 0.7448 (양) -> 규약 의존 (카드가 자백)
std_W = [c for c in reg["sigma2_W_clusters"] if c["multiplicity"] == 4][0]["value"]
std_R = [c for c in reg["rho_R_clusters"] if c["multiplicity"] == 4][0]["value"]
out["std_sector"] = {"sigma2_W": std_W, "rho_R": std_R, "convention_dependent": True}
# sigma^2 < 1 인가 (7.3 전제)
out["sigma2_max_lt_1"] = bool(max(s2) < 1.0)
# R 에서 rho > 1 이 있는가 (1-rho < 0 -> log 발산)
out["rho_max"] = float(np.max(rho))
out["any_rho_gt_1"] = bool(np.any(rho > 1.0))
out["one_minus_rho_min"] = float(np.min(1.0-rho))

print(json.dumps(out, ensure_ascii=True, indent=1, default=float))
(HERE/"c8_spectrum.json").write_text(json.dumps(out, ensure_ascii=True, indent=2, default=float), encoding="utf-8")
