"""C4: content 검사 - 예산의 두 항이 실제로 경쟁하는가, S_c 부호 규약이 lstar 를 만드는가.

Gamma_eff/hbar = kappa*S_hat_c - [m ln(l/lP) + c],  kappa = l^2/(8 pi lP^2)
 (i) S_hat_c > 0 (모듈 규약): kappa S_hat 은 l^2 증가, m ln l 은 로그 증가 -> 최소 lstar 존재.
 (ii) S_hat_c < 0 (GHP S_E = -S_geo/8piG): 두 항 모두 l 증가에 대해 Gamma_eff 감소 -> 정류점 없음, lstar 소멸.
검사: 부호를 뒤집었을 때 정류점 방정식 dGamma/dl = S l/(4pi) - m/l = 0 의 실근 존재 여부.
또 budget parts >= 2, conserved_by 가 항등식인지.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
out = {}
S_pos = 7.895885215817185
m = 39
c_R = -91.22790360446403

def gamma(l, S, c=c_R, m=m):
    return S*l**2/(8*math.pi) - m*math.log(l) - c

ls = np.logspace(-1, 2, 400)
for tag, S in (("S_c>0 (module convention)", S_pos), ("S_c<0 (GHP S_E=-S_geo/8piG)", -S_pos)):
    g = np.array([gamma(l, S) for l in ls])
    # 정류점: S l^2 = 4 pi m
    root = 4*math.pi*m/S
    out.setdefault("stationary", {})[tag] = {
        "lstar2": root if root > 0 else None,
        "real_stationary_point_exists": bool(root > 0),
        "gamma_monotone_decreasing": bool(np.all(np.diff(g) < 0)),
        "gamma_at_l=0.1": gamma(0.1, S), "gamma_at_l=100": gamma(100.0, S),
        "argmin_index_interior": bool(0 < int(np.argmin(g)) < len(ls)-1),
    }
out["sign_flip_kills_lstar"] = (out["stationary"]["S_c<0 (GHP S_E=-S_geo/8piG)"]["real_stationary_point_exists"] is False)

# 두 항이 실제로 경쟁하는 구간이 있는가: |dS/dl| vs |d(m ln l)/dl|
# S l/(4pi) < m/l  <=>  l < lstar : 엔트로피 항이 더 빨리 변함
out["competition"] = {
    "entropy_dominant_range": "l < lstar = 7.878 lP",
    "action_dominant_range": "l > lstar",
    "gamma_min_over_hbar_R": m/2 - 0.5*m*math.log(4*math.pi*m/S_pos) - c_R,
    "gamma_min_positive": bool(m/2 - 0.5*m*math.log(4*math.pi*m/S_pos) - c_R > 0),
    "note": "Gamma_min > 0 : Omega_fold < 1 at the crossover, i.e. the fold volume NEVER beats the action at block level. 'Competition' is only in the derivative, not in which term wins.",
}

# Omega_fold at lstar
lnO_at_lstar = m*0.5*math.log(4*math.pi*m/S_pos) + c_R
out["competition"]["ln_Omega_at_lstar_R"] = lnO_at_lstar
out["competition"]["Omega_lt_1_at_lstar"] = bool(lnO_at_lstar < 0)

# budget parts
out["budget_parts_count"] = 2
out["conserved_by_is_identity"] = {
    "claim": "(2pi)^{-m/2} det(K^T N^-1 K)^{-1/2} kappa^{m/2} and Schur det ratio = prod(1-sigma^2)",
    "verified_numerically_in_c1": True,
    "tautology_risk": "The Schur ratio identity is a linear-algebra theorem (true for any N,K), not a physical constraint. It constrains nothing about the physics; it only certifies the code computed the same object two ways.",
}

# lstar 의 c 무관성 확인 (해석적)
out["lstar_independent_of_c"] = {"dGamma/dl": "S l/(4pi) - m/l, c drops out", "true": True,
                                 "but": "lstar depends on m (convention: gauge in numerator) and on |S_c| (sign convention)"}

print(json.dumps(out, ensure_ascii=True, indent=1, default=float))
(HERE/"c4_content_sign.json").write_text(json.dumps(out, ensure_ascii=True, indent=2, default=float), encoding="utf-8")
