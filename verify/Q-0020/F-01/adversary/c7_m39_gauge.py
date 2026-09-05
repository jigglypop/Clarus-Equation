"""C7: m=39 이 게이지를 분자에만 거는 규약에 의존하는가 + hidden assumptions 수치 확인.

카드 scope: '게이지를 분모에도 걸면 m=35 가 되고 두 단계 가법성(K1)이 깨진다 - 분자에만 거는 것이 가법성을 주는 정의다.'
이 문장은 K1 결과를 이미 안다고 전제한다(two_level 미실행). 즉 '가법성이 m=39 를 고른다' 는
아직 관측이 아니라 가정이다 -> hidden assumption.

또한: lstar^2 = 4 pi m / S_hat 은 m 에 선형이므로 m 이 규약이면 lstar 도 규약이다.
'규약 무관' 은 오직 {R,W} 두 부호 규약에 대한 무관성이며, 게이지 규약·측도 규약에는 의존.
검사: 게이지 4방향이 정말 N 의 영방향인가(g^T N g ~ 0), 그렇다면 분모의 Gaussian 이 그 방향으로 발산 -> 정규화 필요.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
sys.path.insert(0, str(SRC))
import predict_fold_budget as P
from regge_one_to_five_refinement import BOUNDARY_VERTICES

out = {}
points = P.points_from_squared(np.full(10, 2.0))
cells = P.refine([tuple(BOUNDARY_VERTICES)], points)
kap = P.equal_split_kappas(cells, tuple(BOUNDARY_VERTICES), np.full(10, math.pi))
lengths = [P.cell_lengths(c, points) for c in cells]
hess = [P.simplex_hessian(l, k) for l, k in zip(lengths, kap)]
N = np.zeros((50,50))
for a,h in enumerate(hess):
    N[10*a:10*a+10, 10*a:10*a+10] = h
g = P.gauge_directions(cells, points, [5])
Gam = P.gluing_rows(cells)

out["gauge_gTNg"] = (g.T@N@g).tolist()
out["gauge_gTNg_norm"] = float(np.linalg.norm(g.T@N@g))
out["gauge_Ng_norm"] = float(np.linalg.norm(N@g))
out["gauge_is_null_of_unglued_N"] = bool(np.linalg.norm(g.T@N@g)/np.linalg.norm(N@g) < 1e-6)
out["gauge_Ng_itself_norm"] = float(np.linalg.norm(N@g))
out["gauge_is_exact_null_vector"] = bool(np.linalg.norm(N@g) < 1e-6)
out["gauge_note"] = ("g^T N g ~ 0 (2.5e-8) but N g is NOT zero (norm below). So g spans a NULL CONE direction of the "
                     "indefinite form, not a kernel. The unglued Gaussian does not literally diverge along g, but the "
                     "quadratic form vanishes there: exp(-kappa/2 * 0) = 1, flat direction => the DENOMINATOR integral "
                     "is divergent along g regardless. Both numerator and denominator are formal.")

# m 의 규약 의존 표
tab = {}
_,s,vt = np.linalg.svd(np.vstack([Gam, g.T])); r=int(np.sum(s>1e-9*s[0]))
tab["gauge_in_numerator_only"] = {"m": r, "lstar2": 4*math.pi*r/7.895885215817185}
_,sg,vtg = np.linalg.svd(Gam); rg=int(np.sum(sg>1e-9*sg[0]))
tab["gauge_in_both (cancels)"] = {"m": rg, "lstar2": 4*math.pi*rg/7.895885215817185}
out["m_by_gauge_convention"] = tab
out["lstar2_relative_spread"] = (tab["gauge_in_numerator_only"]["lstar2"]-tab["gauge_in_both (cancels)"]["lstar2"])/tab["gauge_in_numerator_only"]["lstar2"]

# 접착 제약이 게이지와 독립인지 (rank 39 = 35+4)
out["rank_check"] = {"rank_Gam": rg, "rank_[Gam;g]": r, "independent": bool(r == rg + g.shape[1])}

# hidden assumption: kappa = l^2/(8 pi lP^2) 의 8pi 와 (m/2)ln16pi^2 는 어디서?
out["normalization_constants"] = {
    "kappa_prefactor": "1/(8 pi)  from S = S_geo/(8 pi G)",
    "16pi2_origin": "delta normalization convention: (2pi)^{-m/2} kappa^{m/2} with kappa=l^2/(8 pi lP^2) gives -(m/2)ln(16 pi^2) when written as m ln(l/lP)",
    "check_2pi_times_8pi": 2*math.pi*8*math.pi,
    "equals_16pi2": bool(abs(2*math.pi*8*math.pi - 16*math.pi**2) < 1e-12),
    "verdict": "16 pi^2 = 2pi * 8pi exactly: the constant is entirely a delta-normalization + action-normalization artifact, as the card admits. It fixes Gamma_min and l_Omega but not lstar.",
}

print(json.dumps(out, ensure_ascii=True, indent=1, default=float))
(HERE/"c7_m39_gauge.json").write_text(json.dumps(out, ensure_ascii=True, indent=2, default=float), encoding="utf-8")
