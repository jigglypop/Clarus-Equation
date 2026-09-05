"""C3: recovers[1] 'glued 제한 -> Schur H_eff = H_c' 를 정확히 재검사.

C2 에서 rel=1.0 로 실패했다. 원인 후보:
 (a) 나의 T 사상이 틀림 (fine 길이 -> glued 길이의 pullback 은 T^T N T 가 맞는가)
 (b) glued fine 작용의 Hessian 은 sum_a H_a 의 pullback 이 아니라,
     glued 15 길이 위에서 refined 작용 S_fine(l_15) 을 직접 미분해야 한다 (kappa 등분할 때문에 같아야 함)
 (c) 카드 recovers[1] 의 check=0 은 sympy 스칼라 항등이라 이 극한을 실제로 실행하지 않았다.
여기서는 (b) 경로로 S_fine(glued 15 길이) 의 Richardson Hessian 을 직접 잡아 Schur 소거한다.
"""
from __future__ import annotations
import json, math, sys
from itertools import combinations
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
sys.path.insert(0, str(SRC))
import predict_fold_budget as P
import regge_one_to_five_boundary_hessian as RG
from regge_one_to_five_refinement import BOUNDARY_VERTICES

out = {}
points = P.points_from_squared(np.full(10, 2.0))
cells = P.refine([tuple(BOUNDARY_VERTICES)], points)
kap = P.equal_split_kappas(cells, tuple(BOUNDARY_VERTICES), np.full(10, math.pi))
# RG.BOUNDARY_EDGES 순서(= combinations(range(5),2)) 를 앞 10개로 강제, 내부변 (i,5) 5개를 뒤에
edges15 = [tuple(sorted(e)) for e in RG.BOUNDARY_EDGES] + [tuple(sorted((i, 5))) for i in range(5)]
idx = {e: k for k, e in enumerate(edges15)}
# fine 길이 순서: cell a 의 combinations(cell,2)
maps = []
for a, cell in enumerate(cells):
    m = []
    for (i, j) in combinations(cell, 2):
        m.append(idx[tuple(sorted((i, j)))])
    maps.append(m)

def S_fine_glued(l15: np.ndarray) -> float:
    tot = 0.0
    for a, cell in enumerate(cells):
        lv = np.asarray([l15[k] for k in maps[a]])
        tot += P.simplex_action(lv, kap[a])
    return float(tot)

l15_0 = np.zeros(15)
for a, cell in enumerate(cells):
    lv = P.cell_lengths(cell, points)
    for r, k in enumerate(maps[a]):
        l15_0[k] = lv[r]
out["l15_background"] = l15_0.tolist()

H15 = P.richardson_hessian(S_fine_glued, l15_0)
b0 = np.sqrt(np.full(10, 2.0))
Hc = P.richardson_hessian(RG.coarse_euclidean_regge_boundary_action, b0)
out["boundary_lengths_match"] = float(np.max(np.abs(l15_0[:10]-b0)))

Abb = H15[:10, :10]; Bbi = H15[:10, 10:]; Cii = H15[10:, 10:]
Heff = Abb - Bbi @ np.linalg.pinv(Cii) @ Bbi.T
out["direct_fine_glued"] = {"max_abs_diff": float(np.max(np.abs(Heff-Hc))),
                            "rel": float(np.max(np.abs(Heff-Hc))/np.max(np.abs(Hc))),
                            "pass": bool(np.max(np.abs(Heff-Hc)) < 1e-5)}
out["Cii_eigs"] = np.linalg.eigvalsh(Cii).tolist()
out["Cii_rank"] = int(np.sum(np.abs(np.linalg.eigvalsh(Cii)) > 1e-8))
out["Hc_norm"] = float(np.max(np.abs(Hc)))
out["Heff_norm"] = float(np.max(np.abs(Heff)))
out["Abb_norm"] = float(np.max(np.abs(Abb)))
# H_c 는 사실상 0 인가? (flat 1->5 는 perfect action => S_fine(glued)=S_coarse 이면 Heff=Hc)
out["S_fine_at_bg"] = S_fine_glued(l15_0)
out["S_coarse_at_bg"] = float(RG.coarse_euclidean_regge_boundary_action(b0))
out["action_match"] = out["S_fine_at_bg"] - out["S_coarse_at_bg"]
# 정류성: dS_fine/dl_internal = 0 ?
gr, _ = RG._gradient_and_hessian(S_fine_glued, l15_0, 2e-3)
out["grad_internal_norm"] = float(np.linalg.norm(gr[10:]))
out["grad_boundary_norm"] = float(np.linalg.norm(gr[:10]))

print(json.dumps(out, ensure_ascii=True, indent=1, default=float))
(HERE/"c3_glued_schur.json").write_text(json.dumps(out, ensure_ascii=True, indent=2, default=float), encoding="utf-8")
