"""C5: kill 실행 가능성·검정력 (kill 자체는 실행하지 않는다 - 6·7단 몫).

K1/K4 창 1e-2 가 250차원 FD 조건수에서 의미 있는가:
 d = -1/2 sum log|eig(K^T N^-1 K)| 이므로 FD 잡음 eps 가 N 에 들어가면
 delta_d ~ (1/2) * m * (relative eigenvalue error).  m=234 -> 잡음이 234배 증폭.
여기서는 level-1 (m=39) 에서 FD step 을 바꿔 d 의 실제 변동을 재고, 234차원으로 외삽한다.
K3 창 [0.8,1.25] 검정력: S_hat_c(비정규)/S_hat_c(regular) 와 m 이 그대로일 때 비율이 창 안인지
 -> 실제 irregular 실행 없이, coarse 작용만 저렴하게 재서 추정 (fold 는 안 돌림).
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
sys.path.insert(0, str(SRC))
import predict_fold_budget as P
import regge_one_to_five_boundary_hessian as RG
from regge_one_to_five_refinement import BOUNDARY_VERTICES

out = {}

# ---- FD 잡음: level-1 d 를 step 별로
points = P.points_from_squared(np.full(10, 2.0))
cells = P.refine([tuple(BOUNDARY_VERTICES)], points)
kap = P.equal_split_kappas(cells, tuple(BOUNDARY_VERTICES), np.full(10, math.pi))
lengths = [P.cell_lengths(c, points) for c in cells]

def d_at_step(step):
    hs = []
    for l, k in zip(lengths, kap):
        _, h1 = RG._gradient_and_hessian(lambda v: P.simplex_action(v, k), l, step)
        _, h2 = RG._gradient_and_hessian(lambda v: P.simplex_action(v, k), l, step/2.0)
        hs.append((4.0*h2-h1)/3.0)
    N = np.zeros((50, 50)); NW = np.zeros((50, 50))
    for a, h in enumerate(hs):
        w, v = np.linalg.eigh(h)
        sl = slice(10*a, 10*a+10)
        N[sl, sl] = h; NW[sl, sl] = v@np.diag(np.abs(w))@v.T
    _, s, vt = np.linalg.svd(np.vstack([P.gluing_rows(cells), P.gauge_directions(cells, points, [5]).T]))
    r = int(np.sum(s > 1e-9*s[0])); K = vt[:r].T
    wR = np.linalg.eigvalsh(K.T@np.linalg.inv(N)@K)
    wW = np.linalg.eigvalsh(K.T@np.linalg.inv(NW)@K)
    return -0.5*float(np.sum(np.log(np.abs(wR)))), -0.5*float(np.sum(np.log(wW))), r, float(np.linalg.cond(N))

steps = [4e-3, 2e-3, 1e-3, 5e-4]
tab = {}
for st in steps:
    dR, dW, r, cond = d_at_step(st)
    tab[f"h={st}"] = {"d_R": dR, "d_W": dW, "m": r, "cond_N": cond}
out["fd_step_scan_level1"] = tab
dRs = [tab[f"h={st}"]["d_R"] for st in steps]
dWs = [tab[f"h={st}"]["d_W"] for st in steps]
out["fd_spread_level1"] = {"d_R_max_minus_min": max(dRs)-min(dRs), "d_W_max_minus_min": max(dWs)-min(dWs)}

# 잡음 외삽: d 의 잡음은 대략 m 에 비례 (고윳값당 독립 오차의 합)
sp_R = max(dRs)-min(dRs); sp_W = max(dWs)-min(dWs)
# two_level 잔차 r = d(direct,234) - d(level1,39) - sum_a d(sub,39)  -> 총 m 기여 234+39+5*39=468
out["k1_noise_estimate"] = {
    "level1_m": 39,
    "fd_spread_per_39_modes_R": sp_R,
    "per_mode_R": sp_R/39,
    "total_modes_in_residual": 234 + 39 + 5*39,
    "projected_residual_noise_R": sp_R/39 * (234 + 39 + 5*39),
    "K1_tol": 1e-2,
    "noise_exceeds_tol": bool(sp_R/39*(234+39+5*39) > 1e-2),
}
out["k4_noise_estimate"] = {
    "per_mode_W": sp_W/39,
    "projected_residual_noise_W": sp_W/39 * (234 + 39 + 5*39),
    "K4_min": 1e-2,
    "noise_alone_can_exceed_K4_min": bool(sp_W/39*(234+39+5*39) > 1e-2),
    "note": "K4 kills the card only if |r_W| < 1e-2. If FD noise alone is >1e-2, K4 can NEVER fire => K4 has zero power (auto-pass).",
}

# 250차원 조건수: level-2 N 은 25 cell 직합, cond 는 cell 별과 같음 (블록대각)
out["cond_note"] = "N is block diagonal (per-cell 10x10); cond(N_250) = max over cells ~ same as level-1. The 250-dim issue is mode COUNT (234), not conditioning."

# ---- K3 검정력: 비정규 coarse 작용만 (fold 안 돌림, 싸다)
rng = np.random.default_rng(20260902)
signs = rng.choice([-1.0, 1.0], size=10)
for amp in (0.05, 0.1, 0.15, 0.2):
    sq = 2.0*(1.0+amp*signs)
    b = np.sqrt(sq)
    try:
        s_irr = float(RG.coarse_euclidean_regge_boundary_action(b))/float(np.mean(sq))
        ratio = (4*math.pi*39/s_irr)/62.06884225954214 if s_irr > 0 else float("nan")
    except Exception as e:
        s_irr, ratio = float("nan"), float("nan")
    out.setdefault("k3_scan", {})[f"amp={amp}"] = {"S_hat_c": s_irr, "lstar2_ratio": ratio,
                                                   "in_window": bool(0.8 <= ratio <= 1.25) if ratio == ratio else None}
# 임의 방향 20개 (씨앗 20260902) 로 검정력 통계
rng2 = np.random.default_rng(20260902)
ratios = []
for _ in range(20):
    sg = rng2.choice([-1.0, 1.0], size=10)
    sq = 2.0*(1.0+0.1*sg)
    b = np.sqrt(sq)
    try:
        s = float(RG.coarse_euclidean_regge_boundary_action(b))/float(np.mean(sq))
        if s > 0:
            ratios.append((4*math.pi*39/s)/62.06884225954214)
    except Exception:
        pass
out["k3_random20"] = {"n": len(ratios), "min": min(ratios) if ratios else None, "max": max(ratios) if ratios else None,
                      "frac_in_window": float(np.mean([(0.8 <= r <= 1.25) for r in ratios])) if ratios else None,
                      "note": "ratio depends ONLY on S_hat_c(irregular) since m is fixed at 39; K3 tests the coarse action, not the fold volume."}

print(json.dumps(out, ensure_ascii=True, indent=1, default=float))
(HERE/"c5_kill_power.json").write_text(json.dumps(out, ensure_ascii=True, indent=2, default=float), encoding="utf-8")
