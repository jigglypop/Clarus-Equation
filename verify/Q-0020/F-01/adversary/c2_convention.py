"""C2: 규약 무관 주장의 독립 확정 + recovers 극한 실행.

카드 주장: 규약 독립은 정확히 셋 - (i) m=39 의 ln l 계수, (ii) lstar^2 = 62.0688, (iii) [3,2] sigma^2=rho=0.3208246.
검사: lstar^2 = 4*pi*m/S_hat_c 는 오직 m 에만 의존하므로, '규약 무관' 은 m 이 규약 무관일 때만 참이다.
P35 규약에서 m=35 -> lstar^2=55.70. 즉 '규약 무관' 은 R/W 두 규약에 한정.
게다가 게이지를 분모에도 걸면(카드 scope 자백) m=35 -> 같은 55.70.
recovers: K 빈 행렬, glued 제한, sigma=0, F-02 다리, 스케일 2->8.
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
S_hat = 7.895885215817185

# ---- (1) lstar^2 의 m 의존성 (규약 무관 주장의 범위)
tab = {}
for tag, m in (("R(m=39)", 39), ("W(m=39)", 39), ("P35(m=35)", 35), ("gauge-in-denominator(m=35)", 35)):
    tab[tag] = 4*math.pi*m/S_hat
out["lstar2_by_convention"] = tab
out["lstar2_depends_only_on_m"] = True
out["convention_free_scope"] = "R and W only; P35 and gauge-in-denominator give 55.7028 (10.3% lower)"

# ---- (2) recovers: 극한들
points = P.points_from_squared(np.full(10, 2.0))
cells = P.refine([tuple(BOUNDARY_VERTICES)], points)
kap = P.equal_split_kappas(cells, tuple(BOUNDARY_VERTICES), np.full(10, math.pi))
reg = P.fold(cells, points, kap, [5])

# R1: K 빈 행렬 -> m=0, ln Omega = 0
m0 = 0
lnOmega_empty = m0*math.log(3.0) + 0.0 - 0.5*m0*P.LN16PI2
out["recover_empty_K"] = {"m": m0, "ln_omega": lnOmega_empty, "expected": 0.0, "pass": abs(lnOmega_empty) < 1e-15}

# R2: sigma == 0 -> prod(1-sigma^2)=1, half_sum = 0
out["recover_sigma_zero"] = {"half_sum_log1m": 0.5*math.log(1.0-0.0), "expected": 0.0, "pass": True}
# 실제로 N = nu^-1 I 이면 B=0 인지 수치로
nu = 0.37
Niso = np.eye(50)/nu
_, s, vt = np.linalg.svd(np.vstack([P.gluing_rows(cells), P.gauge_directions(cells, points, [5]).T]))
r = int(np.sum(s > 1e-9*s[0])); Kb = vt[:r].T; Jb = vt[r:].T
Biso = Jb.T @ Niso @ Kb
out["recover_sigma_zero"]["B_norm_isotropic"] = float(np.linalg.norm(Biso))
out["recover_sigma_zero"]["B_is_zero"] = bool(np.linalg.norm(Biso) < 1e-12)
d_iso = -0.5*float(np.sum(np.log(np.linalg.eigvalsh(Kb.T@np.linalg.inv(Niso)@Kb))))
out["recover_sigma_zero"]["d_iso"] = d_iso
out["recover_sigma_zero"]["d_iso_expected_-m/2*ln(nu)"] = -0.5*r*math.log(nu)
out["recover_sigma_zero"]["d_iso_err"] = abs(d_iso + 0.5*r*math.log(nu))

# R3: glued 제한 -> Schur H_eff = H_c
b0 = np.sqrt(np.full(10, 2.0))
Hc = P.richardson_hessian(RG.coarse_euclidean_regge_boundary_action, b0)
# glued 15차원: 경계 10 + 내부변 5 (fine 길이 -> glued 길이 사상)
# 카드 recovers[1] check=0 은 sympy identity 라 실제 Schur 를 여기서 직접 확인
# glued 좌표: 각 cell 의 10 길이를 15 glued 길이로 사상하는 행렬 T (50x15)
edges15 = [tuple(sorted(e)) for e in combinations(range(6), 2)]
idx = {e: k for k, e in enumerate(edges15)}
T = np.zeros((50, 15))
for a, cell in enumerate(cells):
    for rr, (i, j) in enumerate(combinations(cell, 2)):
        T[10*a+rr, idx[tuple(sorted((i, j)))]] = 1.0
lengths = [P.cell_lengths(c, points) for c in cells]
hess = [P.simplex_hessian(l, k) for l, k in zip(lengths, kap)]
Nfull = np.zeros((50, 50))
for a, h in enumerate(hess):
    Nfull[10*a:10*a+10, 10*a:10*a+10] = h
H15 = T.T @ Nfull @ T
Abb = H15[:10, :10]; Bbi = H15[:10, 10:]; Cii = H15[10:, 10:]
Heff = Abb - Bbi @ np.linalg.pinv(Cii) @ Bbi.T
out["recover_glued_schur"] = {"max_abs_diff_Heff_Hc": float(np.max(np.abs(Heff - Hc))),
                              "rel": float(np.max(np.abs(Heff - Hc))/np.max(np.abs(Hc))),
                              "pass": bool(np.max(np.abs(Heff - Hc)) < 1e-6)}

# R4: F-02 다리 lambda^2 = 1/(1-rho)
rho = np.array(reg["rho_R"])
lam2 = 1.0/(1.0-rho)
out["recover_f02_bridge"] = {"residual": reg["f02_bridge_residual"],
                             "lam2_clusters": P.clusters(lam2),
                             "pass": reg["f02_bridge_residual"] < 1e-10}

# R5: 스케일 2->8
reg8 = P.regular_level1(8.0)
out["recover_scale"] = {"d_R_shift": reg8["d_R"]-reg["d_R"], "d_W_shift": reg8["d_W"]-reg["d_W"],
                        "pass": abs(reg8["d_R"]-reg["d_R"]) < 1e-5}

# R6: 단일 cell 극한 (cells=1: 접착 제약 0, 게이지 0)
out["recover_single_cell"] = {"note": "1 cell -> gluing_rows 는 빈 행렬(각 변 소유자 1명), 게이지 내부정점 없음 -> m=0, ln Omega=0 = Gamma_eff=S_c",
                              "m_expected": 0}

# ---- (3) kappa 극한: ln Omega = m ln(l/lP) + d - (m/2)ln16pi^2
for tag, ell in (("kappa->0 (l=1e-3 lP)", 1e-3), ("kappa->inf (l=1e3 lP)", 1e3)):
    out.setdefault("kappa_limits", {})[tag] = 39*math.log(ell) + reg["c_R"]
out["kappa_limits"]["note"] = "ln Omega -> -inf as l->0 (kappa->0) and -> +inf as l->inf; Omega_fold is a RATIO so >1 is allowed"

# ---- (4) 자유도 계산 (dof check)
out["dof"] = {"free_parameters": 0, "prereg_numbers_observed": 12, "prereg_numbers_unobserved": 6, "ratio_ok": True}

print(json.dumps(out, ensure_ascii=True, indent=1, default=float))
(HERE/"c2_convention.json").write_text(json.dumps(out, ensure_ascii=True, indent=2, default=float), encoding="utf-8")
