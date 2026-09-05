"""C1: Gaussian 제약 부피 항등식의 독립 재계산.

카드 주장(사다리 2단): ln Omega = (m/2)ln(kappa/2pi) - (1/2)ln det(K^T N^-1 K)  ... 정규화 비.
카드 구현은 rows=[Gam; g^T] 의 SVD 우기저 K 를 쓴다. 여기서는
 (a) QR 기반 다른 정규직교 기저 K2 (부호·회전 다름)
 (b) 직접 marginalization: 제약을 좌표변환으로 풀어 Schur 보수로 계산
 (c) Cholesky 기반 det (W 규약, 정부호일 때만)
세 경로로 d = -1/2 ln det(K^T N^-1 K) 를 재계산하고, 기저 회전 불변성을 확인한다.
또한 R 규약(N 부정부호)에서 실제 Gaussian 적분이 발산하는 것을 스칼라 예제로 보인다.
"""
from __future__ import annotations
import json, math, sys
from itertools import combinations
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
sys.path.insert(0, str(SRC))
import predict_fold_budget as P  # 배경 기하만 재사용(독립 유도는 선형대수 쪽)
from regge_one_to_five_refinement import BOUNDARY_VERTICES

out = {}

# ---- 배경: regular 1->5
points = P.points_from_squared(np.full(10, 2.0))
cells = P.refine([tuple(BOUNDARY_VERTICES)], points)
kap = P.equal_split_kappas(cells, tuple(BOUNDARY_VERTICES), np.full(10, math.pi))
lengths = [P.cell_lengths(c, points) for c in cells]
hess = [P.simplex_hessian(l, k) for l, k in zip(lengths, kap)]
dof = 50
N = np.zeros((dof, dof)); NW = np.zeros((dof, dof))
for a, h in enumerate(hess):
    w, v = np.linalg.eigh(h)
    sl = slice(10*a, 10*a+10)
    N[sl, sl] = h
    NW[sl, sl] = v @ np.diag(np.abs(w)) @ v.T

Gam = P.gluing_rows(cells)
g = P.gauge_directions(cells, points, [5])
rows = np.vstack([Gam, g.T])

# (a) QR 기반 기저 (카드의 SVD 기저와 다른 회전)
Q, Rq = np.linalg.qr(rows.T)   # rows.T is 50x39
rank_qr = int(np.sum(np.abs(np.diag(Rq)) > 1e-9*np.max(np.abs(np.diag(Rq)))))
K_qr = Q[:, :rank_qr]
# 카드 기저
_, s, vt = np.linalg.svd(rows)
rank_svd = int(np.sum(s > 1e-9*s[0]))
K_svd = vt[:rank_svd].T

out["rank_qr"] = rank_qr
out["rank_svd"] = rank_svd
out["subspace_agreement"] = float(np.linalg.norm(K_qr@K_qr.T - K_svd@K_svd.T))

Ninv = np.linalg.inv(N)
NWinv = np.linalg.inv(NW)

def d_from(Kb, Minv, use_abs):
    w = np.linalg.eigvalsh(Kb.T @ Minv @ Kb)
    val = np.abs(w) if use_abs else w
    return -0.5*float(np.sum(np.log(val))), int(np.sum(w < 0))

d_R_qr, nneg_qr = d_from(K_qr, Ninv, True)
d_R_svd, nneg_svd = d_from(K_svd, Ninv, True)
d_W_qr, _ = d_from(K_qr, NWinv, False)
d_W_svd, _ = d_from(K_svd, NWinv, False)
out["d_R_qr"] = d_R_qr; out["d_R_svd"] = d_R_svd
out["d_R_basis_diff"] = d_R_qr - d_R_svd
out["n_neg_qr"] = nneg_qr; out["n_neg_svd"] = nneg_svd
out["d_W_qr"] = d_W_qr; out["d_W_svd"] = d_W_svd
out["d_W_basis_diff"] = d_W_qr - d_W_svd
out["card_d_R"] = 7.482042027338306
out["card_d_W"] = 4.197596512637077
out["reproduce_d_R_err"] = abs(d_R_qr - 7.482042027338306)
out["reproduce_d_W_err"] = abs(d_W_qr - 4.197596512637077)

# (b) 독립 경로: Schur.  [Jp K] 직교완전기저이면
#     K^T N^-1 K = (C - B^T A^-1 B)^-1,  A=Jp^T N Jp, C=K^T N K, B=Jp^T N K
Jp_svd = vt[rank_svd:].T
for tag, M in (("R", N), ("W", NW)):
    A = Jp_svd.T @ M @ Jp_svd
    C = K_svd.T @ M @ K_svd
    B = Jp_svd.T @ M @ K_svd
    S = C - B.T @ np.linalg.solve(A, B)
    lhs = K_svd.T @ np.linalg.inv(M) @ K_svd
    out[f"schur_inverse_residual_{tag}"] = float(np.linalg.norm(lhs @ S - np.eye(rank_svd)))
    # d via Schur: -1/2 ln|det (S^-1)| = +1/2 ln|det S|
    sign, logabs = np.linalg.slogdet(S)
    out[f"d_via_schur_{tag}"] = 0.5*float(logabs)

# (c) Cholesky 경로 (W 정부호)
L = np.linalg.cholesky(NW)          # NW = L L^T
Y = np.linalg.solve(L, K_svd)       # K^T NW^-1 K = Y^T Y ... 아니 = (L^-1 K)^T (L^-1 K)
M2 = Y.T @ Y
sgn, la = np.linalg.slogdet(M2)
out["d_W_cholesky"] = -0.5*float(la)

# (d) 직접 수치 적분: 저차원 장난감으로 항등식 자체를 검증
rng = np.random.default_rng(20260902)
toy = {}
n_dim, m_dim = 4, 2
Araw = rng.normal(size=(n_dim, n_dim)); Npos = Araw@Araw.T + n_dim*np.eye(n_dim)
Kraw = rng.normal(size=(n_dim, m_dim)); Ktoy, _ = np.linalg.qr(Kraw)
kappa = 1.7
# 해석식: Omega = (2pi)^{-m/2} kappa^{m/2} det(K^T N^-1 K)^{-1/2}
ana = (2*math.pi)**(-m_dim/2) * kappa**(m_dim/2) / math.sqrt(np.linalg.det(Ktoy.T@np.linalg.inv(Npos)@Ktoy))
# 몬테카를로: delta^m(K^T xi) 를 좁은 Gaussian 으로 근사, eps->0
mc = []
for eps in (0.05, 0.02, 0.01):
    Neff = Npos*kappa + (Ktoy@Ktoy.T)/eps**2
    # ratio = int e^{-1/2 xi^T Neff xi} * (2pi eps^2)^{-m/2}  / int e^{-kappa/2 xi^T N xi}
    r = math.sqrt(np.linalg.det(Npos*kappa)/np.linalg.det(Neff)) * (2*math.pi*eps**2)**(-m_dim/2)
    mc.append(r)
toy["analytic"] = ana
toy["delta_regularized"] = mc
toy["rel_err_finest"] = abs(mc[-1]-ana)/ana
out["toy_identity"] = toy

# (e) R 규약: 실제 Gaussian 적분의 발산 — 부정부호 1차원 예
out["indefinite_note"] = {
    "n_neg_of_KtNinvK": nneg_svd,
    "signature_N": [int(np.sum(np.linalg.eigvalsh(N) > 0)), int(np.sum(np.linalg.eigvalsh(N) < 0))],
    "explain": "N has 10 positive / 40 negative eigenvalues; exp(-kappa xi^T N xi/2) diverges along 40 directions. R takes |det| of K^T N^-1 K, of which 31/39 eigenvalues are negative: the 'volume ratio' is a product of 31 negative numbers whose absolute value is retained and phase e^{i pi*31/2} discarded. This is a declared convention, not a convergent integral.",
}

# (f) 위상: 만약 위상을 버리지 않으면 Omega 는 복소수. |Omega| 만 남기면 ln Omega 실수.
out["discarded_phase_deg"] = (nneg_svd * 90.0) % 360.0

print(json.dumps(out, ensure_ascii=True, indent=1, default=float))
(HERE/"c1_gaussian_identity.json").write_text(json.dumps(out, ensure_ascii=True, indent=2, default=float), encoding="utf-8")
