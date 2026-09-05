"""Q-0013 F-02 사전등록 시험: master 식 + 바닥 소멸 조건 tl G(Sigma) = 0 (5개 조건)

    eps(n)^2 = delta^4 [ (tr H kappa)^2 ||tl G(Sigma)||_F^2 + 2 ||H kappa H||_F^2 T(Sigma) ] / (12 n^2)

카드: derivations/Q-0013/F-02.formula.md.  자유 파라미터 0개 — PRED의 숫자는 모두 `--mode constants`의
정확 선형대수(해석적 구조상수 M^{ab}, 유리수)에서 나오며 MC 적합이 아니다.  F-01 스크립트를 복사해 고쳤다:
MIN_DET 재추출(Q-0008 F-02 check_modes와 같은 규약), 새 모드(kernel/diag/univ), 영모드 시험 delta<=0.1,
모델 표준오차로 고정한 창.

사전등록 상수(결과를 본 뒤 바꾸지 않는다):
    DELTA = 0.005,  SIZES = (4, 8, 16, 32, 64),  TRIALS = 512,  N2_TRIALS = 3000,  SEED = 20260902
    MIN_DET = 0.05: 어느 cell이든 det(I + delta*label) <= MIN_DET 이면 구성 전체 재추출(재추출 횟수 기록).
    영모드 시험만 delta in (0.005, 0.1).  delta=0.1에서 단일 대각 성분의 정렬 반전(1+delta*g<=MIN_DET)은
    g<=-9.5, 확률 ~1e-21/추출 — 무시 가능. 3-대각 모드는 (1+delta g)^3<=0.05 <=> g<=-6.3, ~1.5e-10/추출.
    통계는 12.4 정규화 잔차 ||tl G||_F/||G||_F 의 trial RMS.
    창 = 정확값 +- (4 x 모델 표준오차 + 1% 계통 여유[O(delta) 정규화 보정]); 기울기·floor_hat은 4 x 모델 표준오차만.
    모델 표준오차: O(delta^2) 가우스 2차형식 모형(기하 없음)을 MODEL_REPLICATES회 반복한 통계의 표준편차.

모드:
    constants  구조상수·5조건 계수표·정확 예측값·모델 표준오차(무작위 tetrad 없음) -> structure_constants.json
    kernel     K1: Sigma_b = 1/2 (e01+e23)(e01+e23)^T + 2 e03 e03^T  (w=(1,0,2) 불균형인데 tl G = 0) -> 바닥 0
    diag       K2: Sigma = I_4(대각 4성분만) / I_12(비대각 12성분만) / I_16, n=2 (3000 trial) + I_4 크기 곡선
    univ       K3: Sigma_o = 1/2 (e01+e23)(e01+e23)^T + 3 e03 e03^T  vs  Sigma_d = e03 e03^T : 같은 tl G = P_3
    zero       K4: e_11 단일 성분 및 (e00+e11+e22)/sqrt3 의 정확 영모드, delta in (0.005, 0.1), MIN_DET
    axis       일관성(kill 아님): (2,3) vs (0,1) 공통 난수 비 = 1 (이산 SO(4) 항등식)
    all        위 다섯(axis 포함) -> result.json
    smoke      작은 크기·적은 trial의 배관 점검(판정에 쓰지 않는다) -> smoke.json

사용: .claude/hooks/python.cmd python verify/Q-0013/F-02/check_floor.py --mode all
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
sys.path.insert(0, str(ROOT))
from examples.physics.gravity.causal_face_simplicity import (  # noqa: E402
    geometric_self_dual_triple,
    simplicity_residual,
    two_form_from_vectors,
    wedge_scalar,
)
from examples.physics.gravity.urbantke_shape_matching_rg import optimal_internal_alignment  # noqa: E402

OUT = ROOT / "verify" / "Q-0013" / "F-02"

# ---------------------------------------------------------------- 사전등록 상수
DELTA = 0.005
ZERO_DELTAS = (0.005, 0.1)
SIZES = (4, 8, 16, 32, 64)
TRIALS = 512
N2_TRIALS = 3000
SEED = 20260902
MIN_DET = 0.05
MODEL_REPLICATES = 200

# 카드의 사전등록 값과 창 (derivations/Q-0013/F-02.formula.md predicts/kill).
# 값은 정확 선형대수(structure_constants.json "exact_pred"), 창은 같은 파일의 "window_from_model"
# (4 x 모델 표준오차 + 1% 계통 여유, MODEL_REPLICATES=200, 씨앗 SEED+1000+r)을 소수 5자리로 옮긴 것. 결과를 본 뒤 바꾸지 않는다.
PRED = {
    "ker_eps64_over_delta2": 0.07733980,
    "ker_slope": -0.45342619,
    "diag_eps2_over_delta2": 0.57735027,
    "diag_eps64_over_delta2": 0.14320549,
    "diag_slope": -0.45342619,
    "off_eps2_over_delta2": 1.08012345,
    "iso_eps2_over_delta2": 1.58113883,
    "cross_eps2_sq_over_delta4": 1.0,
    "univ_o_eps64_over_delta2": 0.15118752,
    "univ_o_eps4_over_delta2": 0.34985116,
    "univ_floor_hat_over_delta2": 0.11785113,
    "univ_d_eps64_over_delta2": 0.11783674,
    "zero_max_residual": 0.0,
}
WINDOW = {
    "ker_eps64_over_delta2": (0.06959, 0.08509),
    "ker_slope": (-0.5056, -0.4013),
    "diag_eps2_over_delta2": (0.52739, 0.62731),
    "diag_eps64_over_delta2": (0.12819, 0.15822),
    "diag_slope": (-0.5037, -0.4032),
    "off_eps2_over_delta2": (1.01470, 1.14554),
    "iso_eps2_over_delta2": (1.49767, 1.66461),
    "cross_eps2_sq_over_delta4": (0.71025, 1.28975),
    "univ_o_eps64_over_delta2": (0.13748, 0.16490),
    "univ_o_eps4_over_delta2": (0.29748, 0.40222),
    "univ_floor_hat_over_delta2": (0.09661, 0.13909),
    "univ_d_eps64_over_delta2": (0.11280, 0.12287),
    "zero_max_residual": (-1.0e-12, 1.0e-12),
}

REF = geometric_self_dual_triple(np.eye(4))
NORM_G0 = 2.0 * math.sqrt(3.0)  # ||G(Sigma_0)||_F
I4 = np.eye(4)
EPS3 = np.zeros((3, 3, 3))
EPS3[0, 1, 2] = EPS3[1, 2, 0] = EPS3[2, 0, 1] = 1.0
EPS3[0, 2, 1] = EPS3[2, 1, 0] = EPS3[1, 0, 2] = -1.0
E3 = np.eye(3)
P_BASIS = [0.5 * (np.outer(E3[k], E3[k]) - E3 / 3.0) for k in range(3)]
S_BASIS = {"12": np.outer(E3[0], E3[1]) + np.outer(E3[1], E3[0]),
           "23": np.outer(E3[1], E3[2]) + np.outer(E3[2], E3[1]),
           "31": np.outer(E3[2], E3[0]) + np.outer(E3[0], E3[2])}
NAMES = ["%d%d" % (a // 4, a % 4) for a in range(16)]
CLASS = {1: ["01", "10", "23", "32"], 2: ["02", "20", "31", "13"], 3: ["03", "30", "12", "21"]}
DIAG = ["00", "11", "22", "33"]


# ---------------------------------------------------------------- 라벨 방향
def e(mu: int, nu: int) -> np.ndarray:
    v = np.zeros(16)
    v[4 * mu + nu] = 1.0
    return v


def idx(name: str) -> int:
    return 4 * int(name[0]) + int(name[1])


SQ2, SQ3 = math.sqrt(2.0), math.sqrt(3.0)
SPECS = {
    # 이름: [(16-벡터 방향, 진폭)]  ->  라벨 l = sum_j g_j * 진폭_j * 방향_j,  g_j ~ N(0,1) 독립
    "kernel": [((e(0, 1) + e(2, 3)) / SQ2, 1.0), (e(0, 3), SQ2)],
    "diag4": [(e(m, m), 1.0) for m in range(4)],
    "off12": [(e(m, n), 1.0) for m in range(4) for n in range(4) if m != n],
    "iso16": [(e(m, n), 1.0) for m in range(4) for n in range(4)],
    "univ_o": [((e(0, 1) + e(2, 3)) / SQ2, 1.0), (e(0, 3), SQ3)],
    "univ_d": [(e(0, 3), 1.0)],
    "zero_11": [(e(1, 1), 1.0)],
    "zero_3diag": [((e(0, 0) + e(1, 1) + e(2, 2)) / SQ3, 1.0)],
    "axis_01": [(e(0, 1), 1.0)],
    "axis_23": [(e(2, 3), 1.0)],
    # F-01 반례 3개 (이미 관측, recovers 전용)
    "ce_i": [((e(0, 1) + e(0, 2) + e(0, 3)) / SQ3, 1.0)],
    "ce_ii": [((e(0, 0) + e(1, 1)) / SQ2, 1.0)],
    "ce_iii": [(e(0, 1), 1.0), ((e(0, 0) + e(1, 1)) / 2.0, 1.0)],
}


def factor(spec) -> np.ndarray:
    """A (16 x r): Sigma = A A^T."""
    return np.array([s * v for v, s in spec]).T


def sigma_of(spec) -> np.ndarray:
    A = factor(spec)
    return A @ A.T


# ---------------------------------------------------------------- 블록 잔차 (MIN_DET 재추출)
def cell(label: np.ndarray, delta: float) -> np.ndarray:
    """polar 정렬된 한 cell의 자기쌍대 삼중항 (Q-0008 F-02 check_modes와 같은 규약)."""
    triple = geometric_self_dual_triple(np.eye(4) + delta * label)
    return optimal_internal_alignment(REF, triple).aligned_candidate


def block_residual(labels: np.ndarray, delta: float = DELTA) -> float:
    return simplicity_residual(sum(cell(lab, delta) for lab in labels))


class Resampler:
    """어느 cell이든 det(I + delta*label) <= MIN_DET 이면 구성 전체 재추출."""

    def __init__(self):
        self.resampled = 0

    def draw(self, rng, n: int, A: np.ndarray, delta: float, g=None) -> tuple[np.ndarray, np.ndarray]:
        r = A.shape[1]
        while True:
            gg = rng.normal(size=(n, r)) if g is None else g
            lab = (gg @ A.T).reshape(n, 4, 4)
            dets = np.linalg.det(np.eye(4)[None] + delta * lab)
            if np.all(dets > MIN_DET):
                return lab, gg
            self.resampled += 1
            g = None


def rms(values) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(array * array)))


def loglog_slope(sizes, values) -> float:
    return float(np.polyfit(np.log(np.asarray(sizes, float)), np.log(np.asarray(values, float)), 1)[0])


# ---------------------------------------------------------------- 구조상수 (해석적, 정확)
def wf(u, v):
    return two_form_from_vectors(np.asarray(u, float), np.asarray(v, float))


def d_linear(l: np.ndarray) -> np.ndarray:
    """d/d delta Sigma(I + delta l) at 0 (정렬 전)."""
    l = np.asarray(l, float)
    out = []
    for i in range(3):
        f = wf(l[0], I4[i + 1]) + wf(I4[0], l[i + 1])
        for j in range(3):
            for k in range(3):
                if EPS3[i, j, k]:
                    f = f + 0.5 * EPS3[i, j, k] * (wf(l[j + 1], I4[k + 1]) + wf(I4[j + 1], l[k + 1]))
        out.append(f)
    return np.asarray(out)


def l_tilde(l: np.ndarray) -> np.ndarray:
    """L~(l) = d/d delta [ R_delta Sigma(I + delta l) ] at 0 : 정렬의 1계 회전은 c1의 반대칭 부분."""
    d = d_linear(l)
    c1 = np.array([[wedge_scalar(REF[i], d[j]) for j in range(3)] for i in range(3)])
    return d + ((c1 - c1.T) / 4.0) @ REF


def traceless(matrix: np.ndarray) -> np.ndarray:
    return matrix - np.trace(matrix) / 3.0 * np.eye(3)


def structure_constants() -> tuple[np.ndarray, np.ndarray]:
    basis = [np.zeros((4, 4)) for _ in range(16)]
    for a in range(16):
        basis[a][a // 4, a % 4] = 1.0
    L = [l_tilde(b) for b in basis]
    M = np.zeros((16, 16, 3, 3))
    for a in range(16):
        for b in range(16):
            g = np.array([[wedge_scalar(L[a][i], L[b][j]) for j in range(3)] for i in range(3)])
            M[a, b] = 0.5 * (g + g.T)
    Mt = np.array([[traceless(M[a, b]) for b in range(16)] for a in range(16)])
    return M, Mt


def fd_structure_constants(step: float = 1.0e-5) -> np.ndarray:
    """F-01 방식(중심차분) M — 해석적 M과의 대조용."""
    basis = [np.zeros((4, 4)) for _ in range(16)]
    for a in range(16):
        basis[a][a // 4, a % 4] = 1.0
    L = [(cell(b, step) - cell(b, -step)) / (2.0 * step) for b in basis]
    M = np.zeros((16, 16, 3, 3))
    for a in range(16):
        for b in range(16):
            g = np.array([[wedge_scalar(L[a][i], L[b][j]) for j in range(3)] for i in range(3)])
            M[a, b] = 0.5 * (g + g.T)
    return M


def floor_amplitude(sigma: np.ndarray, Mt: np.ndarray) -> float:
    """F = ||tl G(Sigma)||_F."""
    return float(np.linalg.norm(np.einsum("ab,abij->ij", sigma, Mt)))


def fluctuation_amplitude(sigma: np.ndarray, Mt: np.ndarray) -> float:
    """T(Sigma) = sum_ij sum_abcd tlM^ab_ij Sigma_ac Sigma_bd tlM^cd_ij (Wick 4점)."""
    return float(np.einsum("abij,ac,bd,cdij->", Mt, sigma, sigma, Mt))


def predicted_eps_over_delta2(n: int, F: float, T: float) -> float:
    """i.i.d. cell(kappa=I): tr(H)=||HIH||_F^2=n-1."""
    return math.sqrt((n - 1) * ((n - 1) * F * F + 2.0 * T) / (12.0 * n * n))


def floor_hat(eps4: float, eps64: float) -> float:
    """두 크기에서 F^2 = (A_64 - A_4)/60, A_n = 12 n^2 eps_n^2/(n-1) = (n-1)F^2 + 2T  ->  floor = F/(2 sqrt3)."""
    A4 = 12.0 * 16 * eps4 * eps4 / 3.0
    A64 = 12.0 * 4096 * eps64 * eps64 / 63.0
    return math.sqrt(max((A64 - A4) / 60.0, 0.0)) / NORM_G0


# ---------------------------------------------------------------- 5개 조건 (카드 3단의 닫힌 규칙)
def s_sign(a: str) -> float:
    """클래스 l의 원소 a: 시간-공간 (0l),(l0) 은 +1, 공간-공간은 -1."""
    return 1.0 if "0" in a else -1.0


def card_W_X(sigma: np.ndarray) -> tuple[np.ndarray, dict]:
    """카드 3단의 명시 규칙: tl G(Sigma) = sum_k W_k P_k + 1/4 sum_{l<m} X_lm S_lm."""
    def Sg(a, b):
        return sigma[idx(a), idx(b)]

    cyc = {1: (2, 3), 2: (3, 1), 3: (1, 2)}
    w, p, D, C = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)
    dpairs = {1: [("00", "11"), ("22", "33")], 2: [("00", "22"), ("11", "33")], 3: [("00", "33"), ("11", "22")]}
    for k in (1, 2, 3):
        ts = [a for a in CLASS[k] if "0" in a]
        ss = [a for a in CLASS[k] if "0" not in a]
        w[k] = sum(Sg(a, a) for a in CLASS[k])
        p[k] = Sg(ts[0], ts[1]) + Sg(ss[0], ss[1])
        D[k] = sum(Sg(a, b) for a, b in dpairs[k])
        C[k] = sum(Sg(a, b) for a in ts for b in ss)
    W = np.zeros(3)
    for k in (1, 2, 3):
        l, m = cyc[k]
        W[k - 1] = w[k] + 2 * p[k] - 4 * D[k] + 2 * (C[m] - C[l])
    X = {}
    for k in (1, 2, 3):
        l, m = cyc[k]
        key = "%d%d" % (l, m) if "%d%d" % (l, m) in S_BASIS else "%d%d" % (m, l)
        val = 2.0 * sum(s_sign(a) * Sg(a, b) for a in CLASS[l] for b in CLASS[m])
        for mu in range(4):
            t = -1.0 if mu in (0, l) else 1.0
            for b in CLASS[k]:
                if str(mu) in b:
                    continue
                val += 4.0 * t * Sg("%d%d" % (mu, mu), b)
        X[key] = val
    return W, X


def card_tlG(sigma: np.ndarray) -> np.ndarray:
    W, X = card_W_X(sigma)
    out = sum(W[k] * P_BASIS[k] for k in range(3))
    for key, val in X.items():
        out = out + 0.25 * val * S_BASIS[key]
    return out


# ---------------------------------------------------------------- O(delta^2) 모형 (창의 표준오차용, 기하 없음)
def model_eps(rng, A: np.ndarray, Mt: np.ndarray, n: int, trials: int) -> np.ndarray:
    r = A.shape[1]
    g = rng.normal(size=(trials, n, r))
    Y = g @ A.T
    Y = Y - Y.mean(axis=1, keepdims=True)
    S = np.einsum("tva,tvb->tab", Y, Y)
    G = np.einsum("tab,abij->tij", S, Mt)
    return np.linalg.norm(G.reshape(trials, 9), axis=1) / (NORM_G0 * n)


def model_statistics(rng, Mt: np.ndarray) -> dict:
    """사전등록 실험 전체를 모형으로 1회 실행한 통계."""
    A = {k: factor(v) for k, v in SPECS.items()}
    st = {}
    ker = {n: rms(model_eps(rng, A["kernel"], Mt, n, TRIALS)) for n in SIZES}
    st["ker_eps64_over_delta2"] = ker[64]
    st["ker_slope"] = loglog_slope(SIZES, [ker[n] for n in SIZES])
    d4 = {n: rms(model_eps(rng, A["diag4"], Mt, n, TRIALS)) for n in SIZES}
    st["diag_eps64_over_delta2"] = d4[64]
    st["diag_slope"] = loglog_slope(SIZES, [d4[n] for n in SIZES])
    st["diag_eps2_over_delta2"] = rms(model_eps(rng, A["diag4"], Mt, 2, N2_TRIALS))
    st["off_eps2_over_delta2"] = rms(model_eps(rng, A["off12"], Mt, 2, N2_TRIALS))
    st["iso_eps2_over_delta2"] = rms(model_eps(rng, A["iso16"], Mt, 2, N2_TRIALS))
    st["cross_eps2_sq_over_delta4"] = (st["iso_eps2_over_delta2"] ** 2 - st["diag_eps2_over_delta2"] ** 2
                                       - st["off_eps2_over_delta2"] ** 2)
    uo = {n: rms(model_eps(rng, A["univ_o"], Mt, n, TRIALS)) for n in (4, 64)}
    st["univ_o_eps64_over_delta2"] = uo[64]
    st["univ_o_eps4_over_delta2"] = uo[4]
    st["univ_floor_hat_over_delta2"] = floor_hat(uo[4], uo[64])
    st["univ_d_eps64_over_delta2"] = rms(model_eps(rng, A["univ_d"], Mt, 64, TRIALS))
    return st


# ---------------------------------------------------------------- 모드
def mode_constants() -> dict:
    M, Mt = structure_constants()
    Mfd = fd_structure_constants()
    fd_err = float(np.max(np.abs(M - Mfd)))

    # 5차원 상(像)의 계수표: tl M^{ab} = c1 P1 + c2 P2 + s12 S12 + s23 S23 + s31 S31  (P3 = -P1-P2)
    B = np.array([P_BASIS[0].reshape(9), P_BASIS[1].reshape(9), S_BASIS["12"].reshape(9),
                  S_BASIS["23"].reshape(9), S_BASIS["31"].reshape(9)]).T
    table = {}
    rational_err = 0.0
    for a in range(16):
        for b in range(a, 16):
            if np.linalg.norm(Mt[a, b]) < 1e-9:
                continue
            c, *_ = np.linalg.lstsq(B, Mt[a, b].reshape(9), rcond=None)
            fit_err = float(np.linalg.norm(B @ c - Mt[a, b].reshape(9)))
            fr = [Fraction(float(x)).limit_denominator(8) for x in c]
            rational_err = max(rational_err, fit_err, max(abs(float(f) - float(x)) for f, x in zip(fr, c)))
            table["%s,%s" % (NAMES[a], NAMES[b])] = [str(f) for f in fr]

    # 카드의 닫힌 규칙(W_k, X_lm) 대 구조상수: 무작위 대칭 Sigma 20개
    rng = np.random.default_rng(SEED)
    rule_err = 0.0
    for _ in range(20):
        Z = rng.normal(size=(16, 16))
        sigma = Z @ Z.T
        rule_err = max(rule_err, float(np.linalg.norm(card_tlG(sigma) - np.einsum("ab,abij->ij", sigma, Mt))))

    # SO(3) 동변: l -> R~ l R~^T (R~ = diag(1,Q)) 아래 tl G -> Q tl G Q^T
    cov_err = 0.0
    sigma = sigma_of(SPECS["univ_o"])
    for _ in range(10):
        Q, R_ = np.linalg.qr(rng.normal(size=(3, 3)))
        Q = Q @ np.diag(np.sign(np.diag(R_)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1.0
        Rt = np.eye(4)
        Rt[1:, 1:] = Q
        Pm = np.kron(Rt, Rt)
        V = np.einsum("ab,abij->ij", sigma, Mt)
        Vr = np.einsum("ab,abij->ij", Pm @ sigma @ Pm.T, Mt)
        cov_err = max(cov_err, float(np.linalg.norm(Vr - Q @ V @ Q.T)))

    D_idx = [idx(a) for a in DIAG]
    O_idx = [a for a in range(16) if a not in D_idx]
    T_DD = float(sum(np.sum(Mt[a, b] ** 2) for a in D_idx for b in D_idx))
    T_OO = float(sum(np.sum(Mt[a, b] ** 2) for a in O_idx for b in O_idx))
    T_DO = float(sum(np.sum(Mt[a, b] ** 2) for a in D_idx for b in O_idx))

    exact = {}
    for name, spec in SPECS.items():
        sigma = sigma_of(spec)
        F, T = floor_amplitude(sigma, Mt), fluctuation_amplitude(sigma, Mt)
        W, X = card_W_X(sigma)
        w_axis = np.array([sum(sigma[idx(a), idx(a)] for a in CLASS[k]) for k in (1, 2, 3)])
        exact[name] = {
            "F": F, "T": T,
            "W": W.tolist(), "X": X,
            "w_axis_weights": w_axis.tolist(),
            "floor_over_delta2": F / NORM_G0,
            "F01_closed_form_floor_over_delta2": float(np.linalg.norm(w_axis - w_axis.mean()) / (4 * SQ3)),
            "eps_star_over_delta2": math.sqrt(2.0 * T) / NORM_G0,
            "eps_over_delta2": {str(n): predicted_eps_over_delta2(n, F, T) for n in (2, *SIZES)},
            "slope": (loglog_slope(SIZES, [predicted_eps_over_delta2(n, F, T) for n in SIZES])
                      if (F > 1e-9 or T > 1e-9) else None),
        }

    def e2(k):
        return exact[k]["eps_over_delta2"]["2"]

    exact_pred = {
        "ker_eps64_over_delta2": exact["kernel"]["eps_over_delta2"]["64"],
        "ker_slope": exact["kernel"]["slope"],
        "diag_eps2_over_delta2": e2("diag4"),
        "diag_eps64_over_delta2": exact["diag4"]["eps_over_delta2"]["64"],
        "diag_slope": exact["diag4"]["slope"],
        "off_eps2_over_delta2": e2("off12"),
        "iso_eps2_over_delta2": e2("iso16"),
        "cross_eps2_sq_over_delta4": e2("iso16") ** 2 - e2("diag4") ** 2 - e2("off12") ** 2,
        "univ_o_eps64_over_delta2": exact["univ_o"]["eps_over_delta2"]["64"],
        "univ_o_eps4_over_delta2": exact["univ_o"]["eps_over_delta2"]["4"],
        "univ_floor_hat_over_delta2": floor_hat(exact["univ_o"]["eps_over_delta2"]["4"],
                                                exact["univ_o"]["eps_over_delta2"]["64"]),
        "univ_d_eps64_over_delta2": exact["univ_d"]["eps_over_delta2"]["64"],
        "zero_max_residual": 0.0,
    }
    pred_mismatch = {k: abs(exact_pred[k] - PRED[k]) for k in PRED}

    # 모델 표준오차와 창
    reps = [model_statistics(np.random.default_rng(SEED + 1000 + r), Mt) for r in range(MODEL_REPLICATES)]
    model_se = {k: float(np.std([r[k] for r in reps], ddof=1)) for k in reps[0]}
    model_mean = {k: float(np.mean([r[k] for r in reps])) for k in reps[0]}
    window_from_model = {}
    for k, se in model_se.items():
        v = exact_pred[k]
        half = 4.0 * se + (0.0 if k in ("ker_slope", "diag_slope", "univ_floor_hat_over_delta2",
                                         "cross_eps2_sq_over_delta4") else 0.01 * abs(v))
        if k == "cross_eps2_sq_over_delta4":
            half += 0.02 * exact_pred["iso_eps2_over_delta2"] ** 2  # 1% 계통 여유를 세 항에 전파(iso 항이 지배)
        window_from_model[k] = (v - half, v + half)

    result = {
        "norm_G0": NORM_G0,
        "analytic_vs_fd_M_maxdiff": fd_err,
        "rational_table_maxerr": rational_err,
        "card_rule_vs_M_maxerr": rule_err,
        "so3_equivariance_maxerr": cov_err,
        "nonzero_tlM_pairs_unordered": len(table),
        "T_I16": float(np.einsum("abij,abij->", Mt, Mt)),
        "T_budget": {"DD": T_DD, "OO": T_OO, "DO_plus_OD": 2 * T_DO, "sum": T_DD + T_OO + 2 * T_DO},
        "sum_a_tl_M_aa_norm": float(np.linalg.norm(sum(Mt[a, a] for a in range(16)))),
        "eps_star_isotropic_over_delta2": math.sqrt(2.0 * 60.0) / NORM_G0,
        "table_tlM_in_P1_P2_S12_S23_S31": table,
        "exact": exact,
        "exact_pred": exact_pred,
        "pred_mismatch_vs_script_PRED": pred_mismatch,
        "model_replicates": MODEL_REPLICATES,
        "model_mean": model_mean,
        "model_se": model_se,
        "window_from_model": window_from_model,
        "script_WINDOW": WINDOW,
    }
    (OUT / "structure_constants.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def _sweep(seed: int, sizes, trials, spec, delta=DELTA) -> tuple[dict, int]:
    rng = np.random.default_rng(seed)
    A = factor(spec)
    rs = Resampler()
    out = {}
    for n in sizes:
        vals = []
        for _ in range(trials):
            lab, _ = rs.draw(rng, n, A, delta)
            vals.append(block_residual(lab, delta))
        out[n] = rms(vals)
    return out, rs.resampled


def mode_kernel(sizes, trials) -> dict:
    curve, res = _sweep(SEED + 1, sizes, trials, SPECS["kernel"])
    top = max(sizes)
    return {
        "ker_curve_over_delta2": {str(n): v / DELTA**2 for n, v in curve.items()},
        "ker_eps64_over_delta2": curve[top] / DELTA**2,
        "ker_slope": loglog_slope(sizes, [curve[n] for n in sizes]),
        "ker_resampled": res,
    }


def mode_diag(sizes, trials, n2_trials=N2_TRIALS) -> dict:
    out = {}
    for tag, name, seed in (("diag", "diag4", SEED + 2), ("off", "off12", SEED + 3), ("iso", "iso16", SEED + 4)):
        c2, res = _sweep(seed, (2,), n2_trials, SPECS[name])
        out[f"{tag}_eps2_over_delta2"] = c2[2] / DELTA**2
        out[f"{tag}_n2_resampled"] = res
    out["cross_eps2_sq_over_delta4"] = (out["iso_eps2_over_delta2"] ** 2 - out["diag_eps2_over_delta2"] ** 2
                                        - out["off_eps2_over_delta2"] ** 2)
    curve, res = _sweep(SEED + 5, sizes, trials, SPECS["diag4"])
    top = max(sizes)
    out["diag_curve_over_delta2"] = {str(n): v / DELTA**2 for n, v in curve.items()}
    out["diag_eps64_over_delta2"] = curve[top] / DELTA**2
    out["diag_slope"] = loglog_slope(sizes, [curve[n] for n in sizes])
    out["diag_resampled"] = res
    return out


def mode_univ(sizes, trials) -> dict:
    co, res_o = _sweep(SEED + 6, sizes, trials, SPECS["univ_o"])
    cd, res_d = _sweep(SEED + 7, sizes, trials, SPECS["univ_d"])
    top, bot = max(sizes), min(sizes)
    return {
        "univ_o_curve_over_delta2": {str(n): v / DELTA**2 for n, v in co.items()},
        "univ_d_curve_over_delta2": {str(n): v / DELTA**2 for n, v in cd.items()},
        "univ_o_eps64_over_delta2": co[top] / DELTA**2,
        "univ_o_eps4_over_delta2": co[bot] / DELTA**2,
        "univ_floor_hat_over_delta2": floor_hat(co[bot] / DELTA**2, co[top] / DELTA**2),
        "univ_d_eps64_over_delta2": cd[top] / DELTA**2,
        "univ_resampled": res_o + res_d,
    }


def mode_zero(sizes, trials) -> dict:
    worst = 0.0
    table = {}
    total_res = 0
    for delta in ZERO_DELTAS:
        for name in ("zero_11", "zero_3diag"):
            rng = np.random.default_rng(SEED + 8 + (0 if name == "zero_11" else 1))
            A = factor(SPECS[name])
            rs = Resampler()
            for n in sizes:
                vals = []
                for _ in range(trials):
                    lab, _ = rs.draw(rng, n, A, delta)
                    vals.append(abs(block_residual(lab, delta)))
                key = f"d{delta}_{name}_n{n}"
                table[key] = {"rms": rms(vals), "max": float(max(vals))}
                worst = max(worst, float(max(vals)))
            total_res += rs.resampled
    return {"zero_max_residual": worst, "zero_table": table, "zero_resampled": total_res}


def mode_axis(sizes, trials) -> dict:
    """일관성(kill 아님): 공통 난수로 (0,1)과 (2,3)의 비."""
    rng = np.random.default_rng(SEED + 10)
    A1, A2 = factor(SPECS["axis_01"]), factor(SPECS["axis_23"])
    rs = Resampler()
    v01, v23 = [], []
    for n in sizes:
        a, b = [], []
        for _ in range(trials):
            lab1, g = rs.draw(rng, n, A1, DELTA)
            lab2, _ = rs.draw(rng, n, A2, DELTA, g=g)
            a.append(block_residual(lab1))
            b.append(block_residual(lab2))
        v01.append(rms(a))
        v23.append(rms(b))
    return {"axis_ratio_23_over_01": rms(v23) / rms(v01), "axis_resampled": rs.resampled}


RUNNERS = {
    "kernel": mode_kernel,
    "diag": mode_diag,
    "univ": mode_univ,
    "zero": mode_zero,
    "axis": mode_axis,
}


def verdict(stats: dict) -> dict:
    fired = []
    for key, (low, high) in WINDOW.items():
        if key in stats and not (low <= float(stats[key]) <= high):
            fired.append(key)
    return {"kill_fired": fired, "status": "refuted" if fired else "consistent"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all",
                        choices=["constants", "all", "smoke", *RUNNERS.keys()])
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    if args.mode == "constants":
        result = mode_constants()
        show = {k: v for k, v in result.items() if k not in ("table_tlM_in_P1_P2_S12_S23_S31", "exact")}
        print(json.dumps(show, ensure_ascii=False, indent=2))
        return 0

    if args.mode == "smoke":
        sizes, trials = (4, 8), 48
        stats = {}
        stats.update(mode_kernel(sizes, trials))
        stats.update(mode_diag(sizes, trials, n2_trials=200))
        stats.update(mode_univ(sizes, trials))
        stats.update(mode_zero(sizes, 16))
        stats.update(mode_axis(sizes, 16))
        # F-01 반례 3개 재현(recovers, 이미 관측) — 작은 크기
        _, Mt = structure_constants()
        for name in ("ce_i", "ce_ii", "ce_iii"):
            c, _ = _sweep(SEED + 20, sizes, trials, SPECS[name])
            sg = sigma_of(SPECS[name])
            F, T = floor_amplitude(sg, Mt), fluctuation_amplitude(sg, Mt)
            stats[name] = {"observed_over_delta2": {str(n): v / DELTA**2 for n, v in c.items()},
                           "master": {str(n): predicted_eps_over_delta2(n, F, T) for n in sizes}}
        stats = {("smoke_" + k.replace("eps64", f"eps{max(sizes)}")): v for k, v in stats.items()}
        pred = {}
        for name in ("kernel", "diag4", "off12", "iso16", "univ_o", "univ_d"):
            sg = sigma_of(SPECS[name])
            F, T = floor_amplitude(sg, Mt), fluctuation_amplitude(sg, Mt)
            pred[name] = {str(n): predicted_eps_over_delta2(n, F, T) for n in (2, *sizes)}
        payload = {
            "note": "SMOKE ONLY — 사전등록 크기(SIZES/TRIALS/N2_TRIALS/SEED)가 아니므로 kill 판정에 쓰지 않는다",
            "sizes": list(sizes), "trials": trials, "n2_trials": 200, "delta": DELTA, "min_det": MIN_DET,
            "stats": stats,
            "predicted_at_smoke_sizes": pred,
        }
        (OUT / "smoke.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    modes = list(RUNNERS) if args.mode == "all" else [args.mode]
    stats = {}
    for name in modes:
        stats.update(RUNNERS[name](SIZES, TRIALS))
    payload = {
        "card": "derivations/Q-0013/F-02.formula.md",
        "modes": modes,
        "delta": DELTA, "sizes": list(SIZES), "trials": TRIALS, "n2_trials": N2_TRIALS,
        "seed": SEED, "min_det": MIN_DET,
        "predicted": PRED, "window": WINDOW,
        "stats": stats,
        "verdict": verdict(stats),
    }
    path = OUT / "result.json"
    if path.is_file():
        old = json.loads(path.read_text(encoding="utf-8"))
        merged = dict(old.get("stats", {}))
        merged.update(stats)
        payload["stats"] = merged
        payload["modes"] = sorted(set(old.get("modes", [])) | set(modes))
        payload["verdict"] = verdict(merged)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"],
                      "stats": {k: v for k, v in payload["stats"].items() if k in WINDOW}},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
