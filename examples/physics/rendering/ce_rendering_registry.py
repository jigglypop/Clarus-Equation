"""입력 하나(α_s)로 묶은 양자·거시 행의 공동 채점과 렌더링 규칙. 원장: 43장.

모든 행은 식·관측·지위·선택 비용(bit)을 함께 가진다. 연속 적합 매개변수는 판본 I의
α_s(MS-bar ŝ_Z^2 교정)와 BAO 거리 눈금 A뿐이다. 해석 규칙은 자료를 본 뒤 고른 [경험식]이며
"영감 규칙"으로 표기하고 비용을 bit로 센다. 수치 일치는 물리 증명이 아니다.

렌더링 공리(43장 §43.5):
  E1  3단계 실현 공간 V3 = C^3, 렌더링 연산 R = a I_3.
  E2  det R = α_s.
  E3  m차원 부분공간 W로의 렌더링 진폭 A_m(W) = tr(P_W) det(P_W R P_W |_W).
  E4  sin θ_W = A_2.
  G1  세대 g의 가중치 w_g = (1+δ/2π)^{[g=2]}; 전이 i<j는 w_i/w_j를 받는다.
  S2  m번째로 렌더링된 축은 그 축을 포함하는 새 통로 2^{m-1}개(8통로 중)만큼 분별을 만들고,
      마지막에 렌더링된 것이 "나"다: s13^2 = δ/8, s12^2 = (1-2δ/8)/3, s23^2 = (1-4δ/8)/2.
  T1  ν1은 2·3세대의 분별 이전 상태다: |U_μ1| = |U_τ1| (TM1). δ_PMNS는 이 조건의 해다.
  U1  쿼크 단위 삼각형의 세 각은 8통로의 차수 분할 Λ^0 : Λ^1 : Λ^{2,3} = 1 : 3 : 4에 π를 나눈다:
      (β, γ, α) = (π/8, 3π/8, π/2). |V_ub|, δ_CKM, J는 |V_us|, |V_cb|와 이 삼각형에서 나온다.
      삼각형의 향(α = +π/2)이 전역 시간 화살표 1 bit다.
  M1  렙톤은 쿼크의 거울이다: 렙톤의 원 방향은 쿼크와 반대(sgn J_PMNS = -sgn J_CKM).

python -B -m examples.physics.rendering.ce_rendering_registry
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np
from scipy.optimize import brentq

from examples.physics.darksector.ce_residual_forward_model import (
    DESI_DR2_ALL_COVARIANCE,
    DESI_DR2_ALL_DATA,
)

PI = math.pi
SZ2, SZ2_ERR = 0.23129, 0.00004            # PDG 2024 MS-bar ŝ_Z^2 전역 적합
A_WORLD, A_WORLD_ERR = 0.1180, 0.0009      # PDG 2024 α_s(M_Z)
C_ONSHELL = 1.0349                         # PDG 2024 식 (10.25), SM 복사보정
AEM_INV_MZ, AEM_INV_MZ_ERR = 127.951, 0.009
T_PLANCK_S = 5.391247e-44
KM_PER_MPC = 3.0856775814913673e19
M_E, M_MU, M_TAU, M_TAU_ERR = 0.51099895, 105.6583755, 1776.93, 0.09
V_EW, M_PLANCK = 246.21965, 1.220890e19

BAO_Z = np.array([p.z for p in DESI_DR2_ALL_DATA])
BAO_KIND = tuple(p.kind for p in DESI_DR2_ALL_DATA)
BAO_Y = np.array([p.value for p in DESI_DR2_ALL_DATA])
BAO_CINV = np.linalg.inv(np.array(DESI_DR2_ALL_COVARIANCE))


# ------------------------------------------------------------------ 코어 사슬
@lru_cache(maxsize=4096)
def core(alpha_s: float) -> dict:
    s2 = 4.0 * alpha_s ** (4.0 / 3.0)
    delta = s2 * (1.0 - s2)
    D = 3.0 + delta
    q = brentq(lambda x: x - math.exp(-D * (1.0 - x)), 1e-12, 1.0 / D)
    F = 1.0 + alpha_s * D
    omega_dm = (1.0 - q) * alpha_s * D / F
    return {"a": alpha_s, "s2": s2, "d": delta, "D": D, "q": q, "F": F,
            "Om": q + omega_dm, "Ne": 18.0 * D}


def calibrated_alpha_s() -> tuple[float, float]:
    a = (SZ2 / 4.0) ** 0.75
    return a, 0.75 * a * SZ2_ERR / SZ2


# ------------------------------------------------------------------ ① 렌더링 진폭
def rendering_amplitude(alpha_s: float, basis: np.ndarray) -> float:
    """E1–E3: A_m(W) = tr(P_W) det(P_W R P_W|_W), basis는 W의 정규직교 열."""
    R = alpha_s ** (1.0 / 3.0) * np.eye(3)
    P = basis @ basis.conj().T
    return float(np.trace(P).real * np.linalg.det(basis.conj().T @ R @ basis).real)


def rendering_amplitude_closed(alpha_s: float, m: int) -> float:
    return m * alpha_s ** (m / 3.0)


# ------------------------------------------------------------------ ③ 세대 가중치
def generation_weight(c: dict, g: int) -> float:
    return 1.0 + c["d"] / (2.0 * PI) if g == 2 else 1.0


def transition_factor(c: dict, i: int, j: int) -> float:
    return generation_weight(c, i) / generation_weight(c, j)


# ------------------------------------------------------------------ ④ PMNS 분별 통로
def exterior_channels(m: int) -> int:
    """dim Λ^{>=1}(C^m)."""
    return sum(math.comb(m, k) for k in range(1, m + 1))


def distinction_channels(m: int) -> int:
    """m번째 축을 포함하는 Λ(C^m)의 통로 수 = 2^{m-1}."""
    return exterior_channels(m) - exterior_channels(m - 1)


def pmns_s2(c: dict, m: int) -> float:
    """S2: 분별 통로 비율만큼 최대 섞임에서 벗어나며 마지막 렌더링이 '나'다."""
    frac = distinction_channels(m) / 2.0 ** 3
    if m == 1:
        return c["d"] * frac
    return {2: 1.0 / 3.0, 3: 0.5}[m] * (1.0 - frac * c["d"])


def pmns_matrix(s12sq: float, s23sq: float, s13sq: float, dl: float):
    return mixing_matrix(math.sqrt(s12sq), math.sqrt(s23sq), math.sqrt(s13sq), dl)


def delta_pmns_tm1(c: dict, circulation: int | None = None) -> float:
    """T1: |U_μ1| = |U_τ1|의 해. 기본 방향은 M1: 쿼크 J의 반대 부호."""
    if circulation is None:
        circulation = -1 if ckm_triangle(c["a"])[2] > 0 else +1
    s12, s23, s13 = pmns_s2(c, 2), pmns_s2(c, 3), pmns_s2(c, 1)
    g = lambda x: abs(pmns_matrix(s12, s23, s13, x)[1][0]) ** 2 - abs(pmns_matrix(s12, s23, s13, x)[2][0]) ** 2
    lo, hi = (0.01, PI - 0.01) if circulation > 0 else (PI + 0.01, 2 * PI - 0.01)
    return math.degrees(brentq(g, lo, hi))


def circulant_eigenvector_drift(theta: float, eps: float = 0.1) -> float:
    """CE 순환 질량 행렬의 고유벡터가 이산 푸리에 벡터에서 벗어난 최대량.

    위상 θ는 고유값만 바꾸고 고유벡터(섞임)를 바꾸지 않는다. 그래서 우주 시계 θ는 δ_PMNS의
    크기를 정하지 못하고, 두 원이 공유하는 것은 순환 방향(ω 또는 ω̄) 1 bit뿐이다.
    """
    S = np.roll(np.eye(3), 1, axis=0)
    M = np.eye(3) + eps * (cmath.exp(1j * theta / 3) * S + cmath.exp(-1j * theta / 3) * S.T)
    _, vecs = np.linalg.eigh(M)
    w = cmath.exp(2j * PI / 3)
    F = np.array([[w ** (a * i) for i in range(3)] for a in range(3)]) / math.sqrt(3)
    return float(1.0 - np.abs(F.conj().T @ vecs).max(axis=0).min())


# ------------------------------------------------------------------ ② 세대 진폭의 조립
def flavour_words(c: dict) -> dict:
    """u = (A_2/tr P_2)^2 = α^{4/3}, ε = sqrt(A_1) = α^{1/6}; 조립은 [경험식] 선택.

    |V_ub|는 조립하지 않고 U1 삼각형에서 얻는다(ckm_triangle)."""
    a = c["a"]
    u = (rendering_amplitude_closed(a, 2) / 2.0) ** 2
    eps = math.sqrt(rendering_amplitude_closed(a, 1))
    return {
        "V_us": 4.0 * u * transition_factor(c, 1, 2),
        "V_cb": u * eps * transition_factor(c, 2, 3),
        "m_mu/m_tau": u * transition_factor(c, 2, 3),
    }


def mixing_matrix(s12: float, s23: float, s13: float, dl: float):
    c12, c23, c13 = (math.sqrt(1.0 - x * x) for x in (s12, s23, s13))
    e = cmath.exp(1j * dl)
    return [[c12 * c13, s12 * c13, s13 / e],
            [-s12 * c23 - c12 * s23 * s13 * e, c12 * c23 - s12 * s23 * s13 * e, s23 * c13],
            [s12 * s23 - c12 * c23 * s13 * e, -c12 * s23 - s12 * c23 * s13 * e, c23 * c13]]


def _triangle_angles(V) -> tuple[float, float]:
    alpha = cmath.phase(-V[2][0] * V[2][2].conjugate() / (V[0][0] * V[0][2].conjugate()))
    beta = cmath.phase(-V[1][0] * V[1][2].conjugate() / (V[2][0] * V[2][2].conjugate()))
    return alpha, beta


@lru_cache(maxsize=4096)
def ckm_triangle(alpha_s: float, orientation: int = +1) -> tuple[float, float, float]:
    """U1: (α, β) = (orientation·π/2, orientation·π/8)에서 (|V_ub|, δ_CKM, J)."""
    from scipy.optimize import fsolve

    w = flavour_words(core(alpha_s))
    vus, vcb = w["V_us"], w["V_cb"]

    def parts(x):
        s13, dl = x
        norm = math.sqrt(1.0 - s13 * s13)
        return mixing_matrix(vus / norm, vcb / norm, s13, dl)

    def eqs(x):
        a, b = _triangle_angles(parts(x))
        return [a - orientation * PI / 2, b - orientation * PI / 8]

    s13, dl = fsolve(eqs, [0.0037, orientation * 1.18], xtol=1e-14)
    V = parts((s13, dl))
    J = (V[0][1] * V[1][2] * V[0][2].conjugate() * V[1][1].conjugate()).imag
    return float(s13), float(dl), float(J)


def koide_me_over_mmu(r_mu_tau: float) -> float:
    xt = 1.0 / math.sqrt(r_mu_tau)
    xe = brentq(lambda x: (x * x + 1 + xt * xt) / (x + 1 + xt) ** 2 - 2.0 / 3.0, 0.0, 0.2)
    return xe * xe


# ------------------------------------------------------------------ ⑤ 합규칙
def alpha_em_inv(c: dict, channel_loop: bool = True) -> float:
    """α_s+α_2+α_em = (1/2π)(1 + α_s/(2π)/8); α_em = ŝ^2 α_2."""
    total = (1.0 / (2.0 * PI)) * (1.0 + (c["a"] / (16.0 * PI) if channel_loop else 0.0))
    alpha_2 = (total - c["a"]) / (1.0 + c["s2"])
    return 1.0 / (c["s2"] * alpha_2)


def v_over_mpl(c: dict) -> float:
    """e^{-12D}/F × (1 + α_s/(2π)·4/8): Λ^odd 4통로 고리(영감 규칙)."""
    return math.exp(-12.0 * c["D"]) / c["F"] * (1.0 + c["a"] / (4.0 * PI))


def hubble_kms(c: dict) -> float:
    log_s = (PI ** 2 / 2.0) * c["Ne"] - PI * c["d"] * (1.0 - c["q"])
    return math.sqrt(PI) * math.exp(-log_s / 2.0) / T_PLANCK_S * KM_PER_MPC


# H0 판독 행: (이름, 판독 시기 z, 값, +σ, −σ). DESI BAO+BBN H0는 BAO 블록과 자료가 겹쳐 제외.
H0_READOUTS = (
    ("H0 Planck", 1090.0, 67.36, 0.54, 0.54),
    ("H0 TDCOSMO", 0.5, 71.6, 3.9, 3.3),
    ("H0 TRGB", 0.03, 70.39, 1.94, 1.94),
    ("H0 SH0ES", 0.03, 73.17, 0.86, 0.86),
)


def vacuum_share(c: dict, z: float) -> float:
    """g(z) = Ω_Λ(z)/Ω_Λ(0): 분별하지 않는 진공의 점유율(오늘 1, 이른 우주 0)."""
    om = c["Om"]
    return 1.0 / (om * (1.0 + z) ** 3 + 1.0 - om)


def hubble_readout(c: dict, z: float, amplitude: float) -> float:
    """H1: 국소 판독의 분별 약화. 실제 팽창 H(z)가 아니라 판독만 바뀐다(BAO 반례)."""
    return hubble_kms(c) * math.exp(amplitude * vacuum_share(c, z))


@lru_cache(maxsize=4096)
def readout_amplitude(alpha_s: float) -> float:
    """H0 판독 네 행으로 약화 크기 하나를 맞춘다(연속 적합 매개변수 +1)."""
    from scipy.optimize import minimize_scalar

    c = core(alpha_s)

    def chi2(A: float) -> float:
        total = 0.0
        for _, z, v, up, dn in H0_READOUTS:
            p = hubble_readout(c, z, A)
            total += ((p - v) / (up if p >= v else dn)) ** 2
        return total

    return float(minimize_scalar(chi2, bounds=(-0.5, 0.5), method="bounded").x)


def scalar_amplitude(c: dict) -> float:
    q, D, Ne = c["q"], c["D"], c["Ne"]
    Q = (2.0 / PI) * (1.0 - q) ** (D / (D + 1.0)) * q * (1.0 - q)
    return Q * Q / (1.0 - q) ** 2 * q / (2.0 * PI * Ne ** 2) * 1e9


# ------------------------------------------------------------------ 행
@dataclass(frozen=True)
class Row:
    key: str
    block: str
    f: Callable[[dict], float]
    obs: float
    sig_up: float
    sig_dn: float
    status: str
    bits: float = 0.0
    note: str = ""
    variants: tuple[str, ...] = ("I", "II")


OMEGA_B_PLANCK = 0.02237 / 0.6736 ** 2
OMEGA_B_PLANCK_ERR = OMEGA_B_PLANCK * math.hypot(0.00015 / 0.02237, 2 * 0.0054 / 0.6736)
PMNS_DATA = {  # NuFIT 6.0 NO: (값, +σ, −σ)
    "SK": {"s12": (0.308, 0.012, 0.011), "s23": (0.470, 0.017, 0.013),
           "s13": (0.02215, 0.00056, 0.00058), "dl": (212.0, 26.0, 41.0)},
    "noSK": {"s12": (0.307, 0.012, 0.011), "s23": (0.561, 0.012, 0.015),
             "s13": (0.02195, 0.00054, 0.00058), "dl": (177.0, 19.0, 20.0)},
}


def rows(pmns: str = "SK") -> tuple[Row, ...]:
    p = PMNS_DATA[pmns]
    words = lambda key: (lambda c: flavour_words(c)[key])
    return (
        Row("s_Z^2", "Q", lambda c: c["s2"], SZ2, SZ2_ERR, SZ2_ERR, "공리", variants=("II",)),
        Row("alpha_s world", "Q", lambda c: c["a"], A_WORLD, A_WORLD_ERR, A_WORLD_ERR, "공리", variants=("I",)),
        Row("m_W/m_Z", "Q", lambda c: math.sqrt(1.0 - c["s2"] / C_ONSHELL), 80.3692 / 91.1876,
            0.000152, 0.000152, "SM 상속"),
        Row("M_H/M_Z", "Q", lambda c: c["F"], 125.20 / 91.1876, 0.11 / 91.1876, 0.11 / 91.1876, "경험식"),
        Row("|V_us|", "Q", words("V_us"), 0.22501, 0.00068, 0.00068, "경험식", 1.0, "4u/w (G1)"),
        Row("|V_cb|", "Q", words("V_cb"), 0.04183, 0.00079, 0.00069, "경험식", 2.0, "u sqrt(A_1) w"),
        Row("|V_ub|", "Q", lambda c: ckm_triangle(c["a"])[0], 0.003732, 0.000090, 0.000085, "산출", 0.0,
            "U1 삼각형"),
        Row("delta_CKM", "Q", lambda c: ckm_triangle(c["a"])[1], 1.147, 0.026, 0.026, "경험식", 3.0,
            "U1 차수 분할 + 화살표 1 bit"),
        Row("J_CKM", "Q", lambda c: ckm_triangle(c["a"])[2], 3.12e-5, 0.13e-5, 0.12e-5, "산출", 0.0, "U1"),
        Row("m_mu/m_tau", "Q", words("m_mu/m_tau"), M_MU / M_TAU, M_MU / M_TAU * M_TAU_ERR / M_TAU,
            M_MU / M_TAU * M_TAU_ERR / M_TAU, "경험식", 2.0, "u w (G1)"),
        Row("m_e/m_mu", "Q", lambda c: koide_me_over_mmu(flavour_words(c)["m_mu/m_tau"]), M_E / M_MU,
            1e-10, 1e-10, "산출", 1.0, "Koide 2/3"),
        Row("alpha_em^-1(M_Z)", "Q", alpha_em_inv, AEM_INV_MZ, AEM_INV_MZ_ERR, AEM_INV_MZ_ERR, "경험식", 3.0,
            "합규칙 + 한 통로 고리"),
        Row("v/M_Pl", "Q", v_over_mpl, V_EW / M_PLANCK, V_EW / M_PLANCK * 1.1e-5, V_EW / M_PLANCK * 1.1e-5,
            "경험식", 3.0, "Λ^odd 고리"),
        Row("s13^2", "Q", lambda c: pmns_s2(c, 1), *p["s13"], "경험식", 0.0, "S2"),
        Row("s12^2", "Q", lambda c: pmns_s2(c, 2), *p["s12"], "경험식", 0.0, "S2 (=TM1 1차)"),
        Row("s23^2", "Q", lambda c: pmns_s2(c, 3), *p["s23"], "경험식", 1.0, "S2: 나=마지막 렌더링"),
        Row("delta_PMNS", "Q", delta_pmns_tm1, *p["dl"], "경험식", 2.0, "T1 + M1 거울"),
        Row("muon Da_mu x1e11", "Q", lambda c: 3.7287e-4, 38.5, math.hypot(14.5, 62.0), math.hypot(14.5, 62.0),
            "경험식"),
        Row("Omega_b", "M", lambda c: c["q"], OMEGA_B_PLANCK, OMEGA_B_PLANCK_ERR, OMEGA_B_PLANCK_ERR, "공리"),
        Row("Omega_m", "M", lambda c: c["Om"], 0.3153, 0.0073, 0.0073, "공리"),
        Row("n_s", "M", lambda c: 1.0 - 2.0 / c["Ne"], 0.9649, 0.0042, 0.0042, "경험식"),
        Row("A_s x1e9", "M", scalar_amplitude, 2.0989, 0.0294, 0.0294, "경험식"),
        Row("dn_s/dlnk", "M", lambda c: -2.0 / c["Ne"] ** 2, -0.0045, 0.0067, 0.0067, "경험식"),
    ) + tuple(
        Row(name, "M", (lambda z_: lambda c: hubble_readout(c, z_, readout_amplitude(c["a"])))(z),
            v, up, dn, "경험식", 0.0, f"H1 판독 약화, z={z:g}")
        for name, z, v, up, dn in H0_READOUTS
    )


# ------------------------------------------------------------------ 채점
def bao_chi2(omega_m: float) -> float:
    z = np.linspace(0.0, 2.5, 25001)
    E = np.sqrt(omega_m * (1 + z) ** 3 + (1 - omega_m))
    inv = 1.0 / E
    dM = np.concatenate([[0.0], np.cumsum((inv[1:] + inv[:-1]) / 2.0 * np.diff(z))])
    m = np.interp(BAO_Z, z, dM)
    h = 1.0 / np.interp(BAO_Z, z, E)
    b = np.array([{"dm": mi, "dh": hi, "dv": (zi * mi * mi * hi) ** (1.0 / 3.0)}[k]
                  for mi, hi, zi, k in zip(m, h, BAO_Z, BAO_KIND)])
    A = float(b @ BAO_CINV @ BAO_Y / (b @ BAO_CINV @ b))
    r = A * b - BAO_Y
    return float(r @ BAO_CINV @ r)


def bao_chi2_if_expansion_weakened(c: dict, amplitude: float) -> float:
    """반례 검산: 약화가 실제 팽창 H(z)를 바꾼다면의 BAO χ² (판독만 바뀌는 H1과 비교)."""
    om = c["Om"]
    z = np.linspace(0.0, 2.5, 25001)
    share = 1.0 / (om * (1 + z) ** 3 + 1.0 - om)
    E = np.sqrt(om * (1 + z) ** 3 + (1 - om)) * np.exp(amplitude * (share - 1.0))
    inv = 1.0 / E
    dM = np.concatenate([[0.0], np.cumsum((inv[1:] + inv[:-1]) / 2.0 * np.diff(z))])
    m = np.interp(BAO_Z, z, dM)
    h = 1.0 / np.interp(BAO_Z, z, E)
    b = np.array([{"dm": mi, "dh": hi, "dv": (zi * mi * mi * hi) ** (1.0 / 3.0)}[k]
                  for mi, hi, zi, k in zip(m, h, BAO_Z, BAO_KIND)])
    A = float(b @ BAO_CINV @ BAO_Y / (b @ BAO_CINV @ b))
    r = A * b - BAO_Y
    return float(r @ BAO_CINV @ r)


def score(variant: str = "I", pmns: str = "SK") -> dict:
    if variant == "I":
        a0, sa = calibrated_alpha_s()
    elif variant == "II":
        a0, sa = A_WORLD, A_WORLD_ERR
    else:
        raise ValueError("variant must be I or II")
    c0 = core(a0)
    out = []
    for r in rows(pmns):
        if variant not in r.variants:
            continue
        pred = r.f(c0)
        st = abs(r.f(core(a0 + sa)) - r.f(core(a0 - sa))) / 2.0
        so = r.sig_up if pred >= r.obs else r.sig_dn
        sig = math.hypot(so, st)
        out.append({"key": r.key, "block": r.block, "pred": pred, "obs": r.obs, "sigma": sig,
                    "pull": (pred - r.obs) / sig, "status": r.status, "bits": r.bits, "note": r.note})
    chib = bao_chi2(c0["Om"])
    q = [o for o in out if o["block"] == "Q"]
    m = [o for o in out if o["block"] == "M"]
    cq = sum(o["pull"] ** 2 for o in q)
    cm = sum(o["pull"] ** 2 for o in m) + chib
    nq, nm = len(q), len(m) + 13
    return {"variant": variant, "pmns": pmns, "alpha_s": a0, "rows": out, "bao_chi2": chib,
            "rmse_Q": math.sqrt(cq / nq), "rmse_M": math.sqrt(cm / nm),
            "rmse_all": math.sqrt((cq + cm) / (nq + nm)), "N": nq + nm,
            "bits": sum(o["bits"] for o in out), "k_continuous": (1 if variant == "I" else 0) + 2,
            "readout_amplitude": readout_amplitude(a0)}


def main() -> None:
    for pmns in ("SK", "noSK"):
        for variant in ("I", "II"):
            R = score(variant, pmns)
            print(f"[{variant} PMNS={pmns}] RMSE Q={R['rmse_Q']:.3f} M={R['rmse_M']:.3f} "
                  f"ALL={R['rmse_all']:.3f} N={R['N']} bits={R['bits']:.1f}")
            for o in R["rows"]:
                if abs(o["pull"]) > 2.0:
                    print(f"    {o['key']:18s} {o['pred']:.6g} vs {o['obs']:.6g}  pull {o['pull']:+.2f}  {o['note']}")


if __name__ == "__main__":
    main()
