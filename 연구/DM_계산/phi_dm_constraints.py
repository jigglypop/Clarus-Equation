#!/usr/bin/env python3
"""
SFE φ-DM 제약 조건(09장) 1차 수치 체크용 스크립트.

주의:
- 이 스크립트는 EoS/생성 메커니즘까지 포함한 완전한 Ω_φ 계산이 아니라,
  09장에 정리된 "부등식 기반 PASS/FAIL 조건"을 구현해
  (m_phi, lambda) 파라미터 쌍이 물리적으로 허용 가능한지 1차 판정하는 용도이다.
- Ω_φ (암흑물질 에너지 밀도 비율)는 여기서 직접 계산하지 않는다.
  향후 Boltzmann 코드/배경 진화와 결합 필요.
"""

import math
from dataclasses import dataclass
from typing import Tuple


# 기본 상수 (SI 단위)
c = 2.99792458e8            # m/s
h = 6.62607015e-34          # J·s
hbar = h / (2 * math.pi)    # J·s
eV = 1.602176634e-19        # J
GeV = 1.0e9 * eV
kpc = 3.085677581e19        # m


@dataclass
class PhiDMParams:
    """
    φ-DM 파라미터 묶음
    - m_phi_eV: φ 질량 [eV]
    - lam:     자가상호작용 λ (무차원)
    """
    m_phi_eV: float
    lam: float


def mass_ev_to_kg(m_ev: float) -> float:
    """eV 단위 질량을 kg로 변환 (E = mc^2 사용)."""
    return (m_ev * eV) / (c ** 2)


def de_broglie_condition(params: PhiDMParams, v_kms: float = 200.0) -> Tuple[bool, float]:
    """
    드브로이 파장 조건 (09장 3.4, 11.3 참고):
    λ_dB ~ h / (m v) << kpc  →  m_phi >> 10^{-22} eV * (200 km/s / v)

    반환:
    - ok: 조건 만족 여부
    - m_min_eV: 주어진 v에서 요구되는 최소 질량 (eV)
    """
    m_min_ev = 1.0e-22 * (200.0 / v_kms)
    ok = params.m_phi_eV > m_min_ev
    return ok, m_min_ev


def self_interaction_condition(params: PhiDMParams) -> Tuple[bool, float]:
    """
    자가상호작용 조건 (09장 3.5, 11.3 참고):
    σ_φφ ~ λ^2 / (64 π m_φ^2),  σ/m ∈ [0.1, 1] cm^2/g (dwarf),
    cluster에서는 σ/m ≲ 0.1 cm^2/g 정도를 요구.

    여기서는 간단히:
    - 허용 구간: σ/m ∈ [1e-2, 1] cm^2/g
    - 둘 사이에 들어오면 PASS, 아니면 FAIL
    """
    m_kg = mass_ev_to_kg(params.m_phi_eV)

    # 단면적 σ [m^2] (스크리닝/수치 계수는 09장와 동일한 차원 구조만 유지)
    sigma = (params.lam ** 2) / (64.0 * math.pi * (m_kg ** 2))

    # σ/m [m^2/kg]
    sigma_over_m_SI = sigma / m_kg

    # cm^2/g 로 변환: 1 m^2/kg = (1e4 cm^2)/(1e3 g) = 10 cm^2/g
    sigma_over_m_cgs = sigma_over_m_SI * 10.0

    # 허용 구간 (매우 보수적, 09장 텍스트 범위 내)
    lower = 1.0e-2   # cm^2/g
    upper = 1.0      # cm^2/g
    ok = (lower <= sigma_over_m_cgs <= upper)
    return ok, sigma_over_m_cgs


def example_scan():
    """
    간단한 파라미터 스캔:
    - m_φ: 10^{-22} eV ~ 1 eV (로그 스캔)
    - λ:   10^{-8} ~ 1 (로그 스캔)
    각 점에서 드브로이 + 자가상호작용 조건을 평가하고,
    둘 다 만족하는 대표 점 몇 개를 출력한다.
    """
    masses_ev = [10 ** p for p in range(-22, 1, 3)]  # -22, -19, ..., -1, 0
    lambdas = [10 ** p for p in [-8, -6, -4, -2, 0]]

    print("=== φ-DM 부등식 조건 1차 스캔 (드브로이 + 자가상호작용) ===")
    for m_ev in masses_ev:
        for lam in lambdas:
            params = PhiDMParams(m_phi_eV=m_ev, lam=lam)
            ok_db, m_min = de_broglie_condition(params, v_kms=200.0)
            ok_si, sigma_over_m_cgs = self_interaction_condition(params)
            if ok_db and ok_si:
                print(
                    f"PASS: m_phi = {m_ev:.1e} eV, λ = {lam:.1e}, "
                    f"m_min ≈ {m_min:.1e} eV, σ/m ≈ {sigma_over_m_cgs:.2e} cm^2/g"
                )


if __name__ == "__main__":
    example_scan()


