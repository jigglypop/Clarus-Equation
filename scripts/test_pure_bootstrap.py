"""순수 1D 부트스트랩 방정식 자체의 검증.

사양 (sleep.md, homeomorphism.md):
  ε² = exp(-(1-ε²) · D_eff),  D_eff = 3.178 (= 3 + δ)

  반복: ε²_{n+1} = exp(-(1 - ε²_n) · D_eff)

예측:
  - 비자명 고정점 ε² = 0.0487 (4.87%)
  - 수축률 ρ = D_eff · ε² = 0.155
  - 2회 반복: 잔차 ≈ 2.4%
  - 3회 반복: 잔차 ≈ 0.4%

이 스크립트는 1D 부트스트랩 자체의 수학적 검증만 한다.
3D 확률 분포 p = (active, struct, bg)로의 확장은 paper에 명시 없음.

이 검증이 통과되면:
  -> 1D 부트스트랩은 수학적으로 정확
  -> 3D substrate 다리는 별도 정의/구현 필요

Usage:
  .venv/Scripts/python.exe scripts/test_pure_bootstrap.py
"""
from __future__ import annotations

import math

from clarus.constants import (
    AD, ACTIVE_RATIO, STRUCT_RATIO, BACKGROUND_RATIO,
    BOOTSTRAP_CONTRACTION,
)


D_EFF_SPEC = 3.0 + AD * (1.0 - AD)


def bootstrap_step(eps_sq: float, d_eff: float) -> float:
    """ε²_{n+1} = exp(-(1-ε²_n) · D_eff)."""
    return math.exp(-(1.0 - eps_sq) * d_eff)


def find_fixed_point(d_eff: float, max_iter: int = 1000, tol: float = 1e-12) -> float:
    """비자명 고정점 수치적 탐색 (시작점 0.5)."""
    x = 0.5
    for _ in range(max_iter):
        x_new = bootstrap_step(x, d_eff)
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x


def measure_contraction(eps_star: float, d_eff: float) -> float:
    """B'(ε*) = -ε* · D_eff · (-1) ... 실제로는 |B'(p*)| = D_eff · ε*."""
    return d_eff * eps_star


def main():
    print("=" * 60)
    print(" Pure 1D bootstrap equation validation")
    print("=" * 60)
    print(f"  D_eff (spec)  = 3 + delta = 3 + {AD*(1-AD):.4f} = {D_EFF_SPEC:.4f}")
    print(f"  spec ε²       = {ACTIVE_RATIO:.4f}")
    print(f"  spec ρ        = D_eff · ε² = {BOOTSTRAP_CONTRACTION:.4f}")
    print()

    # (1) 고정점 수치 검증
    eps_star = find_fixed_point(D_EFF_SPEC)
    print(f"[FIXED POINT]")
    print(f"  numerical ε*  = {eps_star:.6f}")
    print(f"  spec ε²       = {ACTIVE_RATIO:.6f}")
    print(f"  difference    = {abs(eps_star - ACTIVE_RATIO):.6f}")
    print()

    # (2) 수축률 검증
    rho = measure_contraction(eps_star, D_EFF_SPEC)
    print(f"[CONTRACTION RATE]")
    print(f"  numerical ρ   = D_eff · ε* = {rho:.6f}")
    print(f"  spec ρ        = {BOOTSTRAP_CONTRACTION:.6f}")
    print(f"  difference    = {abs(rho - BOOTSTRAP_CONTRACTION):.6f}")
    print()

    # (3) 균등 초기화에서 수렴 trajectory (sleep.md 6.2 표 검증)
    print(f"[CONVERGENCE TRAJECTORY] (initial p_0 = 1/3, spec table 5_Sparsity 6.2)")
    p = 1.0 / 3.0
    print(f"  n=0: p = {p:.6f} (initial)")
    spec_traj = {1: 0.0928, 2: 0.0555, 3: 0.0498}
    for n in range(1, 6):
        p = bootstrap_step(p, D_EFF_SPEC)
        residual = p - eps_star
        line = f"  n={n}: p = {p:.6f}  residual = {residual:+.6f}"
        if n in spec_traj:
            line += f"  (spec table: {spec_traj[n]:.4f})"
        print(line)
    print()

    # (4) 5_Sparsity 6.2의 "p_{n+1} = p* + ρ(p_n - p*)" 식 검증
    print(f"[LINEARIZED CONTRACTION] (per docs 5_Sparsity 6.2)")
    print(f"  p_{{n+1}} = p* + ρ(p_n - p*) where p* = {eps_star:.4f}, ρ = {rho:.4f}")
    p = 1.0 / 3.0
    print(f"  n=0: p = {p:.6f}")
    for n in range(1, 6):
        p = eps_star + rho * (p - eps_star)
        print(f"  n={n}: p = {p:.6f}  residual = {p - eps_star:+.6f}")
    print()

    # (5) 두 trajectory 비교: 진짜 부트스트랩 사상 vs 선형화
    print(f"[NONLINEAR vs LINEARIZED comparison]")
    p_nl = 1.0 / 3.0
    p_lin = 1.0 / 3.0
    print(f"  n  | nonlinear B    | linearized B    | error")
    print(f"  ---|----------------|-----------------|--------")
    for n in range(1, 8):
        p_nl = bootstrap_step(p_nl, D_EFF_SPEC)
        p_lin = eps_star + rho * (p_lin - eps_star)
        print(f"  {n}  | {p_nl:.6f}       | {p_lin:.6f}        | {abs(p_nl - p_lin):.6f}")
    print()

    # (6) D_eff sweep: 다른 차원에서는?
    print(f"[D_EFF SWEEP]")
    print(f"  D_eff      ε*           ρ             기능")
    print(f"  ---------- ------------ ------------- ----")
    for d in [1.0, 2.0, 2.5, 3.0, D_EFF_SPEC, 3.5, 4.0, 5.0]:
        try:
            ep = find_fixed_point(d)
            r = d * ep
            note = "(d_eff < 1: trivial)" if d < 1.0 else (
                "(spec)" if abs(d - D_EFF_SPEC) < 0.01 else ""
            )
            print(f"  {d:>8.4f}    {ep:.6f}     {r:.6f}      {note}")
        except Exception as e:
            print(f"  {d:>8.4f}    ERROR: {e}")
    print()

    # 검증 요약
    print("=" * 60)
    print(" VERDICT")
    print("=" * 60)
    eps_match = abs(eps_star - ACTIVE_RATIO) < 1e-3
    rho_match = abs(rho - BOOTSTRAP_CONTRACTION) < 1e-3
    print(f"  ε* numerical == ε² spec  : {eps_match}  "
          f"({eps_star:.4f} vs {ACTIVE_RATIO:.4f})")
    print(f"  ρ numerical == ρ spec    : {rho_match}  "
          f"({rho:.4f} vs {BOOTSTRAP_CONTRACTION:.4f})")
    print()
    if eps_match and rho_match:
        print("  -> 1D 부트스트랩 방정식은 사양 그대로 검증됨.")
        print("     ε* = 0.0487, ρ = 0.155는 수학적 정확값.")
        print()
        print("  남은 격차:")
        print("  - 1D scalar ε² → 3D probability vector p=(active,struct,bg) 확장")
        print("    의 정확한 수학이 paper에 명시되지 않음.")
        print("  - 신경망 hidden state m에서 (active, struct, bg) ratio를 어떻게")
        print("    뽑아낼지의 정의도 명시되지 않음 (TopK? threshold? energy?)")
        print("  - 이 정의를 명확히 한 후에야 spec의 implementation이 가능.")
    else:
        print("  -> 1D 부트스트랩도 수치 검증 실패. 수식 자체에 문제.")


if __name__ == "__main__":
    main()
