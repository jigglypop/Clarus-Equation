"""공동 렌더링 — 화소 요동은 시간축 회전이라 차등 간섭계에서 사라진다. 원장: 43장 §43.85. 예측값을 바꾸지 않는다.

사용자: “양자 중력은 확률로 정의했다. 3개 차원의 렌더링 때문에 시공간축으로 돈다.” 초기부터 있던 원리의 적용(QG1 §43.30 v10, O1).
QG1: 중력의 원천은 확률 무게. 화소 요동은 기록 시각(시간축)의 요동이다. 렌더링은 세 축의 공동 기록(det, α_s = det과 같은
구조)이라 사건 하나에 시계 하나 → 요동은 세 공간 방향에 공통(시간축 회전). 계산 전에 적은 판본과 규칙:

(가) 세 축 요동의 상관 ρ. 마이컬슨 차등(x − y) 응답을 독립일 때 1로 정규화: α_diff = 1 − ρ(몬테카를로 확인).
    공동 렌더링 ρ = 1(CE), 축별 독립 ρ = 0(VZ형).
(나) 잔여: 팔 길이 차 ε = ΔL/L이면 공통 요동도 α ≈ ε²만큼 샌다.
(다) Holometer·LIGO의 3σ 상한(§43.84)과 비교.
새 예측 후보 P42: 차등 간섭계(GQuEST)는 α < 0.1까지 신호가 없다. kill: GQuEST가 α ≥ 0.1 신호를 5σ로 검출.
증거의 지위: v10부터 있던 원리의 적용이나, Holometer·LIGO 무신호 자료가 먼저 나왔으므로 일치는 일관성으로 센다.

python -B -m examples.physics.rendering.ce_rendering_joint_render
"""

from __future__ import annotations

import numpy as np

from examples.physics.rendering import ce_rendering_modular as MOD


def alpha_diff(rho: float, n: int = 400000, seed: int = 4, eps: float = 0.0) -> float:
    """세 축 요동(분산 1, 상관 ρ)에서 팔 x(길이 1+ε)와 y(길이 1)의 차등 분산 / 독립 기준(2)."""
    rng = np.random.default_rng(seed)
    cov = np.full((3, 3), rho) + (1 - rho) * np.eye(3)
    d = rng.multivariate_normal(np.zeros(3), cov, n)
    diff = (1 + eps) * d[:, 0] - d[:, 1]
    return float(diff.var() / 2.0)


def scan() -> dict:
    rows = {rho: alpha_diff(rho) for rho in (0.0, 0.5, 0.9, 0.99, 1.0)}
    resid = {eps: alpha_diff(1.0, eps=eps) for eps in (1e-2, 1e-3)}
    ce = rows[1.0]
    bounds = MOD.CONSTRAINTS_3SIGMA
    return {"alpha_diff_by_rho": rows, "residual_by_mismatch": resid, "alpha_CE": ce,
            "CE_within_all_bounds": all(ce < b for b in bounds.values()),
            "VZ_like_excluded": [k for k, b in bounds.items() if rows[0.0] > b],
            "P42_null_at_gquest": ce < MOD.GQUEST_REACH}


def main() -> None:
    s = scan()
    print("alpha_diff by rho:", {k: round(v, 5) for k, v in s["alpha_diff_by_rho"].items()})
    print("residual by arm mismatch (rho=1):", {k: f"{v:.2e}" for k, v in s["residual_by_mismatch"].items()})
    print({k: v for k, v in s.items() if k not in ("alpha_diff_by_rho", "residual_by_mismatch")})


if __name__ == "__main__":
    main()
