"""E4의 사전 — 결합은 기록의 확률, 섞임은 그 척도 미분. 원장: 43장 §43.83. 예측값을 바꾸지 않는다.

남은 읽기 AP(섞임 = 평면의 렌더링 진폭 tr·det)와 RC(결합 = 가둔 기록의 확률 det)를 하나로 모은다.
야코비: a ∂_a det(a I_m) = m a^m = tr·det. 곧 섞임은 기록 확률의 척도(오일러) 미분이다. 계산 전에 적은 판본과 kill:

검산: 야코비 항등식을 수치 미분으로(m = 1 … 5).
가족: 연산자 {∂_a, a∂_a, (1−a)∂_a, a(1−a)∂_a(로짓), a²∂_a} × m ∈ {1, 2, 3}를 det(a I_m) = a^m에 적용해 sin θ_W에 맞추는 a
   (0 < a < 1, 해가 여럿이면 모두)를 구하고 α_s = a³을 세계 평균 0.1180 ± 0.0009와 비교.
kill: a∂_a(m = 2) 밖의 판본이 1σ 안이면 “척도 미분”의 유일성 기각. E4는 자료로 찾은 식이라 적중은 증거가 아니다.

python -B -m examples.physics.rendering.ce_rendering_dictionary
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from examples.physics.rendering import ce_rendering_registry as R

AS_WORLD = (0.1180, 0.0009)
OPERATORS = {"d_a": lambda a, m: m * a ** (m - 1),
             "a d_a": lambda a, m: m * a ** m,
             "(1-a) d_a": lambda a, m: (1 - a) * m * a ** (m - 1),
             "a(1-a) d_a": lambda a, m: a * (1 - a) * m * a ** (m - 1),
             "a^2 d_a": lambda a, m: a * a * m * a ** (m - 1)}


def jacobi_check(h: float = 1e-6) -> float:
    worst = 0.0
    for m in range(1, 6):
        for a in (0.3, 0.49, 0.5, 0.6):
            det = lambda x: np.linalg.det(x * np.eye(m))
            num = a * (det(a + h) - det(a - h)) / (2 * h)
            worst = max(worst, abs(num - m * a ** m))
    return worst


def _roots(f, target: float) -> list[float]:
    grid = np.linspace(1e-4, 1 - 1e-4, 4001)
    vals = np.array([f(x) - target for x in grid])
    out = []
    for i in np.where(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]:
        out.append(brentq(lambda x: f(x) - target, grid[i], grid[i + 1]))
    return out


def scan() -> dict:
    s = math.sqrt(R.SZ2)
    rows = {}
    for name, op in OPERATORS.items():
        for m in (1, 2, 3):
            roots = _roots(lambda a: op(a, m), s)
            rows[f"{name} | m={m}"] = [{"a": a, "alpha_s": a ** 3, "pull": (a ** 3 - AS_WORLD[0]) / AS_WORLD[1]}
                                       for a in roots]
    hits = sorted(k for k, v in rows.items() if any(abs(r["pull"]) <= 1 for r in v))
    return {"rows": rows, "hits": hits, "killed": hits != ["a d_a | m=2"]}


def main() -> None:
    print("Jacobi worst residual:", jacobi_check())
    s = scan()
    for k, v in s["rows"].items():
        txt = ", ".join(f"a={r['a']:.4f} a_s={r['alpha_s']:.4f} pull={r['pull']:+.1f}" for r in v) or "no solution"
        print(f"  {k:18s} {txt}")
    print("hits:", s["hits"], "killed:", s["killed"])


if __name__ == "__main__":
    main()
