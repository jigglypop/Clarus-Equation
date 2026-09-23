"""E4의 지수 — 차원 2개를 뛰어넘어 4(보른 배가). 원장: 43장 §43.82. 예측값을 바꾸지 않는다.

사용자 가설: “그냥 차원 2개 뛰어넘은 거니까 4.” sin θ_W는 m차원 평면의 렌더링 진폭 A_m = m a^m이고, 관측되는 ŝ²는
진폭 × 켤레 진폭(보른)이다. 켤레는 평면의 거울상(“아닌 나”, 여집합 = 반입자 §43.44)이므로 확률은 평면 ⊕ 켤레, 곧 2m차원의
진폭 A_{2m}이어야 한다. 계산 전에 적은 조건과 kill(m = 0 … 6):

(가) 보른 배가 일관성: A_m² = A_{2m}가 모든 a에서 성립(m² = 2m).
(나) 닻: 공평점 a = ½에서 A_m² = ¼(§43.80의 렙톤 대각합, 따로 유도된 값).
(다) 사용자: 뛰어넘는 차원 2 = 배가, m + 2 = 2m.
kill: 세 조건의 교집합(자명한 m = 0 제외)이 {2}가 아니면 기각.
확인: m = 2이면 sin θ_W = 2a², ŝ² = 4a⁴ = 4α_s^{4/3}(E4)이며 세계 평균 α_s 대비 pull을 보고한다.

python -B -m examples.physics.rendering.ce_rendering_born_doubling
"""

from __future__ import annotations

from fractions import Fraction

from examples.physics.rendering import ce_rendering_registry as R

M_RANGE = range(0, 7)
AS_WORLD = (0.1180, 0.0009)


def born_consistent(m: int) -> bool:
    return m * m == 2 * m


def anchor_ok(m: int) -> bool:
    return (Fraction(m, 2 ** m)) ** 2 == Fraction(1, 4)


def user_jump(m: int) -> bool:
    return m + 2 == 2 * m


def scan() -> dict:
    sets = {"born": {m for m in M_RANGE if born_consistent(m)},
            "anchor": {m for m in M_RANGE if anchor_ok(m)},
            "user_jump": {m for m in M_RANGE if user_jump(m)}}
    inter = (sets["born"] & sets["anchor"] & sets["user_jump"]) - {0}
    return {**{k: sorted(v) for k, v in sets.items()}, "intersection": sorted(inter), "killed": inter != {2}}


def e4_check() -> dict:
    s2 = R.SZ2
    a = (s2 / 4) ** 0.25                     # ŝ² = A₄ = 4a⁴
    alpha = a ** 3
    return {"alpha_s_from_A4": alpha, "pull_world": (alpha - AS_WORLD[0]) / AS_WORLD[1],
            "A2_squared_equals_A4": abs((2 * a * a) ** 2 - 4 * a ** 4) < 1e-15}


def main() -> None:
    print("scan:", scan())
    print("E4 check:", e4_check())


if __name__ == "__main__":
    main()
