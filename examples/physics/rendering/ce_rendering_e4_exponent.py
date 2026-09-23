"""E4의 지수 — 닻에서의 벗어남은 가둔 영역의 T3 상태 넷의 공동 인자다. 원장: 43장 §43.81. 예측값을 바꾸지 않는다.

E4의 약력 쪽: ŝ² = ¼(2a)⁴, 닻 ¼은 가둔 기록(색 단일항) 영역의 대각합(§43.80). 남은 것은 지수 4(sin θ_W로는 2).
계산 전에 적은 판본과 kill:

(가) 대칭 기운 동전(BC, §43.57)을 렙톤 대각합의 상태 무게로: 모든 축 / 약 축만 / 색 축만.
(나) 공동 인자: 가둔 영역에서 T3를 나르는 상태 수 N_T를 세고, 그 상태들이 동시에 공평 배치로 남을 확률 ρ^n
    (ρ = 2a = 1 − λ)을 T3 몫에 곱한다(기록은 곱한다, AP). n = 1 … 6.
λ는 FP에서 가져오지 않는다(FP가 E4의 지수를 쓰므로 순환). 세계 평균 α_s에서 λ = 1 − 2α_s^{1/3}, 오차는 α_s 전파 ⊕ ŝ².
kill: 실제 ŝ²를 1σ 안에서 맞추는 n이 N_T와 다르면 읽기 기각. E4는 자료로 찾은 식이라 n = 4의 적중은 증거가 아니다.

python -B -m examples.physics.rendering.ce_rendering_e4_exponent
"""

from __future__ import annotations

import math

from examples.physics.rendering import ce_rendering_lepton_trace as LT
from examples.physics.rendering import ce_rendering_registry as R

AS_WORLD = (0.1180, 0.0009)


def t3_confined_count() -> int:
    return sum(1 for s in LT.states() if s["k"] in (0, 3) and s["T3"] != 0)


def _lambda(alpha: float) -> float:
    return 1 - 2 * alpha ** (1 / 3)


def symmetric_bias(which: str, lam: float) -> float:
    """렙톤 대각합에 BC 확률 무게: 축이 렌더링이면 ½(1+λ), 아니면 ½(1−λ)."""
    p, a = 0.5 * (1 + lam), 0.5 * (1 - lam)
    t = q = 0.0
    for s in LT.states():
        if s["k"] not in (0, 3):
            continue
        w = 1.0
        if which in ("all", "colour"):
            w *= p ** s["k"] * a ** (3 - s["k"])
        if which in ("all", "weak"):
            w *= p ** s["j"] * a ** (2 - s["j"])
        t += w * float(s["T3"] ** 2)
        q += w * float(s["Q"] ** 2)
    return t / q


def scan() -> dict:
    lam = _lambda(AS_WORLD[0])
    s2, s2e = R.SZ2, R.SZ2_ERR
    out = {"lambda_from_world_alpha_s": lam, "N_T": t3_confined_count(), "symmetric": {}, "joint": {}}
    for which in ("all", "weak", "colour"):
        v = symmetric_bias(which, lam)
        out["symmetric"][which] = {"s2": v, "pull": (v - s2) / s2e}
    dlnrho = (1 / 3) * AS_WORLD[1] / AS_WORLD[0] * (1 - lam) / (1 - lam)   # d ln(2a) = (1/3) d ln α_s
    for n in range(1, 7):
        v = 0.25 * (1 - lam) ** n
        sig = math.hypot(v * n * dlnrho, s2e)
        out["joint"][n] = {"s2": v, "pull": (v - s2) / sig}
    fits = [n for n, d in out["joint"].items() if abs(d["pull"]) <= 1]
    out["fitting_n"] = fits
    out["killed"] = fits != [out["N_T"]]
    return out


def main() -> None:
    s = scan()
    print("lambda:", round(s["lambda_from_world_alpha_s"], 6), "N_T:", s["N_T"])
    for k, v in s["symmetric"].items():
        print(f"  symmetric {k:7s} s2={v['s2']:.5f} pull={v['pull']:+.1f}")
    for n, v in s["joint"].items():
        print(f"  joint n={n}  s2={v['s2']:.5f} pull={v['pull']:+.2f}")
    print("fitting n:", s["fitting_n"], "killed:", s["killed"])


if __name__ == "__main__":
    main()
