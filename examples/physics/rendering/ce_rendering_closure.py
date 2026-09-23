"""마감 판정: 기록 완성 기준, 지평선 미시 상태, 가설 장부. 원장: 43장 §43.33. 예측값을 바꾸지 않는다.

1. 기록 완성 기준(계산 전 고정 두 후보).
   R2 중복도: 경로 정보가 독립 환경 조각 둘 이상에 복사되면 확정("나와 너가 둘 다 읽음").
   RL 란다우어: 경로 정보가 열 저장고로 비가역적으로 버려지면(비트당 k_B ln 2 이상 엔트로피) 확정.
   판정 자료(정성): Page–Geilker, BMV(고립 질량), 양자 지우개(한 표지), 다광자·다큐비트 GHZ 결맞음, 뜨거운 C70.
2. 지평선: 면적 칸 a ℓ_P²마다 정수 d개의 통로면 S = (A / a ℓ_P²) ln d. 1/4 계수는 ln d = a/4.
   칸 넓이 a ∈ {1, 4}(사전 고정)에서 정수 d가 있는지.
3. 가설 장부: 이 장에서 자료를 보고 고른 선택의 bit 합.

python -B -m examples.physics.rendering.ce_rendering_closure
"""

from __future__ import annotations

import math

# (실험, 경로 정보가 복사된 독립 조각 수, 열 저장고로 버려졌는가, 관측: 결맞음이 유지/복원되는가)
RECORD_CASES = (
    ("Page-Geilker (macroscopic detector)", 10 ** 20, True, False),
    ("BMV isolated masses", 0, False, True),
    ("quantum eraser, one marker photon", 1, False, True),
    ("multi-photon / multi-qubit GHZ coherence (>= 3 parties)", 3, False, True),
    ("hot C70 emitting thermal photons", 10, True, False),
    ("cold large-molecule interference", 0, False, True),
)


def record_confirmed(rule: str, copies: int, dumped_to_bath: bool) -> bool:
    if rule == "R2":
        return copies >= 2
    if rule == "RL":
        return dumped_to_bath
    raise ValueError(rule)


def record_rule_verdicts() -> dict:
    """규칙마다 관측과 어긋나는 경우. 확정이면 결맞음이 사라져야 한다."""
    out = {}
    for rule in ("R2", "RL"):
        out[rule] = [name for name, n, bath, coherent in RECORD_CASES
                     if record_confirmed(rule, n, bath) == coherent]
    return out


def horizon_integer_channels(areas: tuple[float, ...] = (1.0, 4.0)) -> dict:
    """ln d = a/4에서 필요한 d와 가장 가까운 정수의 차."""
    res = {}
    for a in areas:
        d = math.exp(a / 4)
        res[a] = {"d_required": d, "nearest_integer": round(d), "mismatch": abs(d - round(d))}
    return res


INSPIRATION_BITS = (   # (규칙, bit, 절)
    ("S2 octant channel", 0.0, "43.3 (counted in registry rows)"),
    ("O1 tilt pi/8 (direct readouts)", 0.5, "43.8"),
    ("W2 frequency choice -> C6 + lowest harmonic + sign", 1.6, "43.26-43.27"),
    ("W3 reading (b)", 1.0, "43.28 (competing branch only)"),
    ("G1m amplitude k = Omega_m", 2.0, "43.32"),
)


def main() -> None:
    print("record rules, cases contradicting observation:", record_rule_verdicts())
    print("horizon integer channels:", {a: {k: round(v, 4) for k, v in r.items()} for a, r in horizon_integer_channels().items()})
    print("inspiration bits in the adopted chain (W2 + G1m + O1):", 1.6 + 2.0 + 0.5)


if __name__ == "__main__":
    main()
