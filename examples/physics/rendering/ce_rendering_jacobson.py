"""강한 중력 — 화소 엔트로피 + 열적 시간 + 기록에서 아인슈타인 방정식(Jacobson). 원장: 43장 §43.72. 예측값을 바꾸지 않는다.

계산 전에 적은 판본과 규칙(단위 G = c = ħ = k_B = 1, ℓ_P² = 1):

전제. (가) 모든 국소 지평선의 엔트로피 = 넓이 × η, 화소 모형 η = 1 nat/(2ℓ_P)²(§43.71).
   (나) 열적 시간 TT(§43.53): 국소 지평선의 기록 위상은 모듈러 흐름(부스트)으로 쌓이므로 운루 온도 T = κ/2π.
   (다) R1: 지평선을 넘는 부스트 에너지 흐름은 기록이다. δQ = T dS(클라우지우스).
정리(Jacobson 1995). (가)–(다)가 모든 점·모든 널 방향에서 성립하면 G_μν + Λ g_μν = 8π G_eff T_μν, G_eff = 1/(4η).
   Λ는 적분 상수로 남는다(CE에서는 가지치기 BR이 정함).
판본: η ∈ {1 nat/(2ℓ_P)², 1 bit/(2ℓ_P)², 1 nat/ℓ_P²} → G_eff/G. 1/4은 뉴턴 상수 정규화이므로 일치는 독립 증거가 아니다.
검산: 화소 엔트로피 S = A/4로 Kerr의 첫째 법칙 dM = T dS + Ω dJ(T = κ/2π)를 유한차분으로 확인. kill: 잔차 > 1e-6.

python -B -m examples.physics.rendering.ce_rendering_jacobson
"""

from __future__ import annotations

import math

ETA_VARIANTS = {"pixel: 1 nat / (2 l_P)^2": 1 / 4, "bit pixel: ln2 / (2 l_P)^2": math.log(2) / 4,
                "single cell: 1 nat / l_P^2": 1.0}


def g_eff() -> dict:
    return {k: 1 / (4 * eta) for k, eta in ETA_VARIANTS.items()}


def _kerr(m: float, j: float) -> dict:
    a = j / m
    rp = m + math.sqrt(m * m - a * a)
    area = 4 * math.pi * (rp * rp + a * a)
    kappa = (rp - m) / (rp * rp + a * a)
    omega = a / (rp * rp + a * a)
    return {"S": area / 4, "T": kappa / (2 * math.pi), "Omega": omega}


def kerr_first_law(spins=(0.0, 0.3, 0.67, 0.9, 0.99), h: float = 1e-6) -> dict:
    """dM = T dS + Ω dJ를 M = 1에서 유한차분으로: (∂S/∂M)_J = 1/T, (∂S/∂J)_M = −Ω/T."""
    out = {}
    for chi in spins:
        m, j = 1.0, chi
        k = _kerr(m, j)
        ds_dm = (_kerr(m + h, j)["S"] - _kerr(m - h, j)["S"]) / (2 * h)
        ds_dj = (_kerr(m, j + h)["S"] - _kerr(m, j - h)["S"]) / (2 * h) if chi > 0 else 0.0
        out[chi] = {"T": k["T"], "res_dM": ds_dm * k["T"] - 1, "res_dJ": ds_dj * k["T"] + k["Omega"]}
    out["max_residual"] = max(max(abs(v["res_dM"]), abs(v["res_dJ"])) for v in out.values() if isinstance(v, dict))
    out["killed"] = out["max_residual"] > 1e-6
    return out


def pixel_energy() -> dict:
    """화소 하나(1 nat)를 더하는 에너지 = T_H: dM = T dS, dS = 1 → dM = T. 슈바르츠실트 M = 1."""
    m = 1.0
    t = 1 / (8 * math.pi * m)
    da = 4.0
    dm = da / (32 * math.pi * m)
    return {"T_H": t, "dM_per_pixel": dm, "ratio": dm / t}


def main() -> None:
    print("G_eff/G:", {k: round(v, 4) for k, v in g_eff().items()})
    fl = kerr_first_law()
    print("Kerr first law max residual:", fl["max_residual"], "killed:", fl["killed"])
    print("pixel energy:", pixel_energy())


if __name__ == "__main__":
    main()
