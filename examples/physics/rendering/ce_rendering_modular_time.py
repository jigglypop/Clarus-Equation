"""경계 밖이 시간을 만든다 — "없지도 있지도 않은 상태"의 모듈러 정식화. 원장: 43장 §43.104. 예측값을 바꾸지 않는다.

사용자(2026-09-25): "없음이 아님. 없지도 않고 있지도 않은 상태임." → "다음". §43.103에서 이 상태는 관계적이었다
(관측자의 렌더링 경계 밖: 무게는 있고 렌더링은 없음). 여기서는 그 상태가 렌더링되는 쪽의 시간(TT 시계)을 만든다는
명제를 식으로 세운다. 수학은 표준(Tomita–Takesaki, Connes–Rovelli 열적 시간, Gibbons–Hawking)이고, CE의 몫은 식별이다.

계산 전 고정(2026-09-25):
정의  관측자의 렌더링 대수 M = 관측자가 평생 기록을 받을 수 있는 영역(세계선 전체의 인과 과거, 사건 지평 안쪽)의
      관측량. 경계 밖 = 교환자 대수 M′. 전체 상태 ψ는 순수하다(QG1·56장: 비관측 성분을 지우지 않는다).
      OBJ-03의 다섯 요구를 이 틀로 채운다: 존재공간 M′, 정규화(ψ의 제한 ρ_B, spec ρ_B = spec ρ_A), 동역학(모듈러 흐름,
      M′ 위에서는 거꾸로), 보존량(ln Δ ψ = 0), 관측 투영(제한 = 부분 대각합).
검사(유한 차원 d = 5, 무작위 표본, 허용오차 1e-10; 수학이므로 통과가 예상되는 일관성 검사다):
MK1 삼분법: 경계 밖이 없음(곱 상태)이면 ρ_A가 순수해 충실하지 않고 모듈러 흐름이 없다. 있음(대수에 포함)이면
    전체 상태가 순수해 역시 흐름이 없다. 없지도 있지도 않음(얽힘, 대수 밖)이면 ρ_A가 충실하고 흐름이 있다.
    반례가 하나라도 있으면 기각한다. 양자장론에서는 레–슐리더 정리 때문에 "없음"이 불가능하다(인용, 계산 아님).
MK2 분별 = 시간: 흐름이 모든 s에서 항등인 것은 ρ_A ∝ 1(무게가 고름, 분별 없음)일 때뿐이다.
MK3 열적 시간: ρ_A = e^{−βH}/Z이면 σ_s(X) = e^{−iβHs} X e^{iβHs}이다. 따라서 물리 시간은 t = β|s|,
    Φ = 2π|s|, dΦ/dt = 2πk_BT/ħ다(TT의 식).
MK4 균형: (ln ρ_A ⊗ 1 − 1 ⊗ ln ρ_B)ψ = 0이다. 경계 밖(M′)의 흐름은 제 모듈러 시간으로 거꾸로 간다.
MK5 절반: 얽힌 전체 상태의 진폭은 e^{−βE/2}(유클리드 반원), 렌더링 쪽 무게는 e^{−βE}(온원)다. H1의 ½(§43.51,
    보른 규칙)과 같은 구조이며 새 유도가 아니다.
MK6 드 시터 속도: T = ħH_Λ/2πk_B이면 2πk_BT/ħ = H_Λ이고, TT V-dS의 V39 0.8407을 그대로 낸다.
MK7 영역 하나: 렌더링 영역은 사건 지평(널 초곡면 하나)으로 둘러싸인 고정 영역이다. 대수가 하나이므로 모듈러 군도
    하나이고, 기하적 근사에서 속도는 상수 하나, 곧 점근 표면 중력 H_Λ(최종 지평)다. 순간값 c/r_e(t)는
    대수의 족이라 모듈러 군이 아니다. 공개: §43.53에서 V-EH가 자료로 +15.8σ 기각된 뒤에 적은 논증이므로 증거로
    세지 않는다(설명).
지위 제안: TT는 공리에서 조건부 정리로 간다. 조건은 셋이다. (가) Connes–Rovelli 가설, (나) 렌더링 영역 상태의 KMS
(드 시터에서는 정확, ΛCDM에서는 점근), (다) 전체 상태가 경계를 가로질러 얽혀 있음(양자장론에서는 레–슐리더).
한계: 유한 모형은 구조의 예시이고, 지평의 대수는 III₁형이다(밀도행렬은 없지만 Tomita–Takesaki는 성립한다).
[정정, 사용자 2026-09-25 "주기는 거리잖아. 위상만 같을 뿐임." / "시공간 축의 순환을 뜻하는데 시공간 자체가 처음으로
돌아가는 것 그 이상도 이하도 아님."] 처음 적은 한계 "모듈러 군은 ℝ이라 실시간 순환은 나오지 않는다"는 철회한다.
순환은 시공간 축이 한 바퀴 돌아 처음 자리로 돌아오는 것이고, 모듈러 흐름 ℝ은 바퀴를 세는 거리다. axis_cycle()은
한 바퀴의 거리와 오늘의 위치만 잰다. 규모·로그 나선·진폭 4π를 순환에 붙였던 첫 정정 판(helix)은 과잉 해석이라
지웠다(축의 회전은 규모와 독립, C5).

python -B -m examples.physics.rendering.ce_rendering_modular_time
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import cumulative_trapezoid

from examples.physics.rendering import ce_rendering_thermal_time as TTM

TOL = 1e-10
D = 5
SAMPLES = 40
S_GRID = (-1.3, -0.4, 0.25, 0.7, 2.0)
HBAR, KB, GYR_S = 1.054571817e-34, 1.380649e-23, 1e9 * 365.25 * 86400


# ------------------------------------------------------------------ 유한 모형
def _haar(d: int, rng: np.random.Generator) -> np.ndarray:
    z = (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) / math.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def state(kind: str, d: int, rng: np.random.Generator, beta: float = 1.3) -> tuple[np.ndarray, np.ndarray | None]:
    """계수 행렬 C(ψ = Σ C_ij |i⟩_A|j⟩_B)와, TFD면 A 쪽 해밀토니안."""
    if kind == "product":                                     # 경계 밖이 없음: 얽힘 없음
        a, b = rng.normal(size=d) + 1j * rng.normal(size=d), rng.normal(size=d) + 1j * rng.normal(size=d)
        c = np.outer(a, b)
    elif kind == "maximal":                                   # 분별 없음: 무게가 고름
        c = _haar(d, rng)
    elif kind == "tfd":                                       # 열적 얽힘: 진폭 e^{−βE/2}
        e = np.sort(rng.uniform(0.0, 3.0, size=d))
        u = _haar(d, rng)
        c = u @ np.diag(np.exp(-beta * e / 2)) @ _haar(d, rng)
        return c / np.linalg.norm(c), u @ np.diag(e) @ u.conj().T
    else:                                                     # 없지도 있지도 않음: 일반 얽힘
        c = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    return c / np.linalg.norm(c), None


def reduced(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ρ_A = C C†, ρ_B = (C† C)ᵀ = Cᵀ C̄."""
    return c @ c.conj().T, c.T @ c.conj()


def _power(rho: np.ndarray, z: complex) -> np.ndarray:
    w, v = np.linalg.eigh(rho)
    return v @ np.diag(np.exp(z * np.log(w))) @ v.conj().T


def flow(rho: np.ndarray, x: np.ndarray, s: float) -> np.ndarray:
    """모듈러 흐름 σ_s(X) = ρ^{is} X ρ^{−is} (ρ가 충실할 때)."""
    return _power(rho, 1j * s) @ x @ _power(rho, -1j * s)


def _faithful(rho: np.ndarray) -> bool:
    return float(np.linalg.eigvalsh(rho).min()) > TOL


def _movement(rho: np.ndarray, rng: np.random.Generator) -> float:
    x = rng.normal(size=rho.shape) + 1j * rng.normal(size=rho.shape)
    return max(float(np.abs(flow(rho, x, s) - x).max()) for s in S_GRID)


def trichotomy(d: int = D, samples: int = SAMPLES, seed: int = 0) -> dict:
    """MK1·MK2."""
    rng = np.random.default_rng(seed)
    out = {"nothing_faithful": 0, "whole_faithful": 0, "neither_faithful": 0, "neither_min_move": math.inf,
           "flat_max_move": 0.0}
    for _ in range(samples):
        c, _ = state("product", d, rng)
        out["nothing_faithful"] += _faithful(reduced(c)[0])
        c, _ = state("neither", d, rng)
        psi = c.reshape(-1)
        out["whole_faithful"] += _faithful(np.outer(psi, psi.conj()))           # 경계 밖까지 대수에 넣으면 순수 상태
        ra = reduced(c)[0]
        out["neither_faithful"] += _faithful(ra)
        out["neither_min_move"] = min(out["neither_min_move"], _movement(ra, rng))
        c, _ = state("maximal", d, rng)
        out["flat_max_move"] = max(out["flat_max_move"], _movement(reduced(c)[0], rng))
    out["samples"] = samples
    out["passed"] = (out["nothing_faithful"] == 0 and out["whole_faithful"] == 0 and out["neither_faithful"] == samples
                     and out["neither_min_move"] > 1e-3 and out["flat_max_move"] < TOL)
    return out


def thermal_time(d: int = D, samples: int = SAMPLES, beta: float = 1.3, seed: int = 1) -> dict:
    """MK3·MK5: TFD에서 모듈러 흐름 = 시간 발전(t = βs), 진폭 e^{−βE/2}, 무게 e^{−βE}."""
    rng = np.random.default_rng(seed)
    err_t, err_half = 0.0, 0.0
    for _ in range(samples):
        c, h = state("tfd", d, rng, beta)
        ra = reduced(c)[0]
        w, v = np.linalg.eigh(h)
        gibbs = v @ np.diag(np.exp(-beta * w)) @ v.conj().T
        gibbs /= np.trace(gibbs).real
        err_half = max(err_half, float(np.abs(ra - gibbs).max()))
        x = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        for s in S_GRID:
            u = v @ np.diag(np.exp(-1j * beta * w * s)) @ v.conj().T
            err_t = max(err_t, float(np.abs(flow(ra, x, s) - u @ x @ u.conj().T).max()))
    return {"beta": beta, "flow_equals_time_err": err_t, "half_amplitude_err": err_half,
            "passed": err_t < 1e-9 and err_half < 1e-9}


def balance(d: int = D, samples: int = SAMPLES, seed: int = 2) -> dict:
    """MK4: (ln ρ_A ⊗ 1 − 1 ⊗ ln ρ_B)ψ = 0, M′ 위의 흐름은 거꾸로."""
    rng = np.random.default_rng(seed)
    err, err_back = 0.0, 0.0
    eye = np.eye(d)
    for _ in range(samples):
        c, _ = state("neither", d, rng)
        ra, rb = reduced(c)
        psi = c.reshape(-1)
        k = np.kron(_log(ra), eye) - np.kron(eye, _log(rb))
        err = max(err, float(np.abs(k @ psi).max()))
        s = 0.37
        delta_is = np.kron(_power(ra, 1j * s), _power(rb, -1j * s))
        y = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        lhs = delta_is @ np.kron(eye, y) @ np.linalg.inv(delta_is)
        rhs = np.kron(eye, flow(rb, y, -s))                   # 경계 밖은 제 모듈러 시간으로 −s
        err_back = max(err_back, float(np.abs(lhs - rhs).max()))
    return {"annihilation_err": err, "commutant_backward_err": err_back, "passed": err < 1e-9 and err_back < 1e-9}


def _log(rho: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(rho)
    return v @ np.diag(np.log(w)) @ v.conj().T


# ------------------------------------------------------------------ 우주
def de_sitter_rate() -> dict:
    """MK6: Gibbons–Hawking 온도의 열적 시간 속도 = H_Λ. TT V-dS 점수를 그대로 낸다."""
    _, om, h0, hl = TTM._cosmo()
    temp = HBAR * (hl / GYR_S) / (2 * math.pi * KB)
    rate = 2 * math.pi * KB * temp / HBAR * GYR_S
    tt = TTM.score("V-dS")
    return {"H_Lambda_per_Gyr": hl, "T_GH_K": temp, "rate_per_Gyr": rate, "rate_err": abs(rate / hl - 1),
            "V39": tt["V39"], "theta0": tt["theta_today"], "passed": abs(rate / hl - 1) < 1e-12}


def one_domain() -> dict:
    """MK7: ΛCDM(TT와 같은 배경)에서 사건 지평의 순간값 c/r_e(t)는 변하고, 끝에서 H_Λ로 간다."""
    _, om, h0, hl = TTM._cosmo()
    n = np.linspace(math.log(1e-6), 40.0, 400001)
    a = np.exp(n)
    hub = h0 * np.sqrt(om / a ** 3 + 1 - om)                  # 1/Gyr, c = 1 Gly/Gyr
    t = cumulative_trapezoid(1 / hub, n, initial=0.0) + 2 / (3 * hub[0])
    f = 1 / (a * hub)                                         # 먼 끝에서 거꾸로 쌓아 큰 수의 뺄셈(상쇄 오차)을 피한다
    chi_e = -cumulative_trapezoid(f[::-1], n[::-1], initial=0.0)[::-1] + math.exp(-n[-1]) / hl
    rate = 1 / (a * chi_e)                                    # c / r_e(t)
    at = lambda tt: float(np.interp(tt, t, rate / hl))
    i0, i20 = int(np.argmin(np.abs(n))), int(np.argmin(np.abs(n - 20.0)))  # 꼬리를 정의한 마지막 점은 피한다
    return {"ratio_at_Gyr": {x: round(at(x), 4) for x in (0.38e-3, 1.0, 5.0, float(t[i0]), 50.0, 200.0)},
            "today_ratio": float(rate[i0] / hl), "late_ratio_N20": float(rate[i20] / hl),
            "rate_varies": float(rate[i0] / hl) > 1.01, "passed": abs(float(rate[i20] / hl) - 1) < 1e-6}


def axis_cycle() -> dict:
    """순환 = 시공간 축이 한 바퀴 돌아 처음 자리로 돌아오는 것(사용자). 한 바퀴의 거리와 오늘의 위치만 잰다."""
    _, om, _, hl = TTM._cosmo()
    phi0 = 2 / 3 * math.asinh(math.sqrt((1 - om) / om))           # TT와 같은 ΛCDM 배경의 오늘 위상
    return {"period_Gyr": 2 * math.pi / hl, "period_length_Gly": 2 * math.pi / hl,   # c = 1 Gly/Gyr: 둘레 2πc/H_Λ
            "phi_today": phi0, "fraction_today": phi0 / (2 * math.pi)}


def verdict() -> dict:
    parts = {"MK1_MK2": trichotomy(), "MK3_MK5": thermal_time(), "MK4": balance(), "MK6": de_sitter_rate(),
             "MK7": one_domain()}
    return {"all_passed": all(p["passed"] for p in parts.values()), **{k: p["passed"] for k, p in parts.items()}}


def main() -> None:
    print("MK1-2 trichotomy:", trichotomy())
    print("MK3,5 thermal time:", thermal_time())
    print("MK4 balance:", balance())
    print("MK6 de Sitter rate:", de_sitter_rate())
    print("MK7 one domain:", one_domain())
    print("verdict:", verdict())
    print("axis cycle:", axis_cycle())


if __name__ == "__main__":
    main()
