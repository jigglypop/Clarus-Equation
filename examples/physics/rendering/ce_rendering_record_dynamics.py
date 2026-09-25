"""기록의 동역학 — 기록은 겹겹이 커지는 대수이고, 되돌릴 수 없으며, 기록 전의 질량은 자기 무게에 끌린다.
원장: 43장 §43.105(OBJ-02 1단계). 예측값을 바꾸지 않는다.

사용자(2026-09-25): "다음" — 허점 장부의 "기록은 언제, 얼마나 빨리 생기는가"(§43.30 한계: 기록 갱신의 동역학은
CE에 아직 없다; 자기 중력(슈뢰딩거–뉴턴)의 크기도 계산하지 않았다).

CE에 이미 있는 것(계산 전 점검, 43장·01장·56장): RL(경로 정보가 열 저장소에 비가역적으로 버려지면, 비트당
k_B ln 2 이상, 기록 완성; 정성 판정), R1(기록은 빛원뿔 안에서만 원천을 바꾸고 c 이하로 퍼짐; 자발 붕괴 없음,
새 속도 상수 없음), QG1(나)(원천 = 국소 ρ의 Tr ρT, 정리 A), P23(BMV 얽힘 0), 화살표 "기록이 쌓이는 쪽이 미래"
1 bit. 기록 대수의 포함을 모듈러 이론으로 다룬 곳은 없다. 01장의 미시 기록 모형은 모두 가역이다.

계산 전 고정(2026-09-25):
정의  관측자의 기록 대수 M(τ) = 세계선의 τ까지의 인과 과거 관측량(R1). τ₁ < τ₂이면 M(τ₁) ⊂ M(τ₂)다.
      관측자에게의 확정 = 전체 순수 상태의 제한이다. τ₁–τ₂ 사이에 새로 확정된 것은 상대 교환자 M(τ₁)′ ∩ M(τ₂)다.
      유한 모형은 옛 기록 A, 새 기록 C, 나머지(경계 밖) R로 둔다.
RD1 비가역성(Takesaki 정리): 상태를 보존하는 "기록 되돌림"(조건부 기댓값 M(τ₂) → M(τ₁))은 M(τ₁)이 M(τ₂)의
    모듈러 흐름에 불변일 때만 있다. 유한 모형에서 그 조건은 ρ_AC = ρ_A ⊗ ρ_C, 곧 새 기록이 옛 기록과 상관없을
    때다. 검산: d_A = d_C = 3, 무작위 40표본. 곱 상태는 누출 0(< 1e-10)이고 E(Y) = Tr_C[(1⊗ρ_C)Y]⊗1이 상태를
    보존한다. 상관 상태(I(A:C) > 1e-3)는 누출 > 1e-3이고 같은 E가 상태를 보존하지 않는다. 반례가 하나라도
    있으면 식별을 기각한다. 되돌릴 수 없음의 양은 I(A:C) = S(ρ_AC ‖ ρ_A⊗ρ_C)로 읽는다.
RD2 방향 1 bit: 모듈러 켤레 J는 M과 M′을 맞바꾸고 흐름을 뒤집는다(§43.104 MK4). 반쪽 포함에서도 J U(a) J = U(−a)
    (Borchers)라 에너지 양수와 KMS만으로는 두 방향이 갈리지 않는다고 예상한다. 이 경우 화살표 1 bit는 남는다.
RD3 언제·얼마나 빨리: CE의 답은 RL(버림 완성) + R1(빛원뿔 도착)이다. 보편 속도 상수는 없다(있으면 자발 붕괴).
    kill: 자발 붕괴 잡음(X선 방출, 가열, 힘 잡음)이 검출되면 기각한다. 문헌 판정만 하고 계산은 없다.
RD4 자기 중력(§43.30 빈칸): QG1(나)에서 기록 전의 고립된 결정은 자기 확률 무게를 원천으로 느낀다. 원자 구름
    둘(가우스, 폭 Δx)의 인력을 전개하면 질량중심 진동수는 ω_SN = √(G m/(6√π Δx³))이다(Yang et al. 2013과 같은 꼴).
    Δx는 디바이 모형의 영점 요동 ⟨u_x²⟩ = 3ħ²/(4 m k_B Θ_D)다. 재료는 Si(645 K), W(400 K), Os(500 K)다.
    R1은 감시되는 진동자에 인과–조건부 처방(Helou et al. 2017)을 준다.
    kill: 기록되지 않은 결정 진동자에서 ω_m ≲ ω_SN의 양자 동역학이 표준 양자역학대로 측정되었으면 QG1(나)+R1을
    기각한다. 문헌 판정(2026-09): 제안만 있다(Großardt 2016, Gan 2016, 2023 슈테른–게를라흐, 2025 부양,
    2026 단층 촬영). 이 경우 등록 후보(P44)로만 적는다(등록은 사용자 승인과 v26).

python -B -m examples.physics.rendering.ce_rendering_record_dynamics
"""

from __future__ import annotations

import math

import numpy as np

TOL = 1e-10
SAMPLES = 40
S_GRID = (-1.1, -0.3, 0.45, 1.7)
G, HBAR, KB, AMU = 6.67430e-11, 1.054571817e-34, 1.380649e-23, 1.66053906660e-27
CRYSTALS = {"Si": (28.0855, 645.0), "W": (183.84, 400.0), "Os": (190.23, 500.0)}   # (원자량 u, 디바이 온도 K)


# ------------------------------------------------------------------ RD1 비가역성
def _rand_rho(d: int, rng: np.random.Generator, rank: int | None = None) -> np.ndarray:
    k = d if rank is None else rank
    z = rng.normal(size=(d, k)) + 1j * rng.normal(size=(d, k))
    r = z @ z.conj().T
    return r / np.trace(r).real


def _power(rho: np.ndarray, z: complex) -> np.ndarray:
    w, v = np.linalg.eigh(rho)
    return v @ np.diag(np.exp(z * np.log(w))) @ v.conj().T


def _ptrace_c(y: np.ndarray, da: int, dc: int) -> np.ndarray:
    return np.einsum("acbc->ab", y.reshape(da, dc, da, dc))


def _entropy(rho: np.ndarray) -> float:
    w = np.linalg.eigvalsh(rho)
    w = w[w > 1e-15]
    return float(-(w * np.log(w)).sum())


def _leakage(rho: np.ndarray, da: int, dc: int, rng: np.random.Generator) -> float:
    """모듈러 흐름이 옛 기록 대수 A⊗1 밖으로 새는 정도(힐베르트–슈미트, 상대값)."""
    x = rng.normal(size=(da, da)) + 1j * rng.normal(size=(da, da))
    y0 = np.kron(x, np.eye(dc))
    worst = 0.0
    for s in S_GRID:
        y = _power(rho, 1j * s) @ y0 @ _power(rho, -1j * s)
        inside = np.kron(_ptrace_c(y, da, dc) / dc, np.eye(dc))
        worst = max(worst, float(np.linalg.norm(y - inside) / np.linalg.norm(y0)))
    return worst


def _expectation_defect(rho: np.ndarray, da: int, dc: int, rng: np.random.Generator) -> float:
    """E(Y) = Tr_C[(1⊗ρ_C)Y]⊗1이 상태를 보존하는가: |ω(E(Y)) − ω(Y)| / ‖Y‖."""
    rc = np.einsum("acad->cd", rho.reshape(da, dc, da, dc))
    y = rng.normal(size=(da * dc, da * dc)) + 1j * rng.normal(size=(da * dc, da * dc))
    ey = np.kron(_ptrace_c(np.kron(np.eye(da), rc) @ y, da, dc), np.eye(dc))
    return float(abs(np.trace(rho @ ey) - np.trace(rho @ y)) / np.linalg.norm(y))


def irreversibility(da: int = 3, dc: int = 3, samples: int = SAMPLES, seed: int = 0) -> dict:
    """RD1: 곱 상태(새 기록이 옛 기록과 무관)만 상태 보존 되돌림을 허락한다."""
    rng = np.random.default_rng(seed)
    prod_leak, prod_defect, corr_leak_min, corr_defect_min, pairs = 0.0, 0.0, math.inf, math.inf, []
    for _ in range(samples):
        ra, rc = _rand_rho(da, rng), _rand_rho(dc, rng)
        rho = np.kron(ra, rc)
        prod_leak = max(prod_leak, _leakage(rho, da, dc, rng))
        prod_defect = max(prod_defect, _expectation_defect(rho, da, dc, rng))
        rho = _rand_rho(da * dc, rng)
        mi = _entropy(_ptrace_c(rho, da, dc)) + _entropy(np.einsum("acad->cd", rho.reshape(da, dc, da, dc))) - _entropy(rho)
        leak = _leakage(rho, da, dc, rng)
        pairs.append((mi, leak))
        if mi > 1e-3:
            corr_leak_min = min(corr_leak_min, leak)
            corr_defect_min = min(corr_defect_min, _expectation_defect(rho, da, dc, rng))
    mis, leaks = np.array(pairs).T
    rank_corr = float(np.corrcoef(np.argsort(np.argsort(mis)), np.argsort(np.argsort(leaks)))[0, 1])
    return {"product_leakage": prod_leak, "product_defect": prod_defect, "correlated_min_leakage": corr_leak_min,
            "correlated_min_defect": corr_defect_min, "spearman_MI_vs_leakage": rank_corr,
            "MI_range": (float(mis.min()), float(mis.max())),
            "passed": prod_leak < TOL and prod_defect < TOL and corr_leak_min > 1e-3 and corr_defect_min > 1e-6}


def correlation_scan(da: int = 3, dc: int = 3, seed: int = 3) -> list[dict]:
    """보조(RD1 읽기): ρ(ε) = (1−ε)ρ_A⊗ρ_C + ε ρ_rand. 상관을 켤수록 누출과 상호정보가 함께 0에서 자란다."""
    rng = np.random.default_rng(seed)
    ra, rc, rr = _rand_rho(da, rng), _rand_rho(dc, rng), _rand_rho(da * dc, rng)
    out = []                                                         # 같은 관측량 X로 비교한다(seed + 1)
    for eps in (1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0):
        rho = (1 - eps) * np.kron(ra, rc) + eps * rr
        mi = _entropy(_ptrace_c(rho, da, dc)) + _entropy(np.einsum("acad->cd", rho.reshape(da, dc, da, dc))) - _entropy(rho)
        out.append({"eps": eps, "MI": mi, "leakage": _leakage(rho, da, dc, np.random.default_rng(seed + 1))})
    return out


# ------------------------------------------------------------------ RD4 자기 중력
def zero_point_dx(mass_u: float, theta_d: float) -> float:
    """디바이 모형의 T = 0 영점 요동(한 축): ⟨u_x²⟩ = 3ħ²/(4 m k_B Θ_D)."""
    m = mass_u * AMU
    return math.sqrt(3 * HBAR ** 2 / (4 * m * KB * theta_d))


def omega_sn(mass_u: float, theta_d: float) -> float:
    """두 가우스 구름의 인력 U(X) = −Gm² erf(X/2σ)/X를 X로 전개한 곡률에서 ω_SN² = G m/(6√π σ³)."""
    dx = zero_point_dx(mass_u, theta_d)
    return math.sqrt(G * mass_u * AMU / (6 * math.sqrt(math.pi) * dx ** 3))


def _gauss_pair_curvature_check(sigma: float = 1.0, m: float = 1.0) -> float:
    """검산: U(X) = −m² erf(X/(2σ))/X(G = 1)의 X = 0 곡률이 m²/(6√π σ³)와 같은가."""
    u = lambda x: -m * m * math.erf(x / (2 * sigma)) / x
    xs = np.array([1e-3, 2e-3, 3e-3]) * sigma
    fit = np.polyfit(xs ** 2, np.array([u(x) for x in xs]), 1)          # U ≈ U0 + (k/2) X²
    return float(abs(2 * fit[0] / (m * m / (6 * math.sqrt(math.pi) * sigma ** 3)) - 1))


def self_gravity() -> dict:
    """RD4: 결정 질량중심의 슈뢰딩거–뉴턴 진동수(기록 전, QG1(나))."""
    table = {k: {"dx_pm": zero_point_dx(*v) * 1e12, "omega_SN_per_s": omega_sn(*v)} for k, v in CRYSTALS.items()}
    return {"formula_check": _gauss_pair_curvature_check(), "crystals": table,
            "note": "Δx는 디바이 영점 추정(±수십 %), ω_SN은 물체 전체 질량과 무관"}


def verdict() -> dict:
    ir = irreversibility()
    sg = self_gravity()
    return {"RD1_passed": ir["passed"], "RD4_formula_ok": sg["formula_check"] < 1e-5,
            "RD3_literature": "자발 붕괴 검출 없음(매개변수 없는 Diosi-Penrose는 배제, CE는 붕괴 없음)",
            "RD4_literature": "제안만 있음(2016–2026), 배제 실험 없음 → 미검증, P44 후보"}


def main() -> None:
    print("RD1 irreversibility:", irreversibility())
    print("RD1 correlation scan:", [{k: f"{v:.3g}" for k, v in r.items()} for r in correlation_scan()])
    print("RD4 self-gravity:", self_gravity())
    print("verdict:", verdict())


if __name__ == "__main__":
    main()
