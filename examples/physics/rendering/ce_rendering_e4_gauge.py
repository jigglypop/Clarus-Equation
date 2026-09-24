"""E4의 재구성 — 동전은 모든 게이지 축에 같고, 사전은 각 게이지 공간에서 읽는다. 원장: 43장 §43.88. 예측값을 바꾸지 않는다.

§43.82의 보른 배가는 조건 (가) m² = 2m과 (다) m + 2 = 2m이 같은 방정식이었고, V₃ = ℂ³에서 정의되지 않는 A₄를 썼다
(§43.87 (라)). 이 모듈은 순환 없이 E4를 다시 세운다. 계산 전에 적은 전제·주장·kill:

E1′ 렌더링 동전은 모든 게이지 축에 같다: R = a·I₅, V = V₃ ⊕ V₂(C6). §43.57(BC: 게이지 축의 동전이 같이 기운다)과
    §43.62(한 동전)가 이미 쓰는 가정을 E1(R은 V₃에만)의 자리에 명시한 것이다.
D   (§43.83) 게이지 인자 G_k가 작용하는 공간 V_k에서 결합 = det(R|V_k), 섞임 진폭 = a∂_a det(R|V_k) = tr·det.
CC  (§43.73) 가두는 힘(SU(3), V₃)은 결합으로, 깨지는 힘(SU(2), V₂)은 섞임으로 읽힌다.
주장 α_s = a³, sin θ_W = 2a²(m = dim V₂ = 2). E1′ 아래 A_m은 m ≤ 5에서 정의되므로 보른 배가 A_m² = A_{2m}이 뜻을 갖는다.
kill (k1) E1′이 등록된 값(V₃ 안의 A₁–A₃, E4)을 하나라도 바꾸면 기각.
     (k2) 2m ≤ 5에서 보른 일관성의 자명하지 않은 해가 2 말고도 있으면 기각.
     (k3) CC 없이 D를 약력 결합에 쓰면 관측과 모순이어야 한다(CC가 필요한 조건인지). 모순이 없으면 CC는 군더더기로 기록.
증거의 지위: D와 CC는 E4를 안 뒤 세운 번역 규칙이다. 논리 구조(순환·정의역)를 고치는 것이지 bit를 줄이지 않는다.

python -B -m examples.physics.rendering.ce_rendering_e4_gauge
"""

from __future__ import annotations

import numpy as np

from examples.physics.rendering import ce_rendering_registry as R

DIM = {"V3": 3, "V2": 2}
GAUGE = {"SU(3)": ("V3", "coupling"), "SU(2)": ("V2", "mixing")}      # CC
AS_WORLD = (0.1180, 0.0009)
ALPHA2_OBS = 0.0338                                                   # §43.70의 관측 α₂(M_Z)
SEED = 20260924


def _random_subspace(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    z = rng.normal(size=(n, m)) + 1j * rng.normal(size=(n, m))
    q, _ = np.linalg.qr(z)
    return q


def amplitude(a: float, basis: np.ndarray, n: int = 5) -> float:
    """E3를 V = ℂⁿ에 편 것: A(W) = tr P_W · det(B† R B), R = a·Iₙ."""
    Rn = a * np.eye(n)
    P = basis @ basis.conj().T
    return float(np.trace(P).real * np.linalg.det(basis.conj().T @ Rn @ basis).real)


def amplitude_scan(a: float = 0.49, trials: int = 50) -> dict:
    rng = np.random.default_rng(SEED)
    worst = {}
    for m in range(1, 6):
        worst[m] = max(abs(amplitude(a, _random_subspace(5, m, rng)) - m * a ** m) for _ in range(trials))
    return worst


def registered_values_unchanged(a: float = 0.49, trials: int = 50) -> float:
    """(k1) V₃ ⊂ V에 놓인 부분공간의 진폭은 E1(ℂ³)과 E1′(ℂ⁵)에서 같다."""
    rng = np.random.default_rng(SEED + 1)
    worst = 0.0
    for m in range(1, 4):
        for _ in range(trials):
            b3 = _random_subspace(3, m, rng)
            b5 = np.vstack([b3, np.zeros((2, m))])
            worst = max(worst, abs(R.rendering_amplitude(a ** 3, b3) - amplitude(a, b5)))
    return worst


def dictionary(a: float) -> dict:
    """D를 각 게이지 공간에서: 결합 det = a^k, 섞임 a∂_a det = k a^k. CC가 어느 쪽을 읽는지 고른다."""
    out = {}
    for g, (space, reading) in GAUGE.items():
        k = DIM[space]
        out[g] = {"space": space, "det": a ** k, "a_da_det": k * a ** k, "reading": reading,
                  "value": a ** k if reading == "coupling" else k * a ** k}
    return out


def e4_from_dictionary() -> dict:
    """ŝ²(M_Z)에서 a를 정하고(sin θ_W = 2a²) 결합 쪽 α_s = a³을 세계 평균과 비교한다."""
    a = (R.SZ2 / 4.0) ** 0.25
    d = dictionary(a)
    alpha_s = d["SU(3)"]["value"]
    return {"a": a, "alpha_s": alpha_s, "sin_theta_W": d["SU(2)"]["value"],
            "s2_check": d["SU(2)"]["value"] ** 2, "pull_world": (alpha_s - AS_WORLD[0]) / AS_WORLD[1]}


def born_consistency(n: int = 5, a: float = 0.49, trials: int = 20) -> dict:
    """(k2) A_m² = A_{2m}: 2m ≤ n인 m만 정의된다. 기호 검사와 무작위 부분공간 검사."""
    rng = np.random.default_rng(SEED + 2)
    defined = [m for m in range(1, n + 1) if 2 * m <= n]
    symbolic = [m for m in defined if m * m == 2 * m]
    numeric = []
    for m in defined:
        dev = max(abs(amplitude(a, _random_subspace(n, m, rng)) ** 2 - amplitude(a, _random_subspace(n, 2 * m, rng)))
                  for _ in range(trials))
        if dev < 1e-10:
            numeric.append(m)
    return {"defined_m": defined, "symbolic": symbolic, "numeric": numeric,
            "undefined_m": [m for m in range(1, n + 1) if 2 * m > n]}


def cc_necessity() -> dict:
    """(k3) CC 없이 D의 결합 읽기를 SU(2)에도 쓰면 α₂ = a²."""
    a = (R.SZ2 / 4.0) ** 0.25
    return {"alpha2_if_coupling": a * a, "alpha2_obs": ALPHA2_OBS, "ratio": a * a / ALPHA2_OBS,
            "contradiction": a * a / ALPHA2_OBS > 2.0}


def singlet_weights_under_e1prime(a: float = 0.49) -> dict:
    """§43.78의 단일항 경로: R이 다섯 축 전부에 작용하면 단일항 무게는 1, a², a³, a⁵로 유일하지 않다."""
    return {"Λ0": 1.0, "Λ2 V2": a ** 2, "Λ3 V3": a ** 3, "Λ5": a ** 5}


def verdict() -> dict:
    k1 = registered_values_unchanged() < 1e-12
    b = born_consistency()
    k2 = b["symbolic"] == [2] and b["numeric"] == [2]
    k3 = cc_necessity()["contradiction"]
    return {"k1_values_unchanged": k1, "k2_born_unique_m2": k2, "k3_cc_needed": k3, "killed": not (k1 and k2)}


def main() -> None:
    print("A_m on random subspaces of C^5, max |A - m a^m|:", {m: f"{v:.1e}" for m, v in amplitude_scan().items()})
    print("(k1) V3 amplitudes unchanged under E1':", f"{registered_values_unchanged():.1e}")
    print("E4 from dictionary:", {k: round(v, 6) for k, v in e4_from_dictionary().items()})
    print("(k2) Born consistency:", born_consistency())
    print("(k3) CC necessity:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in cc_necessity().items()})
    print("singlet weights under E1':", {k: round(v, 4) for k, v in singlet_weights_under_e1prime().items()})
    print("verdict:", verdict())


if __name__ == "__main__":
    main()
