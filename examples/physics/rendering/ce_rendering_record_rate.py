"""기록률의 상한 — 떨림 바다에서 첫 기록은 얼마나 자주 일어날 수 있는가(란다우어 예산), 그리고 무엇이 강제되는가.
원장: 43장 §43.109.

사용자(2026-09-26): "다음"(직전 제안: 남은 빈칸 = 기록률, §43.105 RD3 "보편 속도 없음").

계산 전 고정(2026-09-26):
전제: Q(§43.108: QG1(나)·R1). RL(기록 1비트는 엔트로피 ln 2 이상을 버린다). 바다는 드 시터 평형(정적 조각의 KMS 상태, §43.104)이라
  엔트로피가 이미 가득 차 있다. 그래서 새 기록이 버릴 자리는 지평 엔트로피가 한 e-fold에 느는 만큼뿐이다:
  예산 B = dS/dN = 2/P_ζ nat(허블 부피당, §43.108 항등식).
정의: f = 허블 부피당 e-fold당 첫 기록 수. 기록 하나는 R1로 제 빛원뿔을 확정하고, 빛원뿔은 결국 기록 시각의 허블 부피 하나를
  덮는다. 그래서 확정 안 된 몫(공변 부피)은 e^{−∫f dN}, 물리 부피는 e^{∫(3−f)dN}이다(Guth–Weinberg 꼴). 남은 바다의 차원은 d = 3 − f.
상한: f ≤ f_max = B/c, c = 기록 하나의 비용(최소 ln 2). 민감도: c ∈ {ln 2, 1 nat, 3 ln 2}.
주장:
  T1 지속 띠: f_max < 3 ⟺ P_ζ > 2/(3c)(c = ln 2이면 0.962). F1 경계(P_ζ = 1)가 이 띠 안이면, 어떤 기록률에서도 경계의 바다는
     남는다(d ≥ 3 − 2/c). c가 클수록 띠는 넓어진다.
  T2 우리 영역의 첫 사건은 기록률과 무관하게 하나(과거 빛원뿔 안의 가장 이른 기록)다. 기록률은 뒤에 합류하는 기록의 수, 곧
     지평 너머의 배치만 바꾼다. 우리 영역은 첫 기록 때 그 빛원뿔의 e^{−(N_q − N(a0H0))}배다.
  T3 곡률 연결: 우리 영역이 급팽창 끝 N_w e-fold 안까지 미확정으로 남을 확률은 e^{−∫f dN}. 곡률이 관측 창(10⁻⁵ 위)에 들려면
     평균 기록률이 대략 1/(N_q − N_w) 이하여야 한다.
  T4 [가설] 최대 기록(f = f_max, c = ln 2): 바다 물리 부피가 최대가 되는 ΔN, 그 크기, 처음 부피로 돌아오는 ΔN.
     해석 근사(P ≈ (N/N_q)²): ΔN_max ≈ 0.0193 N_q, ln V_max ≈ 0.00112 N_q, ΔN_0 ≈ 0.0382 N_q.
kill: K1 P_ζ 문턱(c = ln 2) ≥ 1이면 T1 기각(경계에서 기록이 팽창을 따라잡을 수 있음).
  K2 T4 수치가 해석 근사와 5% 넘게 다르면 근사 한계로 보고한다.
  K3(관측, 미래) Ω_k > 0이 3σ로 확정되면 Q에서 f ≲ 10⁻⁶이어야 한다(CE는 이 작은 기록률을 설명해야 함).
     Ω_k < 0이면 Q의 열린 거품이 기각된다(§43.108).
예상(계산 전, 해석): K1 통과(2/(3 ln 2) = 0.962 < 1)는 산술로 이미 안다. 이 절은 위험한 예측이 아니라 구조 정리다.
선행: Arkani-Hamed+2007(영원하지 않은 급팽창에서 지평 넓이는 e-fold마다 플랑크 단위 이상 는다)이 같은 계열의 기준이다.

python -B -m examples.physics.rendering.ce_rendering_record_rate
"""

from __future__ import annotations

import math

from examples.physics.rendering import ce_rendering_pocket_boundary as PKB

COSTS = {"bit": math.log(2), "nat": 1.0, "3bit": 3 * math.log(2)}


def budget(p_zeta: float) -> float:
    """지평 엔트로피 증가 dS/dN = 2/P_ζ(nat, 허블 부피당 e-fold당)."""
    return 2 / p_zeta


def persistence_band() -> dict:
    """T1: f_max = B/c < 3 ⟺ P_ζ > 2/(3c). 경계(P_ζ = 1)가 띠 안인지, 그때 바다 차원의 하한."""
    m = PKB._model()
    n_q = m["n_of"](m["x_at"](1.0))
    out = {}
    for k, c in COSTS.items():
        p_c = 2 / (3 * c)
        n_c = m["n_of"](m["x_at"](p_c))
        out[k] = {"cost_nat": c, "P_threshold": p_c, "N_threshold": n_c, "band_below_boundary_efolds": n_q - n_c,
                  "boundary_inside": p_c < 1.0, "f_max_at_boundary": budget(1.0) / c,
                  "min_dim_at_boundary": 3 - budget(1.0) / c, "f_max_at_pivot": budget(m["a_s"]) / c}
    return out


def curvature_link() -> dict:
    """T2·T3: 우리 영역과 첫 거품의 크기 비, 곡률이 보이려면 필요한 기록률."""
    m = PKB._model()
    qb = PKB.qg1_bubble()
    n_q, n_w, n_k = qb["N_q"], qb["start_needed_for_window"], qb["start_needed_for_killed_curvature"]
    return {"N_q": n_q, "log10_our_region_over_bubble": -(n_q - m["n_hub"]) / math.log(10),
            "f_for_window": 1 / (n_q - n_w), "f_for_killed_curvature": 1 / (n_q - n_k),
            "f_max_boundary_bit": budget(1.0) / math.log(2)}


def maximal_recording() -> dict:
    """T4 [가설]: f = B/ln 2. 경계부터 쌓인 바다 물리 부피 ln V(N) = ∫_N^{N_q}(3 − f)dN′."""
    from scipy.integrate import quad
    from scipy.optimize import brentq
    m = PKB._model()
    x_q = m["x_at"](1.0)
    n_q = m["n_of"](x_q)
    x_of_n = lambda n: brentq(lambda y: m["n_of"](y) - n, m["x_end"] * (1 + 1e-12), x_q * 1.01)
    g = lambda n: 3 - budget(m["p_of"](x_of_n(n))) / math.log(2)
    lnv = lambda n: quad(g, n, n_q, limit=200, epsrel=1e-10)[0]
    n_max = m["n_of"](m["x_at"](2 / (3 * math.log(2))))                            # f = 3인 곳
    n_zero = brentq(lnv, 0.5 * n_max, n_max * (1 - 1e-9))
    a = 2 / math.log(2)
    u_max, u_zero = 1 - math.sqrt(a / 3), 1 - a / 3                                 # 해석 근사
    approx = {"dN_max": u_max * n_q, "lnV_max": n_q * u_max * (3 - a / (1 - u_max)), "dN_zero": u_zero * n_q}
    out = {"N_q": n_q, "dN_max": n_q - n_max, "lnV_max": lnv(n_max), "dN_zero": n_q - n_zero, "approx": approx}
    out["max_rel_dev"] = max(abs(out[k] / approx[k] - 1) for k in approx)
    return out


def verdict() -> dict:
    pb, mr = persistence_band(), maximal_recording()
    return {"K1_boundary_in_band_pass": all(v["boundary_inside"] for v in pb.values()),
            "K2_approx_within_5pct": mr["max_rel_dev"] < 0.05}


def main() -> None:
    for k, v in persistence_band().items():
        print(k, {a: f"{b:.5g}" if isinstance(b, float) else b for a, b in v.items()})
    print("curvature_link", curvature_link())
    print("maximal_recording", maximal_recording())
    print("verdict", verdict())


if __name__ == "__main__":
    main()
