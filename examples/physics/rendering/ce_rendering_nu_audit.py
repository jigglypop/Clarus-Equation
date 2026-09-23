"""중성미자 질량식 감사 — bit 장부 밖에 있던 가장 큰 경험식. 원장: 43장 §43.46. 예측값을 바꾸지 않는다.

식(§43.19): m_{ν_l} = P · m_l^{5/8} m_τ^{3/8},  P = δ⁴(1 − α_s/π) / [(16π²)² 32π³ (1 + R)].
비율은 지수 5/8 하나, 크기는 계수 P 하나가 정한다. 계산 전에 적은 감사:

A1 지수. Δm²₂₁/Δm²₃₁에서 p를 역산해 5/8과 비교, 8분의 k 사전의 우연 적중 확률.
A2 크기. 미리 정한 계수 가족 δ^a (1 − α_s/π)^b (16π²)^{−c} (Nπ^k)^{−1} (1 + R)^{−e},
   a ∈ 2..6, b ∈ {0,1}, c ∈ 1..3, N ∈ {1,2,4,…,64}, k ∈ 0..4, e ∈ {0,1} (2100개)에서 m₃의 창에 드는 수.
A3 변별력. Σm_ν와 최소 정상 순서, m_β, m_ββ 범위(마요라나 위상 자유).
A4 최신 자료. JUNO Δm²₂₁, DESI DR2 + CMB Σm_ν < 64.2 meV(95%).
kill. 지수나 크기가 자료와 3σ 넘게 어긋나면 기각.

python -B -m examples.physics.rendering.ce_rendering_nu_audit
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

from scipy.optimize import brentq

from examples.physics.rendering import ce_rendering_neutrino as NU
from examples.physics.rendering import ce_rendering_registry as R

DM21 = {"NuFIT 6.0": (7.49e-5, 0.19e-5), "JUNO 2025": (7.50e-5, 0.12e-5)}
DM31 = (2.513e-3, 0.020e-3)
DESI_DR2_SUM_95 = 64.2        # meV, DESI DR2 BAO + CMB, arXiv:2503.14738
MEV_TO_MEVNU = 1e9            # MeV → meV


def _ratio(p: float) -> float:
    e, mu = (R.M_E / R.M_TAU) ** (2 * p), (R.M_MU / R.M_TAU) ** (2 * p)
    return (mu - e) / (1 - e)


def exponent_fit(key: str = "NuFIT 6.0") -> dict:
    """A1: (m_μ/m_τ)^{2p}–꼴 비율로 Δm²₂₁/Δm²₃₁을 맞추는 p와 오차."""
    d21, s21 = DM21[key]
    r = d21 / DM31[0]
    sr = r * math.hypot(s21 / d21, DM31[1] / DM31[0])
    p = brentq(lambda x: _ratio(x) - r, 0.3, 0.9)
    lo = brentq(lambda x: _ratio(x) - (r + sr), 0.3, 0.9)
    hi = brentq(lambda x: _ratio(x) - (r - sr), 0.3, 0.9)
    sig = abs(hi - lo) / 2
    eighths = [Fraction(k, 8) for k in range(1, 8)]
    small = sorted({Fraction(n, d) for d in range(2, 13) for n in range(1, d)})
    in_window = lambda fs: [str(f) for f in fs if abs(float(f) - p) <= sig]
    return {"p": p, "sigma": sig, "pull_5_8": (0.625 - p) / sig, "eighths_in_1sigma": in_window(eighths),
            "chance_eighth": min(1.0, 2 * sig * 8), "fractions_den_le_12_in_1sigma": in_window(small)}


def prefactor_target() -> dict:
    """A2 목표: Δm²₃₁에서 역산한 P(m₁은 5/8 지수로 포함)."""
    e = (R.M_E / R.M_TAU) ** (5 / 4)
    m3 = math.sqrt(DM31[0] / (1 - e)) * 1e3          # meV
    p = m3 / (R.M_TAU * MEV_TO_MEVNU)
    return {"m3_meV": m3, "P": p, "rel_sigma": 0.5 * DM31[1] / DM31[0]}


def prefactor_family(c: dict):
    a, d, big_r = c["a"], c["d"], c["a"] * c["D"]
    for pa, pb, pc, n, k, pe in itertools.product(range(2, 7), (0, 1), (1, 2, 3), (1, 2, 4, 8, 16, 32, 64),
                                                  range(5), (0, 1)):
        val = d ** pa * (1 - a / math.pi) ** pb / ((16 * math.pi ** 2) ** pc * n * math.pi ** k * (1 + big_r) ** pe)
        yield (pa, pb, pc, n, k, pe), val


def scale_look_elsewhere(c: dict) -> dict:
    t = prefactor_target()
    members = list(prefactor_family(c))
    logs = [math.log10(v) for _, v in members]
    hit = lambda nsig: [m for m, v in members if abs(v / t["P"] - 1) <= nsig * t["rel_sigma"]]
    h1, h3 = hit(1), hit(3)
    actual = (4, 1, 2, 32, 3, 1)
    return {"family": len(members), "log10_span": (min(logs), max(logs)), "hits_1sigma": len(h1),
            "hits_3sigma": len(h3), "actual_in_family": any(m == actual for m, _ in members),
            "actual_rel_dev": dict(members)[actual] / t["P"] - 1,
            "selection_bits": math.log2(len(members) / max(len(h1), 1)), "hits_1sigma_list": h1}


def distinctiveness(c: dict) -> dict:
    """A3: Σ 대 최소 정상 순서, m_β, m_ββ 범위(S2 각, 마요라나 위상 자유)."""
    m = NU.neutrino_masses_mev(c)
    s12, s13 = R.pmns_s2(c, 2), R.pmns_s2(c, 1)
    ue = ((1 - s12) * (1 - s13), s12 * (1 - s13), s13)
    terms = [u * mi for u, mi in zip(ue, m)]
    sum_min_no = math.sqrt(DM21["NuFIT 6.0"][0]) * 1e3 + math.sqrt(DM31[0]) * 1e3
    return {"masses_meV": m, "sum_meV": sum(m), "sum_min_NO_meV": sum_min_no, "sum_minus_min": sum(m) - sum_min_no,
            "DESI_DR2_95_meV": DESI_DR2_SUM_95, "m_beta_meV": math.sqrt(sum(u * mi ** 2 for u, mi in zip(ue, m))),
            "m_bb_range_meV": (max(0.0, max(terms) - (sum(terms) - max(terms))), sum(terms))}


def juno_outlook(c: dict) -> dict:
    """A4: 예측 Δm²₂₁과 JUNO 현재·최종 정밀도(0.3%, 중심값 유지 가정)의 pull."""
    pred = NU.splittings_ev2(c)[0]
    v, s = DM21["JUNO 2025"]
    return {"pred": pred, "pull_now": (pred - v) / s, "pull_final_0.3pct_if_central_holds": (pred - v) / (0.003 * v),
            "rel_gap": pred / v - 1}


def main() -> None:
    c = R.core(R.calibrated_alpha_s()[0])
    for key in DM21:
        print(f"A1 exponent ({key}):", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in exponent_fit(key).items()})
    print("A2 target:", {k: round(v, 6) for k, v in prefactor_target().items()})
    s = scale_look_elsewhere(c)
    print("A2 family:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k != "hits_1sigma_list"})
    print("   1-sigma hits (a,b,c,N,k,e):", s["hits_1sigma_list"])
    print("A3:", {k: (tuple(round(x, 3) for x in v) if isinstance(v, tuple) else round(v, 3)) for k, v in distinctiveness(c).items()})
    print("A4 JUNO:", {k: round(v, 4) if abs(v) > 1e-3 else v for k, v in juno_outlook(c).items()})


if __name__ == "__main__":
    main()
