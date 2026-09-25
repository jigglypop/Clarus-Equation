"""눈금 없음 NR — 결합에는 고정된 눈금이 없고, 눈금은 분별마다 변한다. 원장: 43장 §43.100. 예측값을 바꾸지 않는다.

사용자 가설(2026-09-25, §43.99 뒤): “눈금 자체가 없어. 변함. 분별임.” §43.99가 남긴 빈칸(왜 MS-bar인가)에 대한 답이다.
NR: 결합 상수에는 고정된 눈금(스킴)이 없다. 눈금은 분별(기록 과정)이 일어나는 곳마다 정해지고, 눈금을 바꾸는 것은 분별의
척도를 옮기는 것과 같다(한 고리에서 α' = α(1 + kα/π)는 Λ' = Λ e^{2k/b₀}와 같다).
눈금 셋: 장부(MS-bar α_s, MS-bar ŝ²), 기록(Z 하드론 폭의 유효 결합 α_R, 유효각), 다른 분별의 예로 정적 퍼텐셜 α_V(한 고리 변환).
계산 전에 적은 판본과 kill:

NR-0 (눈금 = 척도). 눈금마다 E4 교차 척도 μ*를 구한다. 기록 눈금의 μ*가 장부 눈금의 μ*와 1σ(로그, α_s 오차) 안이면
    “눈금을 바꾸면 척도가 옮겨진다”는 전제가 틀린 것이므로 NR-0 기각.
NR-A (분별이 곧 눈금: 눈금 = 그 사건의 기록). §43.99의 S-b2와 같다. 레지스트리 행(판본 I 행 모음)마다 장부 → 기록 눈금의
    pull을 낸다. 계산 전에 적은 순환 행(MS-bar 값을 보며 찾은 식, §43.18·§43.58·§43.60·§43.93): α_s 세계 평균(E4 자신),
    m_μ/m_τ(렙톤 규칙), m_e/m_μ(Koide + 렙톤 규칙), α_em⁻¹(합규칙), v/M_Pl(계층 식).
    kill: 순환 행이 아닌 행에서 기록 눈금의 |pull| ≥ 3이 새로 생기면 NR-A 기각.
NR-B (눈금이 아예 없음: E4는 1차 관계). 장부와 기록 눈금의 차를 고유 이론 폭으로 둔다.
    (i) E4 지수의 증거: §43.3의 기대 우연 적중 0.029(σ_p = 0.0048)를 창 폭에 비례해 다시 센다. 1 bit 아래면 수치 증거 소멸.
    (ii) 행마다 눈금 폭 / 원래 σ. 3을 넘는 행은 “판정 불가”로 적는다.
    (iii) S-b2 긴장과 FP–렙톤 규칙 긴장(−3.75σ)이 눈금 폭으로 풀리는지.

python -B -m examples.physics.rendering.ce_rendering_ruler
"""

from __future__ import annotations

import math

from scipy.optimize import brentq

from examples.physics.rendering import ce_rendering_balance as BAL
from examples.physics.rendering import ce_rendering_boundary_loop as BLP
from examples.physics.rendering import ce_rendering_distinction as DST
from examples.physics.rendering import ce_rendering_event_scale as ES
from examples.physics.rendering import ce_rendering_one_event as OE
from examples.physics.rendering import ce_rendering_registry as R

S2_LEDGER = DST.SCHEMES["MS-bar"]
S2_RECORD = DST.SCHEMES["effective"]
A1_V = 31 / 12 - 5 * 5 / 18                    # α_V = α(1 + A1_V α/π), n_f = 5 한 고리
CIRCULAR = ("alpha_s world", "m_mu/m_tau", "m_e/m_mu", "alpha_em^-1(M_Z)", "v/M_Pl")
E4_CHANCE, E4_SIGMA_P = 0.029, 0.0048           # §43.3 계수 사전


def alpha_v(alpha_s: float) -> float:
    return alpha_s * (1 + A1_V * alpha_s / math.pi)


RULERS = {"ledger (MS-bar)": (lambda a: a, 0.0),
          "record (alpha_R, effective angle)": (BAL.alpha_record, S2_RECORD[0] - S2_LEDGER[0]),
          "static potential (alpha_V)": (alpha_v, 0.0)}


def crossing(ruler: str, as_mz: float = ES.A_WORLD) -> float:
    conv, ds2 = RULERS[ruler]
    g = lambda lm: ES.s2_run(math.exp(lm)) + ds2 - 4 * conv(ES.alpha_s_run(math.exp(lm), as_mz)) ** (4 / 3)
    return math.exp(brentq(g, math.log(20.0), math.log(1000.0)))


def ruler_scales() -> dict:
    """NR-0: 눈금마다 E4가 성립하는 척도(ŝ²는 §43.18 흐름, 기록 눈금은 유효각 차이만큼 옮김)."""
    e = OE.event_scale()
    out = {"sigma_ln": e["sigma_ln"]}
    for name in RULERS:
        mu = crossing(name)
        out[name] = {"mu_star": mu, "pull_ln_vs_ledger": math.log(mu / e["mu_star"]) / e["sigma_ln"],
                     "pull_ln_vs_MZ": math.log(mu / ES.M_Z) / e["sigma_ln"]}
    rec = out["record (alpha_R, effective angle)"]["pull_ln_vs_ledger"]
    out["killed"] = abs(rec) < 1
    return out


def _coin(s2: float) -> tuple[float, float]:
    a = (s2[0] / 4) ** 0.75
    return a, 0.75 * a * s2[1] / s2[0]


def _row_pull(row, a: float, sa: float, obs: float, up: float, dn: float) -> tuple[float, float, float]:
    pred = row.f(R.core(a))
    st = abs(row.f(R.core(a + sa)) - row.f(R.core(a - sa))) / 2
    sig = math.hypot(up if pred >= obs else dn, st)
    return pred, sig, (pred - obs) / sig


def row_audit() -> dict:
    """NR-A·NR-B(ii): 행마다 장부 → 기록 눈금의 pull과, 눈금이 바꾸는 잔차 / 원래 σ."""
    (a_l, sa_l), (a_r, sa_r) = _coin(S2_LEDGER), _coin(S2_RECORD)
    out = {}
    for row in R.rows("SK"):
        if "I" not in row.variants:
            continue
        obs_r, up_r, dn_r = row.obs, row.sig_up, row.sig_dn
        if row.key == "alpha_s world":            # 자료도 기록 눈금으로 옮긴다
            k = BAL.d_alpha_record(row.obs)
            obs_r, up_r, dn_r = BAL.alpha_record(row.obs), row.sig_up * k, row.sig_dn * k
        p_l, s_l, pull_l = _row_pull(row, a_l, sa_l, row.obs, row.sig_up, row.sig_dn)
        p_r, _, pull_r = _row_pull(row, a_r, sa_r, obs_r, up_r, dn_r)
        width = abs((p_r - obs_r) - (p_l - row.obs))
        out[row.key] = {"pull_ledger": pull_l, "pull_record": pull_r, "width_over_sigma": width / s_l,
                        "circular": row.key in CIRCULAR}
    out["bao_chi2"] = {"ledger": R.bao_chi2(R.core(a_l)["Om"]), "record": R.bao_chi2(R.core(a_r)["Om"])}
    return out


def e4_evidence() -> dict:
    """NR-B(i): 눈금 폭을 넣은 E4 지수의 증거(§43.3 방식, 창 폭에 비례)."""
    p_l = math.log(S2_LEDGER[0] / 4) / math.log(ES.A_WORLD)
    p_r = math.log(S2_RECORD[0] / 4) / math.log(BAL.alpha_record(ES.A_WORLD))
    width = abs(p_r - p_l)
    sig = math.hypot(E4_SIGMA_P, width)
    chance = min(1.0, E4_CHANCE * sig / E4_SIGMA_P)
    return {"p_ledger": p_l, "p_record": p_r, "ruler_width": width, "sigma_total": sig,
            "pull_4_3_ledger": (p_l - 4 / 3) / E4_SIGMA_P, "pull_4_3_record_total": (p_r - 4 / 3) / sig,
            "bits_before": -math.log2(E4_CHANCE), "bits_after": -math.log2(chance),
            "killed": -math.log2(chance) < 1}


def tensions() -> dict:
    """NR-B(iii): 눈금 폭이 푸는 긴장과 풀지 못하는 긴장."""
    a_r = _coin(S2_RECORD)[0]
    obs = BAL.alpha_record(ES.A_WORLD)
    err = ES.A_WORLD_ERR * BAL.d_alpha_record(ES.A_WORLD)
    width = abs(BAL.alpha_record(ES.A_WORLD) - ES.A_WORLD)
    lep = BLP.alpha_from_lepton_rule()
    return {"S-b2_pull_without_width": (a_r - obs) / err, "S-b2_pull_with_width": (a_r - obs) / math.hypot(err, width),
            "FP_vs_lepton": lep["FP_vs_lepton"], "E4_vs_lepton": lep["E4_vs_lepton"],
            "FP_lepton_uses_ruler": False}       # 둘 다 동전 값끼리의 비교, 눈금이 들어가지 않는다


def verdict() -> dict:
    s, rows, ev = ruler_scales(), row_audit(), e4_evidence()
    new_bad = [k for k, v in rows.items() if isinstance(v, dict) and "pull_record" in v
               and abs(v["pull_record"]) >= 3 and abs(v["pull_ledger"]) < 3]
    return {"NR0_killed": s["killed"],
            "NRA_new_3sigma_rows": new_bad,
            "NRA_killed": any(k not in CIRCULAR for k in new_bad),
            "NRB_bits_after": ev["bits_after"], "NRB_killed": ev["killed"],
            "NRB_toothless_rows": [k for k, v in rows.items() if isinstance(v, dict)
                                   and v.get("width_over_sigma", 0) > 3]}


def main() -> None:
    print("NR-0 ruler scales:", BAL._r(ruler_scales()))
    for k, v in row_audit().items():
        print("row", k, BAL._r(v, 3))
    print("NR-B(i) E4 evidence:", BAL._r(e4_evidence()))
    print("NR-B(iii) tensions:", BAL._r(tensions()))
    print("verdict:", BAL._r(verdict()))


if __name__ == "__main__":
    main()
