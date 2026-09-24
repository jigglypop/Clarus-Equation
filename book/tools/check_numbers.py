"""book 본문 수치 검사.

book 본문에 적은 핵심 수치를 저장소 모듈과 사전 등록 v24에서 다시 계산해 대조한다.
본문 값은 문자열로 적고, 계산값이 그 문자열의 마지막 자리 반올림 범위(±½단위) 안에 있어야 통과한다.

실행(저장소 루트): python -B book/tools/check_numbers.py
장이 늘 때마다 CHECKS에 항목을 더한다. 항목의 셋째 칸은 그 수치가 처음 나오는 장.절이다.
"""
from __future__ import annotations

import json
import math
import sys
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from examples.physics.rendering import ce_rendering_registry as R  # noqa: E402

V24_PATH = ROOT / "experiments" / "preregistration" / "rendering_predictions_v24.json"
M_Z = 91.1876


@lru_cache(maxsize=None)
def core() -> dict:
    return R.core(R.calibrated_alpha_s()[0])


@lru_cache(maxsize=None)
def v24() -> dict:
    data = json.loads(V24_PATH.read_text(encoding="utf-8"))
    return {p["id"]: p for p in data["predictions"]}


def pred(pid: str, key: str | None = None) -> float:
    value = v24()[pid]["value"]
    return float(value[key] if key is not None else value)


@lru_cache(maxsize=None)
def ckm() -> tuple[float, float, float]:
    return R.ckm_triangle(core()["a"])


@lru_cache(maxsize=None)
def words() -> dict:
    return R.flavour_words(core())


@lru_cache(maxsize=None)
def joint_rmse() -> tuple[float, float]:
    from examples.physics.rendering import ce_rendering_gradient as GD

    return GD.joint_v("G1m"), GD.joint_rows("G1m", "full")


@lru_cache(maxsize=None)
def fixed_point() -> float:
    from examples.physics.rendering import ce_rendering_single_self as SSF

    return SSF.uniqueness()["root"]


@lru_cache(maxsize=None)
def mu_star() -> float:
    from examples.physics.rendering import ce_rendering_one_event as OE

    return math.exp(OE.crossing())


@lru_cache(maxsize=None)
def nu_masses() -> tuple[float, float, float]:
    from examples.physics.rendering import ce_rendering_neutrino as NU

    return NU.neutrino_masses_mev(core())


def w_boundary() -> float:
    return 1.0 + core()["d"] / (2.0 * math.pi)


def mod(name: str):
    """렌더링 모듈을 이름(ce_rendering_ 뒤)으로 불러온다."""
    import importlib

    return importlib.import_module(f"examples.physics.rendering.ce_rendering_{name}")


@lru_cache(maxsize=None)
def bmv() -> dict:
    return mod("probability_weight").bmv()


@lru_cache(maxsize=None)
def horizon() -> dict:
    return mod("light_limit").rendering_horizon(core())


@lru_cache(maxsize=None)
def ledger(rows_from: str) -> dict:
    return mod("ledger").compare(rows_from)


@lru_cache(maxsize=None)
def ledger_blocks() -> dict:
    return mod("ledger").block_scores("IV")


@lru_cache(maxsize=None)
def mdl() -> dict:
    return mod("ledger").mdl_blocks("IV")


@lru_cache(maxsize=None)
def rescore_2026() -> dict:
    return mod("open_checks").rescore_2026()


@lru_cache(maxsize=None)
def pantheon() -> dict:
    return mod("sn_holdout").score()


@lru_cache(maxsize=None)
def clocks() -> dict:
    return mod("modular").clocks()["rows"]


def drift_ratio(weight: str) -> float:
    """BAO 판독 인자의 기울기 비 s(2.33)/s(0.3). weight: 'today' 또는 'record'."""
    import numpy as np

    om = core()["Om"]
    z = np.array([0.3, 2.33])
    half = mod("gradient")._psi(z, om) / 2
    w = om if weight == "today" else om * (1 + z) ** 3 / (om * (1 + z) ** 3 + 1 - om)
    s = 1 - w * (1 - np.cos(half))
    return float(s[1] / s[0])


def live_independent() -> int:
    ids = set(v24())
    not_independent = {"P20", "P33", "P38"}
    killed = {pid for pid, p in v24().items() if any("KILLED" in str(v) for k, v in p.items() if k.startswith("status"))}
    return len(ids - not_independent - killed)


def omega_parts() -> tuple[float, float, float]:
    c = core()
    om_dm = (1.0 - c["q"]) * c["a"] * c["D"] / c["F"]
    om_l = (1.0 - c["q"]) / c["F"]
    return c["q"], om_dm, om_l


# (이름, 본문에 쓴 값, 처음 나오는 장.절, 계산 함수)
CHECKS: list[tuple[str, str, str, callable]] = [
    # --- 01장: 코어
    ("alpha_s (s_Z^2 교정)", "0.117916", "01.04", lambda: core()["a"]),
    ("a = alpha_s^(1/3)", "0.49037", "01.04", lambda: core()["a"] ** (1 / 3)),
    ("A_1", "0.4904", "01.04", lambda: R.rendering_amplitude_closed(core()["a"], 1)),
    ("A_2 = sin theta_W", "0.4809", "01.04", lambda: R.rendering_amplitude_closed(core()["a"], 2)),
    ("A_3 = 3 alpha_s", "0.3537", "01.04", lambda: R.rendering_amplitude_closed(core()["a"], 3)),
    ("s^2 = 4 alpha_s^(4/3)", "0.23129", "01.04", lambda: core()["s2"]),
    ("delta", "0.177795", "01.04", lambda: core()["d"]),
    ("D", "3.177795", "01.04", lambda: core()["D"]),
    ("q = Omega_b", "0.048645", "01.04", lambda: core()["q"]),
    ("F", "1.374713", "01.04", lambda: core()["F"]),
    ("alpha_s D", "0.374713", "01.04", lambda: core()["a"] * core()["D"]),
    ("Omega_m", "0.30796", "01.04", lambda: core()["Om"]),
    ("Omega_DM", "0.25932", "01.04", lambda: omega_parts()[1]),
    ("Omega_Lambda", "0.69204", "01.04", lambda: omega_parts()[2]),
    ("w = 1 + delta/2pi", "1.028297", "01.04", w_boundary),
    ("N_e = 18 D", "57.2003", "01.04", lambda: core()["Ne"]),
    ("FP r*", "0.490352", "01.04", fixed_point),
    ("FP alpha_s (P34)", "0.117903", "01.04", lambda: pred("P34", "alpha_s")),
    ("FP s^2 (P34)", "0.231256", "01.04", lambda: pred("P34", "s2")),
    ("mu* (OE)", "91.6", "01.03", mu_star),
    # --- 01장: 입자
    ("|V_us|", "0.22493", "01.05", lambda: words()["V_us"]),
    ("|V_cb|", "0.041637", "01.05", lambda: words()["V_cb"]),
    ("|V_ub| (P06)", "0.0036759", "01.05", lambda: ckm()[0]),
    ("delta_CKM [rad]", "1.1787", "01.05", lambda: ckm()[1]),
    ("J_CKM", "3.097e-5", "01.05", lambda: ckm()[2]),
    ("m_mu/m_tau", "0.059459", "01.05", lambda: words()["m_mu/m_tau"]),
    ("m_e/m_mu (Koide)", "0.0048374", "01.05", lambda: R.koide_me_over_mmu(words()["m_mu/m_tau"])),
    ("s13^2 (P04)", "0.0222244", "01.05", lambda: R.pmns_s2(core(), 1)),
    ("s12^2 (P03)", "0.318517", "01.05", lambda: R.pmns_s2(core(), 2)),
    ("s23^2 (P01)", "0.455551", "01.05", lambda: R.pmns_s2(core(), 3)),
    ("delta_PMNS [deg] (P02)", "258.76", "01.05", lambda: R.delta_pmns_tm1(core())),
    ("sin^2 2theta_23 (P30)", "0.992097", "01.05", lambda: 1 - core()["d"] ** 2 / 4),
    ("alpha_em^-1(M_Z)", "127.933", "01.05", lambda: R.alpha_em_inv(core())),
    ("v/M_Pl", "2.0168e-17", "01.05", lambda: R.v_over_mpl(core())),
    ("M_H = M_Z F", "125.36", "01.05", lambda: M_Z * core()["F"]),
    ("m_t = M_Z F^2 (P39)", "172.33", "01.05", lambda: M_Z * core()["F"] ** 2),
    ("M_Z/M_H = 1/F", "0.72742", "01.05", lambda: 1 / core()["F"]),
    ("P36 (동결값)", "0.2570745", "01.05", lambda: pred("P36")),
    ("(1 + delta/2pi)/4 (코어 재계산)", "0.257074", "01.05", lambda: w_boundary() / 4),
    ("m_1 [meV]", "0.307", "01.06", lambda: nu_masses()[0]),
    ("m_2 [meV]", "8.61", "01.06", lambda: nu_masses()[1]),
    ("m_3 [meV]", "50.24", "01.06", lambda: nu_masses()[2]),
    ("Sum m_nu [meV] (P17)", "59.16", "01.06", lambda: sum(nu_masses())),
    # --- 01장: 우주
    ("H_CE = 100h (P08)", "67.772", "01.06", lambda: R.hubble_kms(core()) * (1 + core()["a"] / (4 * math.pi))),
    ("H0 direct (P07)", "73.356", "01.06", lambda: pred("P07")),
    ("1/cos(pi/8) (P11)", "1.08239", "01.06", lambda: 1 / math.cos(math.pi / 8)),
    ("omega_b h^2 (P12)", "0.022343", "01.06", lambda: pred("P12")),
    ("omega_c h^2 (P13)", "0.11847", "01.06", lambda: pred("P13")),
    ("h r_d (P14)", "99.987", "01.06", lambda: pred("P14")),
    ("S8 (P16)", "0.8161", "01.06", lambda: pred("P16")),
    ("n_s (P10)", "0.965035", "01.06", lambda: 1 - 2 / core()["Ne"]),
    ("r (P09)", "0.00366762", "01.06", lambda: 12 / core()["Ne"] ** 2),
    ("w(z=0) (P21)", "-0.9894", "01.06", lambda: pred("P21")),
    ("w(z=1) (P22)", "-0.9974", "01.06", lambda: pred("P22")),
    ("BAO drift (P25)", "1.0126", "01.06", lambda: pred("P25")),
    ("A_s x1e9", "2.1035", "01.06", lambda: R.scalar_amplitude(core())),
    ("Omega_m from Higgs (P35)", "0.3071", "01.06", lambda: pred("P35", "CMB_only_LambdaCDM")),
    # --- 01장: 중력
    ("ringdown M omega = 1/(8 pi) (P40)", "0.039789", "01.07", lambda: 1 / (8 * math.pi)),
    ("BMV concurrence (섭동 양자 중력)", "0.0330", "01.07", lambda: bmv()["concurrence_quantum_gravity"]),
    # --- 01장: 점수
    ("joint RMSE, 39 rows", "0.834", "01.08", lambda: joint_rmse()[0]),
    ("joint RMSE, 43 rows", "0.878", "01.08", lambda: joint_rmse()[1]),
    # --- 13–14장
    ("Omega_k pull (P24)", "-0.37", "13.03",
     lambda: mod("probability_weight").cosmic_weights(core())["Omega_k pull vs Planck2018+BAO 0.0007+/-0.0019"]),
    ("반인과 판독", "62.61", "14.02", lambda: mod("causal_lock").direction()["anti_causal"]),
    ("반인과 SH0ES pull", "-12.3", "14.02", lambda: mod("causal_lock").direction()["anti_causal_SH0ES"]),
    ("H_Lambda [km/s/Mpc]", "56.38", "16.03", lambda: pred("P08") * math.sqrt(omega_parts()[2])),
    # --- 15장
    ("BAO chi2 (G1m)", "10.99", "15.02", lambda: mod("gradient").bao_rows("G1m")["bao_chi2_fixed"]),
    ("CMB-BAO 척도 긴장 (G1m)", "-0.22", "15.02", lambda: mod("gradient").bao_rows("G1m")["tension_sigma"]),
    ("BAO 기울기 비, 오늘 무게", "1.0126", "15.02", lambda: drift_ratio("today")),
    ("W1 w_a", "-0.214", "15.03",
     lambda: -2 * core()["a"] ** (2 / 3) * (1 - omega_parts()[2]) / omega_parts()[2]),
    ("xi^2 = alpha_s^(2/3)", "0.24046", "15.03", lambda: core()["a"] ** (2 / 3)),
    ("Delta m^2_21 [eV^2] (P19)", "7.403e-5", "15.05", lambda: mod("neutrino").splittings_ev2(core())[0]),
    ("Delta m^2_31 [eV^2]", "2.524e-3", "15.05", lambda: mod("neutrino").splittings_ev2(core())[1]),
    ("JUNO pull (P19)", "-0.81", "15.05", lambda: mod("nu_cosmo").juno_pulls()["dm21 (P19)"]),
    ("JUNO pull (P03)", "1.07", "15.05", lambda: mod("nu_cosmo").juno_pulls()["s12sq (P03)"]),
    # --- 16장
    ("가지 무게 신호(벨 쌍)", "0.500", "16.05", lambda: mod("probability_weight").signalling("branch")),
    ("가지 무게 무작위 최대 신호", "0.624", "16.01", lambda: mod("axiom_proofs").qg1_signalling_scan()["branch"]),
    ("BMV 얽힘 위상 [rad]", "0.0661", "16.02", lambda: bmv()["entangling_phase_rad"]),
    ("BMV 가장 가까운 쌍 위상 [rad]", "0.3516", "16.02", lambda: bmv()["closest_pair_phase_rad"]),
    ("c/H_Lambda [Mpc]", "5317", "16.03", lambda: horizon()["radius_mpc"]),
    ("유클리드 주기 [Gyr]", "109", "16.03", lambda: horizon()["euclidean_period_gyr"]),
    ("드 시터 온도 [K]", "2.221e-30", "16.03", lambda: horizon()["temperature_k"]),
    ("드 시터 엔트로피", "3.2e122", "16.03", lambda: horizon()["entropy_over_kB"]),
    ("실수 분기 beta / 쌍극자", "336.8", "16.03", lambda: mod("complex_boost").real_branch()["beta_over_dipole"]),
    ("실수 분기 gamma", "1.0987", "16.03", lambda: mod("complex_boost").real_branch()["gamma"]),
    # --- 17장
    ("로그 보정 기울기", "-0.49999", "17.02", lambda: mod("horizon_pixel").log_correction()["slope_vs_lnN"]),
    ("QNM / 선 간격", "9.39", "17.05", lambda: mod("horizon_pixel").ringdown()["QNM_in_units_of_spacing"]),
    ("화소 하나의 질량 / T_H", "1.000", "17.03", lambda: mod("jacobson").pixel_energy()["ratio"]),
    ("Kerr 첫째 법칙 최대 잔차", "4.3e-9", "17.03", lambda: mod("jacobson").kerr_first_law()["max_residual"]),
    ("G_eff/G, bit 화소", "1.443", "17.04",
     lambda: mod("jacobson").g_eff()["bit pixel: ln2 / (2 l_P)^2"]),
    # --- 18장
    ("C/S 감마(2)", "0.73", "18.01", lambda: clocks()["gamma(2, mean 1)"]["C_over_S"]),
    ("C/S 반정규", "0.53", "18.01", lambda: clocks()["half-normal(mean 1)"]["C_over_S"]),
    ("C/S 가우스", "0.35", "18.01", lambda: clocks()["gaussian(mean 1, sd 1)"]["C_over_S"]),
    ("alpha_diff, rho = 0.9", "0.100", "18.03", lambda: mod("joint_render").alpha_diff(0.9)),
    ("팔 길이 차 잔여, eps = 0.01", "5.0e-5", "18.03", lambda: mod("joint_render").alpha_diff(1.0, eps=1e-2)),
    # --- 19장
    ("chi2, 39행", "27.14", "19.01", lambda: ledger("IV")["CE"]["chi2"]),
    ("chi2, 43행", "33.18", "19.01", lambda: ledger("full")["CE"]["chi2"]),
    ("기준 RMSE, 39행", "1.442", "19.03", lambda: ledger("IV")["baseline"]["rmse"]),
    ("기준 RMSE, 43행", "1.471", "19.03", lambda: ledger("full")["baseline"]["rmse"]),
    ("S 이론, 39행", "58.1", "19.03", lambda: ledger("IV")["CE"]["S"]),
    ("S 기준, 39행", "119.1", "19.03", lambda: ledger("IV")["baseline"]["S"]),
    ("손익분기 bit, 39행", "64.9", "19.03", lambda: mod("ledger").break_even_bits(ledger("IV"))),
    ("AIC 거시 이론", "20.0", "19.03", lambda: ledger_blocks()["M"]["AIC_CE"]),
    ("AIC 거시 기준", "89.6", "19.03", lambda: ledger_blocks()["M"]["AIC_base"]),
    ("설명 길이 양자, 이론", "38.6", "19.03", lambda: mdl()["Q"]["L_CE"]),
    ("설명 길이 양자, 기준", "168.1", "19.03", lambda: mdl()["Q"]["L_base"]),
    ("2026 점검, 39행", "0.876", "19.02", lambda: rescore_2026()["V39"]["after"]),
    ("2026 점검, 43행", "0.914", "19.02", lambda: rescore_2026()["43rows"]["after"]),
    ("Pantheon L0", "39.58", "19.04", lambda: pantheon()["L0"]),
    ("Pantheon W2", "39.77", "19.04", lambda: pantheon()["W2"]),
    ("Pantheon W3", "39.69", "19.04", lambda: pantheon()["W3"]),
    ("Pantheon 최적 ΛCDM", "39.30", "19.04", lambda: pantheon()["best"]["chi2"]),
    # --- 20–21장
    ("살아 있는 독립 예측", "37", "20.00", live_independent),
    ("BAO 기울기 비, 기록 시기 무게", "1.0188", "21.02", lambda: drift_ratio("record")),
]


def matches(printed: str, value: float) -> bool:
    d = Decimal(printed)
    half_ulp = 0.5 * 10.0 ** d.as_tuple().exponent
    return abs(value - float(d)) <= half_ulp * (1 + 1e-9)


def main() -> int:
    failed = 0
    for name, printed, where, fn in CHECKS:
        value = float(fn())
        ok = matches(printed, value)
        failed += not ok
        mark = "ok  " if ok else "FAIL"
        print(f"{mark} {where:>6}  {name:<34} 본문 {printed:<12} 계산 {value:.10g}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} 통과")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
