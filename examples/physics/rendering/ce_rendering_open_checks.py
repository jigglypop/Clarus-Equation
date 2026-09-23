"""남은 항목 점검 — 정리 E의 투영 차수, O1의 방향, 2026 자료 재채점, 가설 장부 검산. 원장: 43장 §43.45.

예측값과 사전 등록을 바꾸지 않는다. 계산 전에 적은 판본:

(A) 정리 E. BAO 관측량 D/r_d는 길이의 비(1차)라서 틀 투영은 cos¹이다. cos²는 에너지 밀도(2차)의 투영이다.
    C5(§43.16)의 불변은 “같은 틀에서 새겨진” 비에 한정된다. 자 r_s, r_d는 탄생 틀(기울기 ~0), 은하 기록은 z의 틀이다.
(B) O1의 방향. 판독자 시계의 연속 시간 팽창이면 H_direct = H·cos(π/8), 현재 틀 팽창의 기록 틀 투영이면 H/cos(π/8).
(C) 2026 자료(JUNO, DES Y6, P-ACT)로 행만 바꿔 다시 채점한다(보고만).
(D) §43.33의 χ²(39행)를 다시 계산한다.

python -B -m examples.physics.rendering.ce_rendering_open_checks
"""

from __future__ import annotations

import math

import numpy as np

from examples.physics.rendering import ce_rendering_gradient as GD
from examples.physics.rendering import ce_rendering_nu_ledger as NL
from examples.physics.rendering import ce_rendering_registry as R
from examples.physics.rendering import ce_rendering_vacuum_tilt as VT

H0_DIRECT = {"SH0ES": (73.17, 0.86), "TRGB": (70.39, 1.94), "TDCOSMO": (71.6, 3.9)}
UPDATES_2026 = {  # 행 키 → (관측값, σ, 출처)
    "s12^2": (0.3092, 0.0087, "JUNO 2025, arXiv:2511.14593"),
    "Delta m^2_21": (7.50e-5, 0.12e-5, "JUNO 2025"),
    "S8 DES Y3 3x2pt": (0.789, 0.012, "DES Y6 3x2pt, arXiv:2601.14559"),
    "n_s": (0.9709, 0.0038, "P-ACT (Planck + ACT DR6, CMB only), arXiv:2503.14452"),
}


def _bao_setup():
    c = R.core(R.calibrated_alpha_s()[0])
    _, _, h = NL.early_densities(c)
    rd, _ = NL.rd_and_h(c)
    b0 = VT.bao_vectors(c["Om"], VT.density(c, VT.ADOPTED_NU))
    base = VT.score(VT.ADOPTED_NU)
    return c, GD.C_KM_S / (100 * h * rd) * b0, base


def projection_rank_scan() -> dict:
    """(A) 인자 = w_m cos^r θ + (1 − w_m), θ = ψ(z)/2. r = 1(길이), 2(에너지); 무게 = 오늘 / 기록 시기."""
    c, pred0, base = _bao_setup()
    z, om = R.BAO_Z, c["Om"]
    theta = np.array([VT.cycle_phase(1 / (1 + zi), om) for zi in z]) / 2
    chi_rows = base["chi2"] - base["branch"]["bao_chi2_fixed"]
    out = {}
    for r in (1, 2):
        for tag, ep in (("today", False), ("record_epoch", True)):
            w = om * (1 + z) ** 3 / (om * (1 + z) ** 3 + 1 - om) if ep else om
            res = pred0 * (w * np.cos(theta) ** r + 1 - w) - R.BAO_Y
            chi = float(res @ R.BAO_CINV @ res)
            out[f"rank{r}_{tag}"] = {"bao_chi2": chi, "V39": math.sqrt((chi_rows + chi) / base["N"])}
    return out


def frame_epochs() -> dict:
    """(A) C5 한정어: 자의 기록 시기(z*, z_d)와 은하 기록 시기(BAO z)의 기울기 ψ/2."""
    om = R.core(R.calibrated_alpha_s()[0])["Om"]
    tilt = lambda zz: VT.cycle_phase(1 / (1 + zz), om) / 2
    return {"ruler_z*": tilt(1090.0), "ruler_zd": tilt(1060.0), "ruler_diff": abs(tilt(1090.0) - tilt(1060.0)),
            "galaxy_min": min(tilt(zz) for zz in R.BAO_Z), "galaxy_max": max(tilt(zz) for zz in R.BAO_Z)}


def o1_direction() -> dict:
    """(B) 두 메커니즘의 직접 판독 H₀와 pull."""
    c = R.core(R.calibrated_alpha_s()[0])
    _, _, h = NL.early_densities(c)
    ring = 100 * h
    k = math.cos(math.pi / 8)
    out = {"ring": ring}
    for tag, val in (("clock_dilation", ring * k), ("projection", ring / k)):
        out[tag] = {"H0": val, **{name: (val - o) / s for name, (o, s) in H0_DIRECT.items()}}
    return out


def rescore_2026() -> dict:
    """(C) 판본 V(39행)와 43행을 2026 자료로 다시 채점. 예측값은 그대로."""
    bao = GD.bao_rows("G1m")["bao_chi2_fixed"]
    out = {}
    for rows_from, label in (("IV", "V39"), ("full", "43rows")):
        base = VT.score(VT.ADOPTED_NU, rows_from, "fixed")
        rows = [dict(o) for o in base["rows"]]
        before = sum(o["pull"] ** 2 for o in rows) + bao
        changed = {}
        for o in rows:
            if o["key"] in UPDATES_2026:
                obs, sig, _ = UPDATES_2026[o["key"]]
                if o["obs"] > 1e-3 > obs:            # Δm²₂₁ 행의 단위(1e-5) 맞춤
                    obs, sig = obs * 1e5, sig * 1e5
                old = o["pull"]
                o["pull"] = (o["pred"] - obs) / sig
                changed[o["key"]] = (round(old, 2), round(o["pull"], 2))
        after = sum(o["pull"] ** 2 for o in rows) + bao
        n = base["N"]
        out[label] = {"N": n, "before": math.sqrt(before / n), "after": math.sqrt(after / n), "changed": changed,
                      "max_abs_pull": max(abs(o["pull"]) for o in rows)}
    return out


def ledger_chi2() -> dict:
    """(D) 가설 사슬(O1·W2·G1m) 없는 판본(Λ, 그라데이션 없음)과 채택 판본의 39행 χ²."""
    lam = VT.score(None)
    base = VT.score(VT.ADOPTED_NU)
    adopted = base["chi2"] - base["branch"]["bao_chi2_fixed"] + GD.bao_rows("G1m")["bao_chi2_fixed"]
    gain = lam["chi2"] - adopted
    return {"N": lam["N"], "chi2_without_chain": lam["chi2"], "chi2_adopted": adopted, "gain": gain,
            "cost_chi2_4.1bit": 4.1 * 2 * math.log(2), "net": gain - 4.1 * 2 * math.log(2)}


def main() -> None:
    print("(A) projection rank:", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in projection_rank_scan().items()})
    print("(A) frame epochs (tilt psi/2):", {k: f"{v:.2e}" for k, v in frame_epochs().items()})
    print("(B) O1 direction:", o1_direction())
    print("(C) 2026 rescoring:", rescore_2026())
    print("(D) ledger:", {k: round(v, 2) for k, v in ledger_chi2().items()})


if __name__ == "__main__":
    main()
