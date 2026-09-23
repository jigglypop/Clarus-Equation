"""O1의 관측 사상 — 틀 사이 비의 1차 투영. 원장: 43장 §43.48. 예측값을 바꾸지 않는다.

계산 전에 적은 사상과 kill:

사상 M. 관측량은 잰 길이(또는 시간) ÷ 교정 단위의 비다. 둘이 기울기 θ만큼 다른 틀에서 기록되었으면 비에 1차 투영
   cos θ가 곱해진다(길이는 1차, §43.45). 같은 틀의 비는 불변(C5). 중력 판독은 무게로 평균(정리 E).
   O1 = 탄생 틀의 자(r_s, r_d)를 현재 틀의 막대(시차, 메이저, GW 진폭, 렌즈 시간 지연)로 읽는 단일 틀 판독(π/8 전부).
   G1m = 같은 투영을 물질 무게로 평균한 BAO 판독. 두 규칙은 한 메커니즘이다.
K1 사상 아래 θ*·BAO D/r_d가 불변. K2 투영이 옳은 방향(직접 > 나이테). K3 BAO + 현재 교정 H₀가 추론하는 r_d가
   r_d(CE)·cos(π/8)과 2σ 안. K4 교정 종류별로 모임(이른 기록 → 67.77, 현재 막대 → 73.36).
관측 파일 benchmarks/cosmology/observations_v1.json의 값은 scientific_score_eligible = false라 표시에만 쓴다.

python -B -m examples.physics.rendering.ce_rendering_o1_map
"""

from __future__ import annotations

import json
import math

import numpy as np

from examples.physics.rendering import ce_rendering_gradient as GD
from examples.physics.rendering import ce_rendering_nu_ledger as NL
from examples.physics.rendering import ce_rendering_open_checks as OC
from examples.physics.rendering import ce_rendering_registry as R
from examples.physics.rendering import ce_rendering_vacuum_tilt as VT

TILT = math.pi / 8
SH0ES = (0.7317, 0.0086)
REPO_OBS = "benchmarks/cosmology/observations_v1.json"
# 교정 종류: 이른 기록(r_s, r_d, BBN ω_b) 또는 현재 막대·시계. 43장 채점 행은 σ가 고정되어 있다.
CHAPTER_READOUTS = [("Planck 2018 (CMB)", "early", 67.36, 0.54), ("DESI BAO+BBN", "early", 68.51, 0.58),
                    ("SH0ES 2024", "present", 73.17, 0.86), ("CCHP TRGB", "present", 70.39, 1.94),
                    ("TDCOSMO 2025", "present", 71.6, 3.6)]
FILE_CLASS = {"Planck2018_base": "early", "ACT_DR6_DESI_reported": "early", "SPT3G_CMBSPA": "early",
              "Planck_ACT_SPT_combined": "early", "SH0ES_2024_FOUR_ANCHOR": "present", "TDCOSMO_2025_CANONICAL": "present"}


def _ce():
    c = R.core(R.calibrated_alpha_s()[0])
    rd, h = NL.rd_and_h(c)
    b = VT.bao_vectors(c["Om"], VT.density(c, VT.ADOPTED_NU)) * GD.factor("G1m", R.BAO_Z, c["Om"])
    return c, rd, h, b


def ring_invariance() -> dict:
    """K1: (h → h/cos, 이른 자 → 자·cos)에서 BAO D/r_d와 θ* = r_s/D_M의 최대 상대 변화."""
    c, rd, h, b = _ce()
    k = math.cos(TILT)
    bao_ring = GD.C_KM_S / (100 * h * rd) * b
    bao_present = GD.C_KM_S / (100 * (h / k) * (rd * k)) * b
    dm_ratio = (1 / (h / k)) / (1 / h)                    # D_M ∝ 1/h (Ω_m 고정)
    theta_ratio = k / dm_ratio                            # r_s·cos ÷ D_M·cos
    return {"bao_max_rel_change": float(np.abs(bao_present / bao_ring - 1).max()), "theta_ratio": theta_ratio}


def present_units_rd() -> dict:
    """K3: BAO의 자유 눈금(G1m, 채택 진공)에서 hr_d를 얻고 SH0ES h로 r_d를 추론해 CE의 r_d·cos(π/8)와 비교."""
    c, rd, h, b = _ce()
    cinv, y = R.BAO_CINV, R.BAO_Y
    fisher = float(b @ cinv @ b)
    a = float(b @ cinv @ y) / fisher
    sa = 1 / math.sqrt(fisher)
    hrd = GD.C_KM_S / (100 * a)
    s_hrd = hrd * sa / a
    rd_inf = hrd / SH0ES[0]
    s_rd = rd_inf * math.hypot(s_hrd / hrd, SH0ES[1] / SH0ES[0])
    pred = rd * math.cos(TILT)
    return {"hrd_bao": hrd, "hrd_sigma": s_hrd, "rd_inferred_present": rd_inf, "rd_sigma": s_rd,
            "rd_CE_ring": rd, "rd_CE_present": pred, "pull": (pred - rd_inf) / s_rd}


def classification() -> dict:
    """K4: 교정 종류별 예측과 43장 채점 행의 pull, 관측 파일 값(표시만)."""
    c, _, h, _ = _ce()
    pred = {"early": 100 * h, "present": 100 * h / math.cos(TILT)}
    rows = [(name, cls, v, s, (pred[cls] - v) / s) for name, cls, v, s in CHAPTER_READOUTS]
    display = []
    try:
        data = json.loads((R.REPO_ROOT / REPO_OBS).read_text(encoding="utf-8")) if hasattr(R, "REPO_ROOT") else None
    except OSError:
        data = None
    if data is None:
        from pathlib import Path
        data = json.loads((Path(__file__).resolve().parents[3] / REPO_OBS).read_text(encoding="utf-8"))
    for o in data["observations"]:
        cls = FILE_CLASS.get(o["observation_id"])
        if cls and "H0" in o["values"]:
            display.append((o["observation_id"], cls, o["values"]["H0"], round(o["values"]["H0"] - pred[cls], 2),
                            o["validity"]["scientific_score_eligible"]))
    return {"pred": pred, "chapter_rows": rows, "file_display_only": display}


def main() -> None:
    print("K1 ring invariance:", ring_invariance())
    print("K2 direction:", {k: (round(v["H0"], 2) if isinstance(v, dict) else round(v, 2)) for k, v in OC.o1_direction().items()})
    print("K3 present-units r_d:", {k: round(v, 3) for k, v in present_units_rd().items()})
    cl = classification()
    print("K4 predictions:", {k: round(v, 2) for k, v in cl["pred"].items()})
    for r in cl["chapter_rows"]:
        print(f"   {r[0]:20s} {r[1]:8s} obs {r[2]:6.2f} ± {r[3]:.2f}  pull {r[4]:+.2f}")
    for r in cl["file_display_only"]:
        print(f"   [display] {r[0]:26s} {r[1]:8s} H0 {r[2]:6.2f}  obs − pred {r[3]:+.2f}  score-eligible={r[4]}")


if __name__ == "__main__":
    main()
