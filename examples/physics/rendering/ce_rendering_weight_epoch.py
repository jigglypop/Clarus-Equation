"""나이테 판독의 무게 시점 — 오늘, 기록 시기, 빛 경로 평균. 원장: 43장 §43.76. 예측값을 바꾸지 않는다.

정리 E(§43.36): BAO 판독 인자 = 1 − w(1 − cos θ), θ = ψ/2(TT + H1, §43.53). 남은 1 bit는 무게 w의 시점이다.
계산 전에 적은 판본과 판정:

W0(채택)  w = 오늘의 Ω_m.
Wz        w = 기록 시기 Ω_m(z)(§43.36 주: 판본 V 0.903).
Wpath     원리 후보: 거리는 경로 적분이므로 투영 손실도 경로를 따라 쌓인다. D_M은 빛 경로 평균 ⟨Ω_m⟩(dz/E 가중),
          D_H = c/H(z)는 국소량이라 기록 시기 Ω_m(z), D_V는 정의대로 (D_M² D_H)^{1/3} 섞음.
판정. Wpath가 W0와 Δχ²(BAO) ≤ 1이면 “경로 평균”이 1 bit를 대신하는 원리로 선다. 더 나쁘면 오늘 무게는 선택(1 bit)으로 남는다.
검산. W0는 기울기 모듈의 G1m(BAO χ², 공동 V)을 재현해야 한다.

python -B -m examples.physics.rendering.ce_rendering_weight_epoch
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad

from examples.physics.rendering import ce_rendering_gradient as GD
from examples.physics.rendering import ce_rendering_nu_ledger as NL
from examples.physics.rendering import ce_rendering_registry as R
from examples.physics.rendering import ce_rendering_vacuum_tilt as VT


def _setup() -> dict:
    c = R.core(R.calibrated_alpha_s()[0])
    f = VT.density(c, VT.ADOPTED_NU)
    om = c["Om"]
    e = lambda z: math.sqrt(om * (1 + z) ** 3 + (1 - om) * f(1 / (1 + z)))
    om_z = lambda z: om * (1 + z) ** 3 / e(z) ** 2
    _, _, h = NL.early_densities(c)
    rd, _ = NL.rd_and_h(c)
    return {"c": c, "f": f, "om": om, "E": e, "om_z": om_z, "a_ce": GD.C_KM_S / (100 * h * rd)}


def _path_mean(s: dict, z: float) -> float:
    num = quad(lambda x: s["om_z"](x) / s["E"](x), 0, z)[0]
    den = quad(lambda x: 1 / s["E"](x), 0, z)[0]
    return num / den


def factors(version: str) -> np.ndarray:
    s = _setup()
    loss = 1 - np.cos(GD._psi(R.BAO_Z, s["om"]) / 2)
    out = []
    for z, kind, eps in zip(R.BAO_Z, R.BAO_KIND, loss):
        if version == "W0":
            out.append(1 - s["om"] * eps)
        elif version == "Wz":
            out.append(1 - s["om_z"](z) * eps)
        else:
            fm = 1 - _path_mean(s, z) * eps
            fh = 1 - s["om_z"](z) * eps
            out.append({"dm": fm, "dh": fh, "dv": (fm * fm * fh) ** (1 / 3)}[kind])
    return np.array(out)


def score(version: str) -> dict:
    s = _setup()
    b = VT.bao_vectors(s["om"], s["f"]) * factors(version)
    r = s["a_ce"] * b - R.BAO_Y
    chi = float(r @ R.BAO_CINV @ r)
    base = VT.score(VT.ADOPTED_NU)
    v = math.sqrt((base["chi2"] - base["branch"]["bao_chi2_fixed"] + chi) / base["N"])
    return {"bao_chi2": chi, "V": v}


def verdict() -> dict:
    out = {k: score(k) for k in ("W0", "Wz", "Wpath")}
    out["check_W0_vs_G1m"] = out["W0"]["bao_chi2"] - GD.bao_rows("G1m")["bao_chi2_fixed"]
    out["d_chi2_path"] = out["Wpath"]["bao_chi2"] - out["W0"]["bao_chi2"]
    out["path_replaces_bit"] = out["d_chi2_path"] <= 1.0
    return out


def main() -> None:
    for k, v in verdict().items():
        print(k, {kk: round(vv, 4) for kk, vv in v.items()} if isinstance(v, dict) else v)


if __name__ == "__main__":
    main()
