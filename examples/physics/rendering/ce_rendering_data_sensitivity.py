"""자료 판본 민감도: 결론이 관측 자료의 판본 선택에 기대는지. 원장: 43장 §43.40. 잠긴 모듈은 import만 한다.

계산 전에 고정한 규칙: 판정은 바꾸지 않고 보고만 한다. 어떤 판본 조합에서든 |pull| > 3인 행이 생기거나
W2와 W3(전달 보정)의 순위가 뒤집히면 따로 표시한다.
축: NuFIT 6.0 (SK 대기 자료 포함/제외) × 약한 렌즈 S8 행 (둘 다/DES Y3만/KiDS-Legacy만/없음) × H0 직접 판독
(셋 다/TRGB 제외). 두 판본(W2+G1m, W3 보정+G1m) 모두 FD θ*와 고정 눈금 BAO를 쓴다.

python -B -m examples.physics.rendering.ce_rendering_data_sensitivity
"""

from __future__ import annotations

import itertools
import math

from examples.physics.rendering import ce_rendering_gradient as GD  # noqa: F401  (G1m via W3 module)
from examples.physics.rendering import ce_rendering_mimetic_vacuum as MV
from examples.physics.rendering import ce_rendering_nu_ledger as NL
from examples.physics.rendering import ce_rendering_registry as R
from examples.physics.rendering import ce_rendering_theta_nu as TN
from examples.physics.rendering import ce_rendering_vacuum_tilt as VT
from examples.physics.rendering import ce_rendering_w3_growth as W3

LENS = {"both": ("S8 KiDS-Legacy", "S8 DES Y3 3x2pt"), "DES": ("S8 DES Y3 3x2pt",),
        "KiDS": ("S8 KiDS-Legacy",), "none": ()}
H0SETS = {"all": ("H0 TDCOSMO", "H0 TRGB", "H0 SH0ES"), "noTRGB": ("H0 TDCOSMO", "H0 SH0ES")}


def _models() -> dict:
    c = R.core(R.calibrated_alpha_s()[0])
    tn = TN.pulls()
    wb, wc, h = NL.early_densities(c)
    f2 = VT.density(c, VT.ADOPTED_NU)
    m = MV.model("b")
    return {
        "W2": {"theta": tn["W2"], "wc": wc, "S8": W3.compare()["W2"]["S8"],
               "bao": W3._bao_chi2_g1m(VT.bao_vectors(c["Om"], f2), wb, wc, h, c["Om"])},
        "W3": {"theta": tn["W3b"], "wc": m["wc"], "S8": W3.s8_corrected(m),
               "bao": W3._bao_chi2_g1m(MV.bao_vectors(m), m["wb"], m["wc"], m["h"], c["Om"])},
    }


def score(model: dict, pmns: str, lens: str, h0set: str) -> dict:
    rows = [dict(o) for o in NL.score("full", "fixed", pmns)["rows"]]
    keep = []
    for o in rows:
        k = o["key"]
        if k.startswith("S8 ") and k not in LENS[lens]:
            continue
        if k.startswith("H0 ") and k not in H0SETS[h0set]:
            continue
        if k == "100 theta*":
            o["pull"] = model["theta"]
        elif k == "omega_c h^2":
            o["pull"] = (model["wc"] - o["obs"]) / o["sigma"]
        elif k.startswith("S8 "):
            o["pull"] = (model["S8"] - o["obs"]) / o["sigma"]
        keep.append(o)
    chi = sum(o["pull"] ** 2 for o in keep) + model["bao"]
    worst = max(keep, key=lambda o: abs(o["pull"]))
    return {"rmse": math.sqrt(chi / (len(keep) + 13)), "N": len(keep) + 13, "worst": (worst["key"], worst["pull"])}


def table() -> list[dict]:
    models = _models()
    out = []
    for pmns, lens, h0set in itertools.product(("SK", "noSK"), LENS, H0SETS):
        s2, s3 = score(models["W2"], pmns, lens, h0set), score(models["W3"], pmns, lens, h0set)
        out.append({"pmns": pmns, "lens": lens, "h0": h0set, "N": s2["N"], "W2": s2["rmse"], "W3": s3["rmse"],
                    "worst_W2": s2["worst"], "flag_3sigma": abs(s2["worst"][1]) > 3 or abs(s3["worst"][1]) > 3,
                    "better": "W2" if s2["rmse"] < s3["rmse"] else "W3"})
    return out


def main() -> None:
    for r in table():
        print(f"{r['pmns']:5s} lens={r['lens']:5s} H0={r['h0']:7s} N={r['N']}  W2={r['W2']:.3f}  W3={r['W3']:.3f}  "
              f"better={r['better']}  worst(W2)={r['worst_W2'][0]} {r['worst_W2'][1]:+.2f}{'  >3sigma' if r['flag_3sigma'] else ''}")


if __name__ == "__main__":
    main()
