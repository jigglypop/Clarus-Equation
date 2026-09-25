"""경계 소실 — 질량 보존을 깨는 별도 갈래. 원장: 43장 §43.103. 주 배경(W2 + TT)과 예측값은 바꾸지 않는다.

사용자 가설 BL(2026-09-25): "전 우주 질량 보존의 법칙부터 깨면? 우주 경계면에서 질량이 '분별'되며 흩어지고
사라지는 것은?" 사용자 지시: 완전 전환이 아니라 별도 갈래로 두고 최대한 강하게 시험한다. 계산 전에 받은 구조
가설(형체 없는 바닥, 경계면에서 굳는 층, 늦게 생긴 층이 먼저 무너짐, 앞선 기록이 다음 진공을 만듦)은 아래
받는 쪽의 주 판본, 가족 B, 가족 C로 옮겼다.

CE 공리 점검(외부 모형을 대입하기 전):
- QG1·R1(§43.30–43.31): 중력의 원천은 국소 확률 무게이고 기록은 빛원뿔 안에서만 원천을 바꾼다. 원격 선택에
  조건화되지 않은 국소 소실률(붕괴와 같은 꼴)은 신호를 만들지 않으므로 허용된다.
- W3(§43.28): CE는 일반 공변 작용(비앙키 항등식)을 채택했다. 받는 쪽 없이 사라지는 것(∇T ≠ 0)은 계량 이론에서
  허용되지 않고, 단모듈 중력에서는 위반량이 Λ로 쌓인다(Josset–Perez–Sudarsky 2017). 받는 쪽은 둘뿐이다.
    S-Λ  진공(주 판본: 무너짐은 요동이 없는 바닥에서 멈춘다)
    S-r  복사("흩어짐": 에너지는 보존되고 질량만 사라진다)
- 균질성: 경계는 관측자마다 다르고 균질한 우주의 모든 점은 누군가의 경계 위에 있다. 평균장 구현은 모든 곳의
  균일한 소실률 = 경계 통과율 Γ × 통과한 질량이 분별되어 사라질 확률 β다.
- 방향: W3는 진공→먼지(확정), BL은 먼지→진공(흐려짐)이다. 전체 무게는 보존된다(Ω_k = 0, P24).

모형: 잃는 성분 ρ_L에 대해 ρ̇_L + 3Hρ_L = −βΓρ_L.  S-Λ: ρ̇_Λ = +βΓρ_L.  S-r: ρ̇_x + 4Hρ_x = +βΓρ_L.
경계 통과율 Γ(β = 1이면 통과한 질량이 모두 사라진다):
  dS  3H_Λ           TT 시계의 드 시터 렌더링 지평(c/H_Λ)을 공동 물질이 빠져나가는 비율
  EH  3c/r_e         실제 사건 지평의 탈출률(미래에 의존하므로 인과 주의를 적되 kill은 아님)
  HX  3 max(0, ä/ȧ)  허블 구 탈출률(가속기에만 켜짐)
  TT  H_Λ tan(Φ/2)   TT 흐려짐의 물리판: β = 1이면 ρ_L a³ ∝ cos²(H_Λ t/2)
  H   3H             문헌 표준 대조군(Q ∝ Hρ_m). 통과율이 아니다.
H_Λ = H0 √Ω_Λ0(오늘의 진공 무게, TT 기준과 같다). CE Ω_m의 해석 1 bit(W3 선례):
  (a) 오늘의 물질(주): 이른 우주의 ω_b·ω_c가 소실만큼 크다.   (b) 이른 기록(z*)의 물질: 오늘의 물질이 작다.
가족 A  모든 물질(바리온 + 암흑물질)이 같은 비율로 잃는다: 5 모양 × 2 받는 쪽 × 2 해석 = 20판본.
가족 B  늦게 생긴 층이 먼저 무너진다: 확정된 무게 q(바리온)만 잃는다. S-Λ, 5 × 2 = 10판본.
가족 C  닫힘: 진공 전체가 이번 순환의 쌓인 소실이다(Λ_bare = 0, W2 흔들림 없음). 소실 적분이 탄생에서 수렴하는
        모양은 TT뿐이다(초기 ρ ∝ t^{-3/2}에 대해 dS·EH·H는 발산하고, HX는 가속이 없으면 0이라 Λ = 0만 남는다).
        TT × {A, B} × {a, b} = 4판본. β는 닫힘 조건이 정한다(연속 매개변수 0).
나이테 H0(CE 식)와 W2 진공 모양(A·B)은 그대로다. TT 판독(θ = H_Λ t/2, G1m 인자의 Ω_m = 오늘의 물질 몫)은 바뀐
배경에서 다시 계산한다.

채점: TT 기준(§43.53, V39 0.8407, χ² 27.563)과 같은 39행. θ*·r_d·BAO 모양·TT 위상은 같은 코드에서 β와 β = 0의
차이(θ*) 또는 비(나머지)를 기준값에 옮긴다(W3와 같은 방식). 기준 행 값(θ* +0.56σ, ω_c −1.26σ, BAO χ² 10.99)은
계산 전에 알려져 있었다.

판정(계산 전 고정):
K1  문자 그대로(β = 1): A·B의 dS·EH·HX·TT 각각 V39 > 0.909 또는 바뀐 행(θ*, ω_b, ω_c, H0 셋)의 |pull| > 3이면 기각.
K2  자료가 소실을 원하는가: A·B 30판본에서 β ≥ 0 적합. Δχ² ≤ −9.87(30판본 Bonferroni 95%)이면 증거, Δχ² < −2
    (ΔAIC < 0)이면 기미, 그 외는 증거 없음. 한쪽 95% 상한(Δχ² = +2.71)의 β와 z* 이후·z = 1 이후 사라진 비율을 보고한다.
K3  원리 후보 β ∈ {1, δ, q, α_s, λ = α_s/2π, ξ² = α_s^{2/3}}(2.6 bit): K1과 같은 kill 규칙. 채택 수준은
    Δχ² ≤ −10.4(판본 4.9 + 후보 2.6 bit, bit당 2 ln 2).
K4  닫힘 C: K1과 같은 kill 규칙. 채택 수준은 Δχ² ≤ −2.8(4판본 2 bit). β_C > 1이면 확률 해석 밖으로 적는다.
    C가 모두 기각되면 "진공은 이번 순환의 쌓인 소실일 수 없다(탄생 이전에서 물려받아야 한다)"고 적는다.
진단: A·S-Λ의 부호 자유 β(음수 = 진공→물질, W3 방향), 기록 기준 실효 w(z), 나이.
보강: 각 가족의 최적 판본과 주 판본(A·S-Λ)의 상한에 43행(S8 성장)과 Pantheon 보류(§43.37 규칙)를 계산한다.

추가 사전 등록(2026-09-25, 위 결과를 본 뒤 사용자 정정 "없음이 아님. 없지도 않고 있지도 않은 상태임"):
가족 D  렌더링된 무게 q(바리온)가 경계에서 "없지도 있지도 않은" 무게로 넘어간다. CE 56장(QG1)은 비관측 성분을
        삭제하지 않고 무게를 그대로 센다. 그래서 넘어간 몫은 에너지와 꼴(먼지, w = 0)을 유지한 채 중력만 내고
        기록(빛)에서 빠진다. 받는 쪽 "d", 5 모양 × 2 해석 = 10판본.
계산 전 해석: 총 물질과 꼴이 그대로라 팽창과 성장이 바뀌지 않는다. (b)는 39·43행 어디에도 보이지 않고(Δχ² ≡ 0),
(a)만 이른 ω_b·ω_c로 시험된다. K2 문턱은 10판본 Bonferroni 95%(Δχ² ≤ −7.88)다.
실험실 점검(계산 전 문헌): 평균장 과정이면 실험실의 핵자도 같은 비율로 렌더링을 잃어 보이지 않는 핵자 소멸로
나타난다. SNO+(Phys. Rev. D 105, 112012, 2022)의 한계는 τ(p→inv) > 9.6×10²⁹년, τ(n→inv) > 9.0×10²⁹년(90%)이다.
이 한계는 가족 A·B·D의 바리온 몫에 모두 적용된다. 관계적 읽기(관측자의 렌더링 경계 밖은 그 관측자에게만
"없지도 있지도 않음")는 새 물리가 없고 geometric_exit()가 그 크기를 준다.

python -B -m examples.physics.rendering.ce_rendering_boundary_loss
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.integrate import cumulative_trapezoid, quad, solve_ivp
from scipy.optimize import brentq, minimize_scalar

from examples.physics.rendering import ce_rendering_bao_ruler as BR
from examples.physics.rendering import ce_rendering_derivations as DV
from examples.physics.rendering import ce_rendering_growth as GR
from examples.physics.rendering import ce_rendering_nu_ledger as NL
from examples.physics.rendering import ce_rendering_registry as R
from examples.physics.rendering import ce_rendering_sn_holdout as SN
from examples.physics.rendering import ce_rendering_thermal_time as TTM
from examples.physics.rendering import ce_rendering_vacuum_tilt as VT

C_KM_S = 299792.458
GYR = 977.79                                    # 1/(km/s/Mpc) → Gyr
SHAPES = ("dS", "EH", "HX", "TT", "H")
LITERAL = ("dS", "EH", "HX", "TT")
FAMILIES = {"A": (("L", "r"), "all"), "B": (("L",), "b")}
READINGS = ("a", "b")
N_LO, N_HI, DN = math.log(1e-8), 30.0, 0.002
V_KILL, PULL_KILL = 0.909, 3.0
EVIDENCE, HINT, ADOPT, ADOPT_C, UPPER = -9.87, -2.0, -10.4, -2.8, 2.71
EVIDENCE_D = -7.88                              # 가족 D 10판본 Bonferroni 95%
TAU_P_INVISIBLE_YR = 9.6e29                     # SNO+ 2022, p → invisible, 90%
BETAS = np.logspace(-6, 0, 61)
H0_KEYS = ("H0 TDCOSMO", "H0 TRGB", "H0 SH0ES")
TOUCHED = ("100 theta*", "omega_b h^2", "omega_c h^2") + H0_KEYS
Z_W = (0.0, 0.5, 1.0, 2.0)
Z_SN = np.linspace(0.0, 2.5, 2501)
BASE_KEY = ("H", "L", "a", "all", 0.0, False)


def variants() -> list[tuple[str, str, str, str]]:
    """(가족, 모양, 받는 쪽, 해석) — A 20 + B 10."""
    return [(fam, shape, sink, rd) for fam, (sinks, _) in FAMILIES.items() for shape in SHAPES
            for sink in sinks for rd in READINGS]


@lru_cache(maxsize=None)
def _base() -> dict:
    """TT 기준(§43.53)의 값과 배경 격자(N = ln a, BAO와 오늘을 정확히 격자에 넣음)."""
    c = R.core(R.calibrated_alpha_s()[0])
    wb, wc, h = NL.early_densities(c)
    rd, _ = NL.rd_and_h(c)
    om = c["Om"]
    vt = VT.score(VT.ADOPTED_NU)
    rows = {o["key"]: o for o in vt["rows"]}
    _, om_t, h0, hl = TTM._cosmo()
    hub = lambda x: h0 * math.sqrt(om_t / x ** 3 + 1 - om_t)
    a = np.logspace(-6, 0, 20001)
    t = cumulative_trapezoid(1 / (a * np.array([hub(x) for x in a])), a, initial=0.0)
    t += quad(lambda x: 1 / (x * hub(x)), 0, a[0])[0]
    phi = hl * t
    n = np.unique(np.concatenate([np.arange(N_LO, N_HI + DN / 2, DN), -np.log1p(R.BAO_Z), [0.0]]))
    f_w2 = VT.density(c, VT.ADOPTED_NU)
    v43 = VT.score(VT.ADOPTED_NU, "full")
    rows43 = {o["key"]: o for o in v43["rows"]}
    s8_keys = tuple(name for name, *_ in GR.LENSING_S8)
    return {"c": c, "om": om, "q": c["q"], "h": h, "wb": wb, "wc": wc, "w_nu": NL.omega_nu_h2(c), "rd": rd,
            "b": VT.bao_vectors(om, f_w2), "rows": rows, "N": vt["N"],
            "chi_rest": sum(o["pull"] ** 2 for k, o in rows.items() if k not in TOUCHED),
            "rows43": rows43, "N43": v43["N"], "s8_keys": s8_keys,
            "chi_rest43": sum(o["pull"] ** 2 for k, o in rows43.items() if k not in TOUCHED + s8_keys),
            "th_z": np.interp(1 / (1 + R.BAO_Z), a, phi) / 2, "th0": float(phi[-1]) / 2, "t0": float(t[-1]),
            "ring": 100 * h, "n": n, "fw": np.array([f_w2(x) for x in np.exp(n)]),
            "i0": int(np.searchsorted(n, 0.0)), "i_bao": np.searchsorted(n, -np.log1p(R.BAO_Z)),
            "n_star": -math.log1p(DV.z_star(wb, wb + wc))}


def candidates() -> dict:
    c = _base()["c"]
    return {"1": 1.0, "delta": c["d"], "q": c["q"], "alpha_s": c["a"], "lambda": c["a"] / (2 * math.pi),
            "xi2": c["a"] ** (2 / 3)}


# ------------------------------------------------------------------ 배경
def _rate(shape: str, n: np.ndarray, a: np.ndarray, H: np.ndarray, t: np.ndarray, lam0: float, h0: float) -> np.ndarray:
    """Γ/H(무차원). H는 km/s/Mpc, t는 Gyr."""
    hl = h0 * math.sqrt(max(lam0, 1e-12))
    if shape == "H":
        return np.full_like(n, 3.0)
    if shape == "dS":
        return 3 * hl / H
    if shape == "EH":
        f = C_KM_S / (a * H)
        chi_e = np.trapezoid(f, n) - cumulative_trapezoid(f, n, initial=0.0) + C_KM_S * math.exp(-n[-1]) / H[-1]
        return 3 * f / chi_e
    if shape == "HX":
        return 3 * np.maximum(0.0, 1 + np.gradient(np.log(H), n))
    if shape == "TT":
        x = np.minimum(hl / GYR * t / 2, math.pi / 2 - 1e-9)
        return np.minimum(hl * np.tan(x) / H, 1e6)
    raise ValueError(shape)


def solve(shape: str, sink: str, reading: str, lossy: str, beta: float, closure: bool = False) -> dict:
    """자기 일관 배경(반복). 밀도는 오늘 임계 밀도 단위, E(1) = 1(평탄, P24)."""
    B = _base()
    n, i0, fw = B["n"], B["i0"], B["fw"]
    a = np.exp(n)
    om, q, h = B["om"], B["q"], B["h"]
    orad = DV.OMEGA_R_H2 / h ** 2
    h0 = 100 * h
    lossy_w = om if lossy == "all" else q
    keep_w = om - lossy_w
    lam0 = 1 - om - orad
    E = np.sqrt(orad / a ** 4 + om / a ** 3 + lam0 * fw)
    ok, err, kappa, it = True, 1.0, 0.0, 0
    for it in range(1, 3001):
        H = h0 * E
        t = cumulative_trapezoid(GYR / H, n, initial=0.0) + GYR / (2 * H[0])
        g = beta * _rate(shape, n, a, H, t, lam0, h0)
        L = cumulative_trapezoid(g, n, initial=0.0)
        l_anc = L[i0] if reading == "a" else float(np.interp(B["n_star"], n, L))
        m_loss = lossy_w * np.exp(np.clip(l_anc - L, -700.0, 700.0))
        rho_m = (m_loss + keep_w) / a ** 3
        y = g * m_loss / a ** 3
        rho_x = np.zeros_like(n)
        if sink == "L" and closure:
            G = cumulative_trapezoid(y, n, initial=0.0)
            need = 1 - rho_m[i0] - orad
            kappa = G[i0] / need - 1
            rho_l = G / G[i0] * need if G[i0] > 0 else np.zeros_like(n)
        elif sink == "d":                                          # 가족 D: 무게와 꼴은 그대로, 렌더링만 잃는다
            rho_m = np.full_like(n, om) / a ** 3
            rho_l = (1 - om - orad) * fw
        elif sink == "L":
            G = np.empty_like(n)                                   # 오늘에서 바깥으로 쌓아 상쇄 오차를 피한다
            G[i0:] = cumulative_trapezoid(y[i0:], n[i0:], initial=0.0)
            G[:i0 + 1] = cumulative_trapezoid(y[:i0 + 1][::-1], n[:i0 + 1][::-1], initial=0.0)[::-1]
            rho_l = (1 - rho_m[i0] - orad) * fw + G
        else:
            rho_x = cumulative_trapezoid(y * a ** 4, n, initial=0.0) / a ** 4
            rho_l = (1 - rho_m[i0] - orad - rho_x[i0]) * fw
        lam_new = float(rho_l[i0])
        e2 = orad / a ** 4 + rho_m + rho_l + rho_x
        if lam_new <= 0 or not np.all(np.isfinite(e2)) or not np.all(e2 > 0):
            ok = False
            break
        e_new = np.sqrt(e2)
        err = float(np.max(np.abs(e_new / E - 1)))
        E = e_new if it < 40 else 0.5 * (E + e_new)
        lam0 = lam_new
        if err < 1e-11:
            break
    return {"ok": ok and err < 1e-8, "err": err, "it": it, "n": n, "a": a, "E": E, "H": h0 * E, "t": t,
            "m_loss": m_loss, "lossy_w": lossy_w, "rho_m": rho_m, "rho_l": rho_l, "rho_x": rho_x, "lam0": lam0,
            "om0": float(rho_m[i0]), "kappa": kappa, "orad": orad}


@lru_cache(maxsize=None)
def observables(shape: str, sink: str, reading: str, lossy: str, beta: float, closure: bool = False) -> dict:
    B = _base()
    s = solve(shape, sink, reading, lossy, beta, closure)
    if not s["ok"]:
        return {"ok": False, "kappa": s["kappa"]}
    n, a, H, t, m_loss = s["n"], s["a"], s["H"], s["t"], s["m_loss"]
    i0, ib, h, q, om = B["i0"], B["i_bao"], B["h"], B["q"], B["om"]
    f_loss = float(np.interp(B["n_star"], n, m_loss)) / s["lossy_w"]
    wb = B["wb"] * f_loss
    wc = (om - q) * h * h * (f_loss if lossy == "all" else 1.0) - B["w_nu"]
    if sink == "d":                                                # 총 물질 보존: 이른 암흑 몫 = om − 이른 바리온
        wc = (om - q * f_loss) * h * h - B["w_nu"]
    if not (wb > 0 and wc > 0):
        return {"ok": False, "kappa": s["kappa"]}
    n_star = -math.log1p(DV.z_star(wb, wb + wc))
    n_drag = -math.log1p(BR.z_drag(wb, wb + wc))
    if not (n[0] + 1 < n_drag < 0 and n[0] + 1 < n_star < 0):          # 이른 밀도가 비현실적인 해는 무효
        return {"ok": False, "kappa": s["kappa"]}
    rb = 0.75 * B["wb"] * (m_loss / s["lossy_w"]) * a / DV.OMEGA_GAMMA_H2
    rs_cum = cumulative_trapezoid(C_KM_S / np.sqrt(3 * (1 + rb)) / (a * H), n, initial=0.0)
    dc_cum = cumulative_trapezoid(C_KM_S / (a * H), n, initial=0.0)
    d_star = float(dc_cum[i0] - np.interp(n_star, n, dc_cum))
    h0 = 100 * h
    dm = (dc_cum[i0] - dc_cum[ib]) * h0 / C_KM_S
    dh = h0 / H[ib]
    b = np.array([{"dm": x, "dh": y, "dv": (z * x * x * y) ** (1 / 3)}[k]
                  for x, y, z, k in zip(dm, dh, R.BAO_Z, R.BAO_KIND)])
    m_rec = float(np.interp(B["n_star"], n, m_loss))
    m_z1 = float(np.interp(-math.log(2.0), n, m_loss))
    anchor = float(np.interp(B["n_star"], n, s["rho_m"] * a ** 3))
    dark = (H / h0) ** 2 - anchor / a ** 3 - s["orad"] / a ** 4
    with np.errstate(divide="ignore", invalid="ignore"):
        w = -1 - np.gradient(np.log(np.abs(dark)), n) / 3
    return {"ok": True, "theta": 100 * float(np.interp(n_star, n, rs_cum)) / d_star,
            "rd": float(np.interp(n_drag, n, rs_cum)), "b": b, "wb": wb, "wc": wc, "hl": math.sqrt(s["lam0"]),
            "t_bao": t[ib], "t0": float(t[i0]), "om0": s["om0"], "lost_rec": 1 - m_loss[i0] / m_rec,
            "lost_z1": 1 - m_loss[i0] / m_z1, "w": tuple(float(np.interp(-math.log1p(z), n, w)) for z in Z_W),
            "e_sn": np.interp(-np.log1p(Z_SN), n, H / h0), "kappa": s["kappa"], "it": s["it"],
            "min_vac": float(np.min(s["rho_l"][:i0 + 1]))}


# ------------------------------------------------------------------ 채점
def score(shape: str, sink: str, reading: str, lossy: str, beta: float, closure: bool = False) -> dict:
    B = _base()
    o = observables(shape, sink, reading, lossy, float(beta), closure)
    if not o["ok"]:
        return {"ok": False, "chi2": math.inf, "V39": math.inf, "killed": True, "obs": o}
    o0 = observables(*BASE_KEY)
    rows = B["rows"]
    pred = {"100 theta*": rows["100 theta*"]["pred"] + o["theta"] - o0["theta"],
            "omega_b h^2": o["wb"], "omega_c h^2": o["wc"]}
    k = o["hl"] / o0["hl"]
    th0 = B["th0"] * k * o["t0"] / o0["t0"]
    th_z = B["th_z"] * k * o["t_bao"] / o0["t_bao"]
    direct = B["ring"] / math.cos(th0)
    pred.update({key: direct for key in H0_KEYS})
    pulls = {key: (v - rows[key]["obs"]) / rows[key]["sigma"] for key, v in pred.items()}
    rd = B["rd"] * o["rd"] / o0["rd"]
    b = B["b"] * o["b"] / o0["b"]
    r = C_KM_S / (100 * B["h"] * rd) * b * (1 - o["om0"] * (1 - np.cos(th_z))) - R.BAO_Y
    bao = float(r @ R.BAO_CINV @ r)
    chi = B["chi_rest"] + sum(p * p for p in pulls.values()) + bao
    v39 = math.sqrt(chi / B["N"])
    killed = v39 > V_KILL or th0 >= math.pi / 2 or any(abs(p) > PULL_KILL for p in pulls.values())
    return {"ok": True, "chi2": chi, "V39": v39, "bao": bao, "pulls": pulls, "direct_H0": direct, "th0": th0,
            "killed": killed, "age": B["t0"] * o["t0"] / o0["t0"], "obs": o}


def baseline_check() -> dict:
    """β = 0이면 TT 기준(§43.53)과 같아야 한다."""
    s = score(*BASE_KEY)
    tt = TTM.score("V-dS")
    return {"chi2": s["chi2"], "V39": s["V39"], "TT_V39": tt["V39"], "diff": s["V39"] - tt["V39"]}


@lru_cache(maxsize=None)
def fit(shape: str, sink: str, reading: str, lossy: str) -> dict:
    """β ≥ 0 적합, Δχ² = 1 구간, 한쪽 95% 상한."""
    chi0 = score(shape, sink, reading, lossy, 0.0)["chi2"]
    f = lambda lb: score(shape, sink, reading, lossy, 10.0 ** lb)["chi2"]
    grid = np.log10(BETAS)
    chis = np.array([f(lb) for lb in grid])
    j = int(np.argmin(chis))
    if chis[j] >= chi0:
        beta, best = 0.0, chi0
    else:
        lo, hi = grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)]
        res = minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": 1e-4})
        beta, best = (10.0 ** res.x, float(res.fun)) if res.fun < chis[j] else (10.0 ** grid[j], float(chis[j]))

    def crossing(level: float, upward: bool) -> float:
        if upward:
            idx = [i for i in range(len(grid)) if 10.0 ** grid[i] > beta and chis[i] > level]
            if not idx:
                return math.inf
            i = idx[0]
            lo = max(grid[i - 1], math.log10(beta)) if (i > 0 and beta > 0) else (grid[i - 1] if i > 0 else grid[0] - 3)
            return 10.0 ** brentq(lambda lb: f(lb) - level, lo, grid[i], xtol=1e-6)
        if beta == 0 or chi0 <= level:
            return 0.0
        idx = [i for i in range(len(grid)) if 10.0 ** grid[i] < beta and chis[i] > level]
        i = idx[-1]
        return 10.0 ** brentq(lambda lb: f(lb) - level, grid[i], math.log10(beta), xtol=1e-6)
    b95 = crossing(best + UPPER, True)
    d = best - chi0
    out = {"beta": beta, "d_chi2": d, "V39": math.sqrt(best / _base()["N"]), "lo1": crossing(best + 1, False),
           "hi1": crossing(best + 1, True), "beta_95": b95,
           "verdict": "evidence" if d <= EVIDENCE else ("hint" if d < HINT else "none")}
    for tag, bb in (("best", beta), ("95", b95)):
        o = observables(shape, sink, reading, lossy, float(bb)) if math.isfinite(bb) else {"ok": False}
        out[f"lost_rec_{tag}"] = o["lost_rec"] if o["ok"] else math.nan
        out[f"lost_z1_{tag}"] = o["lost_z1"] if o["ok"] else math.nan
    return out


def scan() -> dict:
    return {v: fit(v[1], v[2], v[3], FAMILIES[v[0]][1]) for v in variants()}


def candidate_table() -> dict:
    """K1(β = 1)과 K3(원리 후보)."""
    out = {}
    for fam, shape, sink, rd in variants():
        lossy = FAMILIES[fam][1]
        chi0 = score(shape, sink, rd, lossy, 0.0)["chi2"]
        for name, beta in candidates().items():
            s = score(shape, sink, rd, lossy, beta)
            out[(fam, shape, sink, rd, name)] = {
                "beta": beta, "V39": s["V39"], "d_chi2": s["chi2"] - chi0, "killed": s["killed"],
                "max_pull": max((abs(p) for p in s["pulls"].values()), default=math.inf) if s["ok"] else math.inf,
                "worst": max(s["pulls"], key=lambda k: abs(s["pulls"][k])) if s["ok"] else "invalid",
                "lost_rec": s["obs"].get("lost_rec", math.nan), "lost_z1": s["obs"].get("lost_z1", math.nan)}
    return out


def closure() -> dict:
    """가족 C: Λ_bare = 0. κ(β) = (쌓인 진공 / 필요한 진공) − 1 = 0의 근."""
    out = {}
    chi0 = score(*BASE_KEY)["chi2"]
    for lossy in ("all", "b"):
        for rd in READINGS:
            kap = lambda lb: solve("TT", "L", rd, lossy, 10.0 ** lb, True)["kappa"]
            lb = brentq(kap, -4.0, 1.5, xtol=1e-7)
            beta = 10.0 ** lb
            s = score("TT", "L", rd, lossy, beta, True)
            o = s["obs"]
            out[(lossy, rd)] = {"beta_C": beta, "V39": s["V39"], "d_chi2": s["chi2"] - chi0, "killed": s["killed"],
                                "pulls": {k: round(v, 2) for k, v in s.get("pulls", {}).items()},
                                "bao": s.get("bao"), "lost_rec": o.get("lost_rec"), "lost_z1": o.get("lost_z1"),
                                "w": o.get("w"), "beyond_probability": beta > 1}
    return out


def signed_scan() -> dict:
    """진단: A·S-Λ에서 β의 부호를 풀면 자료는 어느 방향(음수 = 진공→물질)을 원하는가."""
    grid = np.concatenate([-np.logspace(-1, -6, 26), [0.0], np.logspace(-6, -1, 26)])
    out = {}
    for shape in SHAPES:
        for rd in READINGS:
            chi = np.array([score(shape, "L", rd, "all", float(b))["chi2"] for b in grid])
            j = int(np.argmin(chi))
            out[(shape, rd)] = {"beta": float(grid[j]), "d_chi2": float(chi[j] - chi[26])}
    return out


# ------------------------------------------------------------------ 보강
def _growth(s: dict) -> float:
    """D(a = 1), D = a(a = 10⁻³)에서 시작. 잃는 비율이 질량에 비례하므로(δQ/Q = δ) 밀도 대비에 희석 항이 없다."""
    n = s["n"]
    e2 = s["rho_m"] + s["rho_l"] + s["rho_x"]
    om_a = s["rho_m"] / e2
    dln = np.gradient(np.log(e2), n) / 2
    rhs = lambda x, y: [y[1], -(2 + np.interp(x, n, dln)) * y[1] + 1.5 * np.interp(x, n, om_a) * y[0]]
    return float(solve_ivp(rhs, (math.log(1e-3), 0.0), [1e-3, 1e-3], rtol=1e-9, atol=1e-13).y[0, -1])


def rows43(shape: str, sink: str, reading: str, lossy: str, beta: float, closure: bool = False) -> dict:
    """43행: 39행 + S8 두 행(+ 가지 행). S8 = 기준 × 이른 전달 비 × 늦은 성장 비 × √(오늘 Ω_m / CE Ω_m)."""
    B = _base()
    c = B["c"]
    sc = score(shape, sink, reading, lossy, beta, closure)
    s0c = score(*BASE_KEY)
    o = sc["obs"]
    raw = lambda wb, wc: GR.sigma8_raw(R.scalar_amplitude(c) * 1e-9, 1 - 2 / c["Ne"], wb, wc, B["h"])
    g_in = lambda wb, wc: GR.growth_today((wb + wc) / B["h"] ** 2)
    early = raw(o["wb"], o["wc"]) / raw(B["wb"], B["wc"]) * g_in(B["wb"], B["wc"]) / g_in(o["wb"], o["wc"])
    late = _growth(solve(shape, sink, reading, lossy, beta, closure)) / _growth(solve(*BASE_KEY))
    ratio = early * late * math.sqrt(o["om0"] / B["om"])

    def chi43(s: dict, rr: float) -> tuple[float, dict]:
        s8 = {k: (B["rows43"][k]["pred"] * rr - B["rows43"][k]["obs"]) / B["rows43"][k]["sigma"] for k in B["s8_keys"]}
        return B["chi_rest43"] + sum(p * p for p in s["pulls"].values()) + sum(p * p for p in s8.values()) + s["bao"], s8
    chi, s8 = chi43(sc, ratio)
    chi_0, _ = chi43(s0c, 1.0)
    return {"S8_ratio": ratio, "s8_pulls": {k: round(v, 2) for k, v in s8.items()},
            "rmse43": math.sqrt(chi / B["N43"]), "rmse43_base": math.sqrt(chi_0 / B["N43"]), "d_chi2_43": chi - chi_0}


@lru_cache(maxsize=1)
def _sn_ref() -> dict:
    B = _base()
    f = VT.density(B["c"], VT.ADOPTED_NU)
    z = SN._data()[0]
    w2 = lambda zz: np.sqrt(B["om"] * (1 + zz) ** 3 + (1 - B["om"]) * np.array([f(1 / (1 + x)) for x in np.atleast_1d(zz)]))
    return {"z": z, "w2": w2, "chi_w2": SN.profiled_chi2(SN.comoving_distance(z, w2)), "best": SN.best_lcdm()["chi2"]}


def supernova(shape: str, sink: str, reading: str, lossy: str, beta: float, closure: bool = False) -> dict:
    """§43.37 규칙: 최적 평탄 ΛCDM 대비 Δχ² ≤ 4 통과, > 9 기각. W2 기준과의 차이도 적는다."""
    ref = _sn_ref()
    o, o0 = observables(shape, sink, reading, lossy, float(beta), closure), observables(*BASE_KEY)
    ratio = o["e_sn"] / o0["e_sn"]
    e = lambda zz: ref["w2"](zz) * np.interp(zz, Z_SN, ratio)
    chi = SN.profiled_chi2(SN.comoving_distance(ref["z"], e))
    d = chi - ref["best"]
    return {"chi2": chi, "d_vs_best": d, "d_vs_W2": chi - ref["chi_w2"],
            "verdict": "pass" if d <= 4 else ("fail" if d > 9 else "tension")}


def scan43(betas: np.ndarray = np.logspace(-4, 0, 25)) -> dict:
    """[목격 후 확장] 39행 결과와 문자 TT(r, b)의 S8 기각을 본 뒤 43행으로 β를 다시 맞춘다. 판정은 K2와 같은 문턱."""
    out = {}
    for fam, shape, sink, rd in variants():
        lossy = FAMILIES[fam][1]
        d = [(b, rows43(shape, sink, rd, lossy, float(b))["d_chi2_43"]) for b in betas
             if score(shape, sink, rd, lossy, float(b))["ok"]]
        b, dc = min(d, key=lambda x: x[1]) if d else (math.nan, math.nan)
        out[(fam, shape, sink, rd)] = {"beta": b if dc < 0 else 0.0, "d_chi2_43": min(dc, 0.0),
                                       "verdict": "evidence" if dc <= EVIDENCE else ("hint" if dc < HINT else "none")}
    return out


def geometric_exit() -> dict:
    """새 물리 없이: 우리 사건 지평(렌더링 가능한 영역) 안의 공동 물질이 기하적으로 빠져나간 비율(β = 0 배경)."""
    B = _base()
    s = solve(*BASE_KEY[:5])
    n, a, H = s["n"], s["a"], s["H"]
    f = C_KM_S / (a * H)
    chi_e = np.trapezoid(f, n) - cumulative_trapezoid(f, n, initial=0.0) + C_KM_S * math.exp(-n[-1]) / H[-1]
    ce = lambda x: float(np.interp(x, n, chi_e))
    ah = a * H
    i_min = int(np.argmin(np.where(n < 0.5, ah, np.inf)))
    return {"chi_e_today_Mpc": ce(0.0), "chi_e_rec_Mpc": ce(B["n_star"]),
            "exited_EH_since_rec": 1 - (ce(0.0) / ce(B["n_star"])) ** 3,
            "exited_EH_since_z1": 1 - (ce(0.0) / ce(-math.log(2.0))) ** 3,
            "accel_onset_z": float(np.exp(-n[i_min]) - 1),
            "exited_hubble_since_onset": 1 - (ah[i_min] / ah[B["i0"]]) ** 3}


def neither() -> dict:
    """가족 D(무게는 두고 렌더링만 잃음). (b)가 행에 보이지 않는지 확인하고, (a)는 적합과 상한을 낸다."""
    out = {}
    for shape in SHAPES:
        chi0 = score(shape, "d", "b", "b", 0.0)["chi2"]
        invisible = max(abs(score(shape, "d", "b", "b", b)["chi2"] - chi0) for b in (1e-3, 1e-2, 0.1, 1.0))
        f = fit(shape, "d", "a", "b")
        lit_a = score(shape, "d", "a", "b", 1.0)
        out[shape] = {"invisible_b_max_dchi2": invisible, "d_chi2_a": f["d_chi2"], "beta_95_a": f["beta_95"],
                      "lost_rec_95_a": f["lost_rec_95"], "literal_a_killed": lit_a["killed"],
                      "literal_b_rendered_today": 1 - observables(shape, "d", "b", "b", 1.0)["lost_rec"],
                      "verdict_a": "evidence" if f["d_chi2"] <= EVIDENCE_D else ("hint" if f["d_chi2"] < HINT else "none")}
    return out


def lab_bound() -> dict:
    """평균장 소실이면 실험실 핵자도 같은 비율로 렌더링을 잃는다. 우주 나이 동안 허용되는 바리온 소실 비율의 상한."""
    age_yr = _base()["t0"] * 1e9
    cosmo = fit("EH", "L", "a", "all")["lost_rec_95"]
    frac = age_yr / TAU_P_INVISIBLE_YR
    return {"tau_yr": TAU_P_INVISIBLE_YR, "max_baryon_fraction_over_age": frac,
            "cosmology_bound_EH": cosmo, "orders_stronger": math.log10(cosmo / frac)}


def verdict() -> dict:
    sc = scan()
    tab = candidate_table()
    cl = closure()
    lit = {k[:4]: v["killed"] for k, v in tab.items() if k[4] == "1" and k[1] in LITERAL}
    best = min(sc.items(), key=lambda kv: kv[1]["d_chi2"])
    return {"K1_literal_survivors": [k for k, dead in lit.items() if not dead],
            "K2_best": (best[0], round(best[1]["d_chi2"], 3)),
            "K2_any_evidence": any(v["verdict"] == "evidence" for v in sc.values()),
            "K2_any_hint": any(v["verdict"] == "hint" for v in sc.values()),
            "K3_adopted": [k for k, v in tab.items() if not v["killed"] and v["d_chi2"] <= ADOPT],
            "K3_survivors": sum(not v["killed"] for v in tab.values()),
            "K4_all_killed": all(v["killed"] for v in cl.values()),
            "K4_beta_C": {k: round(v["beta_C"], 4) for k, v in cl.items()}}


def main() -> None:
    print("baseline check:", {k: round(v, 5) for k, v in baseline_check().items()})
    print(f"{'variant':16s} {'beta':>9s} {'dchi2':>7s} {'beta95':>9s} {'lost rec':>8s} {'lost z1':>8s}  (95% upper)")
    for (fam, shape, sink, rd), f in scan().items():
        print(f"{fam} {shape:3s} {sink} {rd}        {f['beta']:9.3g} {f['d_chi2']:+7.3f} {f['beta_95']:9.3g} "
              f"{f['lost_rec_95']:8.4f} {f['lost_z1_95']:8.4f}  {f['verdict']}")
    print("closure C:", {k: (round(v["beta_C"], 4), round(v["V39"], 3), v["killed"]) for k, v in closure().items()})
    print("geometric exit:", {k: round(v, 4) for k, v in geometric_exit().items()})
    lit = rows43("TT", "r", "b", "all", 1.0)
    print("literal TT fade (r, b) in 43 rows:", lit["s8_pulls"], f"d_chi2_43={lit['d_chi2_43']:+.2f}")
    print("verdict:", verdict())
    for shape, v in neither().items():
        print(f"neither D {shape}:", {k: (round(x, 5) if isinstance(x, float) else x) for k, x in v.items()})
    print("lab bound:", {k: f"{x:.3g}" for k, x in lab_bound().items()})


if __name__ == "__main__":
    main()
