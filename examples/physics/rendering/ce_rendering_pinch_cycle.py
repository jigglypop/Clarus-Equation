"""끼임 순환 PC — 분별의 순환이 우주의 순환이라는 추측을 끼임의 촉진·소멸 모형으로 채점한다. 원장: 43장 §43.102.
예측값을 바꾸지 않는다.

사용자 추측(2026-09-25): 우주의 시작은 분별의 시작이다(시작공리). 인과(= 끼임, 참조/5_유도 §0.2, 07장 §7.6)가 쌓여
절정에 이르고, 분별이 없어질 때까지 흐려진 뒤 다시 시작한다. 이 분별의 순환이 우주의 순환이다. 빛은 전파(렌더링)
한계속도다. 사용자 요청: 식을 최대한 유연하게, 경우의 수를 넓게, RMSE를 기준으로.

모형(§0.2.3 이웃 촉진의 평균장). 켜진 끼임의 비율 A(t):
    dA/dt = κ ν(a) A (1 − A) − γ A,   A(시작) = seed(첫 끼임)
ν(a)는 빛원뿔로 닿는 이웃의 수(오늘 1로 정규화한 공동 부피)다. 빛이 전파 한계라서 인과 지평이 이웃을 정한다.
기록 위상은 쌓인 분별에 비례한다: Φ(t) = ω ∫ A dt, 기울기 θ = Φ/2(H1). 채점은 §43.53 score와 같은 39행이다
(H₀ 넷은 나이테/cos θ₀, BAO는 1 − Ω_m(1 − cos θ(z))).

계산 전에 적은 판본 축:
    ν: H(허블 구 공동 부피 (aH)⁻³), E(사건 지평 공동 부피), P(입자 지평 공동 부피), 1(대조: 이웃 수 불변)
    ω: 한 순환 전체의 위상이 2π(한 바퀴), π, π/2가 되도록 정한다(연속 매개변수 없음). 또는 자유(연속 매개변수 1).
    연속 매개변수: κ, γ(로그 격자 각 36점, 1/Gyr), seed ∈ {1e-3, 1e-6, 1e-9}
    → 16판본. 매개변수 수 k = 3(κ, γ, seed) 또는 4(ω 자유 추가).
기준: TT(V-dS, 열적 시간 시계, 위상 시계에 자유 매개변수 0). V39 = 0.8407.
판정:
    K1: V39 > 0.909 또는 H₀ 행 |pull| > 3이면 그 판본 기각(§43.53과 같다).
    K2(순환 성립): 최적점에서 A가 절정 뒤 적분 끝까지 절정의 1e-3 아래로 떨어지지 않으면 “순환 불성립”(추측 위반)으로 적는다.
    K3(우위): AIC = χ² + 2k로 TT(k = 0)와 비교한다. ΔAIC < −2 우위, |ΔAIC| ≤ 2 동등, > 2 열세. BIC(χ² + k ln 39)도 적는다.
    모든 맞춤은 개발 자료 위의 적합이며 holdout이 아니다. 판본 16개를 훑었으므로 최선값의 우위에는 look-elsewhere가 붙는다.
정정(첫 실행 결과를 본 뒤, 공개): 최선점 다수가 A가 씨앗에서 한 번도 늘지 않고 줄어드는 퇴화 해였다(절정 시각 0).
    추측의 “인과가 쌓여 절정에 이른다”를 K2에 명시해 진짜 순환 = (절정 ≥ 씨앗의 100배) 그리고 (절정 뒤 1e-3 아래로 소멸)으로
    고친다. 격자 끝에 걸린 최적점이 있어 κ·γ 격자를 넓혔다(κ 1e-2–1e3, γ 1e-5–1e1, 각 48점). 퇴화 해의 결과도 함께 보고한다.
극한(결과를 본 뒤 계산, 공개): 허블 구 판본의 최적 κ가 격자 끝이라 κ → ∞(순간 평형) 극한을 따로 푼다.
    A = max(0, 1 − 1/R0(t)), R0(t) = R0_today ν(t). 매개변수는 R0_today 하나(k = 1)다. 분별의 시작은 R0가 1을 넘을 때,
    소멸은 1 아래로 떨어질 때다.

python -B -m examples.physics.rendering.ce_rendering_pinch_cycle
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import cumulative_trapezoid

from examples.physics.rendering import ce_rendering_gradient as GD
from examples.physics.rendering import ce_rendering_nu_ledger as NL
from examples.physics.rendering import ce_rendering_registry as R
from examples.physics.rendering import ce_rendering_thermal_time as TTM
from examples.physics.rendering import ce_rendering_vacuum_tilt as VT

KAPPAS = np.logspace(-2, 3, 48)              # 1/Gyr
GAMMAS = np.logspace(-5, 1, 48)              # 1/Gyr
RISE = 100.0                                 # 진짜 순환: 절정 ≥ 씨앗 × RISE
SEEDS = (1e-3, 1e-6, 1e-9)
NORMS = {"2pi": 2 * math.pi, "pi": math.pi, "pi/2": math.pi / 2, "free": None}
OMEGA_FREE = np.logspace(-3, 1, 81)          # 자유 ω의 격자(1/Gyr당 rad)
NU_KINDS = ("H", "E", "P", "1")
N_MIN, N_MAX, DN = -13.8, 175.0, 0.01
N_BAO = -np.log1p(R.BAO_Z)


def background() -> dict:
    """N = ln a 격자에서 t, H, 공동 허블 반지름, 사건·입자 지평(단위 Gyr, Gly; c = 1).
    BAO 적색편이와 오늘(N = 0)을 격자에 정확히 넣는다."""
    _, om, h0, hl = TTM._cosmo()
    n = np.unique(np.concatenate([np.arange(N_MIN, N_MAX + DN / 2, DN), N_BAO, [0.0]]))
    a = np.exp(n)
    hub = h0 * np.sqrt(om / a ** 3 + 1 - om)
    t = cumulative_trapezoid(1 / hub, n, initial=0.0) + 2 / (3 * hub[0])          # 물질 시대 앞머리
    f = np.exp(-n) / hub                                                         # dχ/dN
    chi_p = cumulative_trapezoid(f, n, initial=0.0) + 2 * math.sqrt(a[0]) / (h0 * math.sqrt(om))
    tail = np.exp(-n[-1]) / hl
    chi_e = (np.trapezoid(f, n) - cumulative_trapezoid(f, n, initial=0.0)) + tail
    i0 = int(np.searchsorted(n, 0.0))
    nu = {"H": (1 / (a * hub) / (1 / hub[i0])) ** 3, "E": (chi_e / chi_e[i0]) ** 3,
          "P": (chi_p / chi_p[i0]) ** 3, "1": np.ones_like(a)}
    return {"n": n, "a": a, "t": t, "H": hub, "nu": nu, "i0": i0, "om": om, "hl": hl}


def _scorer():
    """§43.53 score와 같은 39행 채점을 θ₀·θ(z) 배열에 벡터로 적용한다."""
    c, om, _, _ = TTM._cosmo()
    ring = 100 * NL.early_densities(c)[2]
    rd, h = NL.rd_and_h(c)
    b = VT.bao_vectors(om, VT.density(c, VT.ADOPTED_NU))
    pref = GD.C_KM_S / (100 * h * rd) * b
    base = VT.score(VT.ADOPTED_NU)
    h0_rows = [(o["obs"], o["sigma"], o["key"]) for o in base["rows"] if o["key"].startswith("H0 ")]
    chi_rest = sum(o["pull"] ** 2 for o in base["rows"] if not o["key"].startswith("H0 "))

    def score(th0: np.ndarray, thz: np.ndarray) -> dict:
        bad = (th0 >= math.pi / 2) | np.any(thz >= math.pi / 2, axis=-1) | (th0 < 0)
        direct = ring / np.cos(np.clip(th0, 0, 1.5))
        pulls = np.stack([(direct - o) / s for o, s, _ in h0_rows], axis=-1)
        factor = 1 - om * (1 - np.cos(np.clip(thz, 0, 1.5)))
        r = pref * factor - R.BAO_Y
        bao = np.einsum("...i,ij,...j->...", r, R.BAO_CINV, r)
        chi = chi_rest + np.sum(pulls ** 2, axis=-1) + bao
        chi = np.where(bad, np.inf, chi)
        return {"chi2": chi, "V39": np.sqrt(chi / base["N"]), "h0_pulls": pulls, "bao": bao, "direct": direct}
    return score, [k for _, _, k in h0_rows], base["N"]


def tt_baseline() -> dict:
    """검산: 같은 채점기에 TT 위상(Φ = H_Λ t)을 넣으면 §43.53의 V39가 나와야 한다."""
    bg = background()
    score, _, _ = _scorer()
    phi = bg["hl"] * bg["t"]
    a_bao = 1 / (1 + R.BAO_Z)
    thz = np.interp(a_bao, bg["a"], phi) / 2
    s = score(np.array(phi[bg["i0"]] / 2), thz)
    return {"theta0": float(phi[bg["i0"]] / 2), "V39": float(s["V39"]), "chi2": float(s["chi2"])}


def _solve(bg: dict, nu: np.ndarray, seed: float) -> dict:
    """모든 (κ, γ)를 한꺼번에 푼다. 계수를 한 걸음 안에서 상수로 두는 로지스틱 정확해(무조건 안정)."""
    k, g = np.meshgrid(KAPPAS, GAMMAS, indexing="ij")
    A = np.full(k.shape, seed)
    integ = np.zeros_like(A)
    idx_bao = np.searchsorted(bg["n"], N_BAO)
    rec = {int(i): None for i in set(idx_bao.tolist()) | {bg["i0"]}}
    peak = A.copy()
    t_peak = np.zeros_like(A)
    t = bg["t"]
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        nv = 0.5 * (nu[i] + nu[i + 1])
        r = k * nv - g
        bb = k * nv
        e = np.exp(np.clip(r * dt, -700, 700))
        with np.errstate(divide="ignore", invalid="ignore"):
            nxt = np.where(np.abs(r * dt) > 1e-12, r * A * e / (r + bb * A * (e - 1)), A / (1 + bb * A * dt))
        nxt = np.clip(np.nan_to_num(nxt, nan=0.0, posinf=1.0), 0.0, 1.0)
        integ = integ + 0.5 * (A + nxt) * dt
        A = nxt
        up = A > peak
        peak = np.where(up, A, peak)
        t_peak = np.where(up, t[i + 1], t_peak)
        if (i + 1) in rec:
            rec[i + 1] = integ.copy()
    return {"k": k, "g": g, "I_end": integ, "A_end": A, "peak": peak, "t_peak": t_peak,
            "I0": rec[bg["i0"]], "I_bao": np.stack([rec[int(i)] for i in idx_bao], axis=-1)}


def scan(genuine_only: bool = True) -> dict:
    """16판본 × (κ, γ, seed[, ω]) 격자의 최선점과 AIC·BIC. genuine_only면 진짜 순환(쌓임 + 소멸)만 채점한다."""
    bg = background()
    score, h0_keys, n_rows = _scorer()
    tt = tt_baseline()
    out = {}
    sols = {(nu_k, s): _solve(bg, bg["nu"][nu_k], s) for nu_k in NU_KINDS for s in SEEDS}
    for nu_k in NU_KINDS:
        for norm_k, total in NORMS.items():
            best = None
            for s in SEEDS:
                sol = sols[(nu_k, s)]
                cyc = (sol["A_end"] < 1e-3 * sol["peak"]) & (sol["peak"] >= RISE * s)
                if total is None:
                    w = OMEGA_FREE[:, None, None]
                    th0 = w * sol["I0"][None] / 2
                    thz = w[..., None] * sol["I_bao"][None] / 2
                else:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        w = np.where(sol["I_end"] > 0, total / sol["I_end"], 0.0)
                    th0 = w * sol["I0"] / 2
                    thz = w[..., None] * sol["I_bao"] / 2
                sc = score(th0, thz)
                chi = np.where(np.broadcast_to(cyc, sc["chi2"].shape), sc["chi2"], np.inf) if genuine_only else sc["chi2"]
                if not np.isfinite(chi).any():
                    continue
                j = np.unravel_index(int(np.argmin(chi)), chi.shape)
                if best is None or chi[j] < best["chi2"]:
                    jk = j[-2:]
                    best = {"chi2": float(chi[j]), "V39": float(sc["V39"][j]), "seed": s,
                            "kappa": float(sol["k"][jk]), "gamma": float(sol["g"][jk]),
                            "omega": float(OMEGA_FREE[j[0]]) if total is None else float(w[jk]),
                            "theta0": float(th0[j]), "direct_H0": float(sc["direct"][j]),
                            "h0_pulls": dict(zip(h0_keys, np.round(sc["h0_pulls"][j], 2).tolist())),
                            "bao_chi2": float(sc["bao"][j]), "cycle_closes": bool(cyc[jk]),
                            "t_peak_Gyr": float(sol["t_peak"][jk]),
                            "today_fraction": float(sol["I0"][jk] / sol["I_end"][jk]) if sol["I_end"][jk] > 0 else float("nan")}
            if best is None:
                out[f"{nu_k}|{norm_k}"] = None
                continue
            kpar = 3 + (1 if total is None else 0)
            best["k"] = kpar
            best["dAIC"] = best["chi2"] + 2 * kpar - tt["chi2"]
            best["dBIC"] = best["chi2"] + kpar * math.log(n_rows) - tt["chi2"]
            best["K1_killed"] = best["V39"] > 0.909 or any(abs(v) > 3 for v in best["h0_pulls"].values())
            out[f"{nu_k}|{norm_k}"] = best
    return {"TT": tt, "variants": out}


def trajectory(nu_k: str, kappa: float, gamma: float, seed: float) -> dict:
    """한 매개변수 조합의 A(t) 궤적(구조 판정용): 절정, 절정 시각, 끝 값."""
    bg = background()
    nu, t = bg["nu"][nu_k], bg["t"]
    A, peak, t_peak = seed, seed, t[0]
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        nv = 0.5 * (nu[i] + nu[i + 1])
        r, bb = kappa * nv - gamma, kappa * nv
        e = math.exp(max(min(r * dt, 700.0), -700.0))
        A = r * A * e / (r + bb * A * (e - 1)) if abs(r * dt) > 1e-12 else A / (1 + bb * A * dt)
        A = min(max(A, 0.0), 1.0)
        if A > peak:
            peak, t_peak = A, t[i + 1]
    return {"peak": peak, "t_peak_Gyr": float(t_peak), "A_end": A}


def instant_equilibrium(r0_grid: np.ndarray = np.logspace(0, 9, 361)) -> dict:
    """κ → ∞ 극한: A = max(0, 1 − 1/(R0_today ν)). 판본 H·E × 위상 총량 2π·π·π/2, k = 1."""
    bg = background()
    score, _, n_rows = _scorer()
    tt = tt_baseline()
    idx_bao = np.searchsorted(bg["n"], N_BAO)
    i0, t = bg["i0"], bg["t"]
    out = {}
    for nu_k in ("H", "E"):
        A = np.clip(1 - 1 / (r0_grid[:, None] * bg["nu"][nu_k][None, :]), 0, 1)
        integ = cumulative_trapezoid(A, t, axis=1, initial=0.0)
        for norm_k in ("2pi", "pi", "pi/2"):
            w = np.where(integ[:, -1] > 0, NORMS[norm_k] / integ[:, -1], 0.0)
            th0 = w * integ[:, i0] / 2
            sc = score(th0, w[:, None] * integ[:, idx_bao] / 2)
            chi = np.where(A[:, -1] == 0, sc["chi2"], np.inf)
            j = int(np.argmin(chi))
            on = np.nonzero(A[j] > 0)[0]
            out[f"{nu_k}|{norm_k}"] = {
                "V39": float(sc["V39"][j]), "chi2": float(chi[j]), "dAIC": float(chi[j] + 2 - tt["chi2"]),
                "dBIC": float(chi[j] + math.log(n_rows) - tt["chi2"]), "R0_today": float(r0_grid[j]),
                "theta0": float(th0[j]), "direct_H0": float(sc["direct"][j]), "t_start_Gyr": float(t[on[0]]),
                "t_peak_Gyr": float(t[int(np.argmax(A[j]))]), "t_end_Gyr": float(t[on[-1]]),
                "today_fraction": float(integ[j, i0] / integ[j, -1])}
    return {"TT": tt, "variants": out}


def zero_parameter() -> dict:
    """매개변수 0개 판본 — 식 안에서 고른다(사용자 “최대한 식 안에서 골라봐”, 2026-09-25).

    계산 전에 적은 원리 선택(판정은 이 하나로만 한다):
        ν = 허블 구(추측의 모양 “쌓임 → 절정 → 소멸”; 사건 지평은 절정이 시작이라 쌓임이 없다),
        R₀(t) = D·ν_H(t)/max ν_H(인과의 절정에서 끼임이 제 깊이 D = 3 + δ만큼 번식, BR1의 평균 자손),
        한 순환의 위상 = 2π(한 바퀴).
    비교용 나머지 7개: 기준 시점 “오늘”, 위상 총량 π·π/2, 위상 속도 H_Λ(TT 속도 × A). look-elsewhere 8.
    판정: K1(§43.53), K2(쌓임 + 소멸), TT(k = 0)와 χ²로 비교."""
    bg = background()
    score, h0_keys, _ = _scorer()
    tt = tt_baseline()
    d_core = R.core(R.calibrated_alpha_s()[0])["D"]
    idx_bao = np.searchsorted(bg["n"], N_BAO)
    i0, t, nu = bg["i0"], bg["t"], bg["nu"]["H"]
    out = {}
    for ref_k, ref in (("peak", float(nu.max())), ("today", float(nu[i0]))):
        A = np.clip(1 - ref / (d_core * nu), 0, 1)
        integ = cumulative_trapezoid(A, t, initial=0.0)
        on = np.nonzero(A > 0)[0]
        for map_k in ("2pi", "pi", "pi/2", "H_L"):
            w = bg["hl"] if map_k == "H_L" else NORMS[map_k] / integ[-1]
            th0 = w * integ[i0] / 2
            sc = score(np.array(th0), w * integ[idx_bao] / 2)
            pulls = dict(zip(h0_keys, np.round(sc["h0_pulls"], 2).tolist()))
            out[f"{ref_k}|{map_k}"] = {
                "V39": float(sc["V39"]), "chi2": float(sc["chi2"]), "d_chi2_vs_TT": float(sc["chi2"] - tt["chi2"]),
                "theta0": float(th0), "direct_H0": float(sc["direct"]), "h0_pulls": pulls, "bao_chi2": float(sc["bao"]),
                "A_peak": float(A.max()), "t_start_Gyr": float(t[on[0]]), "t_peak_Gyr": float(t[int(np.argmax(A))]),
                "t_end_Gyr": float(t[on[-1]]), "today_fraction": float(integ[i0] / integ[-1]),
                "K1_killed": bool(sc["V39"] > 0.909 or any(abs(v) > 3 for v in pulls.values())),
                "cycle_closes": bool(A[-1] == 0 and A.max() > 0)}
    return {"TT": tt, "D": d_core, "principled": "peak|2pi", "variants": out}


def closure() -> dict:
    """큰 R₀의 원리 — 식 안에서 R₀를 정한다(사용자 “정리는 없음 계속”, 2026-09-25). 이웃은 허블 구.

    계산 전에 적은 판본(원리 선택은 C-rest, look-elsewhere 3):
        C-rest: 분별의 순환 = 우주의 순환(CE의 TT 순환). 끼임 순환은 TT가 다시 시작하는 때 t = 2π/H_Λ에 끝난다.
                R₀ = 1/ν_H(2π/H_Λ), 한 순환의 위상 = 2π.
        C-blur: 끼임 순환은 TT 기록 창이 닫히는 때 t = (π/2)/H_Λ에 끝난다. R₀ = 1/ν_H((π/2)/H_Λ), 위상 = π/2.
        C-pix : 끼임 하나가 닿는 이웃 = 지평선 화소 수(HP). R₀ = D·S_dS, 위상 = 2π.
                (화소 수에 TT 속도를 곱하는 판본은 A ≈ 1이라 TT와 같으므로 계산하지 않는다.)
    판정: K1(§43.53), TT(k = 0)와 χ²로 비교."""
    bg = background()
    score, h0_keys, _ = _scorer()
    tt = tt_baseline()
    c = R.core(R.calibrated_alpha_s()[0])
    idx_bao = np.searchsorted(bg["n"], N_BAO)
    i0, t, nu, hl = bg["i0"], bg["t"], bg["nu"]["H"], bg["hl"]
    s_ds = 3.2e122                                              # §43.29 렌더링 지평선 엔트로피
    specs = {"C-rest": (1 / float(np.interp(2 * math.pi / hl, t, nu)), 2 * math.pi),
             "C-blur": (1 / float(np.interp((math.pi / 2) / hl, t, nu)), math.pi / 2),
             "C-pix": (c["D"] * s_ds, 2 * math.pi)}
    out = {}
    for name, (r0, total) in specs.items():
        with np.errstate(divide="ignore"):
            A = np.clip(1 - 1 / (r0 * nu), 0, 1)
        integ = cumulative_trapezoid(A, t, initial=0.0)
        w = total / integ[-1]
        th0 = w * integ[i0] / 2
        sc = score(np.array(th0), w * integ[idx_bao] / 2)
        pulls = dict(zip(h0_keys, np.round(sc["h0_pulls"], 2).tolist()))
        on = np.nonzero(A > 0)[0]
        out[name] = {"R0_today": r0, "V39": float(sc["V39"]), "d_chi2_vs_TT": float(sc["chi2"] - tt["chi2"]),
                     "theta0": float(th0), "direct_H0": float(sc["direct"]), "h0_pulls": pulls,
                     "bao_chi2": float(sc["bao"]), "t_start_Gyr": float(t[on[0]]), "t_end_Gyr": float(t[on[-1]]),
                     "A_today": float(A[i0]), "today_fraction": float(integ[i0] / integ[-1]),
                     "K1_killed": bool(sc["V39"] > 0.909 or any(abs(v) > 3 for v in pulls.values()))}
    return {"TT": tt, "principled": "C-rest", "variants": out}


def present_share_link() -> dict:
    """교차 식별(계산 전에 적음): 현재가 한 고리의 c*/λ만 가져가는 이유 = 지금 켜진 끼임 비율 A(t₀) = 1 − 1/R₀.
    R₀ = 1/(1 − c*/λ)를 입자 쪽(§43.101의 Koide·계층 식 자)에서 정해 우주 쪽 위상을 예측한다. 위상 총량 2π·π·π/2(look-elsewhere 3)."""
    from examples.physics.rendering import ce_rendering_present_share as PSH
    e = PSH.exploration()
    fracs = {"Koide": e["present_over_lambda"], "M5": 1 + e["eps_star"]["M5 hierarchy"] / e["lambda"]}
    bg = background()
    score, h0_keys, _ = _scorer()
    tt = tt_baseline()
    idx_bao = np.searchsorted(bg["n"], N_BAO)
    i0, t, nu = bg["i0"], bg["t"], bg["nu"]["H"]
    out = {}
    for meter, frac in fracs.items():
        r0 = 1 / (1 - frac)
        A = np.clip(1 - 1 / (r0 * nu), 0, 1)
        integ = cumulative_trapezoid(A, t, initial=0.0)
        on = np.nonzero(A > 0)[0]
        for norm_k in ("2pi", "pi", "pi/2"):
            w = NORMS[norm_k] / integ[-1]
            th0 = w * integ[i0] / 2
            sc = score(np.array(th0), w * integ[idx_bao] / 2)
            pulls = dict(zip(h0_keys, np.round(sc["h0_pulls"], 2).tolist()))
            out[f"{meter}|{norm_k}"] = {"R0_today": r0, "V39": float(sc["V39"]), "theta0": float(th0),
                                        "d_chi2_vs_TT": float(sc["chi2"] - tt["chi2"]), "t_end_Gyr": float(t[on[-1]]),
                                        "K1_killed": bool(sc["V39"] > 0.909 or any(abs(v) > 3 for v in pulls.values()))}
    return {"TT": tt, "variants": out}


def indra() -> dict:
    """인다라망 IN — 서로가 서로를 켠다. 원장: 43장 §43.102.

    사용자(2026-09-25): “여기서 나와. 인다라망. 서로가 서로를 트리거하는거.”(화엄: 一卽一切 一切卽一, 법계연기, 初發心時便正覺)
    계산 전에 적은 판본(원리 선택은 IN-inf):
        끼임 하나는 망의 다른 모든 끼임을 비추고(켜고), 비친 모습이 다시 서로를 비춘다. 한 겹마다 끼임의 깊이 D만큼 번식한다.
        반사 깊이 k의 번식률 R₀(k) = Σ_{j=1..k} D^j. 망은 모두를 잇는다(ν = 1). 순간 평형 A = 1 − 1/R₀는 시간에 무관하다.
        위상은 지평선의 열적 속도로, 켜진 만큼 돈다: dΦ/dt = H_Λ·A(TT × A).
        IN-inf(원리 선택, 고전 인다라망: 비침이 끝없이 되비친다): R₀ = ∞, A = 1 → Φ = H_Λ t. 곧 TT가 그대로 나와야 한다.
        IN-1·IN-2·IN-3: 반사를 1–3겹에서 자른 민감도 판본(look-elsewhere 3).
    공개: 계산 전 암산으로 IN-2(D + D² = 13.3, A = 0.925)가 자료가 선호하는 기울기(≈ 0.37) 근처임을 보았다. IN-2의 적중은
    증거로 세지 않는다.
    판정: K1(§43.53). IN-inf는 TT와 같은 점수를 내야 한다(구현 검산)."""
    bg = background()
    score, h0_keys, _ = _scorer()
    tt = tt_baseline()
    d_core = R.core(R.calibrated_alpha_s()[0])["D"]
    idx_bao = np.searchsorted(bg["n"], N_BAO)
    i0, t, hl = bg["i0"], bg["t"], bg["hl"]
    out = {}
    for name, depth in (("IN-inf", None), ("IN-1", 1), ("IN-2", 2), ("IN-3", 3)):
        r0 = math.inf if depth is None else sum(d_core ** j for j in range(1, depth + 1))
        A = 1.0 if depth is None else 1 - 1 / r0
        phi = hl * A * t
        sc = score(np.array(phi[i0] / 2), phi[idx_bao] / 2)
        pulls = dict(zip(h0_keys, np.round(sc["h0_pulls"], 2).tolist()))
        out[name] = {"R0": r0, "A": A, "V39": float(sc["V39"]), "d_chi2_vs_TT": float(sc["chi2"] - tt["chi2"]),
                     "theta0": float(phi[i0] / 2), "direct_H0": float(sc["direct"]), "h0_pulls": pulls,
                     "bao_chi2": float(sc["bao"]),
                     "K1_killed": bool(sc["V39"] > 0.909 or any(abs(v) > 3 for v in pulls.values()))}
    return {"TT": tt, "D": d_core, "principled": "IN-inf", "variants": out}


def main() -> None:
    ind = indra()
    print(f"Indra net (principled pick = {ind['principled']}):")
    for name, v in ind["variants"].items():
        print(f"  {name:6s}", {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items()})
    cl = closure()
    print(f"closure (principled pick = {cl['principled']}):")
    for name, v in cl["variants"].items():
        print(f"  {name:7s}", {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items()})
    z = zero_parameter()
    print(f"zero-parameter (D = {z['D']:.6f}; principled pick = {z['principled']}):")
    for name, v in z["variants"].items():
        print(f"  {name:11s}", {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items()})
    print("instant equilibrium (k = 1):")
    for name, v in instant_equilibrium()["variants"].items():
        print(f"  {name:7s}", {k: round(x, 4) for k, x in v.items()})
    for genuine in (False, True):
        res = scan(genuine_only=genuine)
        print("\n=== genuine cycles only (rise x100 + fade) ===" if genuine else "=== unconstrained (includes degenerate) ===")
        print("TT baseline:", {k: round(v, 4) for k, v in res["TT"].items()})
        print(f"{'variant':10s} {'V39':>7s} {'chi2':>7s} {'k':>2s} {'dAIC':>7s} {'dBIC':>7s} {'theta0':>7s} "
              f"{'H0dir':>6s} {'BAO':>6s} {'peak Gyr':>9s} {'today frac':>10s} cycle  kappa    gamma    seed")
        rows = [(n, b) for n, b in res["variants"].items() if b is not None]
        for name, b in sorted(rows, key=lambda kv: kv[1]["chi2"]):
            print(f"{name:10s} {b['V39']:7.4f} {b['chi2']:7.2f} {b['k']:2d} {b['dAIC']:+7.2f} {b['dBIC']:+7.2f} "
                  f"{b['theta0']:7.4f} {b['direct_H0']:6.2f} {b['bao_chi2']:6.2f} {b['t_peak_Gyr']:9.2f} "
                  f"{b['today_fraction']:10.4f} {str(b['cycle_closes']):5s} {b['kappa']:.3g} {b['gamma']:.3g} {b['seed']:.0e}"
                  f"{'  KILLED' if b['K1_killed'] else ''}")
        missing = [n for n, b in res["variants"].items() if b is None]
        if missing:
            print("no genuine cycle on the grid:", missing)


if __name__ == "__main__":
    main()
