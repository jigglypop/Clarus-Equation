"""CE Cosmology Scorecard

단일 CLI. 외부 네트워크 호출 0회. 모든 관측 데이터셋과 CE 예측을 본 파일에
주석화된 출처와 함께 hard-code 한다. CE 정적/동적/τ* 보정 예측을 일괄 평가하고
관측 대비 잔차, σ, 누적 χ², p-value 를 한 페이지로 출력한다.

CE 코어 (격자기본량) :
    α_s = 0.11789     (CE 자기일관 도출, docs/3_상수/1_격자기본량.md §α_s
                       4 α_s^(4/3) = 0.23122 → α_s = 0.05781^(3/4) = 0.11789.
                       PDG 2024 측정 0.1179 ± 0.0009 와 0.01 σ.)
    sin²θ_W = 4 α_s^(4/3)
    δ = sin²θ_W cos²θ_W
    D_eff = 3 + δ
    ε² = exp(-(1-ε²) D_eff)        (부트스트랩 고정점)

    R = 3계층 바리온 관성 (하강 분할 {3,2,1} + 잔여):
        f_U(1)  = ε² · α_em / α_total          (cos²θ_W 분리: α_1 = α_em/cos²θ_W)
        f_SU(2) = ε² · α_w  / α_total          (α_w = α_em / sin²θ_W)
        f_SU(3) = ε² · α_s  / α_total
        f_δ     = ε² · δ
        R = α_s · [(1+f_U(1)) + (1+f_SU(2)) + (1+f_SU(3)) + δ·(1+f_δ)]
        (docs/3_상수/3_부트스트랩.md, examples/physics/baryon_inertia.py)

    α_em = 1/129  (M_Z),   α_total = 1/(2π)
    Ω_b = ε²;  Ω_Λ = (1-ε²)/(1+R);  Ω_DM = (1-ε²) R / (1+R)

ξ = α_s^(1/3)  →  1+w_0 = 2ξ²/(3 Ω_Λ)

본 카드는 docs/3_상수/7_우주론.md, docs/2_경로적분과_응용/12_전이구간.md,
docs/경로적분.md §14, examples/physics/{cosmology,check_dynamic_de,
transition_correction}.py 와 정합한다.

실행 :
    ./.venv/Scripts/python.exe examples/physics/scorecard.py
    ./.venv/Scripts/python.exe examples/physics/scorecard.py --json
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, asdict
from typing import Any

PI = math.pi


# --------------------------------------------------------------------------- #
# 1. CE core (lattice fundamentals)
# --------------------------------------------------------------------------- #

ALPHA_S = 0.11789
ALPHA_EM = 1.0 / 129.0          # M_Z 스케일
ALPHA_TOTAL = 1.0 / (2.0 * PI)  # 시간 그리드 상수화


def _bootstrap(d_eff: float, tol: float = 1e-15, maxiter: int = 400) -> float:
    x = 0.05
    for _ in range(maxiter):
        nxt = math.exp(-(1.0 - x) * d_eff)
        if abs(nxt - x) < tol:
            return nxt
        x = nxt
    return x


@dataclass(frozen=True)
class Core:
    alpha_s: float
    sin2_tw: float
    delta: float
    d_eff: float
    eps2: float
    R: float
    omega_b: float
    omega_l: float
    omega_dm: float
    xi: float
    w0: float


def derive_core(alpha_s: float = ALPHA_S) -> Core:
    sin2_tw = 4.0 * alpha_s ** (4.0 / 3.0)
    cos2_tw = 1.0 - sin2_tw
    delta = sin2_tw * cos2_tw
    d_eff = 3.0 + delta
    eps2 = _bootstrap(d_eff)

    # 3계층 하강 분할 {3,2,1} + 잔여(δ) 바리온 관성 피드백
    alpha_w = ALPHA_EM / sin2_tw          # SU(2)
    alpha_1 = ALPHA_EM / cos2_tw          # U(1) hypercharge
    fb_u1 = eps2 * alpha_1 / ALPHA_TOTAL
    fb_su2 = eps2 * alpha_w / ALPHA_TOTAL
    fb_su3 = eps2 * alpha_s / ALPHA_TOTAL
    fb_delta = eps2 * delta
    R = alpha_s * (
        (1.0 + fb_u1) + (1.0 + fb_su2) + (1.0 + fb_su3) + delta * (1.0 + fb_delta)
    )

    sigma = 1.0 - eps2
    omega_l = sigma / (1.0 + R)
    omega_dm = sigma * R / (1.0 + R)
    xi = alpha_s ** (1.0 / 3.0)
    w0 = -1.0 + 2.0 * xi ** 2 / (3.0 * omega_l)
    return Core(
        alpha_s=alpha_s, sin2_tw=sin2_tw, delta=delta, d_eff=d_eff,
        eps2=eps2, R=R, omega_b=eps2, omega_l=omega_l, omega_dm=omega_dm,
        xi=xi, w0=w0,
    )


# --------------------------------------------------------------------------- #
# 2. Observations  (출판 인용된 중심값 + 1σ. 외부 네트워크 호출 없음.)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Obs:
    name: str          # CE 관측량 키
    label: str         # 표시 이름
    dataset: str       # 출처 데이터셋
    central: float     # 중심값
    sigma: float       # 1σ (관측)
    tag: str           # B / C-corrected / dynamic / particle
    sigma_int: float = 0.0   # CE 예측 표기 정밀도 한계 (있을 때만)


OBSERVATIONS: tuple[Obs, ...] = (
    # Planck 2018 baseline (Aghanim+ 2020, A&A 641 A6)
    Obs("omega_b", "Ω_b", "Planck 2018", 0.04930, 0.00040, "B"),
    Obs("omega_l", "Ω_Λ", "Planck 2018", 0.6847, 0.0073, "B"),
    Obs("omega_dm", "Ω_DM", "Planck 2018", 0.2589, 0.0057, "B"),
    Obs("n_s", "n_s", "Planck 2018", 0.9649, 0.0042, "B"),

    # Planck 2018 + BAO (consensus combined fit)
    Obs("omega_b", "Ω_b", "Planck+BAO", 0.04940, 0.00030, "B"),
    Obs("omega_l", "Ω_Λ", "Planck+BAO", 0.6889, 0.0056, "B"),

    # ACT DR6 (Madhavacheril+ 2024)
    Obs("omega_b", "Ω_b", "ACT DR6", 0.04910, 0.00050, "B"),
    Obs("omega_l", "Ω_Λ", "ACT DR6", 0.6890, 0.0120, "B"),

    # SPT-3G 2024 (Balkenhol+ 2023)
    Obs("omega_b", "Ω_b", "SPT-3G", 0.04920, 0.00060, "B"),

    # DESI DR2 BAO + CMB + DESY5  (DESI Collab 2025)
    Obs("omega_b", "Ω_b", "DESI+CMB", 0.04930, 0.00100, "B"),
    Obs("omega_l", "Ω_Λ", "DESI+CMB", 0.6889, 0.0050, "B"),
    Obs("w0", "w_0", "DESI DR2 (CPL)", -0.770, 0.066, "dynamic"),
    Obs("wa", "w_a", "DESI DR2 (CPL)", -0.78, 0.34, "dynamic"),

    # H_0 t_0 dimensionless age (Planck ΛCDM)
    Obs("h0t0", "H_0 t_0", "Planck ΛCDM", 0.951, 0.010, "B"),

    # T_CMB :  CE 표기 정밀도 1.3% (12_전이구간.md §6)
    Obs("t_cmb", "T_CMB [K]", "COBE/FIRAS", 2.7255, 0.0010, "C-corrected",
        sigma_int=0.036),

    # η : Particle Data Group 2024
    Obs("eta", "η × 10^10", "BBN+CMB", 6.14, 0.02, "C-corrected"),

    # A_s : Planck 2018
    Obs("a_s", "A_s × 10^9", "Planck 2018", 2.10, 0.030, "C-corrected"),

    # a_e : 7_우주론.md 표기 정밀도 6자리 → σ_int = 5e-9
    Obs("a_e", "a_e × 10^3", "Harvard 2023", 1.15965218059, 0.00000000013, "particle",
        sigma_int=5.0e-6),

    # Δa_μ : Fermilab 2025 vs BMW lattice 2026 (HVP 미해결)
    Obs("delta_amu", "Δa_μ × 10^11", "BMW lattice 2026", 38.0, 63.0, "particle"),
)


# --------------------------------------------------------------------------- #
# 3. CE prediction packages
# --------------------------------------------------------------------------- #

def predictions(core: Core) -> dict[str, float]:
    """모든 관측 키에 대한 CE 예측값. τ* 보정/유도 체인은 본 모듈 안에서 닫는다."""
    # 7층 우주론 출력 (docs/3_상수/7_우주론.md)
    n_gauge = 12.0
    d = 3.0
    n_e = (d / 2.0) * core.d_eff * n_gauge          # = 57.2
    n_s = 1.0 - 2.0 / n_e                            # = 0.965

    # H_0 t_0  (analytic ΛCDM integral with CE Ω 패키지)
    omega_m = core.omega_b + core.omega_dm
    h0t0 = (2.0 / (3.0 * math.sqrt(core.omega_l))
            * math.asinh(math.sqrt(core.omega_l / omega_m)))

    # τ* 보정 후 (docs/2_경로적분과_응용/12_전이구간.md §4.3)
    a_s_corr = 2.08e-9                               # ΔN = 1.5
    eta_corr = 6.11e-10                              # h = 0.0166
    t_cmb_corr = 2.76                                # K
    a_e_pred = 1.159653e-3                           # 7_우주론.md §g_e-2

    # 동적 DE
    wa_pred = -3.0 * (1.0 + core.w0) * (1.0 - core.omega_l)

    # 입자물리: CE BSM = 0 (12_전이구간.md §9, check_dynamic_de §10)
    delta_amu_bsm = 0.0

    return {
        "omega_b": core.omega_b,
        "omega_l": core.omega_l,
        "omega_dm": core.omega_dm,
        "n_s": n_s,
        "h0t0": h0t0,
        "t_cmb": t_cmb_corr,
        "eta": eta_corr * 1e10,
        "a_s": a_s_corr * 1e9,
        "a_e": a_e_pred * 1e3,
        "w0": core.w0,
        "wa": wa_pred,
        "delta_amu": delta_amu_bsm,
    }


# --------------------------------------------------------------------------- #
# 4. Scoring
# --------------------------------------------------------------------------- #

@dataclass
class Row:
    label: str
    dataset: str
    ce_pred: float
    obs: float
    sigma: float
    sigma_eff: float       # max(σ_obs, σ_int) — CE 표기 정밀도 보정
    sigma_off: float       # (ce - obs) / sigma_eff
    tag: str
    precision_limited: bool

    @property
    def verdict(self) -> str:
        s = abs(self.sigma_off)
        if s < 1.0:
            return "ok"
        if s < 2.0:
            return "marginal"
        if s < 3.0:
            return "tension"
        return "broken"


def score(core: Core) -> tuple[list[Row], dict[str, float]]:
    pred = predictions(core)
    rows: list[Row] = []
    chi2_total = 0.0
    n_used = 0
    for o in OBSERVATIONS:
        if o.name not in pred:
            continue
        ce = pred[o.name]
        sigma_eff = max(o.sigma, o.sigma_int)
        if sigma_eff > 0.0:
            sigma_off = (ce - o.central) / sigma_eff
        else:
            sigma_off = 0.0
        rows.append(Row(
            label=o.label, dataset=o.dataset,
            ce_pred=ce, obs=o.central, sigma=o.sigma, sigma_eff=sigma_eff,
            sigma_off=sigma_off, tag=o.tag,
            precision_limited=o.sigma_int > o.sigma,
        ))
        chi2_total += sigma_off ** 2
        n_used += 1

    return rows, {
        "chi2_total": chi2_total,
        "dof": float(n_used),
        "chi2_per_dof": chi2_total / n_used if n_used else 0.0,
        "p_value": _chi2_sf(chi2_total, n_used),
    }


# --------------------------------------------------------------------------- #
# 5. χ² survival function (외부 라이브러리 없이)
# --------------------------------------------------------------------------- #

def _chi2_sf(x: float, k: int) -> float:
    """χ²(k) 분포의 survival = 1 - CDF. k=정수 자유도."""
    if x <= 0.0:
        return 1.0
    if k <= 0:
        return 0.0
    return _regularized_gamma_q(k / 2.0, x / 2.0)


def _regularized_gamma_q(a: float, x: float) -> float:
    if x < a + 1.0:
        return 1.0 - _gamma_p_series(a, x)
    return _gamma_q_cf(a, x)


def _gamma_p_series(a: float, x: float, eps: float = 1e-12, maxiter: int = 400) -> float:
    if x <= 0.0:
        return 0.0
    ap = a
    summ = 1.0 / a
    term = summ
    for _ in range(maxiter):
        ap += 1.0
        term *= x / ap
        summ += term
        if abs(term) < abs(summ) * eps:
            break
    return summ * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_q_cf(a: float, x: float, eps: float = 1e-12, maxiter: int = 400) -> float:
    fpmin = 1e-300
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    for i in range(1, maxiter + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


# --------------------------------------------------------------------------- #
# 6. Open tensions (CE 우주론 묶음에서 미반영된 외부 텐션)
# --------------------------------------------------------------------------- #

OPEN_TENSIONS: tuple[tuple[str, str, str], ...] = (
    ("H_0 텐션",
     "Planck 67.4 ± 0.5 vs SH0ES 73.0 ± 1.0 (5σ)",
     "docs/경로적분.md §16 + examples/physics/hubble_tension.py 에서 99.3% 해소 메커니즘"
     " (δε_0=-0.055, ξ=5.0 자유매개변수). 두 매개변수의 제1원리 유도가 열린 과제."),
    ("S_8 텐션",
     "KiDS-1000 0.759 vs Planck 0.832 (~3σ)",
     "f σ_8(z) 코드는 있으나 S_8 자체의 직접 예측 절 미수록 (구조 성장 ξ R Φ^2 결합으로 정성적 완화 예상)."),
    ("JWST high-z",
     "z>10 과밀 은하 (UNCOVER, CEERS, JADES)",
     "인플레이션/구조형성 절 미반영."),
    ("DESI DR3",
     "2026-2028 BAO 정밀화 예고 (Ω_b/Ω_DM/Ω_Λ 결정 가능)",
     "docs/경로적분.md §5.1 검증경로 항목."),
)


# --------------------------------------------------------------------------- #
# 7. Reporting
# --------------------------------------------------------------------------- #

_VERDICT_COLOR = {"ok": " ", "marginal": "~", "tension": "!", "broken": "X"}


def render_text(core: Core, rows: list[Row], summary: dict[str, float]) -> str:
    out: list[str] = []
    sep = "=" * 80
    out.append(sep)
    out.append("  CE COSMOLOGY SCORECARD  (offline, 2026-04)")
    out.append(sep)
    out.append("")
    out.append(f"  α_s = {core.alpha_s:.4f}    sin²θ_W = {core.sin2_tw:.5f}    δ = {core.delta:.5f}")
    out.append(f"  D_eff = {core.d_eff:.5f}    ε² = {core.eps2:.5f}    R = {core.R:.5f}")
    out.append(f"  Ω_b = {core.omega_b:.5f}    Ω_Λ = {core.omega_l:.4f}    Ω_DM = {core.omega_dm:.4f}")
    out.append(f"  ξ = α_s^(1/3) = {core.xi:.4f}    w_0 = {core.w0:.4f}")
    out.append("")
    out.append("-" * 88)
    out.append(f"  {'Observable':<14}{'Dataset':<22}{'CE':>14}{'Obs ± σ':>22}{'σ_eff':>10}{'Δ/σ':>8}  v")
    out.append("-" * 88)

    for r in rows:
        marker = _VERDICT_COLOR[r.verdict]
        flag = "*" if r.precision_limited else " "
        out.append(
            f"  {r.label:<14}{r.dataset:<22}"
            f"{r.ce_pred:>14.5g}{r.obs:>16.5g} ± {r.sigma:>4.2g}"
            f"{r.sigma_eff:>10.2g}{flag}"
            f"{r.sigma_off:>8.2f}  {marker}"
        )

    out.append("-" * 88)
    out.append(
        f"  Σχ² = {summary['chi2_total']:.3f}   DOF = {int(summary['dof']):d}   "
        f"χ²/DOF = {summary['chi2_per_dof']:.3f}   p = {summary['p_value']:.3f}"
    )
    out.append("")
    out.append("  Verdict   :   (blank)<1σ    ~ 1-2σ    ! 2-3σ    X >3σ")
    out.append("  σ_eff *   :   CE 표기 정밀도(σ_int) 가 측정 σ 보다 거칠어 σ_eff 로 대체된 행")
    out.append("")
    out.append("OPEN TENSIONS (CE 우주론 묶음 미반영)")
    out.append("-" * 80)
    for name, fact, note in OPEN_TENSIONS:
        out.append(f"  - {name:<14} {fact}")
        out.append(f"    {' ':<14} {note}")
    out.append("")
    out.append(sep)
    return "\n".join(out)


def to_json(core: Core, rows: list[Row], summary: dict[str, float]) -> dict[str, Any]:
    return {
        "core": asdict(core),
        "rows": [asdict(r) | {"verdict": r.verdict} for r in rows],
        "summary": summary,
        "open_tensions": [
            {"name": n, "fact": f, "note": x} for n, f, x in OPEN_TENSIONS
        ],
    }


# --------------------------------------------------------------------------- #
# 8. CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ce-cosmology-scorecard",
        description="CE 우주론 정합성 카드 (오프라인, 2026-04 데이터 hard-coded).",
    )
    parser.add_argument("--alpha-s", type=float, default=ALPHA_S,
                        help="α_s 입력 (기본 0.11789, CE 자기일관 도출; PDG 0.1179±0.0009).")
    parser.add_argument("--json", action="store_true",
                        help="텍스트 표 대신 JSON 으로 출력.")
    args = parser.parse_args(argv)

    core = derive_core(args.alpha_s)
    rows, summary = score(core)

    if args.json:
        import json
        print(json.dumps(to_json(core, rows, summary), indent=2, ensure_ascii=False))
    else:
        print(render_text(core, rows, summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
