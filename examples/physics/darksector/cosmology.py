"""평탄 FLRW 배경 커널, CE 밀도비 감사, 성장률 수치 계산을 한 모듈에 모은다.

세 절로 구성한다.
1. 배경 커널(background kernel): 항등식만 담는다. 밀도 파라미터를 고르거나 관측에
   맞추지 않는다. 모든 밀도는 현재 분율이고 ``E(a) = H(a) / H0`` 는 무차원이다.
2. 밀도비 감사(ratio audit): 현재 코드가 실제로 지지하는 부분, 곧 현재 밀도비만
   검사하는 얇은 검사기다. 저장소에 전방 모델이 없어 범위 밖인 현대 우주론·암흑물질
   주장도 함께 기록한다.
3. 성장률 수치(growth numerics): 격자·적분·보간 유틸리티, 선형 성장 방정식 풀이,
   광도 거리, epsilon_grav 보정, 명령줄 진입점.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# 1. 배경 커널
# ---------------------------------------------------------------------------


def _require_scale_factor(a: float) -> None:
    if not math.isfinite(a) or a <= 0.0:
        raise ValueError("scale factor a must be finite and > 0")


def cpl_w(a: float, w0: float = -1.0, wa: float = 0.0) -> float:
    """CPL 상태방정식 ``w(a) = w0 + wa (1 - a)`` 를 돌려준다."""

    _require_scale_factor(a)
    return w0 + wa * (1.0 - a)


def cpl_density_scale(a: float, w0: float = -1.0, wa: float = 0.0) -> float:
    """CPL 상태방정식에 대한 ``rho_de(a) / rho_de0`` 를 돌려준다."""

    _require_scale_factor(a)
    power = -3.0 * (1.0 + w0 + wa)
    return a**power * math.exp(3.0 * wa * (a - 1.0))


@dataclass(frozen=True)
class FlatFLRW:
    """복사 + 무압력 물질 + CPL 암흑에너지 배경.

    곡률 항은 없다. 물리적으로 정규화된 평탄 모형은
    ``omega_r0 + omega_m0 + omega_de0 == 1`` 을 만족한다. 항등식 메서드는 정규화되지
    않은 중간 계산에도 쓰이므로 입력을 조용히 재정규화하지 않는다.
    """

    omega_m0: float
    omega_de0: float
    omega_r0: float = 0.0
    w0: float = -1.0
    wa: float = 0.0

    @property
    def density_sum0(self) -> float:
        return self.omega_r0 + self.omega_m0 + self.omega_de0

    def e2_of_a(self, a: float) -> float:
        """설정된 유체에 대한 정확한 무차원 ``E(a)^2`` 를 돌려준다."""

        _require_scale_factor(a)
        return (
            self.omega_r0 * a ** (-4.0)
            + self.omega_m0 * a ** (-3.0)
            + self.omega_de0 * cpl_density_scale(a, self.w0, self.wa)
        )

    def e_of_a(self, a: float) -> float:
        e2 = self.e2_of_a(a)
        if e2 <= 0.0:
            raise ValueError("E(a)^2 must be positive")
        return math.sqrt(e2)

    def omega_r_of_a(self, a: float) -> float:
        return self.omega_r0 * a ** (-4.0) / self.e2_of_a(a)

    def omega_m_of_a(self, a: float) -> float:
        return self.omega_m0 * a ** (-3.0) / self.e2_of_a(a)

    def omega_de_of_a(self, a: float) -> float:
        return self.omega_de0 * cpl_density_scale(a, self.w0, self.wa) / self.e2_of_a(a)

    def dlnh_dln_a(self, a: float) -> float:
        """수치 차분 없이 ``d ln(H) / d ln(a)`` 를 돌려준다."""

        omega_r = self.omega_r_of_a(a)
        omega_m = self.omega_m_of_a(a)
        omega_de = self.omega_de_of_a(a)
        return 0.5 * (
            -4.0 * omega_r
            - 3.0 * omega_m
            - 3.0 * (1.0 + cpl_w(a, self.w0, self.wa)) * omega_de
        )

    def ricci_over_h2(self, a: float) -> float:
        """평탄 FLRW의 정확한 리치 스칼라(Ricci scalar) 비 ``R / H^2`` 를 돌려준다.

        우주상수(``w0=-1, wa=0``)에서는 ``12 - 9 Omega_m(a) - 12 Omega_r(a)`` 로 준다.
        """

        return 6.0 * (2.0 + self.dlnh_dln_a(a))


# ---------------------------------------------------------------------------
# 2. 밀도비 감사
# ---------------------------------------------------------------------------


ROOT = Path(__file__).resolve().parents[3]
OBSERVATION_MANIFEST_PATH = ROOT / "benchmarks" / "cosmology" / "observations_v1.json"

# reality_stone 런타임 제거 뒤 남긴 호환 경계다.
# 출처와 정규화된 과학적 밀도와의 구분은 paper/검증_원장/상수_코어_원장.md §2.3 에 있다.
LEGACY_ROUNDED_RUNTIME_RATIOS = (0.0487, 0.2623, 0.6891)


def load_ce_ratios_from_constants() -> dict[str, float]:
    """문서화된 유산 호환 삼중항(legacy compatibility triplet)을 돌려준다.

    이것은 역사적 모형 경계이지 CE 예측이나 정규화된 과학적 밀도 사후분포가 아니다.
    유산 전방 모델 경계를 아직 재현하는 호출자를 위해 공개 이름을 유지한다.
    """

    omega_b, omega_c, omega_lambda = LEGACY_ROUNDED_RUNTIME_RATIOS
    return {
        "omega_b": omega_b,
        "omega_c": omega_c,
        "omega_lambda": omega_lambda,
    }


def load_observation_manifest() -> dict[str, Any]:
    """판본이 붙은 관측 매니페스트(observation manifest)를 읽고 최소한으로 검증한다."""

    payload = json.loads(OBSERVATION_MANIFEST_PATH.read_text(encoding="utf-8"))
    if payload.get("manifest_id") != "CE_COSMOLOGY_OBSERVATIONS_V1":
        raise ValueError("unexpected cosmology observation manifest_id")
    required = set(payload["provenance_policy"]["required_entry_fields"])
    observations = payload.get("observations")
    if not isinstance(observations, list):
        raise ValueError("cosmology observation manifest requires an observations list")
    for entry in observations:
        missing = required - set(entry)
        if missing:
            raise ValueError(
                f"observation {entry.get('observation_id', '<unknown>')} "
                f"is missing provenance fields: {sorted(missing)}"
            )
    return payload


@dataclass(frozen=True)
class DensityBaseline:
    name: str
    omega_b_h2: float
    omega_c_h2: float
    h0: float
    note: str
    validity_status: str = "VALID_REFERENCE"
    scientific_score_eligible: bool = False

    @property
    def h(self) -> float:
        return self.h0 / 100.0

    @property
    def omega_b(self) -> float:
        return self.omega_b_h2 / (self.h * self.h)

    @property
    def omega_c(self) -> float:
        return self.omega_c_h2 / (self.h * self.h)

    @property
    def omega_lambda_flat(self) -> float:
        return 1.0 - self.omega_b - self.omega_c


@dataclass(frozen=True)
class RatioComparison:
    baseline: str
    omega_b_diff: float
    omega_c_diff: float
    omega_lambda_diff: float
    omega_b_rel: float
    omega_c_rel: float
    omega_lambda_rel: float

    @property
    def max_abs_relative_error(self) -> float:
        return max(abs(self.omega_b_rel), abs(self.omega_c_rel), abs(self.omega_lambda_rel))


@dataclass(frozen=True)
class CoverageVerdict:
    density_ratios_close: bool
    has_background_expansion_model: bool
    has_growth_model_for_s8: bool
    has_particle_dark_matter_model: bool
    has_detector_likelihood: bool
    scientific_score_eligible: bool = False
    closure_role: str = "exploratory_ratio_diagnostic"

    @property
    def summary(self) -> str:
        if not self.density_ratios_close:
            return "density-ratio mismatch"
        if any((
            self.has_background_expansion_model,
            self.has_growth_model_for_s8,
            self.has_particle_dark_matter_model,
            self.has_detector_likelihood,
        )):
            return "partial physical forward model"
        return "density ratios match; modern likelihood physics not implemented"


CE_RATIOS = load_ce_ratios_from_constants()

_BASELINE_NOTES = {
    "Planck2018_base": "Planck base-LambdaCDM reference.",
    "Planck_ACT_SPT_combined": (
        "Historical mixed tuple; no single official posterior or covariance."
    ),
    "ACT_DR6_DESI_reported": "ACT DR6.02 tagged compressed reference.",
    "SPT3G_CMBSPA": "Corrected SPT-3G D1 CMB-SPA compressed reference.",
}


def _load_recent_baselines() -> tuple[DensityBaseline, ...]:
    manifest = load_observation_manifest()
    by_id = {entry["observation_id"]: entry for entry in manifest["observations"]}
    baselines: list[DensityBaseline] = []
    for observation_id in manifest["legacy_ratio_baseline_ids"]:
        try:
            entry = by_id[observation_id]
            values = entry["values"]
            validity = entry["validity"]
            baselines.append(
                DensityBaseline(
                    name=observation_id,
                    omega_b_h2=float(values["omega_b_h2"]),
                    omega_c_h2=float(values["omega_c_h2"]),
                    h0=float(values["H0"]),
                    note=_BASELINE_NOTES[observation_id],
                    validity_status=str(validity["status"]),
                    scientific_score_eligible=bool(validity["scientific_score_eligible"]),
                )
            )
        except KeyError as exc:
            raise ValueError(f"invalid ratio baseline manifest entry: {observation_id}") from exc
    return tuple(baselines)


RECENT_BASELINES = _load_recent_baselines()


def relative_error(predicted: float, observed: float) -> float:
    if observed == 0.0:
        return 0.0
    return predicted / observed - 1.0


def compare_density_ratios(baseline: DensityBaseline) -> RatioComparison:
    return RatioComparison(
        baseline=baseline.name,
        omega_b_diff=CE_RATIOS["omega_b"] - baseline.omega_b,
        omega_c_diff=CE_RATIOS["omega_c"] - baseline.omega_c,
        omega_lambda_diff=CE_RATIOS["omega_lambda"] - baseline.omega_lambda_flat,
        omega_b_rel=relative_error(CE_RATIOS["omega_b"], baseline.omega_b),
        omega_c_rel=relative_error(CE_RATIOS["omega_c"], baseline.omega_c),
        omega_lambda_rel=relative_error(CE_RATIOS["omega_lambda"], baseline.omega_lambda_flat),
    )


def compare_all_density_ratios(
    baselines: tuple[DensityBaseline, ...] = RECENT_BASELINES,
) -> tuple[RatioComparison, ...]:
    return tuple(compare_density_ratios(baseline) for baseline in baselines)


def coverage_verdict(max_relative_tolerance: float = 0.04) -> CoverageVerdict:
    valid_names = {
        baseline.name
        for baseline in RECENT_BASELINES
        if baseline.validity_status != "EXCLUDED_HISTORICAL"
    }
    comparisons = tuple(
        comparison
        for comparison in compare_all_density_ratios()
        if comparison.baseline in valid_names
    )
    if not comparisons:
        raise ValueError("no valid density reference remains after provenance filtering")
    density_close = all(c.max_abs_relative_error <= max_relative_tolerance for c in comparisons)
    return CoverageVerdict(
        density_ratios_close=density_close,
        has_background_expansion_model=False,
        has_growth_model_for_s8=False,
        has_particle_dark_matter_model=False,
        has_detector_likelihood=False,
    )


def print_report() -> None:
    print("# CE Cosmology Ratio Audit")
    print()
    print("CE ratios")
    print(f"  Omega_b      {CE_RATIOS['omega_b']:.6f}")
    print(f"  Omega_DM     {CE_RATIOS['omega_c']:.6f}")
    print(f"  Omega_Lambda {CE_RATIOS['omega_lambda']:.6f}")
    print(f"  Omega_m      {CE_RATIOS['omega_b'] + CE_RATIOS['omega_c']:.6f}")
    print()
    print("Historical/excluded rows are displayed for parity but omitted from the verdict.")
    print("| baseline | dOmega_b | rel_b | dOmega_DM | rel_DM | dOmega_Lambda | rel_Lambda |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    baseline_by_name = {baseline.name: baseline for baseline in RECENT_BASELINES}
    for comparison in compare_all_density_ratios():
        baseline = baseline_by_name[comparison.baseline]
        display_name = comparison.baseline
        if baseline.validity_status == "EXCLUDED_HISTORICAL":
            display_name += " [EXCLUDED_HISTORICAL]"
        print(
            f"| {display_name} "
            f"| {comparison.omega_b_diff:+.6f} | {100.0 * comparison.omega_b_rel:+.2f}% "
            f"| {comparison.omega_c_diff:+.6f} | {100.0 * comparison.omega_c_rel:+.2f}% "
            f"| {comparison.omega_lambda_diff:+.6f} | {100.0 * comparison.omega_lambda_rel:+.2f}% |"
        )
    print()
    verdict = coverage_verdict()
    print("verdict", verdict.summary)
    print("density_ratios_close", verdict.density_ratios_close)
    print("has_background_expansion_model", verdict.has_background_expansion_model)
    print("has_growth_model_for_s8", verdict.has_growth_model_for_s8)
    print("has_particle_dark_matter_model", verdict.has_particle_dark_matter_model)
    print("has_detector_likelihood", verdict.has_detector_likelihood)
    print("scientific_score_eligible", verdict.scientific_score_eligible)
    print("closure_role", verdict.closure_role)


def ratio_audit_main() -> int:
    """밀도비 감사 보고서를 출력하는 진입점이다(옛 cosmology_ratio_audit.main)."""

    print_report()
    return 0


# ---------------------------------------------------------------------------
# 3. 성장률 수치
# ---------------------------------------------------------------------------


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def linspace(a: float, b: float, n: int) -> list[float]:
    if n <= 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


def logspace(a_min: float, a_max: float, n: int) -> list[float]:
    if a_min <= 0.0 or a_max <= 0.0:
        raise ValueError("a_min and a_max must be > 0")
    if n <= 1:
        return [a_min]
    la = math.log(a_min)
    lb = math.log(a_max)
    step = (lb - la) / (n - 1)
    return [math.exp(la + i * step) for i in range(n)]


def simpson(y: list[float], x: list[float]) -> float:
    """표로 주어진 자료를 비균등 2차 패널(quadratic panel)로 적분한다.

    연속한 두 구간 쌍을 세 표본을 지나는 유일한 2차 보간 다항식으로 적분한다.
    표본 수가 짝수이면 마지막 구간을 버리지 않고 사다리꼴로 더한다.
    """

    n = len(x)
    if n != len(y):
        raise ValueError("x and y length mismatch")
    if n < 2:
        return 0.0

    direction = x[1] - x[0]
    if direction == 0.0:
        raise ValueError("x values must be strictly monotonic")
    for i in range(1, n - 1):
        if (x[i + 1] - x[i]) * direction <= 0.0:
            raise ValueError("x values must be strictly monotonic")

    panel_stop = n if n % 2 == 1 else n - 1
    total = 0.0
    for i in range(0, panel_stop - 2, 2):
        h0 = x[i + 1] - x[i]
        h1 = x[i + 2] - x[i + 1]
        hsum = h0 + h1
        total += (hsum / 6.0) * (
            (2.0 - h1 / h0) * y[i]
            + (hsum * hsum / (h0 * h1)) * y[i + 1]
            + (2.0 - h0 / h1) * y[i + 2]
        )

    if n % 2 == 0:
        total += 0.5 * (x[-1] - x[-2]) * (y[-2] + y[-1])
    return total


def interp_linear(x_grid: list[float], y_grid: list[float], x: float) -> float:
    if len(x_grid) != len(y_grid):
        raise ValueError("x_grid and y_grid length mismatch")
    if not x_grid:
        raise ValueError("empty grid")
    if x <= x_grid[0]:
        return y_grid[0]
    if x >= x_grid[-1]:
        return y_grid[-1]
    lo = 0
    hi = len(x_grid) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if x_grid[mid] <= x:
            lo = mid
        else:
            hi = mid
    x0 = x_grid[lo]
    x1 = x_grid[hi]
    if x1 == x0:
        return y_grid[lo]
    w = (x - x0) / (x1 - x0)
    return (1.0 - w) * y_grid[lo] + w * y_grid[hi]


def parse_fsigma8_triplets(spec: str) -> list[tuple[float, float, float]]:
    """"z:fs8:sigma,z:fs8:sigma,..." 문자열을 삼중항 목록으로 파싱한다.

    sigma 가 0 이면 "미상/무시"를 뜻한다.
    """
    out: list[tuple[float, float, float]] = []
    s = spec.strip()
    if not s:
        return out
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        fields = [f.strip() for f in p.split(":")]
        if len(fields) != 3:
            raise ValueError(f"invalid triplet '{p}': expected z:fs8:sigma")
        z = float(fields[0])
        fs8 = float(fields[1])
        sig = float(fields[2])
        if z < 0.0:
            raise ValueError("z must be >= 0")
        if sig < 0.0:
            raise ValueError("sigma must be >= 0")
        out.append((z, fs8, sig))
    out.sort(key=lambda t: t[0])
    return out


def is_close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


@dataclass(frozen=True)
class Background:
    """FlatFLRW 커널에 위임하는 이전 호환 배경 래퍼다."""

    omega_m0: float
    omega_l0: float
    omega_r0: float = 0.0
    w0: float = -1.0
    wa: float = 0.0

    def _kernel(self) -> FlatFLRW:
        return FlatFLRW(
            omega_m0=self.omega_m0,
            omega_de0=self.omega_l0,
            omega_r0=self.omega_r0,
            w0=self.w0,
            wa=self.wa,
        )

    def e2_of_a(self, a: float) -> float:
        return self._kernel().e2_of_a(a)

    def e_of_a(self, a: float) -> float:
        return self._kernel().e_of_a(a)

    def dlnh_dln_a(self, a: float) -> float:
        return self._kernel().dlnh_dln_a(a)

    def omega_m_of_a(self, a: float) -> float:
        return self._kernel().omega_m_of_a(a)

    def omega_l_of_a(self, a: float) -> float:
        return self._kernel().omega_de_of_a(a)

    def omega_r_of_a(self, a: float) -> float:
        return self._kernel().omega_r_of_a(a)

    def ricci_over_h2(self, a: float) -> float:
        return self._kernel().ricci_over_h2(a)


def compute_s_of_a(bg: Background, a_grid: list[float]) -> list[float]:
    omegals = [bg.omega_l_of_a(a) for a in a_grid]
    denom = simpson(omegals, a_grid)
    if denom <= 0.0:
        return [0.0 for _ in a_grid]
    out = []
    for i in range(len(a_grid)):
        num = simpson(omegals[: i + 1], a_grid[: i + 1])
        out.append(clamp01(num / denom))
    return out


def compute_s_of_a_ratio(bg: Background, a_grid: list[float]) -> list[float]:
    omega_l0 = bg.omega_l_of_a(1.0)
    if omega_l0 <= 0.0:
        return [0.0 for _ in a_grid]
    out = []
    for a in a_grid:
        out.append(clamp01(bg.omega_l_of_a(a) / omega_l0))
    return out


def solve_growth(bg: Background, a_grid: list[float], mu_of_a: list[float]) -> tuple[list[float], list[float]]:
    """ln a 를 독립변수로 선형 성장 방정식을 RK4 로 풀어 (정규화 D, f) 를 돌려준다."""
    if len(a_grid) != len(mu_of_a):
        raise ValueError("a_grid and mu_of_a length mismatch")
    if len(a_grid) < 3:
        raise ValueError("need at least 3 points")
    if any(not math.isfinite(a) or a <= 0.0 for a in a_grid):
        raise ValueError("a_grid values must be finite and > 0")
    if any(a_grid[i + 1] <= a_grid[i] for i in range(len(a_grid) - 1)):
        raise ValueError("a_grid values must be strictly increasing")

    d = [0.0 for _ in a_grid]
    dp = [0.0 for _ in a_grid]

    a0 = a_grid[0]
    d[0] = a0
    dp[0] = a0

    lna = [math.log(a) for a in a_grid]

    def mu_at_ln_a(x: float) -> float:
        if x <= lna[0]:
            return mu_of_a[0]
        if x >= lna[-1]:
            return mu_of_a[-1]
        lo = 0
        hi = len(lna) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if lna[mid] <= x:
                lo = mid
            else:
                hi = mid
        x0 = lna[lo]
        x1 = lna[hi]
        w = (x - x0) / (x1 - x0) if x1 != x0 else 0.0
        return (1.0 - w) * mu_of_a[lo] + w * mu_of_a[hi]

    def rhs(x: float, d_val: float, dp_val: float) -> tuple[float, float]:
        a = math.exp(x)
        om = bg.omega_m_of_a(a)
        mu = mu_at_ln_a(x)
        term = 2.0 + bg.dlnh_dln_a(a)
        dd = dp_val
        ddp = -(term * dp_val) + 1.5 * om * mu * d_val
        return dd, ddp

    for i in range(len(a_grid) - 1):
        x = lna[i]
        dln = lna[i + 1] - x
        k1 = rhs(x, d[i], dp[i])
        k2 = rhs(x + 0.5 * dln, d[i] + 0.5 * dln * k1[0], dp[i] + 0.5 * dln * k1[1])
        k3 = rhs(x + 0.5 * dln, d[i] + 0.5 * dln * k2[0], dp[i] + 0.5 * dln * k2[1])
        k4 = rhs(x + dln, d[i] + dln * k3[0], dp[i] + dln * k3[1])

        d[i + 1] = d[i] + (dln / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        dp[i + 1] = dp[i] + (dln / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])

    d1 = d[-1]
    if d1 == 0.0:
        d1 = 1.0
    dn = [v / d1 for v in d]
    fn = []
    for i in range(len(a_grid)):
        if dn[i] <= 0.0:
            fn.append(0.0)
        else:
            fn.append((dp[i] / d1) / dn[i])
    return dn, fn


def luminosity_distance_mpc(bg: Background, h0_km_s_mpc: float, z: float, n: int) -> float:
    if z <= 0.0:
        return 0.0
    c_km_s = 299792.458
    z_grid = linspace(0.0, z, n)
    integrand = []
    for zz in z_grid:
        a = 1.0 / (1.0 + zz)
        e = bg.e_of_a(a)
        integrand.append(1.0 / e)
    chi = simpson(integrand, z_grid)
    return (c_km_s / h0_km_s_mpc) * (1.0 + z) * chi


def h0_t0(bg: Background, a_min: float, n: int) -> float:
    if a_min <= 0.0:
        raise ValueError("a_min must be > 0")
    ln_a_grid = linspace(math.log(a_min), 0.0, n)
    integrand = []
    for ln_a in ln_a_grid:
        a = math.exp(ln_a)
        e = bg.e_of_a(a)
        integrand.append(1.0 / e)
    return simpson(integrand, ln_a_grid)


def make_mu_grid(s_grid: list[float], epsilon_grav: float) -> list[float]:
    return [1.0 - epsilon_grav * ss for ss in s_grid]


def predict_fsigma8_at_z(
    bg: Background,
    ln_a_grid: list[float],
    a_grid: list[float],
    s_grid: list[float],
    epsilon_grav: float,
    sigma8_0: float,
    z: float,
) -> float:
    mu_grid = make_mu_grid(s_grid, epsilon_grav)
    d_norm, f_ln = solve_growth(bg, a_grid, mu_grid)
    a = 1.0 / (1.0 + z)
    ln_a = math.log(a)
    d = interp_linear(ln_a_grid, d_norm, ln_a)
    fz = interp_linear(ln_a_grid, f_ln, ln_a)
    return fz * (sigma8_0 * d)


def calibrate_epsilon_grav_bisect(
    bg: Background,
    ln_a_grid: list[float],
    a_grid: list[float],
    s_grid: list[float],
    sigma8_0: float,
    z_cal: float,
    fs8_target: float,
    eps_min: float,
    eps_max: float,
    max_iter: int = 80,
    tol_abs: float = 1.0e-6,
) -> float:
    """이분법(bisection)으로 fsigma8 목표값에 맞는 epsilon_grav 를 찾는다."""
    if eps_max <= eps_min:
        raise ValueError("invalid epsilon_grav bracket")
    if not (z_cal >= 0.0):
        raise ValueError("z_cal must be >= 0")

    def f(epsg: float) -> float:
        return predict_fsigma8_at_z(bg, ln_a_grid, a_grid, s_grid, epsg, sigma8_0, z_cal) - fs8_target

    f_lo = f(eps_min)
    f_hi = f(eps_max)
    if f_lo == 0.0:
        return eps_min
    if f_hi == 0.0:
        return eps_max
    if f_lo * f_hi > 0.0:
        raise ValueError("calibration target not bracketed by eps_min/eps_max")

    lo = eps_min
    hi = eps_max
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) <= tol_abs:
            return mid
        if f_lo * f_mid <= 0.0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return 0.5 * (lo + hi)


def main() -> int:
    p = argparse.ArgumentParser(prog="cosmology")
    p.add_argument("--model", choices=["bootstrap", "calibrate"], default="bootstrap")
    p.add_argument("--alpha-s", type=float, default=0.11789)
    p.add_argument("--omega-lambda", type=float, default=0.685)
    p.add_argument("--omega-m", type=float, default=0.315)
    p.add_argument("--mu", choices=["lcdm", "sfe"], default="sfe")
    p.add_argument("--sdef", choices=["ratio", "cumulative"], default="ratio")
    p.add_argument("--epsilon-grav", type=float, default=0.0)
    p.add_argument("--calibrate-epsilon-grav", action="store_true")
    p.add_argument("--cal-z", type=float, default=float("nan"))
    p.add_argument("--cal-fsigma8", type=float, default=float("nan"))
    p.add_argument("--cal-pick", choices=["first", "last"], default="first")
    p.add_argument("--eps-min", type=float, default=-1.0)
    p.add_argument("--eps-max", type=float, default=1.0)
    p.add_argument("--fsigma8-data", type=str, default="")
    p.add_argument("--sigma8-0", type=float, default=0.811)
    p.add_argument("--h0", type=float, default=67.4)
    p.add_argument("--zmax", type=float, default=2.0)
    p.add_argument("--nz", type=int, default=11)
    p.add_argument("--z-list", type=str, default="")
    p.add_argument("--na", type=int, default=2001)
    p.add_argument("--print-h0t0", action="store_true")
    p.add_argument("--extended", action="store_true")
    p.add_argument("--compare-fsigma8", action="store_true")
    args = p.parse_args()

    if args.model == "bootstrap":
        alpha_s = args.alpha_s
        sin2_tw = 4.0 * alpha_s ** (4.0 / 3.0)
        delta = sin2_tw * (1.0 - sin2_tw)
        d_eff = 3.0 + delta
        x = 0.05
        for _ in range(200):
            x = math.exp(-(1.0 - x) * d_eff)
        eps2 = x
        r_lo = alpha_s * d_eff
        omega_l0 = (1.0 - eps2) / (1.0 + r_lo)
        omega_m0 = 1.0 - omega_l0
    else:
        omega_l0 = args.omega_lambda
        omega_m0 = args.omega_m

    s = omega_l0 + omega_m0
    if s <= 0.0:
        raise SystemExit("invalid density parameters")
    omega_l0 /= s
    omega_m0 /= s

    bg = Background(omega_m0=omega_m0, omega_l0=omega_l0)

    a_grid = logspace(1.0e-3, 1.0, args.na)
    ln_a_grid = [math.log(a) for a in a_grid]
    if args.sdef == "ratio":
        s_grid = compute_s_of_a_ratio(bg, a_grid)
    else:
        s_grid = compute_s_of_a(bg, a_grid)

    cal_z_used = float("nan")
    cal_fsigma8_used = float("nan")
    cal_source = ""
    cal_z_source = ""

    epsilon_grav = args.epsilon_grav
    if args.calibrate_epsilon_grav:
        if args.mu != "sfe":
            raise SystemExit("--calibrate-epsilon-grav requires --mu sfe")
        triplets = parse_fsigma8_triplets(args.fsigma8_data)
        if math.isfinite(args.cal_z):
            cal_z_used = args.cal_z
            cal_z_source = "explicit"
        else:
            if not triplets:
                raise SystemExit("--calibrate-epsilon-grav requires --cal-z or non-empty --fsigma8-data")
            if args.cal_pick == "last":
                cal_z_used = triplets[-1][0]
            else:
                cal_z_used = triplets[0][0]
            cal_z_source = "fsigma8_data"

        cal_fsigma8 = args.cal_fsigma8
        cal_source = "explicit"
        if not math.isfinite(cal_fsigma8):
            z_tol = 5.0e-7
            for (zt, fs8, _sig) in triplets:
                if is_close(zt, cal_z_used, z_tol):
                    cal_fsigma8 = fs8
                    cal_source = "fsigma8_data"
                    break
        if not math.isfinite(cal_fsigma8):
            raise SystemExit("--calibrate-epsilon-grav requires --cal-fsigma8 or matching point in --fsigma8-data")
        cal_fsigma8_used = cal_fsigma8
        epsilon_grav = calibrate_epsilon_grav_bisect(
            bg=bg,
            ln_a_grid=ln_a_grid,
            a_grid=a_grid,
            s_grid=s_grid,
            sigma8_0=args.sigma8_0,
            z_cal=cal_z_used,
            fs8_target=cal_fsigma8,
            eps_min=args.eps_min,
            eps_max=args.eps_max,
        )

    mu_grid_lcdm = [1.0 for _ in a_grid]
    d_norm_lcdm, f_ln_lcdm = solve_growth(bg, a_grid, mu_grid_lcdm)

    if args.mu == "lcdm":
        mu_grid = mu_grid_lcdm
        d_norm = d_norm_lcdm
        f_ln = f_ln_lcdm
    else:
        mu_grid = make_mu_grid(s_grid, epsilon_grav)
        d_norm, f_ln = solve_growth(bg, a_grid, mu_grid)

    if args.z_list.strip():
        z_grid = []
        for part in args.z_list.split(","):
            s_part = part.strip()
            if not s_part:
                continue
            z_grid.append(float(s_part))
        if not z_grid:
            z_grid = linspace(0.0, args.zmax, args.nz)
    else:
        z_grid = linspace(0.0, args.zmax, args.nz)
    print("model", args.model)
    print("omega_m0", f"{omega_m0:.9f}")
    print("omega_lambda0", f"{omega_l0:.9f}")
    print("mu", args.mu)
    print("sdef", args.sdef)
    print("epsilon_grav", f"{epsilon_grav:.9f}")
    if args.calibrate_epsilon_grav:
        print("cal_z", f"{cal_z_used:.6f}")
        print("cal_fsigma8", f"{cal_fsigma8_used:.9f}")
        print("cal_z_source", cal_z_source)
        print("cal_source", cal_source)
    print("sigma8_0", f"{args.sigma8_0:.9f}")
    print("h0", f"{args.h0:.6f}")
    if args.print_h0t0:
        print("h0_t0", f"{h0_t0(bg, a_min=1.0e-6, n=20001):.9f}")
    print("")
    if args.extended:
        print("z,E(z),D_L_Mpc,Omega_m(a),Omega_Lambda(a),S(a),mu(a),D(a),f(z),sigma8(z),f_sigma8(z)")
    else:
        print("z,E(z),D_L_Mpc,D(a),f(z),sigma8(z),f_sigma8(z)")

    for z in z_grid:
        a = 1.0 / (1.0 + z)
        ln_a = math.log(a)

        ez = bg.e_of_a(a)
        dl = luminosity_distance_mpc(bg, args.h0, z, n=2001)
        om = bg.omega_m_of_a(a)
        ol = bg.omega_l_of_a(a)
        ss = interp_linear(ln_a_grid, s_grid, ln_a)
        muu = 1.0 if args.mu == "lcdm" else (1.0 - epsilon_grav * ss)
        d = interp_linear(ln_a_grid, d_norm, ln_a)
        fz = interp_linear(ln_a_grid, f_ln, ln_a)
        s8 = args.sigma8_0 * d
        fs8 = fz * s8
        if args.extended:
            print(
                f"{z:.6f},"
                f"{ez:.9f},"
                f"{dl:.6f},"
                f"{om:.9f},"
                f"{ol:.9f},"
                f"{ss:.9f},"
                f"{muu:.9f},"
                f"{d:.9f},"
                f"{fz:.9f},"
                f"{s8:.9f},"
                f"{fs8:.9f}"
            )
        else:
            print(
                f"{z:.6f},"
                f"{ez:.9f},"
                f"{dl:.6f},"
                f"{d:.9f},"
                f"{fz:.9f},"
                f"{s8:.9f},"
                f"{fs8:.9f}"
            )

    if args.compare_fsigma8:
        triplets = parse_fsigma8_triplets(args.fsigma8_data)
        legacy = False
        if not triplets:
            # 유산 예시값(불확도 없음). --fsigma8-data 를 우선한다.
            legacy = True
            triplets = [(0.32, 0.438, 0.0), (0.57, 0.447, 0.0), (0.70, 0.442, 0.0)]
        print("")
        if legacy:
            print("fsigma8_compare_mode legacy")
        else:
            print("fsigma8_compare_mode data")
        print("fsigma8_compare(z,is_cal,pred,target,sigma,delta,delta_over_sigma,delta_percent)")
        chi2_all = 0.0
        n_all = 0
        chi2_holdout = 0.0
        n_holdout = 0
        chi2_lcdm_all = 0.0
        n_lcdm_all = 0
        chi2_lcdm_holdout = 0.0
        n_lcdm_holdout = 0
        n_cal_points = 0
        z_cal = cal_z_used if args.calibrate_epsilon_grav else float("nan")
        z_tol = 5.0e-7  # 출력이 소수 6자리이므로 이 허용오차 안이면 같은 점으로 본다.
        for (zt, tgt, sig) in triplets:
            is_cal = args.calibrate_epsilon_grav and math.isfinite(z_cal) and is_close(zt, z_cal, z_tol)
            if is_cal:
                n_cal_points += 1

            # 모형 예측(--mu 에 따라 lcdm 또는 sfe).
            a = 1.0 / (1.0 + zt)
            ln_a = math.log(a)
            d_m = interp_linear(ln_a_grid, d_norm, ln_a)
            f_m = interp_linear(ln_a_grid, f_ln, ln_a)
            pred = f_m * (args.sigma8_0 * d_m)

            # 기준선 예측(항상 lcdm, mu=1).
            d_b = interp_linear(ln_a_grid, d_norm_lcdm, ln_a)
            f_b = interp_linear(ln_a_grid, f_ln_lcdm, ln_a)
            pred_lcdm = f_b * (args.sigma8_0 * d_b)

            delta = pred - tgt
            pct = (delta / tgt) * 100.0 if tgt != 0.0 else 0.0
            if sig > 0.0:
                d_over_s = delta / sig
                chi2_all += d_over_s * d_over_s
                n_all += 1
                if not is_cal:
                    chi2_holdout += d_over_s * d_over_s
                    n_holdout += 1

                d_over_s_lcdm = (pred_lcdm - tgt) / sig
                chi2_lcdm_all += d_over_s_lcdm * d_over_s_lcdm
                n_lcdm_all += 1
                if not is_cal:
                    chi2_lcdm_holdout += d_over_s_lcdm * d_over_s_lcdm
                    n_lcdm_holdout += 1
                print(f"{zt:.6f},{1 if is_cal else 0:d},{pred:.9f},{tgt:.9f},{sig:.9f},{delta:.9f},{d_over_s:.6f},{pct:.3f}")
            else:
                print(f"{zt:.6f},{1 if is_cal else 0:d},{pred:.9f},{tgt:.9f},{sig:.9f},{delta:.9f},,{pct:.3f}")
        if args.calibrate_epsilon_grav and n_cal_points == 0:
            print("")
            print("fsigma8_note calibration_z_not_in_data")
        if n_all > 0:
            print("")
            dof_all = n_all
            if dof_all <= 0:
                dof_all = 1
            print("fsigma8_chi2_all", f"{chi2_all:.6f}")
            print("fsigma8_n_all", f"{n_all:d}")
            print("fsigma8_dof_all", f"{dof_all:d}")
            print("fsigma8_chi2_red_all", f"{(chi2_all / dof_all):.6f}")
            if args.calibrate_epsilon_grav and n_holdout > 0:
                dof_h = n_holdout
                if dof_h <= 0:
                    dof_h = 1
                print("fsigma8_chi2_holdout", f"{chi2_holdout:.6f}")
                print("fsigma8_n_holdout", f"{n_holdout:d}")
                print("fsigma8_dof_holdout", f"{dof_h:d}")
                print("fsigma8_chi2_red_holdout", f"{(chi2_holdout / dof_h):.6f}")

            # 기준선(lcdm, mu=1) chi2 와 비교용 delta-chi2.
            if n_lcdm_all > 0:
                print("")
                print("fsigma8_baseline mu=1")
                print("fsigma8_chi2_lcdm_all", f"{chi2_lcdm_all:.6f}")
                print("fsigma8_n_lcdm_all", f"{n_lcdm_all:d}")
                print("fsigma8_dof_lcdm_all", f"{dof_all:d}")
                print("fsigma8_chi2_red_lcdm_all", f"{(chi2_lcdm_all / dof_all):.6f}")
                print("fsigma8_delta_chi2_all", f"{(chi2_all - chi2_lcdm_all):.6f}")
                print("fsigma8_delta_chi2_red_all", f"{((chi2_all / dof_all) - (chi2_lcdm_all / dof_all)):.6f}")
                if args.calibrate_epsilon_grav and n_lcdm_holdout > 0 and n_holdout > 0:
                    dof_h = n_holdout
                    if dof_h <= 0:
                        dof_h = 1
                    print("fsigma8_chi2_lcdm_holdout", f"{chi2_lcdm_holdout:.6f}")
                    print("fsigma8_n_lcdm_holdout", f"{n_lcdm_holdout:d}")
                    print("fsigma8_dof_lcdm_holdout", f"{dof_h:d}")
                    print("fsigma8_chi2_red_lcdm_holdout", f"{(chi2_lcdm_holdout / dof_h):.6f}")
                    print("fsigma8_delta_chi2_holdout", f"{(chi2_holdout - chi2_lcdm_holdout):.6f}")
                    print("fsigma8_delta_chi2_red_holdout", f"{((chi2_holdout / dof_h) - (chi2_lcdm_holdout / dof_h)):.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
