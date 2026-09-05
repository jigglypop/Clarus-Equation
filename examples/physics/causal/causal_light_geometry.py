"""인과 순서(causal order)에서 빛 기하와 계량(metric)이 어디까지 복원되는지 유한 반례와 장난감 재구성으로 감사한다.

이 모듈은 두 부분을 한 곳에 둔다.

첫째, 인과-빛 렌더링 가설에 대한 집중 감사다. 서로 섞기 쉬운 네 진술을 분리한다.

1. 연속체 가정 아래 인과 순서는 영(null)·등각(conformal) 구조를 고정한다.
2. 인과 순서만으로는 부피(volume)·곡률(curvature)·차원(dimension)이 고정되지 않는다.
3. 미시적 최대 갱신 속도만으로는 로런츠 대칭(Lorentz symmetry)이 따라오지 않는다.
4. 영 인과 전선(null causal frontier)은 c로 움직여도 질량 있는 기록 운반자는 엄격히 아광속일 수 있다.

아래 계산은 반례와 장난감 재구성 검사다. 성장 동역학, 광자 U(1) 부문, 0차원 씨앗에서
관측 우주를 유도하지 않는다.

둘째, 몫 판독(quotient readout)·인과 순서·계량 복원에 대한 유한 인증서(certificate)다.
이는 의도적으로 상한(ceiling)이지 CE 계량 유도가 아니다. 계수 1(rank-one) 판독은 몫 좌표가
잘 정의되어도 전체 사전 상태를 잃을 수 있고, 인과 순서는 추가 자료 없이는 기껏해야 등각
구조만 결정한다.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import random
from typing import Sequence

import numpy as np


Event = tuple[float, tuple[float, ...]]


@dataclass(frozen=True)
class ConformalCounterexample:
    """H*eta in [-2, -1]과 단위 공간 정육면체 위의 무차원 비교 결과다."""

    causal_order_identical: bool
    minkowski_normalized_four_volume: float
    de_sitter_normalized_four_volume: float
    minkowski_normalized_ricci_scalar: float
    de_sitter_normalized_ricci_scalar: float


@dataclass(frozen=True)
class CountingVolumeAudit:
    """계수 법칙(counting law)을 공급한 뒤 부피 비를 되찾는 장난감 복원 결과다."""

    trials: int
    normalized_event_density: float
    minkowski_mean_count: float
    de_sitter_mean_count: float
    expected_volume_ratio: float
    recovered_volume_ratio: float


def causally_comparable(left: Event, right: Event, *, c: float = 1.0) -> bool:
    """평탄 좌표에서 두 사건이 시간꼴(timelike) 또는 영(null) 관계인지 돌려준다."""

    if c <= 0.0:
        raise ValueError("c must be positive")
    if len(left[1]) != len(right[1]):
        raise ValueError("events must have the same spatial dimension")
    delta_t = abs(right[0] - left[0])
    distance_squared = sum(
        (right_value - left_value) ** 2
        for left_value, right_value in zip(left[1], right[1])
    )
    return c * c * delta_t * delta_t >= distance_squared


def causal_pairs(events: Sequence[Event], *, c: float = 1.0) -> set[tuple[int, int]]:
    """인과 순서로 이어진 비순서 인덱스 쌍의 집합을 돌려준다."""

    pairs: set[tuple[int, int]] = set()
    for left in range(len(events)):
        for right in range(left + 1, len(events)):
            if causally_comparable(events[left], events[right], c=c):
                pairs.add((left, right))
    return pairs


def massive_carrier_speed_ratio(momentum_to_mass_ratio: float) -> float:
    """상대론적 질량 운반자의 ``v_group / c``를 돌려준다.

    입력은 무차원 비 ``kappa = p / (m c)``다. ``E**2 = p**2 c**2 + m**2 c**4``에서
    군속도(group speed) 비는 ``kappa / sqrt(1 + kappa**2)``이며, 유한한 양의 ``kappa``마다
    엄격히 1보다 작다. 이는 모든 물리 기록이 정확히 c로 전파해야 한다는 주장에 대한
    운동학적 반례일 뿐이다.
    """

    if momentum_to_mass_ratio < 0.0 or not math.isfinite(momentum_to_mass_ratio):
        raise ValueError("momentum_to_mass_ratio must be finite and non-negative")
    return momentum_to_mass_ratio / math.sqrt(1.0 + momentum_to_mass_ratio**2)


def conformal_counterexample() -> ConformalCounterexample:
    """같은 영 순서에 다른 부피와 곡률이 붙는 예를 제시한다.

    민코프스키 공간과 등각 드 지터(de Sitter) 조각 ``g_dS = (H eta)^-2 g_M``을
    ``H eta in [-2, -1]``에서 비교한다. 양의 등각 인자는 인과 부호와 매개변수 없는 영 경로를
    바꾸지 않는다. 보고 값은 무차원 조합 ``H**4 * V``와 ``R / H**2``다(좌표 정육면체는
    일관되게 정규화한다).
    """

    events: tuple[Event, ...] = (
        (-1.9, (0.10, 0.10, 0.10)),
        (-1.6, (0.40, 0.10, 0.10)),
        (-1.3, (0.45, 0.40, 0.10)),
        (-1.1, (0.90, 0.90, 0.90)),
    )
    minkowski_order = causal_pairs(events)
    # Omega(eta)^2 > 0 을 곱해도 인과 부호는 바뀌지 않으므로 드 지터 등각 조각은
    # 같은 좌표 인과 관계를 가진다.
    de_sitter_order = causal_pairs(events)

    eta_start = -2.0
    eta_end = -1.0
    minkowski_volume = eta_end - eta_start
    de_sitter_normalized_volume = (
        (-1.0 / (3.0 * eta_end**3))
        - (-1.0 / (3.0 * eta_start**3))
    )

    return ConformalCounterexample(
        causal_order_identical=minkowski_order == de_sitter_order,
        minkowski_normalized_four_volume=minkowski_volume,
        de_sitter_normalized_four_volume=de_sitter_normalized_volume,
        minkowski_normalized_ricci_scalar=0.0,
        de_sitter_normalized_ricci_scalar=12.0,
    )


def _poisson_count(mean: float, generator: random.Random) -> int:
    """단위 비율 지수 도착으로 정확한 푸아송(Poisson) 계수를 뽑는다."""

    if mean < 0.0 or not math.isfinite(mean):
        raise ValueError("Poisson mean must be finite and non-negative")
    elapsed = 0.0
    count = 0
    while True:
        elapsed += generator.expovariate(1.0)
        if elapsed > mean:
            return count
        count += 1


def counting_volume_audit(
    *,
    normalized_event_density: float = 120.0,
    trials: int = 1000,
    seed: int = 20260828,
) -> CountingVolumeAudit:
    """보정된 계수가 등각적으로 축퇴한 부피를 구별하는 방식을 보인다.

    ``N ~ Poisson(rho_c * V_4)``를 유도하지 않고 채택한다. 두 영역은
    :func:`conformal_counterexample`과 같은 등각 인과 순서를 가지지만 정규화 4-부피는
    1과 7/24이다.
    """

    if normalized_event_density <= 0.0:
        raise ValueError("normalized_event_density must be positive")
    if trials < 1:
        raise ValueError("trials must be positive")

    generator = random.Random(seed)
    expected_ratio = 7.0 / 24.0
    minkowski_total = 0
    de_sitter_total = 0
    for _ in range(trials):
        minkowski_total += _poisson_count(normalized_event_density, generator)
        de_sitter_total += _poisson_count(
            normalized_event_density * expected_ratio,
            generator,
        )

    minkowski_mean = minkowski_total / trials
    de_sitter_mean = de_sitter_total / trials
    return CountingVolumeAudit(
        trials=trials,
        normalized_event_density=normalized_event_density,
        minkowski_mean_count=minkowski_mean,
        de_sitter_mean_count=de_sitter_mean,
        expected_volume_ratio=expected_ratio,
        recovered_volume_ratio=de_sitter_mean / minkowski_mean,
    )


def expected_ordering_fraction(spacetime_dimension: float) -> float:
    """평탄 인과 다이아몬드의 뮈르하임--마이어(Myrheim--Meyer) 순서 분율을 돌려준다.

    ``R``이 인과적으로 비교 가능한 비순서 쌍의 수일 때 ``R / comb(N, 2)``의 대표본 기댓값은

        Gamma(d + 1) Gamma(d / 2) / (2 Gamma(3 d / 2))

    이다. 분모가 4인 흔한 표현은 대표본 관계 밀도 ``R / N**2``다. 두 규약을 구분해야
    추정 차원의 2배 오류를 피한다.
    """

    if spacetime_dimension < 1.0:
        raise ValueError("spacetime_dimension must be at least one")
    log_fraction = (
        math.lgamma(spacetime_dimension + 1.0)
        + math.lgamma(spacetime_dimension / 2.0)
        - math.log(2.0)
        - math.lgamma(3.0 * spacetime_dimension / 2.0)
    )
    return math.exp(log_fraction)


def expected_relation_density(spacetime_dimension: float) -> float:
    """R / N**2 의 대표본 기댓값을 돌려준다."""

    return expected_ordering_fraction(spacetime_dimension) / 2.0


def sprinkle_minkowski_diamond(
    spacetime_dimension: int,
    count: int,
    *,
    seed: int,
) -> list[Event]:
    """단위 평탄 알렉산드로프 구간(Alexandrov interval)에 사건을 균일하게 뿌린다.

    구간은 ``-1 <= t <= 1``, 공간 반지름 ``1 - abs(t)``다. 알려진 다양체꼴 표적을
    표본화할 뿐, 한 점에서 다양체를 생성하거나 인과 집합 성장 법칙을 검사하지 않는다.
    """

    if spacetime_dimension < 2:
        raise ValueError("the sprinkling audit requires spacetime_dimension >= 2")
    if count < 1:
        raise ValueError("count must be positive")

    generator = random.Random(seed)
    spatial_dimension = spacetime_dimension - 1
    events: list[Event] = []
    for _ in range(count):
        # 단면 부피는 (1 - |t|) ** (d - 1) 에 비례한다.
        absolute_time = 1.0 - generator.random() ** (1.0 / spacetime_dimension)
        time = absolute_time if generator.random() < 0.5 else -absolute_time
        maximum_radius = 1.0 - absolute_time
        radius = maximum_radius * generator.random() ** (1.0 / spatial_dimension)

        direction = [generator.gauss(0.0, 1.0) for _ in range(spatial_dimension)]
        norm = math.sqrt(sum(component * component for component in direction))
        spatial = tuple(radius * component / norm for component in direction)
        events.append((time, spatial))
    return events


def ordering_fraction(events: Sequence[Event]) -> float:
    """유한 인과 표본의 R / comb(N, 2) 를 돌려준다."""

    count = len(events)
    if count < 2:
        raise ValueError("at least two events are required to estimate dimension")
    related = len(causal_pairs(events))
    return related / math.comb(count, 2)


def estimate_myrheim_meyer_dimension(
    observed_fraction: float,
    *,
    maximum_dimension: float = 32.0,
    iterations: int = 80,
) -> float:
    """단조인 연속체 순서 분율 공식을 이분법으로 역산한다."""

    if not 0.0 < observed_fraction <= 1.0:
        raise ValueError("observed_fraction must lie in (0, 1]")
    if maximum_dimension <= 1.0:
        raise ValueError("maximum_dimension must exceed one")
    if observed_fraction < expected_ordering_fraction(maximum_dimension):
        raise ValueError("observed fraction implies a dimension above the search bound")

    lower = 1.0
    upper = maximum_dimension
    for _ in range(iterations):
        midpoint = (lower + upper) / 2.0
        if expected_ordering_fraction(midpoint) > observed_fraction:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def square_lattice_angular_frequency(
    wavevector: Sequence[float],
    *,
    lattice_spacing: float = 1.0,
    limiting_speed: float = 1.0,
) -> float:
    """우선 좌표계(preferred frame) 대조군으로 쓰는 최근접 이웃 격자 분산 관계다."""

    if not wavevector:
        raise ValueError("wavevector must contain at least one component")
    if lattice_spacing <= 0.0 or limiting_speed <= 0.0:
        raise ValueError("lattice_spacing and limiting_speed must be positive")
    sine_sum = sum(
        math.sin(component * lattice_spacing / 2.0) ** 2
        for component in wavevector
    )
    return 2.0 * limiting_speed * math.sqrt(sine_sum) / lattice_spacing


def lattice_directional_split(
    wavenumber: float,
    *,
    lattice_spacing: float = 1.0,
    limiting_speed: float = 1.0,
) -> float:
    """고정된 2차원 |k| 에서 축 방향과 대각 방향의 상대 진동수 차를 돌려준다."""

    if wavenumber <= 0.0:
        raise ValueError("wavenumber must be positive")
    axis = square_lattice_angular_frequency(
        (wavenumber, 0.0),
        lattice_spacing=lattice_spacing,
        limiting_speed=limiting_speed,
    )
    diagonal_component = wavenumber / math.sqrt(2.0)
    diagonal = square_lattice_angular_frequency(
        (diagonal_component, diagonal_component),
        lattice_spacing=lattice_spacing,
        limiting_speed=limiting_speed,
    )
    return abs(axis - diagonal) / ((axis + diagonal) / 2.0)


def run_audit(*, count: int = 800, seed: int = 20260828) -> dict[str, object]:
    """집중 반례와 재구성 검사를 실행해 JSON 사전을 돌려준다."""

    target_dimension = 4
    events = sprinkle_minkowski_diamond(target_dimension, count, seed=seed)
    observed = ordering_fraction(events)
    estimated = estimate_myrheim_meyer_dimension(observed)
    conformal = conformal_counterexample()
    counting = counting_volume_audit(seed=seed)

    return {
        "claim_boundary": {
            "light_only_volume_curvature": "refuted_by_conformal_counterexample",
            "order_plus_number_route": "toy_reconstruction_only",
            "zero_dimensional_origin": "not_tested",
        },
        "conformal_counterexample": asdict(conformal),
        "counting_volume_audit": asdict(counting),
        "dimension_reconstruction": {
            "sample_count": count,
            "seed": seed,
            "target_spacetime_dimension": target_dimension,
            "continuum_ordering_fraction": expected_ordering_fraction(target_dimension),
            "observed_ordering_fraction": observed,
            "estimated_spacetime_dimension": estimated,
        },
        "lattice_control": {
            "low_wavenumber_directional_split": lattice_directional_split(0.01),
            "finite_wavenumber_directional_split": lattice_directional_split(1.5),
            "interpretation": "a maximum update speed alone does not ensure Lorentz symmetry",
        },
        "record_frontier_control": {
            "null_outer_front_speed_ratio": 1.0,
            "massive_carrier_speed_ratio_at_p_over_mc_1": massive_carrier_speed_ratio(1.0),
            "stored_record_speed_ratio": 0.0,
            "interpretation": "the causal outer envelope can be null while records are timelike",
        },
    }


def main() -> None:
    """빛 기하 감사를 명령줄에서 실행한다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260828)
    arguments = parser.parse_args()
    print(json.dumps(run_audit(count=arguments.count, seed=arguments.seed), indent=2, sort_keys=True))


P0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)


def density(p: float, coherence: complex) -> np.ndarray:
    """검증된 큐비트 상태 ``[[p,z],[z*,1-p]]``를 돌려준다."""

    if not math.isfinite(p) or not 0.0 < p < 1.0:
        raise ValueError("p must be finite and lie in (0, 1)")
    if not math.isfinite(coherence.real) or not math.isfinite(coherence.imag):
        raise ValueError("coherence must be finite")
    state = np.array([[p, coherence], [coherence.conjugate(), 1.0 - p]], dtype=complex)
    if np.linalg.eigvalsh(state).min() < -1.0e-12:
        raise ValueError("density matrix must be positive semidefinite")
    return state


def luders_zero(state: np.ndarray) -> np.ndarray:
    """부분정규화된 계수 1 뤼더스(Lueders) 결과 ``P0 rho P0``를 돌려준다."""

    return P0 @ state @ P0


def posterior_zero(state: np.ndarray) -> np.ndarray:
    """결과 0에 대한 정규화된 사후 상태를 돌려준다."""

    outcome = luders_zero(state)
    probability = float(np.trace(outcome).real)
    if probability <= 0.0:
        raise ValueError("outcome zero has no posterior at zero probability")
    return outcome / probability


def quotient_coordinate(state: np.ndarray) -> float:
    """몫 좌표, 즉 결과 0의 확률을 돌려준다."""

    return float(np.trace(luders_zero(state)).real)


def canonical_section(p: float) -> np.ndarray:
    """몫 좌표 ``p``에 대응하는 정준 단면(canonical section), 즉 결맞음 0인 상태를 돌려준다."""

    return density(p, 0.0)


def conformal_sign(interval: tuple[float, ...], omega: float) -> float:
    """양의 스케일링 ``Omega**2 g`` 뒤의 mostly-plus 간격 부호를 돌려준다."""

    if not math.isfinite(omega) or omega <= 0.0:
        raise ValueError("Omega must be finite and positive")
    if len(interval) < 2:
        raise ValueError("an interval needs time and at least one spatial component")
    if any(not math.isfinite(value) for value in interval):
        raise ValueError("interval components must be finite")
    base = -interval[0] ** 2 + sum(value * value for value in interval[1:])
    return omega * omega * base


def volume_recovery(volume_ratio: float, *, n: int = 4) -> float:
    """공급된 무차원 부피 비 ``Omega**n``에서 등각 인자를 되찾는다."""

    if not math.isfinite(volume_ratio) or volume_ratio <= 0.0:
        raise ValueError("volume_ratio must be finite and positive")
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError("n must be an integer spacetime dimension of at least two")
    return volume_ratio ** (1.0 / n)


def z2_cycle(transports: tuple[int, int, int, int], lengths: tuple[float, ...]) -> dict[str, object]:
    """Z2 홀로노미(holonomy)를 고정해도 변 계량이 자유로운 C4 밑공간 예다."""

    if len(transports) != 4 or any(value not in (-1, 1) for value in transports):
        raise ValueError("Z2 transports must be exactly four values in {-1, +1}")
    if len(lengths) != 4 or any(not math.isfinite(value) or value <= 0.0 for value in lengths):
        raise ValueError("C4 lengths must be four finite positive values")
    return {"base": "C4", "holonomy": math.prod(transports), "perimeter": sum(lengths)}


@dataclass(frozen=True)
class QuotientCertificate:
    """몫 판독·등각 반례·Z2 숨은 연결에 대한 유한 인증서다."""

    p: float
    coherences: tuple[complex, ...]
    prior_eigenvalues: tuple[tuple[float, float], ...]
    identical_subnormalised_readouts: bool
    identical_posteriors: bool
    distinct_priors: bool
    section_roundtrip: bool
    quotient_homeomorphism_conditions: dict[str, bool]
    controls: dict[str, bool]
    conformal: dict[str, object]
    z2_hidden_connection: dict[str, object]
    dimensions: dict[str, bool]
    accounting: dict[str, bool]
    status: dict[str, bool]


def certificate(*, p: float = 0.4, epsilon: float = 0.1, omega: float = 2.0, n: int = 4) -> QuotientCertificate:
    """입력을 닫힌 실패(fail-closed)로 검사하며 결정론적 유한 인증서를 만든다."""

    if not math.isfinite(epsilon) or not 0.0 < epsilon <= p < 1.0:
        raise ValueError("require finite 0 < epsilon <= p < 1")
    if not math.isfinite(omega) or omega <= 0.0:
        raise ValueError("Omega must be finite and positive")
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError("n must be an integer spacetime dimension of at least two")
    bound = math.sqrt(p * (1.0 - p))
    coherences = (0.0j, 0.5 * bound, 0.25j * bound)
    states = tuple(density(p, z) for z in coherences)
    readouts = tuple(luders_zero(state) for state in states)
    posteriors = tuple(posterior_zero(state) for state in states)
    eigens = tuple(tuple(float(x) for x in np.linalg.eigvalsh(state)) for state in states)
    section = canonical_section(p)
    ratio = omega**n
    intervals = {"timelike": (2.0, 1.0), "null": (1.0, 1.0), "spacelike": (1.0, 2.0)}
    signs = {name: conformal_sign(interval, omega) for name, interval in intervals.items()}
    base = {name: conformal_sign(interval, 1.0) for name, interval in intervals.items()}
    conformal = conformal_counterexample()
    plus = z2_cycle((1, 1, 1, 1), (1.0, 1.0, 1.0, 1.0))
    minus = z2_cycle((-1, 1, 1, 1), (1.0, 1.0, 1.0, 1.0))
    stretched = z2_cycle((1, 1, 1, 1), (1.0, 2.0, 1.0, 2.0))
    same_readout = all(np.allclose(readouts[0], item) for item in readouts[1:])
    same_posterior = all(np.allclose(posteriors[0], item) for item in posteriors[1:])
    distinct_priors = any(not np.allclose(states[0], item) for item in states[1:])
    quotient_conditions = {
        "finite_dimensional_density_state_space_compact": True,
        "luders_readout_continuous": True,
        "image_interval_pP0_hausdorff": True,
    }
    # 이는 콤팩트-하우스도르프 몫 상 정리(compact-to-Hausdorff quotient-image theorem)일 뿐이다.
    # 끝점 올(fibre) p=0,1 이 수축하므로 기구 상태 공간의 전역 매끄러운 올다발 구조는 주지 않는다.
    quotient_conditions["induced_quotient_to_image_homeomorphism"] = all(quotient_conditions.values())
    metric_counterexample = (
        conformal.causal_order_identical
        and conformal.minkowski_normalized_four_volume != conformal.de_sitter_normalized_four_volume
        and conformal.minkowski_normalized_ricci_scalar != conformal.de_sitter_normalized_ricci_scalar
    )
    recovered = volume_recovery(ratio, n=n)
    return QuotientCertificate(
        p=p,
        coherences=coherences,
        prior_eigenvalues=eigens,
        identical_subnormalised_readouts=same_readout,
        identical_posteriors=same_posterior,
        distinct_priors=distinct_priors,
        section_roundtrip=np.allclose(luders_zero(section), p * P0) and quotient_coordinate(section) == p,
        quotient_homeomorphism_conditions=quotient_conditions,
        controls={"posterior_sample_satisfies_p_ge_epsilon": p >= epsilon},
        conformal={
            "existing_counterexample_causal_order_identical": conformal.causal_order_identical,
            "existing_minkowski_volume": conformal.minkowski_normalized_four_volume,
            "existing_de_sitter_volume": conformal.de_sitter_normalized_four_volume,
            "existing_minkowski_ricci": conformal.minkowski_normalized_ricci_scalar,
            "existing_de_sitter_ricci": conformal.de_sitter_normalized_ricci_scalar,
            "n": n, "Omega": omega, "volume_ratio": ratio,
            "recovered_Omega": recovered,
            "causal_signs_unchanged": all((base[k] == 0.0 and signs[k] == 0.0) or base[k] * signs[k] > 0.0 for k in base),
        },
        z2_hidden_connection={"supplied_regular_bundle_control": True,
                              "instrument_connection_derived": False,
                              "plus": plus, "minus": minus, "stretched": stretched,
                              "same_base_different_holonomy": plus["holonomy"] != minus["holonomy"],
                              "fixed_holonomy_different_perimeter": plus["perimeter"] != stretched["perimeter"]},
        dimensions={"Omega_dimensionless": True, "volume_ratio_dimensionless": True,
                    "dimensionless_is_not_physical_derivation": True},
        accounting={"rn_weighting_used": False, "energy_or_stress_accounting_present": False},
        status={
            "full_map_injective": not (distinct_priors and same_readout),
            "induced_quotient_homeomorphism_conditional": all(quotient_conditions.values()),
            "homeomorphism_determines_metric": not metric_counterexample,
            "same_causal_order_different_full_metric_witness": metric_counterexample,
            "continuum_causal_order_to_conformal_theorem_proved": False,
            "distinguishing_continuum_assumptions_supplied": False,
            "volume_scale_recovered_for_supplied_toy": math.isclose(recovered, omega), "differentiable_structure_derived": False,
            "closed_posterior_domain_constructed": False,
            "instrument_fibers_global_bundle_derived": False,
            "quotient_smooth_manifold_derived": False,
            "metric_tensor_pullback_derived": False,
            "physical_causal_order_derived": False, "volume_law_derived": False,
            "levi_civita_dynamics_derived": False, "fold_stress_derived": False,
            "gr_lensing_backreaction_derived": False, "holdout_complete": False,
            "success_gates_5_to_8_complete": False,
        },
    )


def run() -> dict[str, object]:
    """소스 전용 점검을 위한 JSON 안전 출력을 돌려준다."""

    result = asdict(certificate())
    result["coherences"] = [[z.real, z.imag] for z in certificate().coherences]
    return result


def quotient_main() -> None:
    """몫 인증서를 명령줄에서 실행한다."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p", type=float, default=0.4)
    args = parser.parse_args()
    print(json.dumps(asdict(certificate(p=args.p)), default=str, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
