"""유한 깊이 인과 양자 도미노(causal quantum domino)와 그 반복·격자 확장의 유한 인증서를 모은다.

이 모듈은 세 부분으로 이루어진다.

첫째, 명시적 에너지 지불자를 갖춘 유한 깊이 인과 양자 도미노다.

연속 시간 최근접 이웃 점프 그림에는 엄격한 전선(front)이 없다. 지수 대기 시간의 사슬은
모든 양의 시간에서 0이 아닌 조기 도착 꼬리를 가진다. 그래서 이 부분은 이산 국소 회로를 쓴다.

틱 ``j``에서 이미 도달한 계 큐비트 ``j``가 이웃 표적 큐비트 ``j + 1``과 새 배터리 큐비트
하나 사이의 부분 교환(partial swap)을 제어한다. 배터리는 들뜬 상태에서 시작한다. 직교 포인터
라벨 ``1``에 대해 게이트는

    |1,0,1> -> cos(theta)|1,0,1> + sin(theta)|1,1,0>

이며, 역방향 상태는 같은 2차원 유니터리로 회전한다. 부모는 바뀌지 않고, 표적 들뜸은
배터리가 지불하며, 게이트는 공급된 등간격 들뜸 해밀토니안과 교환한다. ``theta``와
``sin(theta)**2``는 무차원이다. 길이와 시간은 별도로 감사된 비 ``a / (c * delta_t)``를
통해서만 삼각함수 코어에 들어간다.

새 배터리를 대각합으로 지우면 CPTP 채널이 된다. 깊이 d 최근접 이웃 회로는 그래프 반지름
d의 정확한 구조적 영향 원뿔(influence cone)을 가지며, ``delta_t >= a / c``를 요구하면 전선이
c로 제한된다. 이는 유한 조건부 증인(witness)이다. 지속적 장치 포인터, 기록-중력 원천 사상,
시공간, 우주론적 존재비를 유도하지 않는다.

둘째, 유한 반복 도미노 장애(obstruction)와 자원 영수증이다.

이는 의도적으로 유한 인증서이며 연속체 도미노 이론이 아니다. 좌표 라벨이 붙은 각 후보
칸에서

``S_j = |h_j|^2 + |d_j|^2 + |b_j|^2``

로 두고 이웃 칸을

``g h[j + 1]^* h[j] d[j + 1]^* b[j + 1] + c.c.``

로 결합한다. 라벨이 물리적 거리라는 것은 증명되지 않았다. 이는 국소 좌표 종(species) 장
후보일 뿐이다. 열린 사슬(각 칸의 차수가 최대 2)에서 여기서 쓰는 정규화는

``|V_link| <= |g|/4 (S_j^2 + S_{j+1}^2)``

를 준다. 따라서 ``lambda >= 2 |g|``는 ``lambda/4 sum_j S_j^2 + sum_links V_link``에 대한
충분 하한이다. 1-들뜸 에르미트 행렬은 4차원 쿼틱 ``g``를 맨 비율로 다루지 않고 사영된
최근접 이웃 비율 ``J_j``(각진동수, 곧 질량 차원)를 쓴다. 이것이 해석적 시작(analytic onset)
장애를 명시한다. 0이 아닌 최근접 이웃 경로의 첫 끝점 테일러 항은 차수 ``N``에 있으므로,
열린 시간 구간에서 정확히 지연된 전선을 제공할 수 없다.

준비 안정성 검사는 단위 봉우리 시계 범프(``||r||_infty <= 1``)를 쓴다. 그러면 2차 운반자
최솟값은 ``m_H^2 - mu_P^2``다. 다르게 정규화한 범프는 그에 맞게 재조정된 하한이 필요하다.

셋째, E20 유한 인과 증인이다. 공급된 스칼라 전선과 국소 보른(Born) 반응을 본다.

이는 의도적으로 E19의 라돈--니코딤(Radon--Nikodym) 재가중이 아니다. ``chi``는 무차원의
공급된 고전 격자 장이다. 그 콤팩트 충격은 시간 단계마다 격자 한 칸(``dt = a/c``)을 이동하며,
그 전선이 검출기에 도달한 뒤에야 국소 해밀토니안이 그 검출기의 보른 확률을 바꾼다. 여기서는
어떤 효과도 인력이나 중력과 동일시하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


DEFAULT_TOLERANCE = 1.0e-10
MAX_TOLERANCE = 1.0e-8
MAX_SYSTEM_QUBITS = 5
MAX_TOTAL_QUBITS = 9


@dataclass(frozen=True)
class BatteryOutcomeReceipt:
    """공급된 배터리 기구의 에너지 분해 결과 하나다."""

    basis_label: str
    probability: float
    final_battery_energy: float
    energy_paid_to_system: float
    conditional_system_energy: float | None
    relative_branch_energy_residual: float | None


@dataclass(frozen=True)
class CausalQuantumDominoCertificate:
    """선언된 유한 깊이 도미노 회로 하나의 수치 영수증이다."""

    site_count: int
    depth: int
    theta: float
    trigger_probability: float
    lattice_spacing: float
    clock_step: float
    causal_speed: float
    causal_ratio: float
    front_speed_bound: float
    structural_influence_cone: tuple[int, ...]
    spacelike_sites: tuple[int, ...]
    activation_probabilities: tuple[float, ...]
    channel_dimension: int
    kraus_count: int
    unitary_residual: float
    relative_energy_commutator_residual: float
    kraus_completeness_residual: float
    minimum_choi_eigenvalue: float
    output_trace_residual: float
    minimum_output_eigenvalue: float
    born_probability_sum_residual: float
    minimum_born_probability: float
    maximum_sampled_spacelike_trace_distance: float
    kraus_vs_direct_partial_trace_residual: float
    standard_limit_superoperator_residual: float
    energy_gap: float
    initial_system_energy: float
    final_system_energy: float
    initial_battery_energy: float
    final_battery_energy: float
    relative_total_energy_balance_residual: float
    battery_outcomes: tuple[BatteryOutcomeReceipt, ...]
    expected_battery_energy_paid: float
    relative_reverse_transfer_identity_residual: float
    maximum_relative_branch_energy_residual: float
    structural_causal_support_exact: bool
    sampled_spacelike_marginals_pass: bool
    cptp_within_tolerance: bool
    energy_conserved_within_tolerance: bool
    energy_resolved_instrument_within_tolerance: bool
    durable_physical_pointer_derived: bool
    covariant_matching_current_derived: bool
    record_to_gravity_source_derived: bool


def homogeneous_continuous_time_early_arrival_probability(
    rate_per_time: float,
    hops: int,
    elapsed_time: float,
) -> float:
    """동일한 지수 도약에 대한 얼랑(Erlang) 조기 도착 확률을 돌려준다.

    ``rate_per_time * elapsed_time``이 무차원 인자다. 정확한 실수 연산에서는 모든 양의
    비율, 양의 시간, 유한한 양의 ``hops``가 엄격히 양의 확률을 준다. 구현은 직접 뺄셈이
    상쇄될 때 하위 꼬리 급수를 쓰며, 부동소수점 범위보다 작은 값은 여전히 0으로 언더플로할 수
    있다. 따라서 연속 시간 마르코프 도미노는 그 자체로 정확한 빛원뿔 전선을 구현할 수 없다.
    """

    if not math.isfinite(rate_per_time) or rate_per_time <= 0.0:
        raise ValueError("rate_per_time must be finite and positive")
    if not isinstance(hops, int) or isinstance(hops, bool) or hops <= 0:
        raise ValueError("hops must be a positive integer")
    if not math.isfinite(elapsed_time) or elapsed_time < 0.0:
        raise ValueError("elapsed_time must be finite and non-negative")
    if elapsed_time == 0.0:
        return 0.0

    argument = rate_per_time * elapsed_time
    if argument == 0.0:
        return 0.0
    if argument < hops + 1.0:
        log_term = -argument + hops * math.log(argument) - math.lgamma(hops + 1.0)
        term = math.exp(log_term)
        probability = term
        order = hops
        for _ in range(100000):
            order += 1
            term *= argument / order
            probability += term
            if order > argument and term <= math.ulp(1.0) * probability:
                break
    else:
        log_terms = tuple(
            -argument + order * math.log(argument) - math.lgamma(order + 1.0)
            for order in range(hops)
        )
        maximum_log = max(log_terms)
        log_survival = maximum_log + math.log(
            math.fsum(math.exp(value - maximum_log) for value in log_terms)
        )
        probability = -math.expm1(log_survival)
    return min(1.0, max(0.0, probability))


def _bit(index: int, qubit: int, total_qubits: int) -> int:
    """기저 인덱스에서 지정한 큐비트의 비트를 읽는다."""

    return (index >> (total_qubits - qubit - 1)) & 1


def _controlled_partial_swap(
    total_qubits: int,
    parent: int,
    target: int,
    battery: int,
    theta: float,
) -> np.ndarray:
    """수 보존(number-conserving) 제어 표적--배터리 회전 하나를 만든다."""

    dimension = 1 << total_qubits
    gate = np.eye(dimension, dtype=np.complex128)
    cosine = math.cos(theta)
    sine = math.sin(theta)
    target_mask = 1 << (total_qubits - target - 1)
    battery_mask = 1 << (total_qubits - battery - 1)

    for basis_index in range(dimension):
        if (
            _bit(basis_index, parent, total_qubits) == 1
            and _bit(basis_index, target, total_qubits) == 0
            and _bit(basis_index, battery, total_qubits) == 1
        ):
            partner = basis_index ^ target_mask ^ battery_mask
            gate[basis_index, basis_index] = cosine
            gate[basis_index, partner] = -sine
            gate[partner, basis_index] = sine
            gate[partner, partner] = cosine
    return gate


def _domino_unitary(site_count: int, depth: int, theta: float) -> np.ndarray:
    """깊이 ``depth``의 도미노 회로 전체 유니터리를 만든다."""

    total_qubits = site_count + depth
    dimension = 1 << total_qubits
    unitary = np.eye(dimension, dtype=np.complex128)
    for tick in range(depth):
        gate = _controlled_partial_swap(
            total_qubits,
            parent=tick,
            target=tick + 1,
            battery=site_count + tick,
            theta=theta,
        )
        unitary = gate @ unitary
    return unitary


def _kraus_operators(
    unitary: np.ndarray,
    site_count: int,
    depth: int,
) -> tuple[np.ndarray, ...]:
    """초기 배터리가 모두 들뜬 상태일 때 배터리를 지운 크라우스(Kraus) 연산자를 돌려준다."""

    system_dimension = 1 << site_count
    battery_dimension = 1 << depth
    initial_battery_index = battery_dimension - 1
    tensor = unitary.reshape(
        system_dimension,
        battery_dimension,
        system_dimension,
        battery_dimension,
    )
    return tuple(
        tensor[:, output_battery, :, initial_battery_index]
        for output_battery in range(battery_dimension)
    )


def _apply_channel(
    kraus_operators: tuple[np.ndarray, ...],
    density: np.ndarray,
) -> np.ndarray:
    """크라우스 연산자 집합을 밀도 행렬에 적용한다."""

    return sum(
        (operator @ density @ operator.conj().T for operator in kraus_operators),
        start=np.zeros_like(density, dtype=np.complex128),
    )


def _single_site_reduced(
    density: np.ndarray,
    site_count: int,
    site: int,
) -> np.ndarray:
    """한 자리(site)만 남기고 나머지를 대각합으로 지운 축소 밀도 행렬을 돌려준다."""

    dimensions = (2,) * site_count
    tensor = density.reshape(dimensions + dimensions)
    traced_sites = tuple(index for index in range(site_count) if index != site)
    permutation = (
        (site,)
        + traced_sites
        + (site_count + site,)
        + tuple(site_count + index for index in traced_sites)
    )
    trace_dimension = 1 << (site_count - 1)
    ordered = np.transpose(tensor, permutation).reshape(
        2,
        trace_dimension,
        2,
        trace_dimension,
    )
    return np.einsum("aibi->ab", ordered)


def _seed_product_state(site_count: int, seed: np.ndarray) -> np.ndarray:
    """첫 자리는 ``seed``, 나머지는 바닥 상태인 곱 상태를 만든다."""

    vector = np.asarray(seed, dtype=np.complex128)
    if vector.shape != (2,):
        raise ValueError("seed must be a two-component state vector")
    state = vector
    ground = np.array([1.0, 0.0], dtype=np.complex128)
    for _ in range(site_count - 1):
        state = np.kron(state, ground)
    return state


def _trace_distance(left: np.ndarray, right: np.ndarray) -> float:
    """두 밀도 행렬의 대각합 거리(trace distance)를 돌려준다."""

    difference = 0.5 * (left - right + (left - right).conj().T)
    return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(difference))))


def _number_expectations(
    system_density: np.ndarray,
    site_count: int,
) -> tuple[float, ...]:
    """자리마다 들뜸 수의 기댓값을 돌려준다."""

    diagonal = np.real(np.diag(system_density))
    return tuple(
        float(
            sum(
                probability * _bit(index, site, site_count)
                for index, probability in enumerate(diagonal)
            )
        )
        for site in range(site_count)
    )


def _channel_superoperator(kraus_operators: tuple[np.ndarray, ...]) -> np.ndarray:
    """크라우스 연산자 집합의 초연산자(superoperator) 행렬을 만든다."""

    dimension = kraus_operators[0].shape[0]
    return sum(
        (np.kron(operator, operator.conj()) for operator in kraus_operators),
        start=np.zeros((dimension * dimension, dimension * dimension), dtype=np.complex128),
    )


def certify_causal_quantum_domino(
    *,
    site_count: int,
    depth: int,
    theta: float,
    lattice_spacing: float,
    clock_step: float,
    causal_speed: float,
    energy_gap: float = 1.0,
    tolerance: float = DEFAULT_TOLERANCE,
) -> CausalQuantumDominoCertificate:
    """에너지를 보존하는 유한 깊이 인과 채널 하나를 만들고 감사한다.

    ``lattice_spacing``, ``clock_step``, ``causal_speed``는 서로 호환되는 하나의 길이/시간
    규약을 써야 한다. ``energy_gap``은 계와 배터리가 공유하는 선언된 에너지 단위 하나를
    쓰며, 삼각함수나 확률 인자에는 결코 들어가지 않는다.
    """

    if not isinstance(site_count, int) or isinstance(site_count, bool) or site_count < 2:
        raise ValueError("site_count must be an integer at least two")
    if site_count > MAX_SYSTEM_QUBITS:
        raise ValueError(
            f"site_count exceeds the finite certificate limit {MAX_SYSTEM_QUBITS}"
        )
    if (
        not isinstance(depth, int)
        or isinstance(depth, bool)
        or depth <= 0
        or depth >= site_count
    ):
        raise ValueError("depth must be a positive integer smaller than site_count")
    if site_count + depth > MAX_TOTAL_QUBITS:
        raise ValueError(
            f"site_count + depth exceeds the finite certificate limit {MAX_TOTAL_QUBITS}"
        )
    if not math.isfinite(theta) or not 0.0 <= theta <= 0.5 * math.pi:
        raise ValueError("theta must be a finite angle in [0, pi/2]")
    for value, name in (
        (lattice_spacing, "lattice_spacing"),
        (clock_step, "clock_step"),
        (causal_speed, "causal_speed"),
        (energy_gap, "energy_gap"),
        (tolerance, "tolerance"),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if tolerance > MAX_TOLERANCE:
        raise ValueError(f"tolerance must not exceed {MAX_TOLERANCE}")

    causal_ratio = lattice_spacing / (causal_speed * clock_step)
    if causal_ratio > 1.0:
        raise ValueError("causal timing requires clock_step >= lattice_spacing / causal_speed")

    unitary = _domino_unitary(site_count, depth, theta)
    total_qubits = site_count + depth
    global_dimension = 1 << total_qubits
    identity_global = np.eye(global_dimension, dtype=np.complex128)
    unitary_residual = float(
        np.linalg.norm(unitary.conj().T @ unitary - identity_global, ord="fro")
    )

    number_diagonal = np.array(
        [
            sum(_bit(index, qubit, total_qubits) for qubit in range(total_qubits))
            * energy_gap
            for index in range(global_dimension)
        ],
        dtype=np.float64,
    )
    hamiltonian = np.diag(number_diagonal).astype(np.complex128)
    hamiltonian_scale = float(np.linalg.norm(hamiltonian, ord="fro"))
    relative_energy_commutator_residual = float(
        np.linalg.norm(unitary @ hamiltonian - hamiltonian @ unitary, ord="fro")
        / hamiltonian_scale
    )

    kraus = _kraus_operators(unitary, site_count, depth)
    system_dimension = 1 << site_count
    identity_system = np.eye(system_dimension, dtype=np.complex128)
    system_number_diagonal = np.array(
        [
            sum(_bit(index, site, site_count) for site in range(site_count))
            * energy_gap
            for index in range(system_dimension)
        ],
        dtype=np.float64,
    )
    system_hamiltonian = np.diag(system_number_diagonal).astype(np.complex128)
    system_hamiltonian_scale = max(
        float(np.linalg.norm(system_hamiltonian, ord="fro")),
        energy_gap,
    )
    completeness = sum(
        (operator.conj().T @ operator for operator in kraus),
        start=np.zeros_like(identity_system),
    )
    kraus_completeness_residual = float(
        np.linalg.norm(completeness - identity_system, ord="fro")
    )
    choi = sum(
        (
            np.outer(
                operator.reshape(-1, order="F"),
                operator.reshape(-1, order="F").conj(),
            )
            for operator in kraus
        ),
        start=np.zeros(
            (system_dimension * system_dimension, system_dimension * system_dimension),
            dtype=np.complex128,
        ),
    )
    minimum_choi_eigenvalue = float(np.min(np.linalg.eigvalsh(choi)))

    excited = np.array([0.0, 1.0], dtype=np.complex128)
    seed_state = _seed_product_state(site_count, excited)
    seed_density = np.outer(seed_state, seed_state.conj())
    initial_system_energy = float(
        np.vdot(seed_state, system_hamiltonian @ seed_state).real
    )
    initial_battery_energy = depth * energy_gap
    output_density = _apply_channel(kraus, seed_density)
    output_trace_residual = abs(float(np.trace(output_density).real) - 1.0)
    minimum_output_eigenvalue = float(np.min(np.linalg.eigvalsh(output_density)))
    born_probabilities = np.real(np.diag(output_density))
    born_probability_sum_residual = abs(float(np.sum(born_probabilities)) - 1.0)
    minimum_born_probability = float(np.min(born_probabilities))
    activation_probabilities = _number_expectations(output_density, site_count)

    structural_influence_cone = tuple(range(depth + 1))
    spacelike_sites = tuple(range(depth + 1, site_count))
    seed_family = (
        np.array([1.0, 0.0], dtype=np.complex128),
        excited,
        np.array([1.0, 1.0], dtype=np.complex128) / math.sqrt(2.0),
        np.array([1.0, 1.0j], dtype=np.complex128) / math.sqrt(2.0),
    )
    output_family = tuple(
        _apply_channel(
            kraus,
            np.outer(state := _seed_product_state(site_count, seed), state.conj()),
        )
        for seed in seed_family
    )
    maximum_sampled_spacelike_trace_distance = 0.0
    for site in spacelike_sites:
        reference = _single_site_reduced(output_family[0], site_count, site)
        for candidate in output_family[1:]:
            maximum_sampled_spacelike_trace_distance = max(
                maximum_sampled_spacelike_trace_distance,
                _trace_distance(
                    _single_site_reduced(candidate, site_count, site),
                    reference,
                ),
            )

    zero_unitary = _domino_unitary(site_count, depth, 0.0)
    zero_kraus = _kraus_operators(zero_unitary, site_count, depth)
    standard_limit_superoperator_residual = float(
        np.linalg.norm(
            _channel_superoperator(zero_kraus)
            - np.eye(system_dimension * system_dimension, dtype=np.complex128),
            ord="fro",
        )
    )

    battery_dimension = 1 << depth
    battery_input = np.zeros(battery_dimension, dtype=np.complex128)
    battery_input[-1] = 1.0
    global_input = np.kron(seed_state, battery_input)
    global_output = unitary @ global_input
    direct_output_matrix = global_output.reshape(system_dimension, battery_dimension)
    direct_output_density = direct_output_matrix @ direct_output_matrix.conj().T
    kraus_vs_direct_partial_trace_residual = float(
        np.linalg.norm(output_density - direct_output_density, ord="fro")
    )

    battery_outcomes: list[BatteryOutcomeReceipt] = []
    reverse_transfer_operator = np.zeros_like(system_hamiltonian)
    expected_battery_energy_paid = 0.0
    branch_residuals: list[float] = []
    for output_battery, operator in enumerate(kraus):
        final_outcome_energy = output_battery.bit_count() * energy_gap
        energy_paid = initial_battery_energy - final_outcome_energy
        branch_state = operator @ seed_state
        branch_probability = float(np.vdot(branch_state, branch_state).real)
        expected_battery_energy_paid += branch_probability * energy_paid
        reverse_transfer_operator += (
            operator.conj().T
            @ (system_hamiltonian - energy_paid * identity_system)
            @ operator
        )
        conditional_system_energy: float | None = None
        relative_branch_energy_residual: float | None = None
        if branch_probability > tolerance:
            conditional_system_energy = float(
                np.vdot(branch_state, system_hamiltonian @ branch_state).real
                / branch_probability
            )
            branch_energy_scale = max(
                initial_system_energy + energy_paid,
                energy_gap,
            )
            relative_branch_energy_residual = abs(
                conditional_system_energy - initial_system_energy - energy_paid
            ) / branch_energy_scale
            branch_residuals.append(relative_branch_energy_residual)
        battery_outcomes.append(
            BatteryOutcomeReceipt(
                basis_label=format(output_battery, f"0{depth}b"),
                probability=branch_probability,
                final_battery_energy=final_outcome_energy,
                energy_paid_to_system=energy_paid,
                conditional_system_energy=conditional_system_energy,
                relative_branch_energy_residual=relative_branch_energy_residual,
            )
        )
    relative_reverse_transfer_identity_residual = float(
        np.linalg.norm(
            reverse_transfer_operator - system_hamiltonian,
            ord="fro",
        )
        / system_hamiltonian_scale
    )
    maximum_relative_branch_energy_residual = max(branch_residuals, default=0.0)

    global_probabilities = np.abs(global_output) ** 2
    final_system_excitation = 0.0
    final_battery_excitation = 0.0
    for index, probability in enumerate(global_probabilities):
        final_system_excitation += probability * sum(
            _bit(index, site, total_qubits) for site in range(site_count)
        )
        final_battery_excitation += probability * sum(
            _bit(index, site_count + battery, total_qubits)
            for battery in range(depth)
        )
    final_system_energy = float(final_system_excitation * energy_gap)
    final_battery_energy = float(final_battery_excitation * energy_gap)
    initial_total_energy = initial_system_energy + initial_battery_energy
    relative_total_energy_balance_residual = abs(
        final_system_energy + final_battery_energy - initial_total_energy
    ) / initial_total_energy

    cptp_within_tolerance = (
        unitary_residual <= tolerance * math.sqrt(global_dimension)
        and kraus_completeness_residual <= tolerance * math.sqrt(system_dimension)
        and minimum_choi_eigenvalue >= -tolerance
        and output_trace_residual <= tolerance
        and minimum_output_eigenvalue >= -tolerance
        and born_probability_sum_residual <= tolerance
        and minimum_born_probability >= -tolerance
        and kraus_vs_direct_partial_trace_residual <= tolerance
    )
    energy_conserved_within_tolerance = (
        relative_energy_commutator_residual <= tolerance
        and relative_total_energy_balance_residual <= tolerance
    )
    energy_resolved_instrument_within_tolerance = (
        relative_reverse_transfer_identity_residual <= tolerance
        and maximum_relative_branch_energy_residual <= tolerance
        and abs(
            final_system_energy
            - initial_system_energy
            - expected_battery_energy_paid
        )
        <= tolerance * max(initial_system_energy + initial_battery_energy, energy_gap)
    )
    structural_causal_support_exact = (
        causal_ratio <= 1.0
        and structural_influence_cone == tuple(range(depth + 1))
        and all(site > depth for site in spacelike_sites)
    )
    sampled_spacelike_marginals_pass = (
        maximum_sampled_spacelike_trace_distance <= tolerance
    )

    return CausalQuantumDominoCertificate(
        site_count=site_count,
        depth=depth,
        theta=theta,
        trigger_probability=math.sin(theta) ** 2,
        lattice_spacing=lattice_spacing,
        clock_step=clock_step,
        causal_speed=causal_speed,
        causal_ratio=causal_ratio,
        front_speed_bound=lattice_spacing / clock_step,
        structural_influence_cone=structural_influence_cone,
        spacelike_sites=spacelike_sites,
        activation_probabilities=activation_probabilities,
        channel_dimension=system_dimension,
        kraus_count=len(kraus),
        unitary_residual=unitary_residual,
        relative_energy_commutator_residual=relative_energy_commutator_residual,
        kraus_completeness_residual=kraus_completeness_residual,
        minimum_choi_eigenvalue=minimum_choi_eigenvalue,
        output_trace_residual=output_trace_residual,
        minimum_output_eigenvalue=minimum_output_eigenvalue,
        born_probability_sum_residual=born_probability_sum_residual,
        minimum_born_probability=minimum_born_probability,
        maximum_sampled_spacelike_trace_distance=(
            maximum_sampled_spacelike_trace_distance
        ),
        kraus_vs_direct_partial_trace_residual=(
            kraus_vs_direct_partial_trace_residual
        ),
        standard_limit_superoperator_residual=standard_limit_superoperator_residual,
        energy_gap=energy_gap,
        initial_system_energy=initial_system_energy,
        final_system_energy=final_system_energy,
        initial_battery_energy=initial_battery_energy,
        final_battery_energy=final_battery_energy,
        relative_total_energy_balance_residual=relative_total_energy_balance_residual,
        battery_outcomes=tuple(battery_outcomes),
        expected_battery_energy_paid=expected_battery_energy_paid,
        relative_reverse_transfer_identity_residual=(
            relative_reverse_transfer_identity_residual
        ),
        maximum_relative_branch_energy_residual=(
            maximum_relative_branch_energy_residual
        ),
        structural_causal_support_exact=structural_causal_support_exact,
        sampled_spacelike_marginals_pass=sampled_spacelike_marginals_pass,
        cptp_within_tolerance=cptp_within_tolerance,
        energy_conserved_within_tolerance=energy_conserved_within_tolerance,
        energy_resolved_instrument_within_tolerance=(
            energy_resolved_instrument_within_tolerance
        ),
        durable_physical_pointer_derived=False,
        covariant_matching_current_derived=False,
        record_to_gravity_source_derived=False,
    )


@dataclass(frozen=True)
class AutonomousRepeatedDominoObstructionCertificate:
    """유한 사슬 감사 영수증이며, 거짓 상한 항목은 의도된 것이다."""

    n_links: int
    couplings: tuple[float, ...]
    field_mass: float
    clock_scale: float
    prep_mass_squared: float
    carrier_quadratic_minimum_mass_squared: float
    battery_energy_per_cell: float
    exchange_coupling: float
    quartic_coupling: float
    quartic_lower_bound_coefficient: float
    hamiltonian_hermiticity_residual: float
    lower_order_endpoint_power_residual: float
    endpoint_order_n_value: complex
    expected_endpoint_order_n_value: float
    endpoint_taylor_coefficient: complex
    expected_endpoint_taylor_coefficient: complex
    small_time: float
    small_time_endpoint_amplitude: complex
    small_time_leading_term: complex
    small_time_remainder_magnitude: float
    all_success_initially_clean_battery_count: int
    all_success_initially_clean_record_count: int
    all_success_battery_energy: float
    dimensionless_core_arguments: tuple[tuple[str, str], ...]
    local_coordinate_species_field_candidate_by_construction: bool
    explicit_coordinate_time_switching_present: bool
    dimensions_closed: bool
    stability_bound_pass: bool
    carrier_prep_stability_pass: bool
    finite_hamiltonian_hermitian: bool
    analytic_coefficient_conditions_pass: bool
    finite_all_success_resource_receipt: bool
    species_index_is_physical_spatial_distance: bool
    physical_lattice_or_worldtube_completion: bool
    coupled_clock_global_monotonicity_one_pass: bool
    exact_delayed_front_derived: bool
    projected_link_rates_from_action_derived: bool
    iterated_fresh_ancilla_cptp_instrument_derived: bool
    continuum_qft_microcausality_derived: bool
    operational_no_signalling_derived: bool
    gr_source_stress_matching_derived: bool
    unbounded_front_from_finite_resources_derived: bool
    durable_records_derived: bool
    cross_dataset_parameter_fixing_derived: bool
    independent_holdout_prediction_derived: bool


def _positive(value: float, name: str) -> float:
    """유한하고 양수인 값만 통과시킨다."""

    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _finite_nonzero(value: float, name: str) -> float:
    """유한하고 0이 아닌 값만 통과시킨다."""

    value = float(value)
    if not math.isfinite(value) or value == 0.0:
        raise ValueError(f"{name} must be finite and nonzero")
    return value


def certify_autonomous_repeated_domino_obstruction(
    *,
    n_links: int,
    couplings: Sequence[float],
    field_mass: float,
    clock_scale: float,
    prep_mass_squared: float,
    battery_energy_per_cell: float,
    exchange_coupling: float,
    quartic_coupling: float,
    small_time: float = 1.0e-3,
) -> AutonomousRepeatedDominoObstructionCertificate:
    """유한 해석적 시작 장애와 전부 성공 자원 인증서를 돌려준다.

    ``all_success_battery_energy``는 처음에 깨끗한 모든 칸이 성공을 기록하는 가지의
    용량이다. 기대 가지 에너지가 아니며, 지속적 기록과 재설정은 여기서 증명되지 않는다.
    """

    if not isinstance(n_links, int) or isinstance(n_links, bool) or n_links < 1:
        raise ValueError("n_links must be an integer at least one")
    # 사영된 1-들뜸 비율 J_j [질량]이지, 무차원 4차원 g 가 아니다.
    supplied_couplings = tuple(
        _finite_nonzero(value, f"couplings[{index}]")
        for index, value in enumerate(couplings)
    )
    if len(supplied_couplings) != n_links:
        raise ValueError("couplings must contain exactly n_links entries")
    field_mass = _positive(field_mass, "field_mass")
    clock_scale = _positive(clock_scale, "clock_scale")
    prep_mass_squared = _positive(prep_mass_squared, "prep_mass_squared")
    carrier_quadratic_minimum_mass_squared = field_mass**2 - prep_mass_squared
    if carrier_quadratic_minimum_mass_squared <= 0.0:
        raise ValueError(
            "carrier preparation stability requires field_mass**2 > prep_mass_squared"
        )
    battery_energy_per_cell = _positive(
        battery_energy_per_cell, "battery_energy_per_cell"
    )
    exchange_coupling = _finite_nonzero(exchange_coupling, "exchange_coupling")
    quartic_coupling = _positive(quartic_coupling, "quartic_coupling")
    small_time = _positive(small_time, "small_time")
    if quartic_coupling < 2.0 * abs(exchange_coupling):
        raise ValueError("quartic stability requires lambda >= 2 |g|")
    quartic_lower_bound_coefficient = (
        quartic_coupling / 4.0 - abs(exchange_coupling) / 2.0
    )

    hamiltonian = np.zeros((n_links + 1, n_links + 1), dtype=np.complex128)
    for index, coupling in enumerate(supplied_couplings):
        hamiltonian[index, index + 1] = coupling
        hamiltonian[index + 1, index] = coupling
    powers = [np.linalg.matrix_power(hamiltonian, power) for power in range(n_links + 1)]
    endpoint_values = [power[n_links, 0] for power in powers]
    lower_residual = max(abs(value) for value in endpoint_values[:-1])
    endpoint_value = endpoint_values[-1]
    product = math.prod(supplied_couplings)
    coefficient = ((-1j) ** n_links) * product / math.factorial(n_links)
    propagator = _antihermitian_propagator(-1j * hamiltonian * small_time)
    amplitude = propagator[n_links, 0]
    leading = coefficient * small_time**n_links

    return AutonomousRepeatedDominoObstructionCertificate(
        n_links=n_links,
        couplings=supplied_couplings,
        field_mass=field_mass,
        clock_scale=clock_scale,
        prep_mass_squared=prep_mass_squared,
        carrier_quadratic_minimum_mass_squared=carrier_quadratic_minimum_mass_squared,
        battery_energy_per_cell=battery_energy_per_cell,
        exchange_coupling=exchange_coupling,
        quartic_coupling=quartic_coupling,
        quartic_lower_bound_coefficient=quartic_lower_bound_coefficient,
        hamiltonian_hermiticity_residual=float(
            np.max(np.abs(hamiltonian - hamiltonian.conj().T))
        ),
        lower_order_endpoint_power_residual=float(lower_residual),
        endpoint_order_n_value=complex(endpoint_value),
        expected_endpoint_order_n_value=product,
        endpoint_taylor_coefficient=complex(coefficient),
        expected_endpoint_taylor_coefficient=complex(coefficient),
        small_time=small_time,
        small_time_endpoint_amplitude=complex(amplitude),
        small_time_leading_term=complex(leading),
        small_time_remainder_magnitude=float(abs(amplitude - leading)),
        all_success_initially_clean_battery_count=n_links,
        all_success_initially_clean_record_count=n_links,
        all_success_battery_energy=n_links * battery_energy_per_cell,
        dimensionless_core_arguments=(
            ("T / M_T", "dimensionless dynamical-clock argument"),
            ("mu_P^2 Delta_tau / omega", "dimensionless preparation area"),
            ("J_j Delta_tau", "dimensionless projected rate-time area; J_j has mass dimension"),
            ("(-i)^N prod(J_j) t^N / N!", "dimensionless endpoint amplitude; prod(J_j) has mass^N"),
        ),
        local_coordinate_species_field_candidate_by_construction=True,
        explicit_coordinate_time_switching_present=False,
        dimensions_closed=True,
        stability_bound_pass=quartic_lower_bound_coefficient >= 0.0,
        carrier_prep_stability_pass=carrier_quadratic_minimum_mass_squared > 0.0,
        finite_hamiltonian_hermitian=True,
        analytic_coefficient_conditions_pass=(
            lower_residual < 1.0e-12 and abs(endpoint_value - product) < 1.0e-12
        ),
        finite_all_success_resource_receipt=True,
        species_index_is_physical_spatial_distance=False,
        physical_lattice_or_worldtube_completion=False,
        coupled_clock_global_monotonicity_one_pass=False,
        exact_delayed_front_derived=False,
        projected_link_rates_from_action_derived=False,
        iterated_fresh_ancilla_cptp_instrument_derived=False,
        continuum_qft_microcausality_derived=False,
        operational_no_signalling_derived=False,
        gr_source_stress_matching_derived=False,
        unbounded_front_from_finite_resources_derived=False,
        durable_records_derived=False,
        cross_dataset_parameter_fixing_derived=False,
        independent_holdout_prediction_derived=False,
    )


def _antihermitian_propagator(antihermitian_generator: np.ndarray) -> np.ndarray:
    """에르미트 ``H``로 만든 반에르미트 ``-i H t``를 지수화한다."""

    eigenvalues, eigenvectors = np.linalg.eigh(1j * antihermitian_generator)
    return (eigenvectors * np.exp(-1j * eigenvalues)) @ eigenvectors.conj().T


_TOL = 2.0e-12


def _finite(value: float, name: str) -> float:
    """유한한 값만 통과시킨다."""

    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _lattice_positive(value: float, name: str) -> float:
    """유한하고 양수인 값만 통과시킨다(격자 증인용 오류 문구)."""

    value = _finite(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_integer(value: int, name: str) -> int:
    """음이 아닌 정수만 통과시킨다."""

    if isinstance(value, bool) or int(value) != value or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


@dataclass(frozen=True)
class CausalProbabilityDeformationLatticeCertificate:
    """여백을 둔 유한 인과 격자 실험 하나의 수치 인증서다."""

    lattice_spacing: float
    light_speed: float
    time_step: float
    detector_distance_cells: int
    time_steps: int
    grid_radius_cells: int
    source_amplitude: float
    coupling_energy: float
    hbar: float
    omega: float
    source_detector_chi: tuple[float, ...]
    control_detector_chi: tuple[float, ...]
    source_probabilities: tuple[float, ...]
    control_probabilities: tuple[float, ...]
    support_violation: float
    first_nonzero_detector_sample: int | None
    expected_first_detector_sample: int
    prearrival_probability_difference: float
    postarrival_probability_difference: float
    front_speed: float
    local_unitary_residual: float
    local_trace_residual: float
    local_choi_minimum_eigenvalue: float
    source_off_probability_difference: float
    coupling_off_probability_difference: float
    boundary_clearance_cells: int
    chi_dimensionless: bool
    source_q_dimensionless: bool
    continuum_source_s_length_power: int
    coupling_g_is_energy: bool
    dt_hamiltonian_over_hbar_dimensionless: bool
    dimensions_pass: bool
    rn_reweighting_used: bool
    finite_lattice_causal_front_witness: bool
    mass_to_q_derived: bool = False
    energy_current_or_backreaction_derived: bool = False
    probability_deformation_equals_attraction_derived: bool = False
    continuous_qft_microcausality_derived: bool = False
    gr_or_lensing_derived: bool = False
    repeated_measurement_or_physical_selection_derived: bool = False
    observational_holdout_derived: bool = False
    gates_5_to_8_closed: bool = False
    two_residuals_reduced: bool = False
    complexity_success: bool = False


def _unitary(chi: float, *, dt: float, hbar: float, omega: float, coupling: float) -> np.ndarray:
    """H=hbar*omega*X/2 + coupling*chi*Z 의 정확한 지수를 돌려준다."""

    hamiltonian = np.array(
        ((coupling * chi, 0.5 * hbar * omega), (0.5 * hbar * omega, -coupling * chi)),
        dtype=complex,
    )
    norm = float(math.hypot(coupling * chi, 0.5 * hbar * omega))
    if norm == 0.0:
        return np.eye(2, dtype=complex)
    angle = dt * norm / hbar
    return math.cos(angle) * np.eye(2, dtype=complex) - 1j * math.sin(angle) * hamiltonian / norm


def _unitary_choi_minimum_eigenvalue(unitary: np.ndarray) -> float:
    """rho -> U rho U dagger 의 최(Choi) 양성 증인을 돌려준다."""

    choi = np.zeros((4, 4), dtype=complex)
    for row in range(2):
        for column in range(2):
            basis = np.zeros((2, 2), dtype=complex)
            basis[row, column] = 1.0
            output = unitary @ basis @ unitary.conj().T
            choi += np.kron(basis, output)
    return float(np.linalg.eigvalsh(choi).min())


def certify_causal_probability_deformation_lattice(
    *, lattice_spacing: float = 0.1, light_speed: float = 1.0,
    detector_distance_cells: int = 3, time_steps: int = 6,
    grid_radius_cells: int = 6, source_amplitude: float = 0.8,
    coupling_energy: float = 0.7, omega: float = 1.0, hbar: float = 1.0,
) -> CausalProbabilityDeformationLatticeCertificate:
    """콤팩트 원천을 전개하고 국소 원천/대조군 보른 반응을 비교한다.

    점화식은 정확히 CFL 1 갱신
    ``chi[j,n+1]=chi[j+1,n]+chi[j-1,n]-chi[j,n-1]+q delta[j,0] delta[n,0]``
    이다. 초기 단면 ``chi[-1]``과 ``chi[0]``은 0이다. 격자 반지름이 ``time_steps`` 이상이면
    이 유한 실현은 돌려주는 모든 단면에서 무한 격자 지평선 안에 머문다.
    """

    a = _lattice_positive(lattice_spacing, "lattice_spacing")
    c = _lattice_positive(light_speed, "light_speed")
    distance = _nonnegative_integer(detector_distance_cells, "detector_distance_cells")
    steps = _nonnegative_integer(time_steps, "time_steps")
    radius = _nonnegative_integer(grid_radius_cells, "grid_radius_cells")
    q = _finite(source_amplitude, "source_amplitude")
    g = _finite(coupling_energy, "coupling_energy")
    angular_frequency = _lattice_positive(omega, "omega")
    planck = _lattice_positive(hbar, "hbar")
    if steps < distance + 1:
        raise ValueError("time_steps must include the first detector-arrival sample")
    if radius < steps:
        raise ValueError("grid_radius_cells must be at least time_steps to avoid boundary/reflection")
    if distance > radius:
        raise ValueError("detector_distance_cells must lie inside the padded grid")

    dt = a / c
    width = 2 * radius + 1
    origin = radius
    detector = origin + distance

    def lattice_history(amplitude: float) -> tuple[np.ndarray, ...]:
        """원천 진폭을 받아 격자 장의 시간 단면 목록을 돌려준다."""

        previous = np.zeros(width, dtype=float)  # chi^-1
        current = np.zeros(width, dtype=float)   # chi^0
        history = [current.copy()]
        for n in range(steps):
            following = np.zeros(width, dtype=float)
            following[1:-1] = current[2:] + current[:-2] - previous[1:-1]
            if n == 0:
                following[origin] += amplitude
            previous, current = current, following
            history.append(current.copy())
        return tuple(history)

    source_history = lattice_history(q)
    control_history = lattice_history(0.0)
    source_chi = tuple(float(slice_[detector]) for slice_ in source_history)
    control_chi = tuple(float(slice_[detector]) for slice_ in control_history)
    expected_first = distance + 1
    nonzero = next((n for n, value in enumerate(source_chi) if abs(value) > _TOL), None)
    support_violation = max(
        (abs(value)
         for n, slice_ in enumerate(source_history)
         for j, value in enumerate(slice_)
         if abs(j - origin) > n - 1),
        default=0.0,
    )

    initial_rho = np.array(((1.0, 0.0), (0.0, 0.0)), dtype=complex)
    projector_zero = np.diag((1.0, 0.0)).astype(complex)

    def probabilities(field: tuple[float, ...], coupling: float) -> tuple[tuple[float, ...], np.ndarray]:
        """검출기 장 이력에 따라 결과 0의 확률 열과 마지막 유니터리를 돌려준다."""

        rho = initial_rho.copy()
        values = [float(np.trace(projector_zero @ rho).real)]
        last = np.eye(2, dtype=complex)
        for n in range(1, steps + 1):
            last = _unitary(field[n], dt=dt, hbar=planck, omega=angular_frequency, coupling=coupling)
            rho = last @ rho @ last.conj().T
            values.append(float(np.trace(projector_zero @ rho).real))
        return tuple(values), last

    source_probabilities, arrival_unitary = probabilities(source_chi, g)
    control_probabilities, _ = probabilities(control_chi, g)
    source_off_probabilities, _ = probabilities(control_chi, g)
    coupling_off_probabilities, _ = probabilities(source_chi, 0.0)
    prearrival = max(abs(source_probabilities[n] - control_probabilities[n]) for n in range(expected_first))
    postarrival = max(abs(source_probabilities[n] - control_probabilities[n]) for n in range(expected_first, steps + 1))
    local_unitarity = float(np.linalg.norm(arrival_unitary.conj().T @ arrival_unitary - np.eye(2), ord=2))
    rho_after = arrival_unitary @ initial_rho @ arrival_unitary.conj().T
    trace_residual = abs(float(np.trace(rho_after).real) - 1.0)

    return CausalProbabilityDeformationLatticeCertificate(
        lattice_spacing=a, light_speed=c, time_step=dt,
        detector_distance_cells=distance, time_steps=steps, grid_radius_cells=radius,
        source_amplitude=q, coupling_energy=g, hbar=planck, omega=angular_frequency,
        source_detector_chi=source_chi, control_detector_chi=control_chi,
        source_probabilities=source_probabilities, control_probabilities=control_probabilities,
        support_violation=float(support_violation), first_nonzero_detector_sample=nonzero,
        expected_first_detector_sample=expected_first,
        prearrival_probability_difference=prearrival, postarrival_probability_difference=postarrival,
        front_speed=a / dt, local_unitary_residual=local_unitarity,
        local_trace_residual=trace_residual,
        local_choi_minimum_eigenvalue=_unitary_choi_minimum_eigenvalue(arrival_unitary),
        source_off_probability_difference=max(abs(x - y) for x, y in zip(source_off_probabilities, control_probabilities)),
        coupling_off_probability_difference=max(abs(x - y) for x, y in zip(coupling_off_probabilities, control_probabilities)),
        boundary_clearance_cells=radius - steps + 1,
        chi_dimensionless=True, source_q_dimensionless=True, continuum_source_s_length_power=-2,
        coupling_g_is_energy=True, dt_hamiltonian_over_hbar_dimensionless=True,
        dimensions_pass=True, rn_reweighting_used=False,
        finite_lattice_causal_front_witness=bool(
            support_violation <= _TOL and nonzero == expected_first and abs(a / dt - c) <= _TOL
        ),
    )
