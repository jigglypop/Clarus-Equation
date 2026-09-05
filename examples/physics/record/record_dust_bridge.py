"""표시된 인과 기록에서 먼지 응력 텐서로 가는 구성적 브리지와 단일 시계 작용 허용 판정.

첫째 부분(인과 기록 → 먼지 브리지)은 의도적으로 정방향이다. 유한한 인과 기록
DAG에 먼저 정규화된 전이 핵(transition kernel)과 양의 에너지 표시를 부여한다.
에너지 표시가 그 핵 아래에서 조화(harmonic)이면, 유도된 개수 흐름과
기대 에너지 흐름은 모든 내부 기록에서 정확한 키르히호프(Kirchhoff) 균형을
만족한다.

그 다음 출구 반사슬(exit antichain)을 이미 지정된 코시(Cauchy) 셀 위의 점과
미래 질량껍질(mass-shell) 운동량으로 표시할 수 있다. 그 밀어내기(pushforward)는
양의 운동학적 측도다. 아래 모멘트 공식은 곡면 개수 규약을 쓴다.

    J^mu = integral (p^mu / E_N) dN,
    T^munu = integral (p^mu p^nu / E_N) dN,

여기서 ``E_N = -N.p`` 이다. 따라서 ``-N.J`` 는 곡면에서 측정한 개수 밀도이고
``T_NN`` 은 개수 밀도 곱하기 입자 에너지다.

이 구성은 DAG 자체, 그 물리적 에너지 스케일, 코시 기하, 질량껍질 표시를
맨 단일체(singleton)에서 유도하지 않는다. 그런 형이 있는 자료가 주어지면
전류, 응력, 이중 계수 없음, 단일 운동학적 먼지 극한이 추가 공리가 아니라
귀결임을 증명한다.

둘째 부분(비회전 먼지 작용 허용)은 양자 영수증에서 중력 원천을 구성하지
않는다. 더 좁은 명제를 검사한다. 이미 주어진 단일 운동학적 먼지 흐름이
매끄럽고, 단일 흐름이고, 비회전(irrotational)이고, 코스틱(caustic) 이전이면,
국소적으로

    S_d = -1/2 integral sqrt(-g) lambda ((grad tau)^2 + 1),
    u_mu = -partial_mu tau,

로 쓸 수 있고, 온셸(on-shell) 작용 응력은 기존 운동학적 먼지 응력과 정확히
같다. 따라서 두 응력은 하나의 배분에 대한 대응된 표현이며 더해서는 안 된다.
공식은 공변 작용을 서술하지만, 이 유한 증명서는 주어진 부호 (-,+,+,+)의
민코프스키(Minkowski) 국소 차트만 검사한다. 곡면 시공간 풀이기도, 코스틱
탐지기도 아니다.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


DEFAULT_TOLERANCE = 1.0e-12


def _finite_nonnegative(value: float, name: str) -> float:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _finite_positive(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _close(left: float, right: float, tolerance: float) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


@dataclass(frozen=True)
class CausalRecordNode:
    """물리적 에너지 표시를 가진 0차원 기록 상태 하나."""

    name: str
    label: str
    energy: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("record name must be non-empty")
        if not self.label:
            raise ValueError("record label must be non-empty")
        _finite_positive(self.energy, "record energy")


@dataclass(frozen=True)
class CausalTransition:
    """유향 인과 간선 위의 무차원 전이 확률."""

    source: str
    target: str
    probability: float

    def __post_init__(self) -> None:
        if not self.source or not self.target:
            raise ValueError("transition endpoints must be non-empty")
        if not math.isfinite(self.probability) or not 0.0 <= self.probability <= 1.0:
            raise ValueError("transition probability must lie in [0, 1]")


@dataclass(frozen=True)
class EdgeFlux:
    source: str
    target: str
    number_flux: float
    energy_flux: float


@dataclass(frozen=True)
class NodeBalance:
    node: str
    number_in: float
    number_out: float
    number_injected: float
    number_extracted: float
    number_residual: float
    energy_in: float
    energy_out: float
    energy_injected: float
    energy_extracted: float
    energy_residual: float


@dataclass(frozen=True)
class CausalRecordFlow:
    # 여기서 "양"은 측도론의 의미다. 모든 플럭스가 음이 아니다.
    """정규화된 인과 핵이 유도하는 유일한 양의 흐름."""

    topological_order: tuple[str, ...]
    node_weights: tuple[tuple[str, float], ...]
    edge_fluxes: tuple[EdgeFlux, ...]
    balances: tuple[NodeBalance, ...]
    terminal_nodes: tuple[str, ...]
    terminal_composition: tuple[tuple[str, float], ...]
    initial_number: float
    terminal_number: float
    initial_energy: float
    terminal_energy: float

    def weight(self, node: str) -> float:
        for candidate, value in self.node_weights:
            if candidate == node:
                return value
        raise KeyError(node)


def _topological_order(
    node_names: set[str], transitions: Sequence[CausalTransition]
) -> tuple[str, ...]:
    outgoing: dict[str, list[str]] = {name: [] for name in node_names}
    indegree = {name: 0 for name in node_names}
    seen_edges: set[tuple[str, str]] = set()
    for transition in transitions:
        edge = (transition.source, transition.target)
        if edge in seen_edges:
            raise ValueError("duplicate causal transition")
        seen_edges.add(edge)
        outgoing[transition.source].append(transition.target)
        indegree[transition.target] += 1

    ready = sorted(name for name, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while ready:
        current = ready.pop(0)
        order.append(current)
        for target in sorted(outgoing[current]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort()
    if len(order) != len(node_names):
        raise ValueError("causal record graph must be acyclic")
    return tuple(order)


def construct_conserved_record_flow(
    nodes: Sequence[CausalRecordNode],
    transitions: Sequence[CausalTransition],
    initial_weights: Mapping[str, float],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> CausalRecordFlow:
    """유한 DAG 위에 유일한 개수 흐름과 기대 에너지 흐름을 구성한다.

    전이 핵의 비말단 행은 모두 합이 1이어야 한다. 에너지 표시는 또한 조화여야
    한다.

        E(source) = sum_target P(source, target) E(target).

    초기 가중치는 그래프의 뿌리에만 허용한다. 이 가정 아래 위상 순서
    재귀가 모든 노드와 간선의 가중치를 유일하게 정한다.
    """

    _finite_positive(tolerance, "tolerance")
    node_by_name = {node.name: node for node in nodes}
    if len(node_by_name) != len(nodes) or not node_by_name:
        raise ValueError("record node names must be unique and non-empty")
    node_names = set(node_by_name)
    for transition in transitions:
        if transition.source not in node_names or transition.target not in node_names:
            raise ValueError("every transition endpoint must name a record node")

    order = _topological_order(node_names, transitions)
    outgoing: dict[str, list[CausalTransition]] = {name: [] for name in node_names}
    incoming: dict[str, list[CausalTransition]] = {name: [] for name in node_names}
    for transition in transitions:
        outgoing[transition.source].append(transition)
        incoming[transition.target].append(transition)

    roots = {name for name in node_names if not incoming[name]}
    for name, weight in initial_weights.items():
        if name not in node_names:
            raise ValueError("initial weight names an unknown record node")
        _finite_nonnegative(weight, "initial weight")
        if name not in roots and weight > tolerance:
            raise ValueError("initial weight is allowed only on causal roots")

    for name, row in outgoing.items():
        if not row:
            continue
        probability_sum = sum(edge.probability for edge in row)
        if not _close(probability_sum, 1.0, tolerance):
            raise ValueError("each nonterminal transition row must sum to one")
        expected_energy = sum(
            edge.probability * node_by_name[edge.target].energy for edge in row
        )
        if not _close(expected_energy, node_by_name[name].energy, tolerance):
            raise ValueError("record energy must be harmonic under the transition kernel")

    weights = {name: float(initial_weights.get(name, 0.0)) for name in node_names}
    edge_flux_by_pair: dict[tuple[str, str], EdgeFlux] = {}
    for name in order:
        for edge in outgoing[name]:
            number_flux = weights[name] * edge.probability
            energy_flux = number_flux * node_by_name[edge.target].energy
            edge_flux_by_pair[(edge.source, edge.target)] = EdgeFlux(
                source=edge.source,
                target=edge.target,
                number_flux=number_flux,
                energy_flux=energy_flux,
            )
            weights[edge.target] += number_flux

    terminal_nodes = tuple(sorted(name for name in node_names if not outgoing[name]))
    balances: list[NodeBalance] = []
    for name in order:
        number_in = sum(
            edge_flux_by_pair[(edge.source, edge.target)].number_flux
            for edge in incoming[name]
        )
        number_out = sum(
            edge_flux_by_pair[(edge.source, edge.target)].number_flux
            for edge in outgoing[name]
        )
        number_injected = float(initial_weights.get(name, 0.0))
        number_extracted = weights[name] if name in terminal_nodes else 0.0
        energy_in = sum(
            edge_flux_by_pair[(edge.source, edge.target)].energy_flux
            for edge in incoming[name]
        )
        energy_out = sum(
            edge_flux_by_pair[(edge.source, edge.target)].energy_flux
            for edge in outgoing[name]
        )
        energy_injected = number_injected * node_by_name[name].energy
        energy_extracted = (
            weights[name] * node_by_name[name].energy if name in terminal_nodes else 0.0
        )
        balances.append(
            NodeBalance(
                node=name,
                number_in=number_in,
                number_out=number_out,
                number_injected=number_injected,
                number_extracted=number_extracted,
                number_residual=(
                    number_out - number_in - number_injected + number_extracted
                ),
                energy_in=energy_in,
                energy_out=energy_out,
                energy_injected=energy_injected,
                energy_extracted=energy_extracted,
                energy_residual=(
                    energy_out - energy_in - energy_injected + energy_extracted
                ),
            )
        )

    initial_number = sum(float(initial_weights.get(name, 0.0)) for name in roots)
    terminal_number = sum(weights[name] for name in terminal_nodes)
    initial_energy = sum(
        float(initial_weights.get(name, 0.0)) * node_by_name[name].energy
        for name in roots
    )
    terminal_energy = sum(
        weights[name] * node_by_name[name].energy for name in terminal_nodes
    )
    if not _close(initial_number, terminal_number, tolerance):
        raise ArithmeticError("constructed record flow violates number conservation")
    if not _close(initial_energy, terminal_energy, tolerance):
        raise ArithmeticError("constructed record flow violates energy conservation")
    if any(
        abs(balance.number_residual) > tolerance
        or abs(balance.energy_residual) > tolerance
        for balance in balances
    ):
        raise ArithmeticError("constructed record flow violates a local balance law")

    label_weights: dict[str, float] = {}
    for name in terminal_nodes:
        label = node_by_name[name].label
        label_weights[label] = label_weights.get(label, 0.0) + weights[name]
    terminal_composition = tuple(
        sorted(
            (
                label,
                0.0 if terminal_number == 0.0 else weight / terminal_number,
            )
            for label, weight in label_weights.items()
        )
    )
    return CausalRecordFlow(
        topological_order=order,
        node_weights=tuple((name, weights[name]) for name in order),
        edge_fluxes=tuple(
            edge_flux_by_pair[pair] for pair in sorted(edge_flux_by_pair)
        ),
        balances=tuple(balances),
        terminal_nodes=terminal_nodes,
        terminal_composition=terminal_composition,
        initial_number=initial_number,
        terminal_number=terminal_number,
        initial_energy=initial_energy,
        terminal_energy=terminal_energy,
    )


Vector3 = tuple[float, float, float]
Vector4 = tuple[float, float, float, float]
Tensor4 = tuple[Vector4, Vector4, Vector4, Vector4]


@dataclass(frozen=True)
class ExitPhaseMark:
    """국소 정규직교 코시 틀 하나에 박아 넣은 말단 기록."""

    node: str
    position: Vector3
    mass: float
    spatial_momentum: Vector3
    residual_efficiency: float = 1.0

    def __post_init__(self) -> None:
        if not self.node:
            raise ValueError("phase mark must name a terminal record")
        _finite_positive(self.mass, "mass")
        if len(self.position) != 3 or len(self.spatial_momentum) != 3:
            raise ValueError("position and spatial momentum must have three components")
        if not all(math.isfinite(value) for value in self.position):
            raise ValueError("position components must be finite")
        if not all(math.isfinite(value) for value in self.spatial_momentum):
            raise ValueError("momentum components must be finite")
        if (
            not math.isfinite(self.residual_efficiency)
            or not 0.0 <= self.residual_efficiency <= 1.0
        ):
            raise ValueError("residual efficiency must lie in [0, 1]")

    @property
    def energy(self) -> float:
        return math.sqrt(
            self.mass * self.mass
            + sum(component * component for component in self.spatial_momentum)
        )

    @property
    def four_momentum(self) -> Vector4:
        return (self.energy, *self.spatial_momentum)


@dataclass(frozen=True)
class MatchedExitRecord:
    node: str
    label: str
    position: Vector3
    mass: float
    four_momentum: Vector4
    full_number: float
    residual_number: float
    complement_number: float


@dataclass(frozen=True)
class KineticMatching:
    """코시 셀 하나 위의 양의 출구 측도와 그 개수·응력 모멘트."""

    records: tuple[MatchedExitRecord, ...]
    cell_volume: float
    current: Vector4
    stress: Tensor4
    surface_number_density: float
    complement_number_density: float
    total_number_density: float
    residual_energy_density: float
    complement_energy_density: float
    total_energy_density: float
    no_double_counting_residual: float


def match_exit_antichain(
    flow: CausalRecordFlow,
    nodes: Sequence[CausalRecordNode],
    marks: Sequence[ExitPhaseMark],
    *,
    cell_volume: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> KineticMatching:
    """말단 기록 측도를 미래 질량껍질 측도로 밀어낸다.

    ``residual_efficiency`` 는 명시적 분할을 수행한다. 잔여 가중치와 여집합
    가중치는 기록마다 전체 말단 가중치로 합산된다. 기록의 에너지 표시는
    국소 질량껍질 에너지와 같아야 하므로, 이산 에너지 장부와 코시 자료는
    같은 형의 양이다.
    """

    _finite_positive(cell_volume, "cell volume")
    _finite_positive(tolerance, "tolerance")
    node_by_name = {node.name: node for node in nodes}
    if len(node_by_name) != len(nodes):
        raise ValueError("record node names must be unique")
    mark_by_node = {mark.node: mark for mark in marks}
    if len(mark_by_node) != len(marks):
        raise ValueError("terminal phase marks must be unique by node")
    if set(mark_by_node) != set(flow.terminal_nodes):
        raise ValueError("phase marks must cover exactly the terminal antichain")

    records: list[MatchedExitRecord] = []
    current = [0.0, 0.0, 0.0, 0.0]
    stress = [[0.0 for _ in range(4)] for _ in range(4)]
    residual_number = 0.0
    complement_number = 0.0
    total_number = 0.0
    residual_energy = 0.0
    complement_energy = 0.0
    total_energy = 0.0
    for name in flow.terminal_nodes:
        if name not in node_by_name:
            raise ValueError("terminal flow node is absent from node metadata")
        node = node_by_name[name]
        mark = mark_by_node[name]
        if not _close(node.energy, mark.energy, tolerance):
            raise ValueError("record energy and future mass-shell energy must agree")
        full_weight = flow.weight(name)
        residual_weight = full_weight * mark.residual_efficiency
        complement_weight = full_weight - residual_weight
        momentum = mark.four_momentum
        energy = momentum[0]
        records.append(
            MatchedExitRecord(
                node=name,
                label=node.label,
                position=mark.position,
                mass=mark.mass,
                four_momentum=momentum,
                full_number=full_weight,
                residual_number=residual_weight,
                complement_number=complement_weight,
            )
        )
        total_number += full_weight
        residual_number += residual_weight
        complement_number += complement_weight
        total_energy += full_weight * energy
        residual_energy += residual_weight * energy
        complement_energy += complement_weight * energy
        for mu in range(4):
            current[mu] += residual_weight * momentum[mu] / energy
            for nu in range(4):
                stress[mu][nu] += (
                    residual_weight * momentum[mu] * momentum[nu] / energy
                )

    inverse_volume = 1.0 / cell_volume
    current_tuple: Vector4 = tuple(  # type: ignore[assignment]
        value * inverse_volume for value in current
    )
    stress_tuple: Tensor4 = tuple(  # type: ignore[assignment]
        tuple(value * inverse_volume for value in row) for row in stress
    )
    no_double_counting_residual = total_energy - residual_energy - complement_energy
    if abs(no_double_counting_residual) > tolerance:
        raise ArithmeticError("residual and complement energy double-count or lose weight")
    if not _close(total_number, flow.terminal_number, tolerance):
        raise ArithmeticError("Cauchy matching changed the terminal record number")
    if not _close(total_energy, flow.terminal_energy, tolerance):
        raise ArithmeticError("Cauchy matching changed the terminal record energy")
    return KineticMatching(
        records=tuple(records),
        cell_volume=cell_volume,
        current=current_tuple,
        stress=stress_tuple,
        surface_number_density=residual_number * inverse_volume,
        complement_number_density=complement_number * inverse_volume,
        total_number_density=total_number * inverse_volume,
        residual_energy_density=residual_energy * inverse_volume,
        complement_energy_density=complement_energy * inverse_volume,
        total_energy_density=total_energy * inverse_volume,
        no_double_counting_residual=no_double_counting_residual * inverse_volume,
    )


@dataclass(frozen=True)
class DustInitialData:
    mass: float
    energy: float
    gamma: float
    four_velocity: Vector4
    surface_number_density: float
    rest_number_density: float
    rest_energy_density: float
    current: Vector4
    stress: Tensor4


def monokinetic_dust_data(
    matching: KineticMatching,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> DustInitialData:
    """양의 단일 운동량 운동학적 측도를 정확한 먼지 자료로 환원한다."""

    _finite_positive(tolerance, "tolerance")
    active = [record for record in matching.records if record.residual_number > tolerance]
    if not active:
        raise ValueError("monokinetic dust requires positive residual weight")
    reference = active[0]
    for record in active[1:]:
        if not _close(record.mass, reference.mass, tolerance) or any(
            not _close(left, right, tolerance)
            for left, right in zip(record.four_momentum, reference.four_momentum)
        ):
            raise ValueError("multi-stream or multi-mass data require kinetic/multifluid EFT")

    mass = reference.mass
    energy = reference.four_momentum[0]
    gamma = energy / mass
    four_velocity: Vector4 = tuple(  # type: ignore[assignment]
        component / mass for component in reference.four_momentum
    )
    rest_number_density = matching.surface_number_density / gamma
    rest_energy_density = mass * rest_number_density
    expected_current: Vector4 = tuple(  # type: ignore[assignment]
        rest_number_density * component for component in four_velocity
    )
    expected_stress: Tensor4 = tuple(  # type: ignore[assignment]
        tuple(
            rest_energy_density * four_velocity[mu] * four_velocity[nu]
            for nu in range(4)
        )
        for mu in range(4)
    )
    if any(
        not _close(left, right, tolerance)
        for left, right in zip(expected_current, matching.current)
    ):
        raise ArithmeticError("monokinetic current does not equal n u")
    if any(
        not _close(expected_stress[mu][nu], matching.stress[mu][nu], tolerance)
        for mu in range(4)
        for nu in range(4)
    ):
        raise ArithmeticError("monokinetic stress does not equal rho u tensor u")
    return DustInitialData(
        mass=mass,
        energy=energy,
        gamma=gamma,
        four_velocity=four_velocity,
        surface_number_density=matching.surface_number_density,
        rest_number_density=rest_number_density,
        rest_energy_density=rest_energy_density,
        current=matching.current,
        stress=matching.stress,
    )


def free_stream_exit_marks(
    marks: Sequence[ExitPhaseMark],
    *,
    coordinate_time: float,
    box_length: float,
) -> tuple[ExitPhaseMark, ...]:
    """주기적 공간 상자 위의 정확한 민코프스키 측지선 밀어내기."""

    if not math.isfinite(coordinate_time):
        raise ValueError("coordinate time must be finite")
    _finite_positive(box_length, "box length")
    streamed = []
    for mark in marks:
        velocity = tuple(component / mark.energy for component in mark.spatial_momentum)
        position: Vector3 = tuple(  # type: ignore[assignment]
            (coordinate + coordinate_time * speed) % box_length
            for coordinate, speed in zip(mark.position, velocity)
        )
        streamed.append(
            ExitPhaseMark(
                node=mark.node,
                position=position,
                mass=mark.mass,
                spatial_momentum=mark.spatial_momentum,
                residual_efficiency=mark.residual_efficiency,
            )
        )
    return tuple(streamed)


@dataclass(frozen=True)
class FlatFLRWCauchyWitness:
    """구속 조건을 만족하는 균질 평탄 토러스 먼지 초기 자료."""

    energy_density: float
    newton_constant: float
    hubble_rate: float
    spatial_metric_diagonal: Vector3
    extrinsic_curvature_diagonal: Vector3
    hamiltonian_left: float
    hamiltonian_right: float
    hamiltonian_residual: float
    momentum_residual: Vector3


def flat_flrw_cauchy_witness(
    dust: DustInitialData,
    *,
    newton_constant: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> FlatFLRWCauchyWitness:
    """균질 공동 운동 먼지를 일반상대론 구속 조건을 만족하는 자료에 박아 넣는다.

    증인은 단위 평탄 3-토러스 ``h_ij = delta_ij`` 와 ``K_ij = -H h_ij`` 를 쓴다.
    공동 운동 먼지에서는 운동량 구속이 사라지고, 해밀토니안(Hamiltonian)
    구속이 ``H^2 = 8 pi G rho / 3`` 을 정한다.
    """

    _finite_positive(newton_constant, "Newton constant")
    _finite_positive(tolerance, "tolerance")
    if any(abs(component) > tolerance for component in dust.four_velocity[1:]):
        raise ValueError("flat FLRW witness requires comoving monokinetic dust")
    density = dust.rest_energy_density
    hubble = math.sqrt(8.0 * math.pi * newton_constant * density / 3.0)
    hamiltonian_left = 6.0 * hubble * hubble
    hamiltonian_right = 16.0 * math.pi * newton_constant * density
    return FlatFLRWCauchyWitness(
        energy_density=density,
        newton_constant=newton_constant,
        hubble_rate=hubble,
        spatial_metric_diagonal=(1.0, 1.0, 1.0),
        extrinsic_curvature_diagonal=(-hubble, -hubble, -hubble),
        hamiltonian_left=hamiltonian_left,
        hamiltonian_right=hamiltonian_right,
        hamiltonian_residual=hamiltonian_left - hamiltonian_right,
        momentum_residual=(0.0, 0.0, 0.0),
    )


MINKOWSKI_DIAGONAL = (-1.0, 1.0, 1.0, 1.0)


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _finite_positive_value(value: float, name: str) -> float:
    value = _finite(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _vector4(values: Sequence[float], name: str) -> Vector4:
    if len(values) != 4:
        raise ValueError(f"{name} must have four components")
    return tuple(_finite(value, name) for value in values)  # type: ignore[return-value]


def _tensor4(values: Sequence[Sequence[float]], name: str) -> Tensor4:
    if len(values) != 4 or any(len(row) != 4 for row in values):
        raise ValueError(f"{name} must be a 4 by 4 tensor")
    return tuple(_vector4(row, name) for row in values)  # type: ignore[return-value]


def _max_abs(values: Sequence[float]) -> float:
    return max((abs(value) for value in values), default=0.0)


def lower_minkowski(vector: Sequence[float]) -> Vector4:
    """반변 벡터의 첨자를 diag(-1,1,1,1) 로 내린다."""

    vector4 = _vector4(vector, "vector")
    return tuple(  # type: ignore[return-value]
        sign * component for sign, component in zip(MINKOWSKI_DIAGONAL, vector4)
    )


def raise_minkowski(covector: Sequence[float]) -> Vector4:
    """공변 벡터의 첨자를 역 민코프스키 계량으로 올린다."""

    return lower_minkowski(covector)


def minkowski_covector_inner(
    left: Sequence[float], right: Sequence[float]
) -> float:
    left4 = _vector4(left, "left covector")
    right4 = _vector4(right, "right covector")
    return sum(
        sign * left4[index] * right4[index]
        for index, sign in enumerate(MINKOWSKI_DIAGONAL)
    )


def clock_constraint_residual(clock_gradient_covector: Sequence[float]) -> float:
    """승수 구속 조건의 (grad tau)^2 + 1 을 돌려준다."""

    return minkowski_covector_inner(
        clock_gradient_covector, clock_gradient_covector
    ) + 1.0


def clock_gradient_from_receipt(
    receipt_gradient_covector: Sequence[float],
    *,
    reference_mass_scale: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Vector4:
    """tau=r/M_* 를 적용하고, 단위 시간꼴 시계가 아닌 영수증은 거부한다."""

    scale = _finite_positive_value(reference_mass_scale, "reference_mass_scale")
    tolerance = _finite_positive_value(tolerance, "tolerance")
    receipt_gradient = _vector4(
        receipt_gradient_covector, "receipt gradient covector"
    )
    clock_gradient: Vector4 = tuple(  # type: ignore[assignment]
        component / scale for component in receipt_gradient
    )
    if abs(clock_constraint_residual(clock_gradient)) > tolerance:
        raise ValueError(
            "receipt gradient must obey (grad r)^2=-M_*^2; "
            "a constant receipt is not an admissible clock"
        )
    return clock_gradient


def multiplier_dust_stress_covariant(
    lambda_density: float,
    clock_gradient_covector: Sequence[float],
) -> Tensor4:
    """시계 작용의 정확한 오프셸(off-shell) 계량 변분을 돌려준다."""

    density = _finite(lambda_density, "lambda density")
    gradient = _vector4(clock_gradient_covector, "clock gradient covector")
    constraint = clock_constraint_residual(gradient)
    return tuple(  # type: ignore[return-value]
        tuple(
            density * gradient[mu] * gradient[nu]
            - 0.5
            * density
            * constraint
            * (MINKOWSKI_DIAGONAL[mu] if mu == nu else 0.0)
            for nu in range(4)
        )
        for mu in range(4)
    )


def raise_rank_two_minkowski(tensor_covariant: Sequence[Sequence[float]]) -> Tensor4:
    tensor = _tensor4(tensor_covariant, "covariant tensor")
    return tuple(  # type: ignore[return-value]
        tuple(
            MINKOWSKI_DIAGONAL[mu]
            * MINKOWSKI_DIAGONAL[nu]
            * tensor[mu][nu]
            for nu in range(4)
        )
        for mu in range(4)
    )


def velocity_vorticity_residual(
    velocity_covector_jacobian: Sequence[Sequence[float]],
) -> float:
    """평탄 차트에서 max |partial_mu u_nu-partial_nu u_mu| 를 돌려준다."""

    jacobian = _tensor4(
        velocity_covector_jacobian, "velocity covector jacobian"
    )
    return max(
        abs(jacobian[mu][nu] - jacobian[nu][mu])
        for mu in range(4)
        for nu in range(4)
    )


@dataclass(frozen=True)
class EpsilonReparameterizationReceipt:
    epsilon: float
    multiplier_density: float
    physical_density: float
    absorbed_multiplier_density: float
    exact_stress_reparameterization: bool
    epsilon_is_independent_dust_coupling: bool
    finite_epsilon_gr_limit_derived: bool


def epsilon_reparameterization(
    *, epsilon: float, multiplier_density: float
) -> EpsilonReparameterizationReceipt:
    """0이 아닌 전체 epsilon이 승수에 흡수됨을 보인다."""

    epsilon = _finite_positive_value(epsilon, "epsilon")
    multiplier = _finite(multiplier_density, "multiplier_density")
    physical_density = epsilon * multiplier
    return EpsilonReparameterizationReceipt(
        epsilon=epsilon,
        multiplier_density=multiplier,
        physical_density=physical_density,
        absorbed_multiplier_density=physical_density,
        exact_stress_reparameterization=True,
        epsilon_is_independent_dust_coupling=False,
        finite_epsilon_gr_limit_derived=False,
    )


@dataclass(frozen=True)
class VorticalDustCounterexample:
    kappa: float
    y: float
    four_velocity: Vector4
    norm_residual: float
    continuity_residual: float
    geodesic_residual: Vector4
    vorticity_residual: float
    kinetic_dust_admissible_at_point: bool
    single_clock_admissible: bool


def vortical_monokinetic_counterexample(
    *, kappa: float = 0.2, y: float = 1.0
) -> VorticalDustCounterexample:
    """어떤 단일 스칼라 시계로도 표현되지 않는 정규화된 측지 먼지 장.

    영역 |kappa*y|<1 에서 u^mu=gamma(1,kappa*y,0,0) 을 쓴다. 이 장은 y에만
    의존하고 u^y=0 이므로, 흐름선은 측지선이고 상수 밀도 전류는 보존된다.
    그럼에도 그 공변 벡터의 회전(curl)은 0이 아니다.
    """

    kappa = _finite(kappa, "kappa")
    y = _finite(y, "y")
    speed = kappa * y
    if not abs(speed) < 1.0:
        raise ValueError("vortical witness requires |kappa*y|<1")
    gamma = 1.0 / math.sqrt(1.0 - speed * speed)
    velocity: Vector4 = (gamma, gamma * speed, 0.0, 0.0)
    jacobian = [[0.0 for _ in range(4)] for _ in range(4)]
    jacobian[2][0] = -(gamma**3) * kappa * kappa * y
    jacobian[2][1] = (gamma**3) * kappa
    geodesic = tuple(  # type: ignore[assignment]
        sum(velocity[mu] * jacobian[mu][nu] for mu in range(4))
        for nu in range(4)
    )
    vorticity = velocity_vorticity_residual(jacobian)
    norm = sum(
        sign * component * component
        for sign, component in zip(MINKOWSKI_DIAGONAL, velocity)
    )
    return VorticalDustCounterexample(
        kappa=kappa,
        y=y,
        four_velocity=velocity,
        norm_residual=norm + 1.0,
        continuity_residual=0.0,
        geodesic_residual=geodesic,
        vorticity_residual=vorticity,
        kinetic_dust_admissible_at_point=True,
        single_clock_admissible=False,
    )


@dataclass(frozen=True)
class IrrotationalDustActionReceipt:
    lambda_density: float
    clock_gradient_covector: Vector4
    receipt_gradient_covector: Vector4
    four_velocity: Vector4
    action_stress_covariant: Tensor4
    action_stress_contravariant: Tensor4
    kinetic_stress_contravariant: Tensor4
    action_energy_current: Vector4
    kinetic_mass_current: Vector4
    clock_constraint_residual: float
    continuity_residual: float
    normalization_derivative_residual: float
    vorticity_residual: float
    geodesic_residual: float
    ward_residual: float
    stress_match_residual: float
    mass_current_match_residual: float
    rest_energy_density: float
    isotropic_pressure: float
    equation_of_state: float | None
    coordinate_speed: float
    proper_time_interval_squared: float
    worldline_speed_below_c: bool
    receipt_mass_dimension: int
    reference_scale_mass_dimension: int
    clock_mass_dimension: int
    multiplier_mass_dimension: int
    stress_mass_dimension: int
    action_density_mass_dimension: int
    volume_element_mass_dimension: int
    action_mass_dimension: int
    dimensions_pass: bool
    lambda_nonnegative_admission: bool
    smooth_single_stream_irrotational_precaustic: bool
    matched_not_summed: bool
    action_is_variational_reexpression_not_new_energy: bool
    local_receipt_field_map_derived: bool
    reference_mass_scale_derived: bool
    multiplier_initial_law_derived: bool
    supplied_metric_derived: bool
    gravitational_boson_derived: bool
    cptp_quantum_dynamics_derived: bool
    qft_microcausality_derived: bool
    operational_no_signalling_derived: bool
    finite_coefficient_gr_phenomenology_derived: bool
    independent_holdout_prediction_derived: bool
    two_residual_classes_reduced: bool
    complexity_penalty_success: bool
    curved_metric_continuum_verified: bool
    flow_jacobian_computed_from_dynamics: bool


def admit_irrotational_dust_action(
    dust: DustInitialData,
    *,
    receipt_gradient_covector: Sequence[float],
    reference_mass_scale: float,
    velocity_covector_jacobian: Sequence[Sequence[float]] = (
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
    ),
    density_gradient_covector: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
    lagrangian_flow_jacobian: float = 1.0,
    single_stream: bool = True,
    smooth: bool = True,
    accounting_mode: str = "matched_not_summed",
    proper_time_step: float = 1.0,
    tolerance: float = DEFAULT_TOLERANCE,
) -> IrrotationalDustActionReceipt:
    """주어진 운동학적 먼지 자료와의 정확한 국소 동치를 증명한다.

    ``lagrangian_flow_jacobian`` 은 코스틱 이전 계약값으로 주어진다. 이 함수는
    양이 아닌 값을 거부하지만, 흐름 사상을 유도하거나 그 코스틱을 찾지는
    않는다.
    """

    tolerance = _finite_positive_value(tolerance, "tolerance")
    scale = _finite_positive_value(reference_mass_scale, "reference_mass_scale")
    flow_jacobian = _finite(lagrangian_flow_jacobian, "lagrangian_flow_jacobian")
    proper_time_step = _finite_positive_value(proper_time_step, "proper_time_step")
    if not smooth:
        raise ValueError("single-clock action admission requires a smooth flow")
    if not single_stream:
        raise ValueError("multistream data require a kinetic or multifluid route")
    if flow_jacobian <= tolerance:
        raise ValueError("caustic or shell crossing invalidates the single-clock chart")
    if accounting_mode != "matched_not_summed":
        raise ValueError("kinetic and action stresses must be matched, not summed")

    density = _finite(dust.rest_energy_density, "dust rest energy density")
    if density < 0.0:
        raise ValueError("lambda density must be non-negative by admission")
    mass = _finite_positive_value(dust.mass, "dust mass")
    velocity = _vector4(dust.four_velocity, "dust four velocity")
    velocity_covector = lower_minkowski(velocity)
    norm = sum(
        sign * velocity[index] * velocity[index]
        for index, sign in enumerate(MINKOWSKI_DIAGONAL)
    )
    if not _close(norm, -1.0, tolerance):
        raise ValueError("dust four velocity must be future unit timelike")
    if velocity[0] <= 0.0:
        raise ValueError("dust four velocity must be future directed")

    clock_gradient = clock_gradient_from_receipt(
        receipt_gradient_covector,
        reference_mass_scale=scale,
        tolerance=tolerance,
    )
    expected_clock_gradient: Vector4 = tuple(  # type: ignore[assignment]
        -component for component in velocity_covector
    )
    if any(
        not _close(actual, expected, tolerance)
        for actual, expected in zip(clock_gradient, expected_clock_gradient)
    ):
        raise ValueError("receipt clock must satisfy u_mu=-partial_mu tau")

    jacobian = _tensor4(
        velocity_covector_jacobian, "velocity covector jacobian"
    )
    density_gradient = _vector4(
        density_gradient_covector, "density gradient covector"
    )
    vorticity = velocity_vorticity_residual(jacobian)
    if vorticity > tolerance:
        raise ValueError("single-clock action requires an irrotational velocity field")

    normalization_derivative = max(
        abs(sum(velocity[nu] * jacobian[mu][nu] for nu in range(4)))
        for mu in range(4)
    )
    if normalization_derivative > tolerance:
        raise ValueError("unit velocity normalization must hold through the local flow")

    divergence_velocity = sum(
        MINKOWSKI_DIAGONAL[mu] * jacobian[mu][mu] for mu in range(4)
    )
    continuity = sum(
        velocity[mu] * density_gradient[mu] for mu in range(4)
    ) + density * divergence_velocity
    if abs(continuity) > tolerance:
        raise ValueError("multiplier current must satisfy the continuity equation")

    geodesic_covector: Vector4 = tuple(  # type: ignore[assignment]
        sum(velocity[mu] * jacobian[mu][nu] for mu in range(4))
        for nu in range(4)
    )
    geodesic = _max_abs(geodesic_covector)
    if geodesic > tolerance:
        raise ValueError("admitted dust flow must be geodesic")

    action_stress_covariant = multiplier_dust_stress_covariant(
        density, clock_gradient
    )
    action_stress_contravariant = raise_rank_two_minkowski(
        action_stress_covariant
    )
    kinetic_stress = _tensor4(dust.stress, "kinetic dust stress")
    expected_kinetic_stress: Tensor4 = tuple(  # type: ignore[assignment]
        tuple(density * velocity[mu] * velocity[nu] for nu in range(4))
        for mu in range(4)
    )
    kinetic_internal_residual = max(
        abs(kinetic_stress[mu][nu] - expected_kinetic_stress[mu][nu])
        for mu in range(4)
        for nu in range(4)
    )
    if kinetic_internal_residual > tolerance:
        raise ValueError("supplied kinetic stress is not rho u tensor u")
    stress_match = max(
        abs(action_stress_contravariant[mu][nu] - kinetic_stress[mu][nu])
        for mu in range(4)
        for nu in range(4)
    )
    if stress_match > tolerance:
        raise ArithmeticError("action and kinetic dust stresses do not match")

    kinetic_current = _vector4(dust.current, "kinetic number current")
    action_energy_current: Vector4 = tuple(  # type: ignore[assignment]
        density * component for component in velocity
    )
    kinetic_mass_current: Vector4 = tuple(  # type: ignore[assignment]
        mass * component for component in kinetic_current
    )
    current_match = _max_abs(
        tuple(
            action_energy_current[index] - kinetic_mass_current[index]
            for index in range(4)
        )
    )
    if current_match > tolerance:
        raise ValueError("lambda u must match mass times the kinetic number current")

    ward = max(
        abs(continuity * velocity[nu] + density * geodesic_covector[nu])
        for nu in range(4)
    )
    spatial_projector_contraction = sum(
        (
            (MINKOWSKI_DIAGONAL[mu] if mu == nu else 0.0)
            + velocity[mu] * velocity[nu]
        )
        * action_stress_covariant[mu][nu]
        for mu in range(4)
        for nu in range(4)
    )
    pressure = spatial_projector_contraction / 3.0
    coordinate_speed = math.sqrt(sum(component * component for component in velocity[1:])) / velocity[0]
    displacement = tuple(component * proper_time_step for component in velocity)
    interval_squared = sum(
        sign * component * component
        for sign, component in zip(MINKOWSKI_DIAGONAL, displacement)
    )

    return IrrotationalDustActionReceipt(
        lambda_density=density,
        clock_gradient_covector=clock_gradient,
        receipt_gradient_covector=_vector4(
            receipt_gradient_covector, "receipt gradient covector"
        ),
        four_velocity=velocity,
        action_stress_covariant=action_stress_covariant,
        action_stress_contravariant=action_stress_contravariant,
        kinetic_stress_contravariant=kinetic_stress,
        action_energy_current=action_energy_current,
        kinetic_mass_current=kinetic_mass_current,
        clock_constraint_residual=clock_constraint_residual(clock_gradient),
        continuity_residual=continuity,
        normalization_derivative_residual=normalization_derivative,
        vorticity_residual=vorticity,
        geodesic_residual=geodesic,
        ward_residual=ward,
        stress_match_residual=stress_match,
        mass_current_match_residual=current_match,
        rest_energy_density=density,
        isotropic_pressure=pressure,
        equation_of_state=0.0 if density > 0.0 else None,
        coordinate_speed=coordinate_speed,
        proper_time_interval_squared=interval_squared,
        worldline_speed_below_c=coordinate_speed < 1.0,
        receipt_mass_dimension=0,
        reference_scale_mass_dimension=1,
        clock_mass_dimension=-1,
        multiplier_mass_dimension=4,
        stress_mass_dimension=4,
        action_density_mass_dimension=4,
        volume_element_mass_dimension=-4,
        action_mass_dimension=0,
        dimensions_pass=True,
        lambda_nonnegative_admission=density >= 0.0,
        smooth_single_stream_irrotational_precaustic=True,
        matched_not_summed=True,
        action_is_variational_reexpression_not_new_energy=True,
        local_receipt_field_map_derived=False,
        reference_mass_scale_derived=False,
        multiplier_initial_law_derived=False,
        supplied_metric_derived=False,
        gravitational_boson_derived=False,
        cptp_quantum_dynamics_derived=False,
        qft_microcausality_derived=False,
        operational_no_signalling_derived=False,
        finite_coefficient_gr_phenomenology_derived=False,
        independent_holdout_prediction_derived=False,
        two_residual_classes_reduced=False,
        complexity_penalty_success=False,
        curved_metric_continuum_verified=False,
        flow_jacobian_computed_from_dynamics=False,
    )


def canonical_dust_data() -> DustInitialData:
    """증명서가 쓰는 자명하지 않은 정확한 민코프스키 먼지 자료."""

    mass = 2.0
    rest_number_density = 3.0
    rest_energy_density = mass * rest_number_density
    velocity: Vector4 = (1.25, 0.75, 0.0, 0.0)
    current: Vector4 = tuple(  # type: ignore[assignment]
        rest_number_density * component for component in velocity
    )
    stress: Tensor4 = tuple(  # type: ignore[assignment]
        tuple(
            rest_energy_density * velocity[mu] * velocity[nu]
            for nu in range(4)
        )
        for mu in range(4)
    )
    return DustInitialData(
        mass=mass,
        energy=mass * velocity[0],
        gamma=velocity[0],
        four_velocity=velocity,
        surface_number_density=rest_number_density * velocity[0],
        rest_number_density=rest_number_density,
        rest_energy_density=rest_energy_density,
        current=current,
        stress=stress,
    )


def certificate() -> IrrotationalDustActionReceipt:
    """정본 정확 조건부 허용 증명서를 돌려준다."""

    scale = 2.0
    dust = canonical_dust_data()
    velocity_covector = lower_minkowski(dust.four_velocity)
    receipt_gradient = tuple(
        -scale * component for component in velocity_covector
    )
    return admit_irrotational_dust_action(
        dust,
        receipt_gradient_covector=receipt_gradient,
        reference_mass_scale=scale,
    )
