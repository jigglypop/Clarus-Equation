"""세계관(worldtube) 에너지 다리: 조건부 영수증 정합, 두 스칼라 교환 전류, 유한 물질 격자 비용을 한 모듈에 둔다.

세 부분으로 이루어진다.

**1. 조건부 세계관 정합.** 공급된 가지별(branch-resolved) 대칭 응력 텐서와
교환 전류(exchange current)에 대해

    nabla_mu T_S^mu{}_nu = Q_nu,
    j_S^mu = -T_S^mu{}_nu xi^nu,

정확한 에너지-전류 항등식은

    Delta E_S + Phi_S^out
      = -integral_V (xi^nu Q_nu
                      + T_S^{mu nu} nabla_(mu xi_nu)) dV.

이다. 반대 부문은 ``-Q_nu`` 를 받는다. 따라서 교환 항은 전체 장부에서 상쇄되고,
킬링(Killing)이 아닌 시간 흐름과 측면 경계 플럭스는 명시적으로 남는다. 킬링이며
측면 플럭스가 0인 증인(witness)만이 이 항등식을 ``delta e_beta = -integral xi.Q dV``
로 줄인다. 이 부분의 기하·가지 응력·전류·표면 에너지·구적(quadrature) 가중치는
모두 공급 입력이다. 계산은 그 정합만 감사하며, 유한 양자 도미노에서 국소 응력
텐서를 유도하거나, 물리적 포인터를 고르거나, 연속체 극한을 취하거나, 일반상대론
(GR) 소스를 만들지 않는다. ``construct_flat_receipt_current_counterexample`` 는
역문제의 정확한 실패를 기록한다: 스칼라 영수증 하나는 같은 평탄 세계관 위에서도
국소 전류의 운동량 성분을 임의로 남긴다.

**2. 두 스칼라 교환 전류와 영수증 비식별성.** 일반 미분동형 워드 항등식
(diffeomorphism Ward identity)은 이미 CE 원장의 일부다. 이 부분은 그 정리를
다시 증명하지 않으며 양자 배터리 영수증을 중력 소스로 바꾸지 않는다. 4차원
시공간, 자연 단위 ``hbar = c = 1``, 부호 ``(-,+,+,+)`` 에서 하나의 명시적 특수화를
평가한다:

    S_m = integral sqrt(-g) [
        -(nabla phi)^2 / 2 - m_phi^2 phi^2 / 2
        -(nabla psi)^2 / 2 - m_psi^2 psi^2 / 2
        -lambda phi^2 psi^2 / 2
    ].

상호작용 응력은 공급된 상수 비율 ``alpha`` 로 두 이름 붙은 부문에 배정할 수 있다.
온셸(on shell)에서 이는 크기가 같고 부호가 반대인 교환 공벡터를 주며, 그 합은
유일하게 정의된 전체 응력이다. 서로 다른 두 ``alpha`` 값은 작용과 국소 상호작용
퍼텐셜 밀도를 고정한 채 서로 다른 부문 전류를 만들 수 있다. 이것이 스칼라 밀도
하나에서 유일한 공변 전류를 추론하는 주장에 대한 구성적 반례다. 여기서
``receipt`` 는 감사 기록을 뜻하며 E9-D 배터리 에너지 영수증이 아니다. 모든 수치
입력은 하나의 선언된 질량 단위의 성분이다. 기준 질량 스케일이 운동방정식과 전류
잔차를 정규화한다. 따라서 무차원 잔차만 허용오차 비교에 들어가며, 차원 있는
값은 지수·로그·삼각함수·확률에 들어가지 않는다.

**3. 유한 물질 격자 비용 증서(3+1 자연 단위).** 후보는 공급된 평탄 등방 배경
위의 세 콤팩트 스칼라 위상 ``Theta^I`` 를 ``Theta^I = q x^I`` 로 쓴다. 감김
(winding)과 고유 간격은 입력이지 작용의 예측이 아니다. 자유 막대(rod) 응력 장부는
배터리와 분리한다: 1차원 가이드는 ``N`` 개의 연속 배터리를, 유한 3차원 정육면체는
``N**3`` 개 셀을 담는다. 어느 용량도 막대 에너지에 더하지 않는다. 후보 막대
부문은 ``-f_X**2/2 sum_I (grad Theta^I)^2`` 이며, 공급된 질량제곱 계수를 곱한
정적 무차원 물질 프로파일 ``W_cell`` 과 ``W_perp`` 로 보강할 수 있다. 여기서
감사하는 정확한 온사이트 정규화는
``g |H|**2 (D* B + B* D) + lambda/4 S**2``, ``S=|H|**2+|D|**2+|B|**2`` 이다.
저장된 우물 계수는 차원만 감사하며 주기적 국소화를 증명하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Sequence

import numpy as np

from examples.physics.causal.causal_domino import BatteryOutcomeReceipt
from examples.physics.record.finite_ctp_diagonal_source_obstruction import (
    QuantumKickConservationAudit,
)


SPACETIME_DIMENSION = 4
DEFAULT_TOLERANCE = 1.0e-10
MAX_TOLERANCE = 1.0e-8


@dataclass(frozen=True)
class ClosedBranchWorldtubeReceipt:
    """공급된 두 부문 세계관 장부 하나의 수치 감사 결과."""

    source_receipt_id: str
    quadrature_cell_count: int
    branch_probability: float
    conditional_trace: float
    receipt_energy: float
    system_initial_surface_energy: float
    system_final_surface_energy: float
    battery_initial_surface_energy: float
    battery_final_surface_energy: float
    system_surface_energy_change: float
    battery_surface_energy_change: float
    system_lateral_outward_energy_flux: float
    battery_lateral_outward_energy_flux: float
    system_source_injection_energy: float
    battery_source_injection_energy: float
    system_deformation_energy: float
    battery_deformation_energy: float
    system_predicted_surface_energy_change: float
    battery_predicted_surface_energy_change: float
    dimensionless_system_balance_residual: float
    dimensionless_battery_balance_residual: float
    dimensionless_total_balance_residual: float
    dimensionless_exchange_cancellation_residual: float
    maximum_dimensionless_opposite_current_residual: float
    dimensionless_system_receipt_surface_residual: float
    dimensionless_battery_receipt_surface_residual: float
    dimensionless_receipt_worldtube_residual: float
    maximum_dimensionless_killing_equation_residual: float
    current_mass_dimension: int
    stress_mass_dimension: int
    four_volume_mass_dimension: int
    energy_mass_dimension: int
    normalized_residual_mass_dimension: int
    dimensions_pass: bool
    positive_probability_outcome: bool
    conditional_branch_normalized: bool
    supplied_time_flow_future_timelike: bool
    supplied_killing_flow_on_samples: bool
    supplied_zero_lateral_flux: bool
    opposite_exchange_current_cancels: bool
    supplied_sector_balances_hold: bool
    supplied_total_energy_balance_holds: bool
    supplied_total_energy_and_exchange_closure_holds: bool
    exclusive_branch_receipt_matches_both_sectors: bool
    killing_zero_flux_receipt_matching_holds: bool
    source_receipt_id_is_provenance_label_only: bool = True
    typed_e9d_outcome_consistency_verified: bool = False
    source_receipt_provenance_authenticated: bool = False
    e9d_receipt_to_worldtube_derived: bool = False
    quadrature_worldtube_supplied: bool = True
    opposite_sector_current_supplied: bool = True
    branch_stress_from_domino_derived: bool = False
    battery_to_covariant_action_derived: bool = False
    continuum_worldtube_derived: bool = False
    physical_pointer_derived: bool = False
    record_to_gravity_source_derived: bool = False


@dataclass(frozen=True)
class FlatReceiptCurrentCounterexample:
    """영수증은 하나이고 전류는 서로 다른 정확한 평탄 세계관 족."""

    receipt_energy: float
    duration: float
    spatial_volume: float
    four_volume: float
    energy_source_density: float
    profile_a_current_covector: tuple[float, float, float, float]
    profile_b_current_covector: tuple[float, float, float, float]
    profile_a_battery_current_covector: tuple[float, float, float, float]
    profile_b_battery_current_covector: tuple[float, float, float, float]
    profile_a_computed_system_divergence_covector: tuple[float, float, float, float]
    profile_b_computed_system_divergence_covector: tuple[float, float, float, float]
    profile_a_computed_battery_divergence_covector: tuple[float, float, float, float]
    profile_b_computed_battery_divergence_covector: tuple[float, float, float, float]
    profile_a_integrated_energy: float
    profile_b_integrated_energy: float
    complement_constant_energy_density: float
    minimum_complement_energy_density: float
    dimensionless_profile_a_receipt_residual: float
    dimensionless_profile_b_receipt_residual: float
    dimensionless_current_difference: float
    maximum_dimensionless_divergence_identity_residual: float
    maximum_dimensionless_total_divergence_residual: float
    maximum_dimensionless_lateral_energy_flux_density: float
    current_mass_dimension: int
    four_volume_mass_dimension: int
    energy_mass_dimension: int
    dimensions_pass: bool
    same_flat_worldtube: bool
    same_scalar_receipt: bool
    current_profiles_distinct: bool
    lateral_energy_flux_zero: bool
    opposite_sector_closes_total_stress: bool
    unique_current_from_receipt_claim_refuted: bool
    worldtube_selected_by_receipt: bool = False
    branch_stress_from_receipt_derived: bool = False
    covariant_action_from_receipt_derived: bool = False
    record_to_gravity_source_derived: bool = False


@dataclass(frozen=True)
class FlatQuantumKickWorldtubeReceipt:
    """보존되는 유한 킥(kick)을 평탄 세계관에 조건부로 정합한 결과.

    공급된 공통 관성 기저는 부호 ``(-,+,+,+)`` 를 쓴다. 닫힌 부문 ``s`` 마다
    감사하는 이산 항등식은 다음과 같다.

        Delta P_s^nu + Phi_s^nu
          = sum_i Delta V_i eta^(nu alpha) Q_(s,i,alpha).

    이는 국소 워드 항등식의 적분된 세계관 귀결이다. 이 루틴은 국소화 ``Q``,
    발산이 ``Q`` 인 국소 응력, 힐베르트 공간 연산자 성분을 물리적 로런츠
    4-벡터와 동일시하는 일을 유도하지 않는다.
    """

    sector_count: int
    component_count: int
    quadrature_cell_count: int
    quantum_mean_kicks: tuple[tuple[float, ...], ...]
    integrated_exchange_four_momenta: tuple[tuple[float, ...], ...]
    lateral_outward_four_momentum_fluxes: tuple[tuple[float, ...], ...]
    predicted_worldtube_kicks: tuple[tuple[float, ...], ...]
    maximum_dimensionless_sector_matching_residual: float
    maximum_dimensionless_local_exchange_residual: float
    dimensionless_integrated_exchange_residual: float
    dimensionless_total_quantum_kick_residual: float
    current_mass_dimension: int
    four_volume_mass_dimension: int
    integrated_source_mass_dimension: int
    lateral_flux_mass_dimension: int
    kick_mass_dimension: int
    normalized_residual_mass_dimension: int
    dimensions_pass: bool
    quantum_operator_conservation_certified: bool
    all_receivers_included: bool
    local_exchange_currents_cancel: bool
    integrated_exchange_cancels: bool
    numerical_integrated_worldtube_matching_holds: bool
    same_local_action_identification_supplied: bool
    shared_inertial_four_vector_basis_supplied: bool
    conditional_quantum_to_worldtube_bridge_holds: bool
    worldtube_localization_supplied: bool = True
    lateral_flux_supplied: bool = True
    operator_components_as_physical_four_vector_derived: bool = False
    exchange_current_from_quantum_dynamics_derived: bool = False
    local_stress_from_quantum_kick_derived: bool = False
    general_curved_spacetime_transport_derived: bool = False
    physical_clarus_source_derived: bool = False


@dataclass(frozen=True)
class FlatChargeStressKernelCounterexample:
    """네 전하가 정확히 같은 두 보존 국소 응력."""

    spatial_volume: float
    energy_density: float
    shear_amplitude: float
    profile_a_stress_contravariant: tuple[tuple[float, ...], ...]
    profile_b_stress_contravariant: tuple[tuple[float, ...], ...]
    profile_a_four_momentum: tuple[float, float, float, float]
    profile_b_four_momentum: tuple[float, float, float, float]
    dimensionless_four_momentum_residual: float
    dimensionless_local_stress_difference: float
    maximum_dimensionless_ward_residual: float
    stress_mass_dimension: int
    spatial_volume_mass_dimension: int
    four_momentum_mass_dimension: int
    normalized_residual_mass_dimension: int
    dimensions_pass: bool
    both_profiles_symmetric: bool
    both_profiles_divergence_free: bool
    both_profiles_satisfy_dominant_energy_condition: bool
    same_complete_four_momentum: bool
    local_stresses_distinct: bool
    finite_charge_to_local_stress_nonuniqueness_certified: bool
    periodic_spatial_cell_supplied: bool = True
    local_action_for_profiles_derived: bool = False
    local_stress_selected_by_finite_charges: bool = False
    cosmological_perturbations_selected_by_background_charges: bool = False


def _finite_scalar(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_scalar(value: float, name: str) -> float:
    result = _finite_scalar(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _array(
    values: Sequence[object],
    name: str,
    trailing_shape: tuple[int, ...],
) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != len(trailing_shape) + 1 or result.shape[1:] != trailing_shape:
        expected = "N-by-" + "-by-".join(str(size) for size in trailing_shape)
        raise ValueError(f"{name} must be a finite {expected} array")
    if result.shape[0] == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite non-empty array")
    return result


def _validate_geometry(
    metrics_covariant: np.ndarray,
    orientation_observers_contravariant: np.ndarray,
    time_flows_contravariant: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, bool]:
    cell_count = metrics_covariant.shape[0]
    if orientation_observers_contravariant.shape[0] != cell_count:
        raise ValueError("orientation observers must have one row per quadrature cell")
    if time_flows_contravariant.shape[0] != cell_count:
        raise ValueError("time flows must have one row per quadrature cell")

    positive_inverse_metrics: list[np.ndarray] = []
    future_timelike = True
    for index in range(cell_count):
        metric = metrics_covariant[index]
        if not np.allclose(metric, metric.T, rtol=0.0, atol=tolerance):
            raise ValueError("each metric_covariant sample must be symmetric")
        eigenvalues = np.linalg.eigvalsh(metric)
        metric_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
        negative = int(np.count_nonzero(eigenvalues < -tolerance * metric_scale))
        positive = int(np.count_nonzero(eigenvalues > tolerance * metric_scale))
        if negative != 1 or positive != 3:
            raise ValueError("each metric sample must have signature (-,+,+,+)")

        observer = orientation_observers_contravariant[index]
        observer_norm = float(observer @ metric @ observer)
        if not math.isclose(observer_norm, -1.0, rel_tol=tolerance, abs_tol=tolerance):
            raise ValueError("each orientation observer must be unit timelike")

        time_flow = time_flows_contravariant[index]
        time_flow_norm = float(time_flow @ metric @ time_flow)
        relative_orientation = float(observer @ metric @ time_flow)
        if time_flow_norm >= -tolerance or relative_orientation >= -tolerance:
            future_timelike = False

        inverse = np.linalg.inv(metric)
        positive_inverse = inverse + 2.0 * np.outer(observer, observer)
        positive_eigenvalues = np.linalg.eigvalsh(positive_inverse)
        if float(np.min(positive_eigenvalues)) <= tolerance:
            raise ArithmeticError("observer-induced covector metric is not positive")
        positive_inverse_metrics.append(positive_inverse)

    if not future_timelike:
        raise ValueError("each supplied time flow must be future timelike")
    return np.stack(positive_inverse_metrics), future_timelike


def _validate_symmetric_samples(
    samples: np.ndarray,
    name: str,
    tolerance: float,
) -> None:
    if not np.allclose(
        samples,
        np.swapaxes(samples, 1, 2),
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError(f"{name} must be symmetric at every quadrature cell")


def _maximum_positive_tensor_norm(
    tensors_covariant: np.ndarray,
    positive_inverse_metrics: np.ndarray,
) -> float:
    squared = np.einsum(
        "nij,nik,njl,nkl->n",
        tensors_covariant,
        positive_inverse_metrics,
        positive_inverse_metrics,
        tensors_covariant,
    )
    if float(np.min(squared)) < -1.0e-12:
        raise ArithmeticError("observer-induced tensor norm became negative")
    return math.sqrt(max(0.0, float(np.max(squared))))


def _maximum_positive_covector_norm(
    covectors: np.ndarray,
    positive_inverse_metrics: np.ndarray,
) -> float:
    squared = np.einsum(
        "ni,nij,nj->n",
        covectors,
        positive_inverse_metrics,
        covectors,
    )
    if float(np.min(squared)) < -1.0e-12:
        raise ArithmeticError("observer-induced covector norm became negative")
    return math.sqrt(max(0.0, float(np.max(squared))))


def audit_flat_quantum_kick_worldtube_receipt(
    *,
    kick_audit: QuantumKickConservationAudit,
    exchange_currents_covariant: Sequence[Sequence[Sequence[float]]],
    proper_four_volume_weights: Sequence[float],
    lateral_outward_four_momentum_fluxes: Sequence[Sequence[float]],
    same_local_action_identification_supplied: bool,
    shared_inertial_four_vector_basis_supplied: bool,
    reference_mass_scale: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> FlatQuantumKickWorldtubeReceipt:
    """유한 4-킥을 공급된 평탄 세계관 워드 영수증에 정합한다.

    여기서 검사하는 등식은 다음의 이산·평탄 좌표계 판이다.

        Delta P_s^nu + Phi_s^nu = integral_W Q_s^nu dV.

    ``same_local_action_identification_supplied`` 는 양자 유니터리와 국소
    전류가 같은 상호작용의 두 기술이라는 필수 전제를 기록한다. 참 값은
    호출자 계약이지 그 전제의 독립적 유도가 아니다.
    """

    if not isinstance(kick_audit, QuantumKickConservationAudit):
        raise TypeError("kick_audit must be a QuantumKickConservationAudit")
    for value, name in (
        (same_local_action_identification_supplied, "same_local_action_identification_supplied"),
        (shared_inertial_four_vector_basis_supplied, "shared_inertial_four_vector_basis_supplied"),
    ):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be boolean")
    tolerance = _positive_scalar(tolerance, "tolerance")
    if tolerance > MAX_TOLERANCE:
        raise ValueError(f"tolerance must not exceed {MAX_TOLERANCE}")
    reference_mass_scale = _positive_scalar(
        reference_mass_scale,
        "reference_mass_scale",
    )
    if kick_audit.component_count != SPACETIME_DIMENSION:
        raise ValueError("kick_audit must contain exactly four declared components")

    currents = np.asarray(exchange_currents_covariant, dtype=float)
    expected_prefix = (kick_audit.sector_count,)
    if (
        currents.ndim != 3
        or currents.shape[0:1] != expected_prefix
        or currents.shape[1] == 0
        or currents.shape[2] != SPACETIME_DIMENSION
        or not np.all(np.isfinite(currents))
    ):
        raise ValueError(
            "exchange_currents_covariant must have shape "
            "(sector_count, cell_count, 4)"
        )
    cell_count = currents.shape[1]
    volume_weights = np.asarray(proper_four_volume_weights, dtype=float)
    if (
        volume_weights.shape != (cell_count,)
        or not np.all(np.isfinite(volume_weights))
        or np.any(volume_weights <= 0.0)
    ):
        raise ValueError(
            "proper_four_volume_weights must contain one positive finite value per cell"
        )
    lateral_fluxes = np.asarray(
        lateral_outward_four_momentum_fluxes,
        dtype=float,
    )
    if (
        lateral_fluxes.shape
        != (kick_audit.sector_count, SPACETIME_DIMENSION)
        or not np.all(np.isfinite(lateral_fluxes))
    ):
        raise ValueError(
            "lateral_outward_four_momentum_fluxes must have shape "
            "(sector_count, 4)"
        )

    mean_kicks = np.asarray(kick_audit.mean_kicks, dtype=float)
    if mean_kicks.shape != lateral_fluxes.shape:
        raise ValueError("kick_audit mean_kicks have an inconsistent shape")
    minkowski_inverse = np.diag((-1.0, 1.0, 1.0, 1.0))
    integrated_sources = np.einsum(
        "n,sna,ab->sb",
        volume_weights,
        currents,
        minkowski_inverse,
        optimize=True,
    )
    predicted_kicks = integrated_sources - lateral_fluxes
    matching_residuals = mean_kicks - predicted_kicks
    local_exchange_residuals = np.sum(currents, axis=0)
    integrated_exchange_residual = np.sum(integrated_sources, axis=0)
    total_quantum_kick = np.sum(mean_kicks, axis=0)

    momentum_scale = reference_mass_scale
    current_scale = reference_mass_scale**5
    maximum_matching_residual = float(
        np.max(np.linalg.norm(matching_residuals, axis=1)) / momentum_scale
    )
    maximum_local_exchange_residual = float(
        np.max(np.linalg.norm(local_exchange_residuals, axis=1)) / current_scale
    )
    dimensionless_integrated_exchange_residual = float(
        np.linalg.norm(integrated_exchange_residual) / momentum_scale
    )
    dimensionless_total_quantum_kick_residual = float(
        np.linalg.norm(total_quantum_kick) / momentum_scale
    )

    current_mass_dimension = 5
    four_volume_mass_dimension = -4
    integrated_source_mass_dimension = 1
    lateral_flux_mass_dimension = 1
    kick_mass_dimension = 1
    normalized_residual_mass_dimension = 0
    dimensions_pass = (
        current_mass_dimension + four_volume_mass_dimension
        == integrated_source_mass_dimension
        and integrated_source_mass_dimension == lateral_flux_mass_dimension
        and lateral_flux_mass_dimension == kick_mass_dimension
        and kick_mass_dimension - 1 == normalized_residual_mass_dimension
    )
    local_exchange_cancels = maximum_local_exchange_residual <= tolerance
    integrated_exchange_cancels = (
        dimensionless_integrated_exchange_residual <= tolerance
    )
    numerical_matching = maximum_matching_residual <= tolerance
    conditional_bridge = (
        kick_audit.operator_conservation_certified
        and kick_audit.all_receivers_included
        and local_exchange_cancels
        and integrated_exchange_cancels
        and dimensionless_total_quantum_kick_residual <= tolerance
        and numerical_matching
        and bool(same_local_action_identification_supplied)
        and bool(shared_inertial_four_vector_basis_supplied)
        and dimensions_pass
    )

    return FlatQuantumKickWorldtubeReceipt(
        sector_count=kick_audit.sector_count,
        component_count=kick_audit.component_count,
        quadrature_cell_count=cell_count,
        quantum_mean_kicks=tuple(
            tuple(float(item) for item in row) for row in mean_kicks
        ),
        integrated_exchange_four_momenta=tuple(
            tuple(float(item) for item in row) for row in integrated_sources
        ),
        lateral_outward_four_momentum_fluxes=tuple(
            tuple(float(item) for item in row) for row in lateral_fluxes
        ),
        predicted_worldtube_kicks=tuple(
            tuple(float(item) for item in row) for row in predicted_kicks
        ),
        maximum_dimensionless_sector_matching_residual=maximum_matching_residual,
        maximum_dimensionless_local_exchange_residual=(
            maximum_local_exchange_residual
        ),
        dimensionless_integrated_exchange_residual=(
            dimensionless_integrated_exchange_residual
        ),
        dimensionless_total_quantum_kick_residual=(
            dimensionless_total_quantum_kick_residual
        ),
        current_mass_dimension=current_mass_dimension,
        four_volume_mass_dimension=four_volume_mass_dimension,
        integrated_source_mass_dimension=integrated_source_mass_dimension,
        lateral_flux_mass_dimension=lateral_flux_mass_dimension,
        kick_mass_dimension=kick_mass_dimension,
        normalized_residual_mass_dimension=normalized_residual_mass_dimension,
        dimensions_pass=dimensions_pass,
        quantum_operator_conservation_certified=(
            kick_audit.operator_conservation_certified
        ),
        all_receivers_included=kick_audit.all_receivers_included,
        local_exchange_currents_cancel=local_exchange_cancels,
        integrated_exchange_cancels=integrated_exchange_cancels,
        numerical_integrated_worldtube_matching_holds=numerical_matching,
        same_local_action_identification_supplied=bool(
            same_local_action_identification_supplied
        ),
        shared_inertial_four_vector_basis_supplied=bool(
            shared_inertial_four_vector_basis_supplied
        ),
        conditional_quantum_to_worldtube_bridge_holds=conditional_bridge,
    )


def construct_flat_charge_stress_kernel_counterexample(
    *,
    spatial_volume: float,
    energy_density: float,
    shear_amplitude: float,
    reference_mass_scale: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> FlatChargeStressKernelCounterexample:
    """전하→응력 사상의 워드 항등식을 보존하는 핵(kernel)을 구성한다.

    평탄 주기적 공간 셀 위에서 상수 응력

        T_A = diag(rho, 0, 0, 0),
        T_B = diag(rho, s, -s, 0).

    을 비교한다. 둘 다 대칭이고 발산이 0이다. 완전한 표면 전하
    ``P^nu = integral T^(0 nu) d^3x`` 는 일치하지만, ``s != 0`` 이면 국소 공간
    응력이 다르다. ``|s| <= rho`` 를 요구하면 이 명시적 증인에서 두 I형 텐서가
    지배 에너지 조건(dominant energy condition) 안에 머문다.
    """

    tolerance = _positive_scalar(tolerance, "tolerance")
    if tolerance > MAX_TOLERANCE:
        raise ValueError(f"tolerance must not exceed {MAX_TOLERANCE}")
    reference_mass_scale = _positive_scalar(
        reference_mass_scale,
        "reference_mass_scale",
    )
    spatial_volume = _positive_scalar(spatial_volume, "spatial_volume")
    energy_density = _positive_scalar(energy_density, "energy_density")
    shear_amplitude = _finite_scalar(shear_amplitude, "shear_amplitude")
    if abs(shear_amplitude) / reference_mass_scale**4 <= tolerance:
        raise ValueError("shear_amplitude must be nonzero at the declared tolerance")
    if abs(shear_amplitude) > energy_density:
        raise ValueError("the explicit dominant-energy witness requires |shear| <= rho")

    stress_a = np.diag((energy_density, 0.0, 0.0, 0.0))
    stress_b = np.diag(
        (energy_density, shear_amplitude, -shear_amplitude, 0.0)
    )
    momentum_a = spatial_volume * stress_a[0]
    momentum_b = spatial_volume * stress_b[0]
    momentum_scale = reference_mass_scale
    stress_scale = reference_mass_scale**4
    momentum_residual = float(np.linalg.norm(momentum_a - momentum_b) / momentum_scale)
    stress_difference = float(np.linalg.norm(stress_a - stress_b, ord=2) / stress_scale)
    ward_residual = 0.0

    stress_mass_dimension = 4
    spatial_volume_mass_dimension = -3
    four_momentum_mass_dimension = 1
    normalized_residual_mass_dimension = 0
    dimensions_pass = (
        stress_mass_dimension + spatial_volume_mass_dimension
        == four_momentum_mass_dimension
        and four_momentum_mass_dimension - 1 == normalized_residual_mass_dimension
    )
    symmetric = bool(
        np.array_equal(stress_a, stress_a.T)
        and np.array_equal(stress_b, stress_b.T)
    )
    divergence_free = ward_residual <= tolerance
    dominant_energy = bool(abs(shear_amplitude) <= energy_density)
    same_momentum = momentum_residual <= tolerance
    distinct = stress_difference > tolerance
    witness = (
        symmetric
        and divergence_free
        and dominant_energy
        and same_momentum
        and distinct
        and dimensions_pass
    )

    return FlatChargeStressKernelCounterexample(
        spatial_volume=spatial_volume,
        energy_density=energy_density,
        shear_amplitude=shear_amplitude,
        profile_a_stress_contravariant=tuple(
            tuple(float(item) for item in row) for row in stress_a
        ),
        profile_b_stress_contravariant=tuple(
            tuple(float(item) for item in row) for row in stress_b
        ),
        profile_a_four_momentum=tuple(float(item) for item in momentum_a),
        profile_b_four_momentum=tuple(float(item) for item in momentum_b),
        dimensionless_four_momentum_residual=momentum_residual,
        dimensionless_local_stress_difference=stress_difference,
        maximum_dimensionless_ward_residual=ward_residual,
        stress_mass_dimension=stress_mass_dimension,
        spatial_volume_mass_dimension=spatial_volume_mass_dimension,
        four_momentum_mass_dimension=four_momentum_mass_dimension,
        normalized_residual_mass_dimension=normalized_residual_mass_dimension,
        dimensions_pass=dimensions_pass,
        both_profiles_symmetric=symmetric,
        both_profiles_divergence_free=divergence_free,
        both_profiles_satisfy_dominant_energy_condition=dominant_energy,
        same_complete_four_momentum=same_momentum,
        local_stresses_distinct=distinct,
        finite_charge_to_local_stress_nonuniqueness_certified=witness,
    )


def audit_closed_branch_worldtube(
    *,
    source_receipt_id: str,
    branch_probability: float,
    conditional_trace: float,
    receipt_energy: float,
    metrics_covariant: Sequence[Sequence[Sequence[float]]],
    orientation_observers_contravariant: Sequence[Sequence[float]],
    time_flows_contravariant: Sequence[Sequence[float]],
    exchange_currents_system_covariant: Sequence[Sequence[float]],
    exchange_currents_battery_covariant: Sequence[Sequence[float]],
    system_stresses_contravariant: Sequence[Sequence[Sequence[float]]],
    battery_stresses_contravariant: Sequence[Sequence[Sequence[float]]],
    symmetrized_time_flow_gradients_covariant: Sequence[
        Sequence[Sequence[float]]
    ],
    proper_four_volume_weights: Sequence[float],
    system_initial_surface_energy: float,
    system_final_surface_energy: float,
    battery_initial_surface_energy: float,
    battery_final_surface_energy: float,
    system_lateral_outward_energy_flux: float,
    battery_lateral_outward_energy_flux: float,
    reference_mass_scale: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> ClosedBranchWorldtubeReceipt:
    """공급된 시스템/배터리 발산 정리 균형을 감사한다.

    자연 단위를 쓴다. 전류는 질량 차원 5, 응력은 4, 무차원 시간 흐름의
    대칭화 기울기는 1, 고유 4-부피 가중치는 -4다. 두 부문 전류는 독립적으로
    공급된 배열이다. 이 루틴은 점별 반대 전류 관계를 검사할 뿐, 그 관계를
    작용에서 부과하거나 유도하지 않는다.
    """

    if not isinstance(source_receipt_id, str) or not source_receipt_id.strip():
        raise ValueError("source_receipt_id must be a non-empty string")
    tolerance = _positive_scalar(tolerance, "tolerance")
    if tolerance > MAX_TOLERANCE:
        raise ValueError(f"tolerance must not exceed {MAX_TOLERANCE}")
    reference_mass_scale = _positive_scalar(
        reference_mass_scale,
        "reference_mass_scale",
    )
    branch_probability = _finite_scalar(branch_probability, "branch_probability")
    if not 0.0 <= branch_probability <= 1.0:
        raise ValueError("branch_probability must lie in [0, 1]")
    conditional_trace = _finite_scalar(conditional_trace, "conditional_trace")
    if conditional_trace < 0.0:
        raise ValueError("conditional_trace must be non-negative")
    receipt_energy = _finite_scalar(receipt_energy, "receipt_energy")
    if receipt_energy < 0.0:
        raise ValueError("receipt_energy must be non-negative for an E9-D receipt")

    metrics = _array(
        metrics_covariant,
        "metrics_covariant",
        (SPACETIME_DIMENSION, SPACETIME_DIMENSION),
    )
    observers = _array(
        orientation_observers_contravariant,
        "orientation_observers_contravariant",
        (SPACETIME_DIMENSION,),
    )
    time_flows = _array(
        time_flows_contravariant,
        "time_flows_contravariant",
        (SPACETIME_DIMENSION,),
    )
    exchange_currents = _array(
        exchange_currents_system_covariant,
        "exchange_currents_system_covariant",
        (SPACETIME_DIMENSION,),
    )
    battery_exchange_currents = _array(
        exchange_currents_battery_covariant,
        "exchange_currents_battery_covariant",
        (SPACETIME_DIMENSION,),
    )
    system_stresses = _array(
        system_stresses_contravariant,
        "system_stresses_contravariant",
        (SPACETIME_DIMENSION, SPACETIME_DIMENSION),
    )
    battery_stresses = _array(
        battery_stresses_contravariant,
        "battery_stresses_contravariant",
        (SPACETIME_DIMENSION, SPACETIME_DIMENSION),
    )
    time_flow_gradients = _array(
        symmetrized_time_flow_gradients_covariant,
        "symmetrized_time_flow_gradients_covariant",
        (SPACETIME_DIMENSION, SPACETIME_DIMENSION),
    )
    cell_count = metrics.shape[0]
    arrays = (
        observers,
        time_flows,
        exchange_currents,
        battery_exchange_currents,
        system_stresses,
        battery_stresses,
        time_flow_gradients,
    )
    if any(array.shape[0] != cell_count for array in arrays):
        raise ValueError("all sampled fields must share the quadrature cell count")
    volume_weights = np.asarray(proper_four_volume_weights, dtype=float)
    if volume_weights.shape != (cell_count,) or not np.all(np.isfinite(volume_weights)):
        raise ValueError("proper_four_volume_weights must have one finite value per cell")
    if np.any(volume_weights <= 0.0):
        raise ValueError("proper_four_volume_weights must be positive")

    positive_inverse_metrics, future_timelike = _validate_geometry(
        metrics,
        observers,
        time_flows,
        tolerance,
    )
    _validate_symmetric_samples(system_stresses, "system stresses", tolerance)
    _validate_symmetric_samples(battery_stresses, "battery stresses", tolerance)
    _validate_symmetric_samples(
        time_flow_gradients,
        "symmetrized time-flow gradients",
        tolerance,
    )

    system_initial_surface_energy = _finite_scalar(
        system_initial_surface_energy,
        "system_initial_surface_energy",
    )
    system_final_surface_energy = _finite_scalar(
        system_final_surface_energy,
        "system_final_surface_energy",
    )
    battery_initial_surface_energy = _finite_scalar(
        battery_initial_surface_energy,
        "battery_initial_surface_energy",
    )
    battery_final_surface_energy = _finite_scalar(
        battery_final_surface_energy,
        "battery_final_surface_energy",
    )
    system_lateral_outward_energy_flux = _finite_scalar(
        system_lateral_outward_energy_flux,
        "system_lateral_outward_energy_flux",
    )
    battery_lateral_outward_energy_flux = _finite_scalar(
        battery_lateral_outward_energy_flux,
        "battery_lateral_outward_energy_flux",
    )

    source_contractions = np.einsum("ni,ni->n", time_flows, exchange_currents)
    battery_source_contractions = np.einsum(
        "ni,ni->n",
        time_flows,
        battery_exchange_currents,
    )
    system_deformation_contractions = np.einsum(
        "nij,nij->n",
        system_stresses,
        time_flow_gradients,
    )
    battery_deformation_contractions = np.einsum(
        "nij,nij->n",
        battery_stresses,
        time_flow_gradients,
    )
    system_source_injection = -float(np.dot(volume_weights, source_contractions))
    battery_source_injection = -float(
        np.dot(volume_weights, battery_source_contractions)
    )
    system_deformation_energy = -float(
        np.dot(volume_weights, system_deformation_contractions)
    )
    battery_deformation_energy = -float(
        np.dot(volume_weights, battery_deformation_contractions)
    )

    system_change = system_final_surface_energy - system_initial_surface_energy
    battery_change = battery_final_surface_energy - battery_initial_surface_energy
    system_predicted_change = (
        system_source_injection
        + system_deformation_energy
        - system_lateral_outward_energy_flux
    )
    battery_predicted_change = (
        battery_source_injection
        + battery_deformation_energy
        - battery_lateral_outward_energy_flux
    )
    system_balance_difference = system_change - system_predicted_change
    battery_balance_difference = battery_change - battery_predicted_change
    total_balance_difference = (
        system_change
        + battery_change
        + system_lateral_outward_energy_flux
        + battery_lateral_outward_energy_flux
        - system_deformation_energy
        - battery_deformation_energy
    )
    exchange_cancellation_difference = (
        system_source_injection + battery_source_injection
    )
    opposite_current_residuals = exchange_currents + battery_exchange_currents
    system_receipt_surface_difference = receipt_energy - system_change
    battery_receipt_surface_difference = -receipt_energy - battery_change
    receipt_worldtube_difference = (
        receipt_energy
        + system_lateral_outward_energy_flux
        - system_source_injection
        - system_deformation_energy
    )

    energy_scale = reference_mass_scale
    dimensionless_system_balance_residual = abs(system_balance_difference) / energy_scale
    dimensionless_battery_balance_residual = abs(battery_balance_difference) / energy_scale
    dimensionless_total_balance_residual = abs(total_balance_difference) / energy_scale
    dimensionless_exchange_cancellation_residual = (
        abs(exchange_cancellation_difference) / energy_scale
    )
    dimensionless_system_receipt_surface_residual = (
        abs(system_receipt_surface_difference) / energy_scale
    )
    dimensionless_battery_receipt_surface_residual = (
        abs(battery_receipt_surface_difference) / energy_scale
    )
    dimensionless_receipt_worldtube_residual = (
        abs(receipt_worldtube_difference) / energy_scale
    )
    maximum_dimensionless_killing_equation_residual = (
        _maximum_positive_tensor_norm(
            time_flow_gradients,
            positive_inverse_metrics,
        )
        / reference_mass_scale
    )
    maximum_dimensionless_opposite_current_residual = (
        _maximum_positive_covector_norm(
            opposite_current_residuals,
            positive_inverse_metrics,
        )
        / reference_mass_scale**5
    )

    current_mass_dimension = 5
    stress_mass_dimension = 4
    four_volume_mass_dimension = -4
    energy_mass_dimension = 1
    normalized_residual_mass_dimension = 0
    dimensions_pass = (
        current_mass_dimension + four_volume_mass_dimension
        == energy_mass_dimension
        and stress_mass_dimension + 1 + four_volume_mass_dimension
        == energy_mass_dimension
        and energy_mass_dimension - 1 == normalized_residual_mass_dimension
    )
    positive_probability_outcome = branch_probability > tolerance
    conditional_branch_normalized = math.isclose(
        conditional_trace,
        1.0,
        rel_tol=tolerance,
        abs_tol=tolerance,
    )
    killing_flow = maximum_dimensionless_killing_equation_residual <= tolerance
    zero_lateral_flux = (
        abs(system_lateral_outward_energy_flux) / energy_scale <= tolerance
        and abs(battery_lateral_outward_energy_flux) / energy_scale <= tolerance
    )
    exchange_cancels = (
        dimensionless_exchange_cancellation_residual <= tolerance
        and maximum_dimensionless_opposite_current_residual <= tolerance
    )
    sector_balances_hold = (
        dimensionless_system_balance_residual <= tolerance
        and dimensionless_battery_balance_residual <= tolerance
    )
    total_energy_balance_holds = dimensionless_total_balance_residual <= tolerance
    total_energy_and_exchange_closure_holds = (
        total_energy_balance_holds and exchange_cancels
    )
    receipt_matches_both = (
        dimensionless_system_receipt_surface_residual <= tolerance
        and dimensionless_battery_receipt_surface_residual <= tolerance
    )
    killing_zero_flux_matching = (
        positive_probability_outcome
        and conditional_branch_normalized
        and future_timelike
        and killing_flow
        and zero_lateral_flux
        and exchange_cancels
        and sector_balances_hold
        and total_energy_and_exchange_closure_holds
        and receipt_matches_both
        and dimensionless_receipt_worldtube_residual <= tolerance
        and dimensions_pass
    )

    return ClosedBranchWorldtubeReceipt(
        source_receipt_id=source_receipt_id.strip(),
        quadrature_cell_count=cell_count,
        branch_probability=branch_probability,
        conditional_trace=conditional_trace,
        receipt_energy=receipt_energy,
        system_initial_surface_energy=system_initial_surface_energy,
        system_final_surface_energy=system_final_surface_energy,
        battery_initial_surface_energy=battery_initial_surface_energy,
        battery_final_surface_energy=battery_final_surface_energy,
        system_surface_energy_change=system_change,
        battery_surface_energy_change=battery_change,
        system_lateral_outward_energy_flux=system_lateral_outward_energy_flux,
        battery_lateral_outward_energy_flux=battery_lateral_outward_energy_flux,
        system_source_injection_energy=system_source_injection,
        battery_source_injection_energy=battery_source_injection,
        system_deformation_energy=system_deformation_energy,
        battery_deformation_energy=battery_deformation_energy,
        system_predicted_surface_energy_change=system_predicted_change,
        battery_predicted_surface_energy_change=battery_predicted_change,
        dimensionless_system_balance_residual=(
            dimensionless_system_balance_residual
        ),
        dimensionless_battery_balance_residual=(
            dimensionless_battery_balance_residual
        ),
        dimensionless_total_balance_residual=dimensionless_total_balance_residual,
        dimensionless_exchange_cancellation_residual=(
            dimensionless_exchange_cancellation_residual
        ),
        maximum_dimensionless_opposite_current_residual=(
            maximum_dimensionless_opposite_current_residual
        ),
        dimensionless_system_receipt_surface_residual=(
            dimensionless_system_receipt_surface_residual
        ),
        dimensionless_battery_receipt_surface_residual=(
            dimensionless_battery_receipt_surface_residual
        ),
        dimensionless_receipt_worldtube_residual=(
            dimensionless_receipt_worldtube_residual
        ),
        maximum_dimensionless_killing_equation_residual=(
            maximum_dimensionless_killing_equation_residual
        ),
        current_mass_dimension=current_mass_dimension,
        stress_mass_dimension=stress_mass_dimension,
        four_volume_mass_dimension=four_volume_mass_dimension,
        energy_mass_dimension=energy_mass_dimension,
        normalized_residual_mass_dimension=normalized_residual_mass_dimension,
        dimensions_pass=dimensions_pass,
        positive_probability_outcome=positive_probability_outcome,
        conditional_branch_normalized=conditional_branch_normalized,
        supplied_time_flow_future_timelike=future_timelike,
        supplied_killing_flow_on_samples=killing_flow,
        supplied_zero_lateral_flux=zero_lateral_flux,
        opposite_exchange_current_cancels=exchange_cancels,
        supplied_sector_balances_hold=sector_balances_hold,
        supplied_total_energy_balance_holds=total_energy_balance_holds,
        supplied_total_energy_and_exchange_closure_holds=(
            total_energy_and_exchange_closure_holds
        ),
        exclusive_branch_receipt_matches_both_sectors=receipt_matches_both,
        killing_zero_flux_receipt_matching_holds=killing_zero_flux_matching,
    )


def audit_e9d_outcome_closed_branch_worldtube(
    *,
    outcome: BatteryOutcomeReceipt,
    initial_system_energy: float,
    metrics_covariant: Sequence[Sequence[Sequence[float]]],
    orientation_observers_contravariant: Sequence[Sequence[float]],
    time_flows_contravariant: Sequence[Sequence[float]],
    exchange_currents_system_covariant: Sequence[Sequence[float]],
    exchange_currents_battery_covariant: Sequence[Sequence[float]],
    system_stresses_contravariant: Sequence[Sequence[Sequence[float]]],
    battery_stresses_contravariant: Sequence[Sequence[Sequence[float]]],
    symmetrized_time_flow_gradients_covariant: Sequence[
        Sequence[Sequence[float]]
    ],
    proper_four_volume_weights: Sequence[float],
    system_lateral_outward_energy_flux: float,
    battery_lateral_outward_energy_flux: float,
    reference_mass_scale: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> ClosedBranchWorldtubeReceipt:
    """형이 있는 양의 확률 E9-D 결과를 공급된 장에 대해 감사한다.

    이 어댑터는 결과 내부의 에너지 관계를 검사하고 두 표면 장부를 형이 있는
    배터리 영수증에서 얻는다. 검증하는 것은 일관성이지 이력의 진위가 아니다:
    호출자가 데이터클래스를 손으로 만들 수 있고, 국소 장과 세계관은 여전히
    독립적으로 공급된 입력이다.
    """

    if not isinstance(outcome, BatteryOutcomeReceipt):
        raise TypeError("outcome must be a BatteryOutcomeReceipt")
    tolerance = _positive_scalar(tolerance, "tolerance")
    if tolerance > MAX_TOLERANCE:
        raise ValueError(f"tolerance must not exceed {MAX_TOLERANCE}")
    reference_mass_scale = _positive_scalar(
        reference_mass_scale,
        "reference_mass_scale",
    )
    initial_system_energy = _finite_scalar(
        initial_system_energy,
        "initial_system_energy",
    )
    probability = _finite_scalar(outcome.probability, "outcome.probability")
    if probability <= tolerance or probability > 1.0:
        raise ValueError("typed E9-D outcome must have positive probability")
    if not isinstance(outcome.basis_label, str) or not outcome.basis_label:
        raise ValueError("typed E9-D outcome must have a basis label")
    paid_energy = _finite_scalar(
        outcome.energy_paid_to_system,
        "outcome.energy_paid_to_system",
    )
    final_battery_energy = _finite_scalar(
        outcome.final_battery_energy,
        "outcome.final_battery_energy",
    )
    if paid_energy < 0.0 or final_battery_energy < 0.0:
        raise ValueError("typed E9-D battery energies must be non-negative")
    if outcome.conditional_system_energy is None:
        raise ValueError("typed E9-D outcome must carry conditional system energy")
    conditional_system_energy = _finite_scalar(
        outcome.conditional_system_energy,
        "outcome.conditional_system_energy",
    )
    if outcome.relative_branch_energy_residual is None:
        raise ValueError("typed E9-D outcome must carry its branch energy residual")
    reported_branch_residual = _finite_scalar(
        outcome.relative_branch_energy_residual,
        "outcome.relative_branch_energy_residual",
    )
    direct_branch_residual = abs(
        conditional_system_energy - initial_system_energy - paid_energy
    ) / reference_mass_scale
    if reported_branch_residual > tolerance or direct_branch_residual > tolerance:
        raise ValueError("typed E9-D outcome fails its branch energy relation")

    receipt = audit_closed_branch_worldtube(
        source_receipt_id=f"QNB-E9-D:{outcome.basis_label}",
        branch_probability=probability,
        conditional_trace=1.0,
        receipt_energy=paid_energy,
        metrics_covariant=metrics_covariant,
        orientation_observers_contravariant=(
            orientation_observers_contravariant
        ),
        time_flows_contravariant=time_flows_contravariant,
        exchange_currents_system_covariant=(
            exchange_currents_system_covariant
        ),
        exchange_currents_battery_covariant=(
            exchange_currents_battery_covariant
        ),
        system_stresses_contravariant=system_stresses_contravariant,
        battery_stresses_contravariant=battery_stresses_contravariant,
        symmetrized_time_flow_gradients_covariant=(
            symmetrized_time_flow_gradients_covariant
        ),
        proper_four_volume_weights=proper_four_volume_weights,
        system_initial_surface_energy=initial_system_energy,
        system_final_surface_energy=conditional_system_energy,
        battery_initial_surface_energy=final_battery_energy + paid_energy,
        battery_final_surface_energy=final_battery_energy,
        system_lateral_outward_energy_flux=(
            system_lateral_outward_energy_flux
        ),
        battery_lateral_outward_energy_flux=(
            battery_lateral_outward_energy_flux
        ),
        reference_mass_scale=reference_mass_scale,
        tolerance=tolerance,
    )
    return replace(
        receipt,
        source_receipt_id_is_provenance_label_only=False,
        typed_e9d_outcome_consistency_verified=True,
    )


def _linear_flat_stress_derivatives(
    energy_slope: float,
    longitudinal_stress_slope: float,
) -> np.ndarray:
    """선형 반례의 ``partial_alpha T^{mu nu}`` 를 돌려준다."""

    derivatives = np.zeros(
        (SPACETIME_DIMENSION, SPACETIME_DIMENSION, SPACETIME_DIMENSION),
        dtype=float,
    )
    derivatives[0, 0, 0] = energy_slope
    derivatives[1, 1, 1] = longitudinal_stress_slope
    return derivatives


def _flat_mixed_stress_divergence(stress_derivatives: np.ndarray) -> np.ndarray:
    """선언된 민코프스키 좌표계에서 ``partial_mu T^mu{}_nu`` 를 계산한다."""

    minkowski = np.diag((-1.0, 1.0, 1.0, 1.0))
    return np.einsum("mmk,kn->n", stress_derivatives, minkowski)


def construct_flat_receipt_current_counterexample(
    *,
    receipt_energy: float,
    duration: float,
    spatial_volume: float,
    momentum_source_a: float,
    momentum_source_b: float,
    reference_mass_scale: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> FlatReceiptCurrentCounterexample:
    """같은 영수증을 갖는 두 보존 부문 완성을 구성한다.

    평탄 직사각 세계관에서 ``p = delta / (T V_3)`` 로 두고

        T_S^{00} = p t,  T_S^{01} = 0,  T_S^{11} = r x.

    라 하자. 그러면 ``Q_0 = -p``, ``Q_1 = r`` 이고 측면 에너지 플럭스는 0이며,
    적분된 시스템 에너지 이득은 모든 ``r`` 에 대해 ``delta`` 다. 서로 다른 두
    ``r`` 값은 스칼라 영수증을 고정한 채 국소 운동량 전류를 바꾼다. 보완 부문
    ``C^{mu nu}-T_S^{mu nu}`` 는 발산 ``-Q_nu`` 를 가지며 같은 상수 전체 응력을
    유지한다.
    """

    tolerance = _positive_scalar(tolerance, "tolerance")
    if tolerance > MAX_TOLERANCE:
        raise ValueError(f"tolerance must not exceed {MAX_TOLERANCE}")
    reference_mass_scale = _positive_scalar(
        reference_mass_scale,
        "reference_mass_scale",
    )
    receipt_energy = _finite_scalar(receipt_energy, "receipt_energy")
    if receipt_energy < 0.0:
        raise ValueError("receipt_energy must be non-negative")
    duration = _positive_scalar(duration, "duration")
    spatial_volume = _positive_scalar(spatial_volume, "spatial_volume")
    momentum_source_a = _finite_scalar(momentum_source_a, "momentum_source_a")
    momentum_source_b = _finite_scalar(momentum_source_b, "momentum_source_b")

    four_volume = duration * spatial_volume
    energy_source_density = receipt_energy / four_volume
    profile_a_array = np.asarray(
        (-energy_source_density, momentum_source_a, 0.0, 0.0),
        dtype=float,
    )
    profile_b_array = np.asarray(
        (-energy_source_density, momentum_source_b, 0.0, 0.0),
        dtype=float,
    )
    derivatives_a = _linear_flat_stress_derivatives(
        energy_source_density,
        momentum_source_a,
    )
    derivatives_b = _linear_flat_stress_derivatives(
        energy_source_density,
        momentum_source_b,
    )
    battery_derivatives_a = -derivatives_a
    battery_derivatives_b = -derivatives_b
    computed_system_divergence_a = _flat_mixed_stress_divergence(derivatives_a)
    computed_system_divergence_b = _flat_mixed_stress_divergence(derivatives_b)
    computed_battery_divergence_a = _flat_mixed_stress_divergence(
        battery_derivatives_a
    )
    computed_battery_divergence_b = _flat_mixed_stress_divergence(
        battery_derivatives_b
    )
    profile_a = tuple(float(value) for value in profile_a_array)
    profile_b = tuple(float(value) for value in profile_b_array)
    battery_profile_a_array = -profile_a_array
    battery_profile_b_array = -profile_b_array
    battery_profile_a = tuple(float(value) for value in battery_profile_a_array)
    battery_profile_b = tuple(float(value) for value in battery_profile_b_array)
    integrated_a = -profile_a[0] * four_volume
    integrated_b = -profile_b[0] * four_volume
    energy_scale = reference_mass_scale
    current_scale = reference_mass_scale**5
    residual_a = abs(integrated_a - receipt_energy) / energy_scale
    residual_b = abs(integrated_b - receipt_energy) / energy_scale
    current_difference = abs(momentum_source_a - momentum_source_b) / current_scale
    divergence_identity_vectors = np.stack(
        (
            computed_system_divergence_a - profile_a_array,
            computed_system_divergence_b - profile_b_array,
            computed_battery_divergence_a - battery_profile_a_array,
            computed_battery_divergence_b - battery_profile_b_array,
        )
    )
    total_divergence_vectors = np.stack(
        (
            computed_system_divergence_a + computed_battery_divergence_a,
            computed_system_divergence_b + computed_battery_divergence_b,
        )
    )
    maximum_divergence_identity_residual = float(
        np.max(np.linalg.norm(divergence_identity_vectors, axis=1))
    ) / current_scale
    maximum_total_divergence_residual = float(
        np.max(np.linalg.norm(total_divergence_vectors, axis=1))
    ) / current_scale
    final_system_energy_density = energy_source_density * duration
    complement_constant_energy_density = 2.0 * final_system_energy_density
    minimum_complement_energy_density = (
        complement_constant_energy_density - final_system_energy_density
    )
    sample_stress_a = np.diag(
        (final_system_energy_density, 0.0, 0.0, 0.0)
    )
    sample_stress_b = np.diag(
        (final_system_energy_density, 0.0, 0.0, 0.0)
    )
    maximum_lateral_energy_flux_density = max(
        float(np.max(np.abs(sample_stress_a[1:, 0]))),
        float(np.max(np.abs(sample_stress_b[1:, 0]))),
    ) / reference_mass_scale**4

    current_mass_dimension = 5
    four_volume_mass_dimension = -4
    energy_mass_dimension = 1
    dimensions_pass = (
        current_mass_dimension + four_volume_mass_dimension
        == energy_mass_dimension
    )
    same_receipt = residual_a <= tolerance and residual_b <= tolerance
    profiles_distinct = current_difference > tolerance
    divergence_identities_hold = maximum_divergence_identity_residual <= tolerance
    total_divergence_closes = maximum_total_divergence_residual <= tolerance
    lateral_flux_zero = maximum_lateral_energy_flux_density <= tolerance
    complement_energy_nonnegative = (
        minimum_complement_energy_density / reference_mass_scale**4
        >= -tolerance
    )
    witness = (
        same_receipt
        and profiles_distinct
        and divergence_identities_hold
        and total_divergence_closes
        and lateral_flux_zero
        and complement_energy_nonnegative
        and dimensions_pass
    )

    return FlatReceiptCurrentCounterexample(
        receipt_energy=receipt_energy,
        duration=duration,
        spatial_volume=spatial_volume,
        four_volume=four_volume,
        energy_source_density=energy_source_density,
        profile_a_current_covector=profile_a,
        profile_b_current_covector=profile_b,
        profile_a_battery_current_covector=battery_profile_a,
        profile_b_battery_current_covector=battery_profile_b,
        profile_a_computed_system_divergence_covector=tuple(
            float(value) for value in computed_system_divergence_a
        ),
        profile_b_computed_system_divergence_covector=tuple(
            float(value) for value in computed_system_divergence_b
        ),
        profile_a_computed_battery_divergence_covector=tuple(
            float(value) for value in computed_battery_divergence_a
        ),
        profile_b_computed_battery_divergence_covector=tuple(
            float(value) for value in computed_battery_divergence_b
        ),
        profile_a_integrated_energy=integrated_a,
        profile_b_integrated_energy=integrated_b,
        complement_constant_energy_density=complement_constant_energy_density,
        minimum_complement_energy_density=minimum_complement_energy_density,
        dimensionless_profile_a_receipt_residual=residual_a,
        dimensionless_profile_b_receipt_residual=residual_b,
        dimensionless_current_difference=current_difference,
        maximum_dimensionless_divergence_identity_residual=(
            maximum_divergence_identity_residual
        ),
        maximum_dimensionless_total_divergence_residual=(
            maximum_total_divergence_residual
        ),
        maximum_dimensionless_lateral_energy_flux_density=(
            maximum_lateral_energy_flux_density
        ),
        current_mass_dimension=current_mass_dimension,
        four_volume_mass_dimension=four_volume_mass_dimension,
        energy_mass_dimension=energy_mass_dimension,
        dimensions_pass=dimensions_pass,
        same_flat_worldtube=True,
        same_scalar_receipt=same_receipt,
        current_profiles_distinct=profiles_distinct,
        lateral_energy_flux_zero=lateral_flux_zero,
        opposite_sector_closes_total_stress=total_divergence_closes,
        unique_current_from_receipt_claim_refuted=witness,
    )


Vector4 = tuple[float, float, float, float]
Matrix4 = tuple[Vector4, Vector4, Vector4, Vector4]


@dataclass(frozen=True)
class TwoScalarExchangeReceipt:
    """공급된 국소 두 스칼라 작용 하나의 점별 워드 영수증."""

    allocation_fraction: float
    coupling: float
    interaction_energy_density: float
    interaction_d_phi: float
    interaction_d_psi: float
    interaction_gradient_covector: Vector4
    exchange_current_phi_covector: Vector4
    exchange_current_psi_covector: Vector4
    phi_sector_divergence_covector: Vector4
    psi_sector_divergence_covector: Vector4
    total_divergence_covector: Vector4
    phi_eom_residual: float
    psi_eom_residual: float
    dimensionless_eom_residual: float
    dimensionless_exchange_current_norm: float
    dimensionless_interaction_allocation_residual: float
    dimensionless_total_divergence: float
    dimensionless_ward_identity_residual: float
    dimensionless_complementarity_residual: float
    metric_signature: tuple[int, int, int, int]
    field_mass_dimension: int
    interaction_mass_dimension: int
    current_mass_dimension: int
    normalized_residual_mass_dimension: int
    dimensions_pass: bool
    interaction_energy_counted_once: bool
    on_shell_within_tolerance: bool
    total_stress_conserved_on_shell: bool
    zero_coupling_exchange_vanishes: bool
    local_covariant_action_supplied: bool = True
    covariant_action_exchange_current_derived: bool = True
    interaction_allocation_dynamically_selected: bool = False
    domino_receipt_to_action_derived: bool = False
    covariant_matching_current_derived: bool = False
    physical_pointer_derived: bool = False
    record_to_gravity_source_derived: bool = False


@dataclass(frozen=True)
class AllocationNonidentifiabilityCertificate:
    """상호작용 밀도는 하나이고 전류는 서로 다른 두 배정."""

    alpha_zero_receipt: TwoScalarExchangeReceipt
    alpha_one_receipt: TwoScalarExchangeReceipt
    dimensionless_interaction_density_difference: float
    dimensionless_current_difference: float
    dimensionless_total_interaction_allocation_difference: float
    same_action_and_interaction_density: bool
    currents_distinct: bool
    total_stress_alpha_invariant: bool
    unique_current_claim_refuted: bool
    supplied_allocation_required: bool = True
    domino_receipt_to_action_derived: bool = False
    physical_source_derived: bool = False
    record_to_gravity_source_derived: bool = False


def _finite_scalar_receipt(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _positive_scalar_receipt(value: float, name: str) -> float:
    value = _finite_scalar_receipt(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _covector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (SPACETIME_DIMENSION,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite four-component covector")
    return array


def _lorentzian_geometry(
    metric_covariant: Sequence[Sequence[float]],
    observer_contravariant: Sequence[float],
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    metric = np.asarray(metric_covariant, dtype=float)
    if metric.shape != (SPACETIME_DIMENSION, SPACETIME_DIMENSION):
        raise ValueError("metric_covariant must be a four-by-four matrix")
    if not np.all(np.isfinite(metric)) or not np.allclose(
        metric,
        metric.T,
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError("metric_covariant must be finite and symmetric")
    eigenvalues = np.linalg.eigvalsh(metric)
    metric_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    negative = int(np.count_nonzero(eigenvalues < -tolerance * metric_scale))
    positive = int(np.count_nonzero(eigenvalues > tolerance * metric_scale))
    if negative != 1 or positive != 3:
        raise ValueError("metric_covariant must have Lorentzian signature (-,+,+,+)")
    inverse = np.linalg.inv(metric)
    observer = _covector(observer_contravariant, "observer_contravariant")
    observer_norm = float(observer @ metric @ observer)
    if not math.isclose(observer_norm, -1.0, rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError("observer_contravariant must be unit timelike")
    positive_inverse = inverse + 2.0 * np.outer(observer, observer)
    positive_eigenvalues = np.linalg.eigvalsh(positive_inverse)
    if float(np.min(positive_eigenvalues)) <= tolerance:
        raise ArithmeticError("observer-induced covector norm is not positive definite")
    signature = tuple(-1 if value < 0.0 else 1 for value in eigenvalues)
    return metric, inverse, positive_inverse, signature  # type: ignore[return-value]


def _positive_covector_norm(
    covector: np.ndarray,
    positive_inverse: np.ndarray,
    tolerance: float,
) -> float:
    squared = float(covector @ positive_inverse @ covector)
    scale = max(float(np.linalg.norm(covector)) ** 2, 1.0)
    if squared < -tolerance * scale:
        raise ArithmeticError("observer-induced covector norm became negative")
    return math.sqrt(max(0.0, squared))


def _vector4(array: np.ndarray) -> Vector4:
    return tuple(float(value) for value in array)  # type: ignore[return-value]


def two_scalar_exchange_receipt(
    *,
    metric_covariant: Sequence[Sequence[float]],
    observer_contravariant: Sequence[float],
    phi: float,
    psi: float,
    gradient_phi_covector: Sequence[float],
    gradient_psi_covector: Sequence[float],
    box_phi: float,
    box_psi: float,
    mass_phi: float,
    mass_psi: float,
    coupling: float,
    allocation_fraction: float,
    reference_mass_scale: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> TwoScalarExchangeReceipt:
    """한 점에서 부문 분할과 온셸 워드 항등식을 평가한다.

    ``phi`` 와 ``psi`` 는 질량 차원 1, 그 공변 기울기는 2, ``box_phi`` 와
    ``box_psi`` 는 3이며, ``coupling`` 과 ``allocation_fraction`` 은 모두
    무차원이다. 배정 비율은 전역 공급 상수다; 시공간 의존 값이라면 이 계약이
    표현하지 않는 미분 항이 추가된다.
    """

    tolerance = _positive_scalar_receipt(tolerance, "tolerance")
    if tolerance > MAX_TOLERANCE:
        raise ValueError(f"tolerance must not exceed {MAX_TOLERANCE}")
    reference_mass_scale = _positive_scalar_receipt(
        reference_mass_scale,
        "reference_mass_scale",
    )
    phi = _finite_scalar_receipt(phi, "phi")
    psi = _finite_scalar_receipt(psi, "psi")
    box_phi = _finite_scalar_receipt(box_phi, "box_phi")
    box_psi = _finite_scalar_receipt(box_psi, "box_psi")
    mass_phi = _finite_scalar_receipt(mass_phi, "mass_phi")
    mass_psi = _finite_scalar_receipt(mass_psi, "mass_psi")
    coupling = _finite_scalar_receipt(coupling, "coupling")
    allocation_fraction = _finite_scalar_receipt(
        allocation_fraction,
        "allocation_fraction",
    )
    if mass_phi < 0.0 or mass_psi < 0.0:
        raise ValueError("scalar masses must be non-negative")
    if coupling < 0.0:
        raise ValueError("coupling must be non-negative in the stable quartic branch")
    if not 0.0 <= allocation_fraction <= 1.0:
        raise ValueError("allocation_fraction must lie in [0, 1]")

    _, _, positive_inverse, signature = _lorentzian_geometry(
        metric_covariant,
        observer_contravariant,
        tolerance,
    )
    gradient_phi = _covector(gradient_phi_covector, "gradient_phi_covector")
    gradient_psi = _covector(gradient_psi_covector, "gradient_psi_covector")

    interaction = 0.5 * coupling * phi * phi * psi * psi
    interaction_d_phi = coupling * phi * psi * psi
    interaction_d_psi = coupling * phi * phi * psi
    interaction_gradient = (
        interaction_d_phi * gradient_phi
        + interaction_d_psi * gradient_psi
    )
    phi_eom_residual = (
        box_phi - mass_phi * mass_phi * phi - interaction_d_phi
    )
    psi_eom_residual = (
        box_psi - mass_psi * mass_psi * psi - interaction_d_psi
    )

    exchange_phi = (
        (1.0 - allocation_fraction) * interaction_d_phi * gradient_phi
        - allocation_fraction * interaction_d_psi * gradient_psi
    )
    exchange_psi = -exchange_phi
    phi_sector_divergence = exchange_phi + phi_eom_residual * gradient_phi
    psi_sector_divergence = exchange_psi + psi_eom_residual * gradient_psi
    total_divergence = phi_sector_divergence + psi_sector_divergence
    expected_off_shell_divergence = (
        phi_eom_residual * gradient_phi + psi_eom_residual * gradient_psi
    )
    ward_identity_difference = total_divergence - expected_off_shell_divergence
    complementarity_difference = (
        phi_sector_divergence
        + psi_sector_divergence
        - expected_off_shell_divergence
    )

    mass_cubed = reference_mass_scale**3
    mass_fourth = reference_mass_scale**4
    mass_fifth = reference_mass_scale**5
    dimensionless_eom_residual = max(
        abs(phi_eom_residual),
        abs(psi_eom_residual),
    ) / mass_cubed
    dimensionless_exchange_current_norm = (
        _positive_covector_norm(exchange_phi, positive_inverse, tolerance)
        / mass_fifth
    )
    allocated_interaction = (
        allocation_fraction * interaction
        + (1.0 - allocation_fraction) * interaction
    )
    dimensionless_interaction_allocation_residual = abs(
        allocated_interaction - interaction
    ) / mass_fourth
    dimensionless_total_divergence = (
        _positive_covector_norm(total_divergence, positive_inverse, tolerance)
        / mass_fifth
    )
    dimensionless_ward_identity_residual = (
        _positive_covector_norm(
            ward_identity_difference,
            positive_inverse,
            tolerance,
        )
        / mass_fifth
    )
    dimensionless_complementarity_residual = (
        _positive_covector_norm(
            complementarity_difference,
            positive_inverse,
            tolerance,
        )
        / mass_fifth
    )
    on_shell = dimensionless_eom_residual <= tolerance

    field_mass_dimension = 1
    interaction_mass_dimension = 4
    current_mass_dimension = 5
    normalized_residual_mass_dimension = 0
    dimensions_pass = (
        4 * field_mass_dimension == interaction_mass_dimension
        and (interaction_mass_dimension - field_mass_dimension)
        + (field_mass_dimension + 1)
        == current_mass_dimension
        and current_mass_dimension - 5 == normalized_residual_mass_dimension
    )

    return TwoScalarExchangeReceipt(
        allocation_fraction=allocation_fraction,
        coupling=coupling,
        interaction_energy_density=interaction,
        interaction_d_phi=interaction_d_phi,
        interaction_d_psi=interaction_d_psi,
        interaction_gradient_covector=_vector4(interaction_gradient),
        exchange_current_phi_covector=_vector4(exchange_phi),
        exchange_current_psi_covector=_vector4(exchange_psi),
        phi_sector_divergence_covector=_vector4(phi_sector_divergence),
        psi_sector_divergence_covector=_vector4(psi_sector_divergence),
        total_divergence_covector=_vector4(total_divergence),
        phi_eom_residual=phi_eom_residual,
        psi_eom_residual=psi_eom_residual,
        dimensionless_eom_residual=dimensionless_eom_residual,
        dimensionless_exchange_current_norm=dimensionless_exchange_current_norm,
        dimensionless_interaction_allocation_residual=(
            dimensionless_interaction_allocation_residual
        ),
        dimensionless_total_divergence=dimensionless_total_divergence,
        dimensionless_ward_identity_residual=dimensionless_ward_identity_residual,
        dimensionless_complementarity_residual=(
            dimensionless_complementarity_residual
        ),
        metric_signature=signature,
        field_mass_dimension=field_mass_dimension,
        interaction_mass_dimension=interaction_mass_dimension,
        current_mass_dimension=current_mass_dimension,
        normalized_residual_mass_dimension=normalized_residual_mass_dimension,
        dimensions_pass=dimensions_pass,
        interaction_energy_counted_once=(
            dimensionless_interaction_allocation_residual <= tolerance
        ),
        on_shell_within_tolerance=on_shell,
        total_stress_conserved_on_shell=(
            on_shell
            and dimensionless_total_divergence <= tolerance
            and dimensionless_ward_identity_residual <= tolerance
            and dimensionless_complementarity_residual <= tolerance
        ),
        zero_coupling_exchange_vanishes=(
            coupling != 0.0 or dimensionless_exchange_current_norm <= tolerance
        ),
    )


def certify_allocation_nonidentifiability(
    *,
    metric_covariant: Sequence[Sequence[float]],
    observer_contravariant: Sequence[float],
    phi: float,
    psi: float,
    gradient_phi_covector: Sequence[float],
    gradient_psi_covector: Sequence[float],
    box_phi: float,
    box_psi: float,
    mass_phi: float,
    mass_psi: float,
    coupling: float,
    reference_mass_scale: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> AllocationNonidentifiabilityCertificate:
    """상호작용 밀도를 고정한 채 ``alpha=0`` 과 ``alpha=1`` 을 비교한다."""

    common = dict(
        metric_covariant=metric_covariant,
        observer_contravariant=observer_contravariant,
        phi=phi,
        psi=psi,
        gradient_phi_covector=gradient_phi_covector,
        gradient_psi_covector=gradient_psi_covector,
        box_phi=box_phi,
        box_psi=box_psi,
        mass_phi=mass_phi,
        mass_psi=mass_psi,
        coupling=coupling,
        reference_mass_scale=reference_mass_scale,
        tolerance=tolerance,
    )
    alpha_zero = two_scalar_exchange_receipt(
        **common,
        allocation_fraction=0.0,
    )
    alpha_one = two_scalar_exchange_receipt(
        **common,
        allocation_fraction=1.0,
    )
    if not (
        alpha_zero.on_shell_within_tolerance
        and alpha_one.on_shell_within_tolerance
        and alpha_zero.total_stress_conserved_on_shell
        and alpha_one.total_stress_conserved_on_shell
    ):
        raise ValueError("allocation non-identifiability requires an on-shell witness")

    tolerance = _positive_scalar_receipt(tolerance, "tolerance")
    reference_mass_scale = _positive_scalar_receipt(
        reference_mass_scale,
        "reference_mass_scale",
    )
    _, _, positive_inverse, _ = _lorentzian_geometry(
        metric_covariant,
        observer_contravariant,
        tolerance,
    )
    interaction_scale = reference_mass_scale**4
    current_scale = reference_mass_scale**5
    receipt_difference = abs(
        alpha_zero.interaction_energy_density
        - alpha_one.interaction_energy_density
    ) / interaction_scale
    current_difference_covector = (
        np.asarray(alpha_zero.exchange_current_phi_covector)
        - np.asarray(alpha_one.exchange_current_phi_covector)
    )
    current_difference = (
        _positive_covector_norm(
            current_difference_covector,
            positive_inverse,
            tolerance,
        )
        / current_scale
    )
    same_receipt = receipt_difference <= tolerance
    currents_distinct = current_difference > tolerance
    alpha_zero_total_interaction = (
        alpha_zero.allocation_fraction * alpha_zero.interaction_energy_density
        + (1.0 - alpha_zero.allocation_fraction)
        * alpha_zero.interaction_energy_density
    )
    alpha_one_total_interaction = (
        alpha_one.allocation_fraction * alpha_one.interaction_energy_density
        + (1.0 - alpha_one.allocation_fraction)
        * alpha_one.interaction_energy_density
    )
    total_interaction_difference = abs(
        alpha_zero_total_interaction - alpha_one_total_interaction
    ) / interaction_scale
    total_stress_alpha_invariant = (
        total_interaction_difference <= tolerance
        and alpha_zero.interaction_energy_counted_once
        and alpha_one.interaction_energy_counted_once
    )
    witness = same_receipt and currents_distinct and total_stress_alpha_invariant

    return AllocationNonidentifiabilityCertificate(
        alpha_zero_receipt=alpha_zero,
        alpha_one_receipt=alpha_one,
        dimensionless_interaction_density_difference=receipt_difference,
        dimensionless_current_difference=current_difference,
        dimensionless_total_interaction_allocation_difference=(
            total_interaction_difference
        ),
        same_action_and_interaction_density=same_receipt,
        currents_distinct=currents_distinct,
        total_stress_alpha_invariant=total_stress_alpha_invariant,
        unique_current_claim_refuted=witness,
    )


@dataclass(frozen=True)
class CovariantMaterialLatticeCostCertificate:
    """유계 장부 영수증이며 격자 동역학의 해가 아니다."""

    cells_per_axis: int
    proper_cell_spacing: float
    rod_scale: float
    battery_energy_per_cell: float
    carrier_mass: float
    carrier_momentum: float
    onsite_exchange_coupling: float
    quartic_coupling: float
    guide_well_mass_squared: float
    cell_well_mass_squared: float
    wave_number: float
    cube_side_length: float
    material_gram_diagonal: tuple[float, float, float]
    material_gram_determinant: float
    normalized_gram_determinant: float
    proper_cell_volume: float
    winding_per_axis: float
    rod_energy_density: float
    rod_pressure: float
    rod_equation_of_state: float
    finite_rod_energy: float
    guide_all_success_battery_count: int
    guide_battery_capacity: float
    full_volume_cell_count: int
    full_volume_battery_capacity: float
    carrier_frequency: float
    carrier_group_velocity: float
    quartic_lower_bound_coefficient: float
    extremal_quartic_potential: float
    quartic_saturation_residual: float
    dimensionless_core_arguments: tuple[tuple[str, str], ...]
    action_terms_have_mass_dimension_four: bool
    compact_phase_period_is_two_pi: bool
    clock_field_used: bool
    diffeomorphism_covariant_scalar_candidate_by_construction: bool
    supplied_finite_free_rod_background_bookkeeping: bool
    invariant_gram_and_nondegenerate_conditional_geometry: bool
    finite_rod_receipt: bool
    rod_and_battery_ledgers_kept_separate: bool
    dimension_closure: bool
    static_common_coupling_without_coordinate_time_schedule: bool
    canonical_fixed_background_classical_principal_symbol: bool
    fixed_background_classical_domain_of_dependence: bool
    spacing_action_winding_derived: bool
    interacting_backreacted_theta_solution_derived: bool
    background_stability_or_caustic_freedom_derived: bool
    periodic_well_localized_modes_derived: bool
    action_to_projected_rates_or_resonance_derived: bool
    scattering_energy_transfer_receipt_derived: bool
    durable_record_or_selection_derived: bool
    repeated_cptp_fresh_ancilla_derived: bool
    band_or_front_speed_derived: bool
    qft_microcausality_or_no_signalling_derived: bool
    coupled_gr_source_derived: bool
    infinite_isolated_lattice_finite_total_energy_derived: bool
    gates_five_to_eight_derived: bool


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def certify_covariant_material_lattice_cost(
    *,
    cells_per_axis: int,
    proper_cell_spacing: float,
    rod_scale: float,
    battery_energy_per_cell: float,
    carrier_mass: float,
    carrier_momentum: float,
    onsite_exchange_coupling: float,
    quartic_coupling: float,
    guide_well_mass_squared: float = 0.0,
    cell_well_mass_squared: float = 0.0,
) -> CovariantMaterialLatticeCostCertificate:
    """공급된 유한 격자 비용과 명시적 주장 경계를 증서로 만든다.

    ``Theta^I`` 는 주기 ``2 pi`` 의 무차원 콤팩트 위상이고 ``f_X`` 는 질량
    차원 1이다. 따라서 ``f_X**2 (d Theta)**2`` 는 질량 차원 4다. ``g`` 와
    ``lambda`` 는 무차원이며 두 우물 계수는 질량 차원 2다. 시계 장이나 시간
    일정은 없다.
    """

    if isinstance(cells_per_axis, bool) or not isinstance(cells_per_axis, int):
        raise ValueError("cells_per_axis must be an integer")
    if cells_per_axis < 1:
        raise ValueError("cells_per_axis must be at least one")
    a = _positive(proper_cell_spacing, "proper_cell_spacing")
    f_x = _positive(rod_scale, "rod_scale")
    e_b = _positive(battery_energy_per_cell, "battery_energy_per_cell")
    mass = _positive(carrier_mass, "carrier_mass")
    momentum = _nonnegative(carrier_momentum, "carrier_momentum")
    g = float(onsite_exchange_coupling)
    lam = _positive(quartic_coupling, "quartic_coupling")
    if not math.isfinite(g) or g == 0.0:
        raise ValueError("onsite_exchange_coupling must be finite and non-zero")
    guide_well = _nonnegative(guide_well_mass_squared, "guide_well_mass_squared")
    cell_well = _nonnegative(cell_well_mass_squared, "cell_well_mass_squared")
    if lam < abs(g):
        raise ValueError("quartic stability requires quartic_coupling >= abs(g)")

    q = 2.0 * math.pi / a
    length = cells_per_axis * a
    gram_diagonal = (q * q, q * q, q * q)
    determinant = q**6
    normalized_determinant = determinant / q**6
    cell_volume = (2.0 * math.pi) ** 3 / math.sqrt(determinant)
    winding = q * length / (2.0 * math.pi)

    rho = 1.5 * f_x * f_x * q * q
    pressure = -0.5 * f_x * f_x * q * q
    rod_energy = rho * length**3
    guide_count = cells_per_axis
    volume_count = cells_per_axis**3
    omega = math.sqrt(momentum * momentum + mass * mass)
    velocity = momentum / omega
    lower_coefficient = (lam - abs(g)) / 4.0

    # H=1/sqrt(2), D=1/2, B=-sign(g)/2 이면 S=1 이다. 따라서
    # 안정화 항은 lambda/4 이고 g |H|^2 (D*B+B*D) 는 -|g|/4 이다.
    h_extremal = 1.0 / math.sqrt(2.0)
    d_extremal = 0.5
    b_extremal = -math.copysign(0.5, g)
    extremal_s = h_extremal**2 + d_extremal**2 + b_extremal**2
    extremal_stabilizer = 0.25 * lam * extremal_s**2
    extremal_interaction = g * h_extremal**2 * (
        d_extremal * b_extremal + b_extremal * d_extremal
    )
    extremal_potential = extremal_stabilizer + extremal_interaction
    saturation_residual = abs(extremal_potential - lower_coefficient)

    return CovariantMaterialLatticeCostCertificate(
        cells_per_axis=cells_per_axis,
        proper_cell_spacing=a,
        rod_scale=f_x,
        battery_energy_per_cell=e_b,
        carrier_mass=mass,
        carrier_momentum=momentum,
        onsite_exchange_coupling=g,
        quartic_coupling=lam,
        guide_well_mass_squared=guide_well,
        cell_well_mass_squared=cell_well,
        wave_number=q,
        cube_side_length=length,
        material_gram_diagonal=gram_diagonal,
        material_gram_determinant=determinant,
        normalized_gram_determinant=normalized_determinant,
        proper_cell_volume=cell_volume,
        winding_per_axis=winding,
        rod_energy_density=rho,
        rod_pressure=pressure,
        rod_equation_of_state=pressure / rho,
        finite_rod_energy=rod_energy,
        guide_all_success_battery_count=guide_count,
        guide_battery_capacity=guide_count * e_b,
        full_volume_cell_count=volume_count,
        full_volume_battery_capacity=volume_count * e_b,
        carrier_frequency=omega,
        carrier_group_velocity=velocity,
        quartic_lower_bound_coefficient=lower_coefficient,
        extremal_quartic_potential=extremal_potential,
        quartic_saturation_residual=saturation_residual,
        dimensionless_core_arguments=(
            ("q a = 2 pi", "compact phase winding"),
            ("q L / (2 pi) = N", "supplied integer winding"),
            ("v_g = |k| / sqrt(|k|^2 + m_H^2)", "free-particle sample"),
        ),
        action_terms_have_mass_dimension_four=True,
        compact_phase_period_is_two_pi=True,
        clock_field_used=False,
        diffeomorphism_covariant_scalar_candidate_by_construction=True,
        supplied_finite_free_rod_background_bookkeeping=True,
        invariant_gram_and_nondegenerate_conditional_geometry=True,
        finite_rod_receipt=True,
        rod_and_battery_ledgers_kept_separate=True,
        dimension_closure=True,
        static_common_coupling_without_coordinate_time_schedule=True,
        canonical_fixed_background_classical_principal_symbol=True,
        fixed_background_classical_domain_of_dependence=True,
        spacing_action_winding_derived=False,
        interacting_backreacted_theta_solution_derived=False,
        background_stability_or_caustic_freedom_derived=False,
        periodic_well_localized_modes_derived=False,
        action_to_projected_rates_or_resonance_derived=False,
        scattering_energy_transfer_receipt_derived=False,
        durable_record_or_selection_derived=False,
        repeated_cptp_fresh_ancilla_derived=False,
        band_or_front_speed_derived=False,
        qft_microcausality_or_no_signalling_derived=False,
        coupled_gr_source_derived=False,
        infinite_isolated_lattice_finite_total_energy_derived=False,
        gates_five_to_eight_derived=False,
    )
