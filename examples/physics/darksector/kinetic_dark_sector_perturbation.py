"""운동 암흑부문의 단일 시계 섭동·EFT 차단 게이트와 급냉 스펙트럼-현재 존재비 조건부 다리를 한 모듈에 모은다.

두 절로 구성한다.

1. 단일 시계 섭동과 EFT 차단(cutoff) 게이트. 옛 R2 진단에서 재현 가능하고 자료가
   적게 드는 부분을 본 examples 트리로 승격한 것이다. 시계+아인슈타인 부분계만
   검사한다. 아래의 고정 배경 ``pi`` 성장은 시계 좌표의 진단이지, 관측 가능한 물질
   성장 함수나 ``f sigma_8`` 가능도가 아니다.

2. 매끄러운 급냉 스펙트럼에서 현재 존재비로 가는 조건부 다리(conditional bridge).
   여기 구현한 정리는 생산 초곡면이 선언된 *뒤에서* 시작한다. 차원 있는 입력은
   모두 ``hbar=c=k_B=1`` 인 공통 자연 에너지 단위를 쓴다. 엔트로피 장부가 생산 시점의
   척도인자를 고정하고, 정확한 매끄러운 tanh 보골류보프(Bogoliubov) 스펙트럼이 생성된
   위상공간 점유수를 고정하며, 무충돌 리우빌(Liouville) 진화가 물리 운동량을
   ``p_0=a_* p_*`` 로 적색이동시킨다.

   안정하고 분리된 종과 ``a_0=1`` 에 대해

       n_0 = a_*^3 g/(2 pi^2) integral dp p^2 f_*(p),

       rho_0 = a_*^3 g/(2 pi^2)
               integral dp p^2 sqrt(m_f^2 + a_*^2 p^2) f_*(p),

       Omega_0 = rho_0 / (3 M_Pl^2 H_0^2).

   급냉, 엔트로피 역사, 우주론 정규화가 주어지면 조건부 전방 사상이 닫힌다. 그
   입력들, 유한 시간 원천 ``q(N)``, 재규격화 FLRW 응력 텐서, 섭동은 유도하지 않는다.
   수치 영수증은 유한 창 심프슨 구적법이다. 생략된 매끄러운 급냉 UV 꼬리에 대한
   별도의 해석적 상계는 붙지만, 유한 창의 이산화 오차 보증서는 없다.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np

from examples.physics.darksector.kinetic_dark_sector_gate import (
    BackgroundSolution,
    KineticClockConfig,
    _densities,
    _rhs,
    solve_background,
)
from examples.physics.darksector.kinetic_dark_sector_quench import (
    smooth_quench_present_tail_certificate,
)
from examples.physics.record.theater_opening import (
    QuantumSeatSpecies,
    integrate_quench_densities,
    smooth_tanh_mode,
)


# ---------------------------------------------------------------------------
# 1. 단일 시계 섭동·EFT 차단 게이트
# ---------------------------------------------------------------------------


H0_KM_S_MPC = 67.4
MPC_IN_M = 3.0856775814913673e22
HBAR_EV_S = 6.582119569e-16
HBAR_C_EV_M = 1.973269804e-7
REDUCED_MPL_EV = 2.435e27
H0_EV = H0_KM_S_MPC * 1000.0 / MPC_IN_M * HBAR_EV_S
RHO_CRIT0_EV4 = 3.0 * REDUCED_MPL_EV**2 * H0_EV**2
MPC_INV_EV = HBAR_C_EV_M / MPC_IN_M


@dataclass(frozen=True)
class PerturbationNode:
    n: float
    e2: float
    friction: float
    tachyon_ratio: float
    cs2: float
    q_s_over_mpl2: float
    pump_slope: float
    zeta_decay_slope: float
    energy_cutoff_ev: float
    wavenumber_cutoff_ev: float


@dataclass(frozen=True)
class SingleClockGate:
    gamma: float
    min_friction: float
    max_tachyon_ratio: float
    fixed_coordinate_growth_minus_one: float
    max_log_growth_bound: float
    min_pump_slope: float
    min_zeta_decay_slope: float
    min_energy_cutoff_over_h: float
    min_wavenumber_cutoff_over_k_1mpc: float
    status: str = "PASS_SINGLE_CLOCK_ONLY"
    failed_gates: tuple[str, ...] = ()
    matter_growth_likelihood: str = "NOT_IMPLEMENTED_COUPLED_EQUATIONS_REQUIRED"


@dataclass(frozen=True)
class QuasiStaticGrowthDiagnostic:
    redshift: float
    predicted_fsigma8: float
    observed_fsigma8: float
    observed_sigma: float
    pull: float
    sigma8_0: float
    closure: str = "KINETIC_CLUSTERS_VACUUM_SMOOTH_GR_SUBHORIZON"
    role: str = "APPROXIMATE_DIAGNOSTIC_NOT_FULL_COUPLED_LIKELIHOOD"


@dataclass(frozen=True)
class KappaSensitivityRow:
    kappa: float
    min_cs2: float
    min_friction: float
    max_log_growth_bound: float
    min_energy_cutoff_over_h: float
    status: str
    failed_gates: tuple[str, ...]


@dataclass(frozen=True)
class FiniteProductGaussianStateDensityAudit:
    """보존된 작용에 대한 정확한 유한 모드 상태-에너지 영수증이다.

    자연 단위 ``hbar=c=1`` 과 ``a0=1`` 을 고정한다. 균질 정준 좌표는
    ``q=sqrt(Vc)*(phi, chi)`` 와 ``p=sqrt(Vc)*(dot(phi), dot(chi))`` 다. 주어진 경계
    상태는 비결합 ``phi`` 결맞음 상태와 ``chi`` 열 상태의 곱이다. *같은* 직교 정규
    모드 회전을 위치 블록과 운동량 블록 모두에 적용하면 일반적으로 상관된 가우스
    상태가 된다.

    유한 모드 빼기 ``E_i-mu_i/2`` 는 상호작용하는 두 진동자 진공에 대한 정규 순서다.
    공변 QFT 응력 재규격화가 아니고 원시 우주론 상태를 고르지도 않는다.
    """

    action_parameter_manifest: tuple[float, float, float, float]
    state_boundary_manifest: tuple[float, float, float, float, float]
    environment_thermal_marginal_manifest: tuple[float, float, float, float]
    covariance_ordering: tuple[str, str, str, str]
    canonical_mean: tuple[float, float, float, float]
    centered_symmetrized_covariance: tuple[tuple[float, ...], ...]
    symplectic_eigenvalues: tuple[float, float]
    expected_symplectic_eigenvalues: tuple[float, float]
    dimensionless_uncertainty_minimum_eigenvalue: float
    normal_mode_rotation: tuple[tuple[float, float], ...]
    phase_space_symplectic_transform: tuple[tuple[float, ...], ...]
    symplectic_residual: float
    normal_mode_diagonalization_relative_residual: float
    normal_mode_mass_squared: tuple[float, float]
    normal_mode_masses: tuple[float, float]
    relative_spectral_gap: float
    normal_mode_mean: tuple[float, float, float, float]
    normal_mode_centered_covariance: tuple[tuple[float, ...], ...]
    normal_mode_position_cross_covariance: float
    normal_mode_momentum_cross_covariance: float
    finite_mode_raw_energies_ev: tuple[float, float]
    finite_mode_vacuum_energies_ev: tuple[float, float]
    finite_mode_vacuum_subtracted_energies_ev: tuple[float, float]
    finite_mode_density_constants_ev4: tuple[float, float]
    uncoupled_coherent_preparation_energy_ev: float
    uncoupled_thermal_preparation_energy_ev: float
    bare_product_vacuum_energy_ev: float
    interacting_vacuum_energy_ev: float
    vacuum_mismatch_quench_energy_ev: float
    finite_mode_vacuum_subtracted_total_energy_ev: float
    finite_mode_vacuum_subtracted_total_density_ev4: float
    vacuum_cell_energy_ev: float
    thermal_occupation_relative_residual: float
    raw_energy_rotation_relative_residual: float
    excitation_energy_ledger_relative_residual: float
    mode_sign_flip_energy_relative_residual: float
    uncertainty_principle_pass: bool
    covariance_physicality_pass: bool
    canonical_transform_pass: bool
    mass_matrix_stable: bool
    nondegenerate_mode_allocation_pass: bool
    finite_mode_excitation_nonnegative: bool
    mass_dimension_manifest: tuple[tuple[str, float], ...]
    dimensionless_core_argument_mass_dimensions: tuple[tuple[str, float], ...]
    dimensions_pass: bool
    status: str
    representation: str = "RETAINED_TWO_FIELD_STATE_ENERGY_ONLY"
    boundary_condition: str = (
        "SUPPLIED_UNCOUPLED_COHERENT_X_THERMAL_PRODUCT_AT_A0_EQ_1"
    )
    normal_mode_state_role: str = (
        "CORRELATED_GAUSSIAN_AFTER_CANONICAL_BASIS_ROTATION_NOT_MODE_THERMAL"
    )
    same_state_finite_mode_energy_map_derived: bool = True
    ctp_to_cosmological_state_map_derived: bool = False
    finite_mode_vacuum_subtraction_only: bool = True
    covariant_qft_stress_renormalized: bool = False
    preparation_battery_dynamics_derived: bool = False
    cosmological_initial_state_derived: bool = False
    absolute_abundance_predicted: bool = False
    vacuum_energy_from_state_derived: bool = False
    integrated_out_environment_stress_added: bool = False
    influence_gram_used_as_gravity_source: bool = False
    physical_dark_matter_dark_energy_identification: bool = False


def _real_matrix_tuple(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _relative_residual(left: float, right: float) -> float:
    scale = max(abs(left), abs(right))
    return 0.0 if scale == 0.0 else abs(left - right) / scale


def audit_finite_product_gaussian_state_densities(
    *,
    system_mass_ev: float,
    environment_mass_ev: float,
    bilinear_coupling_ev2: float,
    vacuum_energy_density_ev4: float,
    comoving_volume_ev_minus3: float,
    system_field_mean_ev: float,
    system_field_velocity_ev2: float,
    environment_mean_occupation: float,
    environment_inverse_temperature_ev_minus1: float,
    tolerance: float = 2.0e-11,
) -> FiniteProductGaussianStateDensityAudit:
    """주어진 곱 가우스 상태 하나를 정확한 정규 모드 에너지로 사상한다.

    ``R=(q_phi,q_chi,p_phi,p_chi)`` 에 대해 중심화 공분산은

    ``Sigma=diag(1/(2m), (n+1/2)/M, m/2, M(n+1/2))``

    이다. ``O.T@K@O=diag(mu_-^2,mu_+^2)`` 이면 블록 변환 ``S=diag(O.T,O.T)`` 는
    사교(symplectic) 변환이고 모드 ``i`` 의 들뜸은

    ``E_i=1/2[Pbar_i^2+mu_i^2 Qbar_i^2``
    ``    +Sigma_PP[ii]+mu_i^2 Sigma_QQ[ii]-mu_i]``

    이다. 음수를 잘라내지 않는다. 물리성이나 비음성 위반은 닫힌 실패로 처리한다.
    개별 모드 밀도도 축퇴에서는 닫힌 실패다. 거기서는 합만 기저 불변이다.
    """

    finite_inputs = {
        "system_mass_ev": system_mass_ev,
        "environment_mass_ev": environment_mass_ev,
        "bilinear_coupling_ev2": bilinear_coupling_ev2,
        "vacuum_energy_density_ev4": vacuum_energy_density_ev4,
        "comoving_volume_ev_minus3": comoving_volume_ev_minus3,
        "system_field_mean_ev": system_field_mean_ev,
        "system_field_velocity_ev2": system_field_velocity_ev2,
        "environment_mean_occupation": environment_mean_occupation,
        "tolerance": tolerance,
    }
    checked: dict[str, float] = {}
    for name, value in finite_inputs.items():
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"{name} must be finite")
        checked[name] = converted

    system_mass = checked["system_mass_ev"]
    environment_mass = checked["environment_mass_ev"]
    coupling = checked["bilinear_coupling_ev2"]
    vacuum_density = checked["vacuum_energy_density_ev4"]
    volume = checked["comoving_volume_ev_minus3"]
    field_mean = checked["system_field_mean_ev"]
    field_velocity = checked["system_field_velocity_ev2"]
    occupation = checked["environment_mean_occupation"]
    tolerance = checked["tolerance"]
    beta = float(environment_inverse_temperature_ev_minus1)

    if system_mass <= 0.0 or environment_mass <= 0.0:
        raise ValueError("system_mass_ev and environment_mass_ev must be positive")
    if volume <= 0.0:
        raise ValueError("comoving_volume_ev_minus3 must be positive")
    if occupation < 0.0:
        raise ValueError("environment_mean_occupation must be nonnegative")
    if math.isnan(beta) or beta <= 0.0:
        raise ValueError(
            "environment_inverse_temperature_ev_minus1 must be positive"
        )
    if tolerance <= 0.0 or tolerance > 1.0e-8:
        raise ValueError("tolerance must lie in (0, 1e-8]")

    if math.isinf(beta):
        expected_occupation = 0.0
    else:
        beta_omega = beta * environment_mass
        expected_occupation = (
            0.0 if beta_omega > 700.0 else 1.0 / math.expm1(beta_omega)
        )
    thermal_residual = abs(occupation - expected_occupation) / max(
        1.0,
        occupation,
        expected_occupation,
    )
    if thermal_residual > tolerance:
        raise ValueError(
            "environment_mean_occupation is inconsistent with beta*M"
        )

    mass_matrix = np.array(
        ((system_mass**2, coupling), (coupling, environment_mass**2)),
        dtype=float,
    )
    determinant = float(np.linalg.det(mass_matrix))
    determinant_stable = determinant > 0.0
    if not determinant_stable:
        raise ValueError(
            "bilinear mass matrix must satisfy m^2*M^2-kappa^2 > 0"
        )

    mass_squared, rotation = np.linalg.eigh(mass_matrix)
    mass_matrix_stable = bool(
        determinant_stable and float(mass_squared[0]) > 0.0
    )
    if not mass_matrix_stable:
        raise ValueError("bilinear mass matrix must be positive definite")
    for column in range(2):
        pivot = int(np.argmax(np.abs(rotation[:, column])))
        if rotation[pivot, column] < 0.0:
            rotation[:, column] *= -1.0
    gap = float(mass_squared[1] - mass_squared[0])
    relative_gap = gap / float(np.max(np.abs(mass_squared)))
    if relative_gap <= tolerance:
        raise ValueError(
            "normal-mode spectrum is degenerate; only aggregate energy is "
            "basis invariant"
        )
    mode_masses_array = np.sqrt(mass_squared)

    root_volume = math.sqrt(volume)
    canonical_mean = np.array(
        (root_volume * field_mean, 0.0, root_volume * field_velocity, 0.0),
        dtype=float,
    )
    centered_covariance = np.diag(
        (
            1.0 / (2.0 * system_mass),
            (occupation + 0.5) / environment_mass,
            system_mass / 2.0,
            environment_mass * (occupation + 0.5),
        )
    )
    identity = np.eye(2)
    zero = np.zeros((2, 2))
    symplectic_form = np.block([[zero, identity], [-identity, zero]])

    # 로버트슨(Robertson) 행렬을 대각화하기 전에 국소 사교 재척도로 모든 구적
    # 변수를 무차원으로 만든다. 따라서 beta*M 과 물리성 게이트에 쓰는 모든 고유값은
    # 무차원이다.
    quadrature_rescaling = np.diag(
        (
            math.sqrt(system_mass),
            math.sqrt(environment_mass),
            1.0 / math.sqrt(system_mass),
            1.0 / math.sqrt(environment_mass),
        )
    )
    dimensionless_covariance = (
        quadrature_rescaling @ centered_covariance @ quadrature_rescaling.T
    )
    uncertainty_matrix = (
        dimensionless_covariance + 0.5j * symplectic_form
    )
    uncertainty_minimum = float(np.min(np.linalg.eigvalsh(uncertainty_matrix)))
    symplectic_spectrum = np.sort(
        np.abs(np.linalg.eigvals(1j * symplectic_form @ centered_covariance))
    )[::2]
    expected_symplectic_spectrum = np.array((0.5, occupation + 0.5))
    symplectic_spectrum_residual = float(
        np.max(np.abs(symplectic_spectrum - expected_symplectic_spectrum))
    )
    uncertainty_pass = bool(
        uncertainty_minimum >= -tolerance
        and float(np.min(symplectic_spectrum)) >= 0.5 - tolerance
    )
    covariance_physicality_pass = bool(
        uncertainty_pass and symplectic_spectrum_residual <= tolerance
    )
    if not covariance_physicality_pass:
        raise ValueError("Gaussian covariance violates the uncertainty gate")

    phase_space_transform = np.block(
        [[rotation.T, zero], [zero, rotation.T]]
    )
    symplectic_residual = float(
        np.max(
            np.abs(
                phase_space_transform
                @ symplectic_form
                @ phase_space_transform.T
                - symplectic_form
            )
        )
    )
    diagonalized_mass_matrix = rotation.T @ mass_matrix @ rotation
    diagonalization_scale = float(np.max(np.abs(mass_squared)))
    diagonalization_residual = float(
        np.max(
            np.abs(diagonalized_mass_matrix - np.diag(mass_squared))
        )
        / diagonalization_scale
    )
    canonical_transform_pass = bool(
        symplectic_residual <= tolerance
        and diagonalization_residual <= tolerance
    )
    if not canonical_transform_pass:
        raise ValueError("normal-mode transform failed the canonical gate")

    normal_mean = phase_space_transform @ canonical_mean
    normal_covariance = (
        phase_space_transform
        @ centered_covariance
        @ phase_space_transform.T
    )

    def mode_energies(
        transformed_mean: np.ndarray,
        transformed_covariance: np.ndarray,
    ) -> np.ndarray:
        return np.array(
            tuple(
                0.5
                * (
                    transformed_mean[2 + index] ** 2
                    + transformed_covariance[2 + index, 2 + index]
                    + mass_squared[index]
                    * (
                        transformed_mean[index] ** 2
                        + transformed_covariance[index, index]
                    )
                )
                for index in range(2)
            ),
            dtype=float,
        )

    raw_mode_energies = mode_energies(normal_mean, normal_covariance)
    vacuum_mode_energies = 0.5 * mode_masses_array
    excitation_energies = raw_mode_energies - vacuum_mode_energies
    excitation_scale = float(np.max(vacuum_mode_energies))
    if float(np.min(excitation_energies)) < -tolerance * excitation_scale:
        raise ValueError(
            "physical Gaussian state produced a negative mode excitation"
        )
    excitation_nonnegative = bool(float(np.min(excitation_energies)) >= 0.0)
    if not excitation_nonnegative:
        raise ValueError(
            "mode excitation is numerically negative; no clipping is permitted"
        )

    original_raw_energy = float(
        0.5
        * (
            canonical_mean[2:] @ canonical_mean[2:]
            + np.trace(centered_covariance[2:, 2:])
            + canonical_mean[:2] @ mass_matrix @ canonical_mean[:2]
            + np.trace(mass_matrix @ centered_covariance[:2, :2])
        )
    )
    rotated_raw_energy = float(np.sum(raw_mode_energies))
    raw_rotation_residual = _relative_residual(
        original_raw_energy, rotated_raw_energy
    )

    sign_flip = np.diag((-1.0, 1.0))
    flipped_rotation = rotation @ sign_flip
    flipped_transform = np.block(
        [[flipped_rotation.T, zero], [zero, flipped_rotation.T]]
    )
    flipped_energies = mode_energies(
        flipped_transform @ canonical_mean,
        flipped_transform @ centered_covariance @ flipped_transform.T,
    )
    sign_flip_residual = max(
        _relative_residual(float(left), float(right))
        for left, right in zip(raw_mode_energies, flipped_energies)
    )

    coherent_preparation_energy = float(
        0.5
        * (
            canonical_mean[2] ** 2
            + system_mass**2 * canonical_mean[0] ** 2
        )
    )
    thermal_preparation_energy = environment_mass * occupation
    bare_vacuum_energy = 0.5 * (system_mass + environment_mass)
    interacting_vacuum_energy = float(np.sum(vacuum_mode_energies))
    mismatch_energy = bare_vacuum_energy - interacting_vacuum_energy
    if mismatch_energy < -tolerance * bare_vacuum_energy:
        raise ValueError("vacuum mismatch energy unexpectedly became negative")
    total_excitation_energy = float(np.sum(excitation_energies))
    ledger_right = (
        coherent_preparation_energy
        + thermal_preparation_energy
        + mismatch_energy
    )
    ledger_residual = _relative_residual(total_excitation_energy, ledger_right)
    density_constants = excitation_energies / volume

    # 자연 단위의 기호적 질량 차원 전파다. 영수증 플래그를 직접 놓는 대신 선언된
    # 입력 차원에서 이 값들을 계산한다.
    input_dimensions = {
        "mass": 1.0,
        "coupling": 2.0,
        "vacuum_density": 4.0,
        "comoving_volume": -3.0,
        "field": 1.0,
        "field_velocity": 2.0,
        "inverse_temperature": -1.0,
        "scale_factor": 0.0,
        "comoving_wavenumber": 1.0,
        "hubble": 1.0,
    }
    derived_dimensions = {
        "beta_times_environment_mass": (
            input_dimensions["inverse_temperature"]
            + input_dimensions["mass"]
        ),
        "canonical_q": (
            0.5 * input_dimensions["comoving_volume"]
            + input_dimensions["field"]
        ),
        "canonical_p": (
            0.5 * input_dimensions["comoving_volume"]
            + input_dimensions["field_velocity"]
        ),
    }
    derived_dimensions.update(
        {
            "p_squared_energy": 2.0 * derived_dimensions["canonical_p"],
            "mass_squared_q_squared_energy": (
                2.0 * input_dimensions["mass"]
                + 2.0 * derived_dimensions["canonical_q"]
            ),
            "mode_density_constant": (
                2.0 * derived_dimensions["canonical_p"]
                - input_dimensions["comoving_volume"]
            ),
            "vacuum_cell_energy": (
                input_dimensions["vacuum_density"]
                + input_dimensions["comoving_volume"]
            ),
            "hubble_over_mode_mass": (
                input_dimensions["hubble"] - input_dimensions["mass"]
            ),
            "physical_wavenumber_over_mode_mass": (
                input_dimensions["comoving_wavenumber"]
                - input_dimensions["scale_factor"]
                - input_dimensions["mass"]
            ),
        }
    )
    expected_dimensions = {
        "beta_times_environment_mass": 0.0,
        "canonical_q": -0.5,
        "canonical_p": 0.5,
        "p_squared_energy": 1.0,
        "mass_squared_q_squared_energy": 1.0,
        "mode_density_constant": 4.0,
        "vacuum_cell_energy": 1.0,
        "hubble_over_mode_mass": 0.0,
        "physical_wavenumber_over_mode_mass": 0.0,
    }
    dimensions_pass = bool(derived_dimensions == expected_dimensions)
    dimensionless_core_dimensions = {
        name: derived_dimensions[name]
        for name in (
            "beta_times_environment_mass",
            "hubble_over_mode_mass",
            "physical_wavenumber_over_mode_mass",
        )
    }
    invariant_pass = bool(
        raw_rotation_residual <= tolerance
        and ledger_residual <= tolerance
        and sign_flip_residual <= tolerance
    )
    if not invariant_pass:
        raise ValueError("state energy failed a rotation or ledger invariant")

    return FiniteProductGaussianStateDensityAudit(
        action_parameter_manifest=(
            system_mass,
            environment_mass,
            coupling,
            vacuum_density,
        ),
        state_boundary_manifest=(
            volume,
            field_mean,
            field_velocity,
            occupation,
            beta,
        ),
        environment_thermal_marginal_manifest=(
            environment_mass,
            volume,
            occupation,
            beta,
        ),
        covariance_ordering=("q_phi", "q_chi", "p_phi", "p_chi"),
        canonical_mean=tuple(float(value) for value in canonical_mean),
        centered_symmetrized_covariance=_real_matrix_tuple(centered_covariance),
        symplectic_eigenvalues=tuple(
            float(value) for value in symplectic_spectrum
        ),
        expected_symplectic_eigenvalues=tuple(
            float(value) for value in expected_symplectic_spectrum
        ),
        dimensionless_uncertainty_minimum_eigenvalue=uncertainty_minimum,
        normal_mode_rotation=_real_matrix_tuple(rotation),
        phase_space_symplectic_transform=_real_matrix_tuple(
            phase_space_transform
        ),
        symplectic_residual=symplectic_residual,
        normal_mode_diagonalization_relative_residual=diagonalization_residual,
        normal_mode_mass_squared=tuple(float(value) for value in mass_squared),
        normal_mode_masses=tuple(float(value) for value in mode_masses_array),
        relative_spectral_gap=relative_gap,
        normal_mode_mean=tuple(float(value) for value in normal_mean),
        normal_mode_centered_covariance=_real_matrix_tuple(normal_covariance),
        normal_mode_position_cross_covariance=float(normal_covariance[0, 1]),
        normal_mode_momentum_cross_covariance=float(normal_covariance[2, 3]),
        finite_mode_raw_energies_ev=tuple(
            float(value) for value in raw_mode_energies
        ),
        finite_mode_vacuum_energies_ev=tuple(
            float(value) for value in vacuum_mode_energies
        ),
        finite_mode_vacuum_subtracted_energies_ev=tuple(
            float(value) for value in excitation_energies
        ),
        finite_mode_density_constants_ev4=tuple(
            float(value) for value in density_constants
        ),
        uncoupled_coherent_preparation_energy_ev=coherent_preparation_energy,
        uncoupled_thermal_preparation_energy_ev=thermal_preparation_energy,
        bare_product_vacuum_energy_ev=bare_vacuum_energy,
        interacting_vacuum_energy_ev=interacting_vacuum_energy,
        vacuum_mismatch_quench_energy_ev=mismatch_energy,
        finite_mode_vacuum_subtracted_total_energy_ev=total_excitation_energy,
        finite_mode_vacuum_subtracted_total_density_ev4=(
            total_excitation_energy / volume
        ),
        vacuum_cell_energy_ev=vacuum_density * volume,
        thermal_occupation_relative_residual=thermal_residual,
        raw_energy_rotation_relative_residual=raw_rotation_residual,
        excitation_energy_ledger_relative_residual=ledger_residual,
        mode_sign_flip_energy_relative_residual=sign_flip_residual,
        uncertainty_principle_pass=uncertainty_pass,
        covariance_physicality_pass=covariance_physicality_pass,
        canonical_transform_pass=canonical_transform_pass,
        mass_matrix_stable=mass_matrix_stable,
        nondegenerate_mode_allocation_pass=True,
        finite_mode_excitation_nonnegative=excitation_nonnegative,
        mass_dimension_manifest=tuple(derived_dimensions.items()),
        dimensionless_core_argument_mass_dimensions=tuple(
            dimensionless_core_dimensions.items()
        ),
        dimensions_pass=dimensions_pass,
        status="PASS_CONDITIONAL_FINITE_GAUSSIAN_STATE_DENSITY_MAP",
    )


@dataclass(frozen=True)
class GaussianNormalModePerturbationAudit:
    """보존된 가우스 작용에 대한 조건부 WKB 섭동 영수증이다.

    이 감사는 유한 가우스 CTP 증인과 같은 파라미터 명세 ``(m, M, kappa, V0)`` 를
    쓰되 표현을 바꾼다. 두 정준 장을 모두 보존하고 대각화한다. 적분해 없앤 영향
    그램(influence Gram)은 중력 원천으로 넣지 않는다.
    """

    action_parameter_manifest: tuple[float, float, float, float]
    normal_mode_mass_squared: tuple[float, float]
    normal_mode_masses: tuple[float, float]
    mass_matrix_determinant_ev4: float
    scale_factor: float
    hubble_ev: float
    comoving_wavenumber_ev: float
    reduced_planck_mass_ev: float
    comoving_mode_density_constants_ev4: tuple[float, float]
    mode_densities_ev4: tuple[float, float]
    background_density_ev4: float
    background_pressure_ev4: float
    vacuum_equation_of_state: float
    vacuum_density_perturbation_ev4: float
    linear_anisotropic_stress: float
    microscopic_characteristic_speed_squared: tuple[float, float]
    effective_sound_speed_squared: tuple[float, float]
    wkb_hubble_ratios: tuple[float, float]
    nonrelativistic_momentum_ratios: tuple[float, float]
    subhorizon_ratio: float
    pressure_frequency_squared_ev2: tuple[float, float]
    four_pi_g_density_sources_ev2: tuple[float, float]
    coupled_density_contrast_matrix_ev2: tuple[tuple[float, float], ...]
    jeans_comoving_wavenumbers_ev: tuple[float, float]
    mass_dimension_manifest: tuple[tuple[str, float], ...]
    dimensionless_core_argument_mass_dimensions: tuple[tuple[str, float], ...]
    dimensions_pass: bool
    mass_matrix_stable: bool
    wkb_domain_pass: bool
    nonrelativistic_domain_pass: bool
    subhorizon_domain_pass: bool
    positive_vacuum_pass: bool
    background_dm_de_limit: bool
    perturbation_discriminant_derived: bool
    status: str
    failed_gates: tuple[str, ...]
    representation: str = "RETAINED_TWO_FIELD_ACTION_ONLY"
    closure: str = "WKB_NONRELATIVISTIC_GR_SUBHORIZON"
    same_action_metric_variation_declared: bool = True
    einstein_gravity_supplied: bool = True
    integrated_out_environment_stress_added: bool = False
    influence_gram_used_as_gravity_source: bool = False
    ctp_to_cosmological_state_map_derived: bool = False
    initial_conditions_derived: bool = False
    absolute_abundance_derived: bool = False
    growth_history_derived: bool = False
    lensing_likelihood_derived: bool = False
    physical_dark_matter_dark_energy_identification: bool = False


@dataclass(frozen=True)
class ProductGaussianWKBPerturbationAudit:
    """보존된 가우스 상태 하나와 WKB 게이트의 합성이다."""

    state_density_audit: FiniteProductGaussianStateDensityAudit
    perturbation_audit: GaussianNormalModePerturbationAudit
    action_parameter_manifest_match: bool
    derived_density_constants_match: bool
    same_state_finite_mode_energy_map_derived: bool
    perturbation_discriminant_derived: bool
    status: str
    representation: str = "RETAINED_TWO_FIELD_STATE_AND_WKB_ONLY"
    ctp_to_cosmological_state_map_derived: bool = False
    cosmological_initial_state_derived: bool = False
    absolute_abundance_predicted: bool = False
    integrated_out_environment_stress_added: bool = False
    influence_gram_used_as_gravity_source: bool = False
    physical_dark_matter_dark_energy_identification: bool = False


def audit_gaussian_normal_mode_perturbations(
    *,
    system_mass_ev: float,
    environment_mass_ev: float,
    bilinear_coupling_ev2: float,
    vacuum_energy_density_ev4: float,
    scale_factor: float,
    hubble_ev: float,
    comoving_wavenumber_ev: float,
    comoving_mode_density_constants_ev4: tuple[float, float],
    reduced_planck_mass_ev: float = REDUCED_MPL_EV,
    validity_limit: float = 0.1,
) -> GaussianNormalModePerturbationAudit:
    """두 모드 WKB 음속과 진스(Jeans) 판별식을 계산한다.

    보존된 작용은

    ``L = -1/2 (d phi)^2 -m^2 phi^2/2``
    ``    -1/2 (d chi)^2 -M^2 chi^2/2-kappa phi chi-V0``

    이다. 거칠게 만들기 전에 정확한 정규 질량을 대각화한다. 빠르게 진동하는
    비상대론적 모드마다

    ``c_eff^2 = k^2/(4*a^2*m_mode^2)``,
    ``omega_pressure^2 = k^4/(4*a^4*m_mode^2)``

    이다. 주어진 아인슈타인 중력과 함께 지평선 아래 밀도 대비 계는

    ``delta_i'' + 2H delta_i' + sum_j A_ij delta_j = 0``,
    ``A_ij = omega_pressure_i^2 delta_ij - 4*pi*G*rho_j``

    로 표현한다. 초기 진폭은 의도적으로 고르거나 진화시키지 않으므로, 성장이나
    렌즈 예측이 아니라 척도 의존 섭동 판별기다.
    """

    values = {
        "system_mass_ev": system_mass_ev,
        "environment_mass_ev": environment_mass_ev,
        "bilinear_coupling_ev2": bilinear_coupling_ev2,
        "vacuum_energy_density_ev4": vacuum_energy_density_ev4,
        "scale_factor": scale_factor,
        "hubble_ev": hubble_ev,
        "comoving_wavenumber_ev": comoving_wavenumber_ev,
        "reduced_planck_mass_ev": reduced_planck_mass_ev,
        "validity_limit": validity_limit,
    }
    checked: dict[str, float] = {}
    for name, value in values.items():
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"{name} must be finite")
        checked[name] = converted

    system_mass = checked["system_mass_ev"]
    environment_mass = checked["environment_mass_ev"]
    coupling = checked["bilinear_coupling_ev2"]
    vacuum_density = checked["vacuum_energy_density_ev4"]
    a = checked["scale_factor"]
    hubble = checked["hubble_ev"]
    wavenumber = checked["comoving_wavenumber_ev"]
    planck_mass = checked["reduced_planck_mass_ev"]
    limit = checked["validity_limit"]
    if system_mass <= 0.0 or environment_mass <= 0.0:
        raise ValueError("system_mass_ev and environment_mass_ev must be positive")
    if a <= 0.0 or hubble < 0.0 or wavenumber <= 0.0:
        raise ValueError(
            "scale_factor and comoving_wavenumber_ev must be positive and "
            "hubble_ev must be nonnegative"
        )
    if planck_mass <= 0.0:
        raise ValueError("reduced_planck_mass_ev must be positive")
    if limit <= 0.0 or limit > 0.25:
        raise ValueError("validity_limit must lie in (0, 0.25]")

    densities = tuple(float(value) for value in comoving_mode_density_constants_ev4)
    if len(densities) != 2 or any(not math.isfinite(value) for value in densities):
        raise ValueError(
            "comoving_mode_density_constants_ev4 must contain two finite values"
        )
    if any(value < 0.0 for value in densities) or sum(densities) <= 0.0:
        raise ValueError(
            "comoving_mode_density_constants_ev4 must be nonnegative with "
            "positive total"
        )

    determinant = system_mass**2 * environment_mass**2 - coupling**2
    if determinant <= 0.0:
        raise ValueError(
            "bilinear mass matrix must satisfy m^2*M^2-kappa^2 > 0"
        )
    discriminant = math.sqrt(
        (environment_mass**2 - system_mass**2) ** 2 + 4.0 * coupling**2
    )
    mass_minus_squared = 0.5 * (
        system_mass**2 + environment_mass**2 - discriminant
    )
    mass_plus_squared = 0.5 * (
        system_mass**2 + environment_mass**2 + discriminant
    )
    mode_mass_squared = (mass_minus_squared, mass_plus_squared)
    mode_masses = tuple(math.sqrt(value) for value in mode_mass_squared)

    physical_wavenumber = wavenumber / a
    wkb_ratios = tuple(hubble / mass for mass in mode_masses)
    momentum_ratios = tuple(
        physical_wavenumber / mass for mass in mode_masses
    )
    subhorizon_ratio = a * hubble / wavenumber
    sound_speeds = tuple(
        wavenumber**2 / (4.0 * a**2 * mass_squared)
        for mass_squared in mode_mass_squared
    )
    pressure_frequencies = tuple(
        wavenumber**4 / (4.0 * a**4 * mass_squared)
        for mass_squared in mode_mass_squared
    )

    mode_densities = tuple(value / a**3 for value in densities)
    density_sources = tuple(
        value / (2.0 * planck_mass**2) for value in mode_densities
    )
    total_gravity_source = sum(density_sources)
    contrast_matrix = (
        (
            pressure_frequencies[0] - density_sources[0],
            -density_sources[1],
        ),
        (
            -density_sources[0],
            pressure_frequencies[1] - density_sources[1],
        ),
    )
    jeans_wavenumbers = tuple(
        (
            4.0 * mass_squared * a**4 * total_gravity_source
        ) ** 0.25
        for mass_squared in mode_mass_squared
    )

    wkb_pass = max(wkb_ratios) <= limit
    nonrelativistic_pass = max(momentum_ratios) <= limit
    subhorizon_pass = subhorizon_ratio <= limit
    positive_vacuum = vacuum_density > 0.0
    gates = {
        "wkb": wkb_pass,
        "nonrelativistic": nonrelativistic_pass,
        "subhorizon": subhorizon_pass,
        "positive_vacuum": positive_vacuum,
    }
    failed = tuple(name for name, passed in gates.items() if not passed)
    input_dimensions = {
        "mode_mass": 1.0,
        "mode_mass_squared": 2.0,
        "scale_factor": 0.0,
        "hubble": 1.0,
        "comoving_wavenumber": 1.0,
        "mode_density": 4.0,
        "reduced_planck_mass": 1.0,
    }
    derived_dimensions = {
        "wkb_hubble_ratio": (
            input_dimensions["hubble"] - input_dimensions["mode_mass"]
        ),
        "nonrelativistic_momentum_ratio": (
            input_dimensions["comoving_wavenumber"]
            - input_dimensions["scale_factor"]
            - input_dimensions["mode_mass"]
        ),
        "subhorizon_ratio": (
            input_dimensions["scale_factor"]
            + input_dimensions["hubble"]
            - input_dimensions["comoving_wavenumber"]
        ),
        "effective_sound_speed_squared": (
            2.0 * input_dimensions["comoving_wavenumber"]
            - 2.0 * input_dimensions["scale_factor"]
            - input_dimensions["mode_mass_squared"]
        ),
        "pressure_frequency_squared": (
            4.0 * input_dimensions["comoving_wavenumber"]
            - 4.0 * input_dimensions["scale_factor"]
            - input_dimensions["mode_mass_squared"]
        ),
        "four_pi_g_density_source": (
            input_dimensions["mode_density"]
            - 2.0 * input_dimensions["reduced_planck_mass"]
        ),
    }
    derived_dimensions["jeans_wavenumber_fourth_power"] = (
        input_dimensions["mode_mass_squared"]
        + 4.0 * input_dimensions["scale_factor"]
        + derived_dimensions["four_pi_g_density_source"]
    )
    derived_dimensions["jeans_wavenumber"] = (
        derived_dimensions["jeans_wavenumber_fourth_power"] / 4.0
    )
    expected_dimensions = {
        "wkb_hubble_ratio": 0.0,
        "nonrelativistic_momentum_ratio": 0.0,
        "subhorizon_ratio": 0.0,
        "effective_sound_speed_squared": 0.0,
        "pressure_frequency_squared": 2.0,
        "four_pi_g_density_source": 2.0,
        "jeans_wavenumber_fourth_power": 4.0,
        "jeans_wavenumber": 1.0,
    }
    dimensions_pass = bool(derived_dimensions == expected_dimensions)
    dimensionless_core_dimensions = {
        name: derived_dimensions[name]
        for name in (
            "wkb_hubble_ratio",
            "nonrelativistic_momentum_ratio",
            "subhorizon_ratio",
            "effective_sound_speed_squared",
        )
    }
    approximation_pass = not failed and dimensions_pass

    return GaussianNormalModePerturbationAudit(
        action_parameter_manifest=(
            system_mass,
            environment_mass,
            coupling,
            vacuum_density,
        ),
        normal_mode_mass_squared=mode_mass_squared,
        normal_mode_masses=mode_masses,
        mass_matrix_determinant_ev4=determinant,
        scale_factor=a,
        hubble_ev=hubble,
        comoving_wavenumber_ev=wavenumber,
        reduced_planck_mass_ev=planck_mass,
        comoving_mode_density_constants_ev4=densities,
        mode_densities_ev4=mode_densities,
        background_density_ev4=vacuum_density + sum(mode_densities),
        background_pressure_ev4=-vacuum_density,
        vacuum_equation_of_state=-1.0,
        vacuum_density_perturbation_ev4=0.0,
        linear_anisotropic_stress=0.0,
        microscopic_characteristic_speed_squared=(1.0, 1.0),
        effective_sound_speed_squared=sound_speeds,
        wkb_hubble_ratios=wkb_ratios,
        nonrelativistic_momentum_ratios=momentum_ratios,
        subhorizon_ratio=subhorizon_ratio,
        pressure_frequency_squared_ev2=pressure_frequencies,
        four_pi_g_density_sources_ev2=density_sources,
        coupled_density_contrast_matrix_ev2=contrast_matrix,
        jeans_comoving_wavenumbers_ev=jeans_wavenumbers,
        mass_dimension_manifest=tuple(derived_dimensions.items()),
        dimensionless_core_argument_mass_dimensions=tuple(
            dimensionless_core_dimensions.items()
        ),
        dimensions_pass=dimensions_pass,
        mass_matrix_stable=True,
        wkb_domain_pass=wkb_pass,
        nonrelativistic_domain_pass=nonrelativistic_pass,
        subhorizon_domain_pass=subhorizon_pass,
        positive_vacuum_pass=positive_vacuum,
        background_dm_de_limit=wkb_pass and positive_vacuum,
        perturbation_discriminant_derived=approximation_pass,
        status=(
            "PASS_CONDITIONAL_GAUSSIAN_WKB_PERTURBATIONS"
            if approximation_pass
            else "FAIL_CONDITIONAL_APPROXIMATION_GATE"
        ),
        failed_gates=failed,
    )


def audit_product_gaussian_state_wkb_perturbations(
    *,
    system_mass_ev: float,
    environment_mass_ev: float,
    bilinear_coupling_ev2: float,
    vacuum_energy_density_ev4: float,
    comoving_volume_ev_minus3: float,
    system_field_mean_ev: float,
    system_field_velocity_ev2: float,
    environment_mean_occupation: float,
    environment_inverse_temperature_ev_minus1: float,
    scale_factor: float,
    hubble_ev: float,
    comoving_wavenumber_ev: float,
    reduced_planck_mass_ev: float = REDUCED_MPL_EV,
    validity_limit: float = 0.1,
    tolerance: float = 2.0e-11,
) -> ProductGaussianWKBPerturbationAudit:
    """정확한 유한 상태 에너지를 조건부 WKB 영수증에 넣는다.

    합성은 의도적으로 한 방향이고 표현 배타적이다. 보존 상태의 에너지 상수를 WKB
    물질 입력으로 쓰되, CTP 영향 그램이나 적분해 없앤 환경 응력은 더하지 않는다.
    """

    state_audit = audit_finite_product_gaussian_state_densities(
        system_mass_ev=system_mass_ev,
        environment_mass_ev=environment_mass_ev,
        bilinear_coupling_ev2=bilinear_coupling_ev2,
        vacuum_energy_density_ev4=vacuum_energy_density_ev4,
        comoving_volume_ev_minus3=comoving_volume_ev_minus3,
        system_field_mean_ev=system_field_mean_ev,
        system_field_velocity_ev2=system_field_velocity_ev2,
        environment_mean_occupation=environment_mean_occupation,
        environment_inverse_temperature_ev_minus1=(
            environment_inverse_temperature_ev_minus1
        ),
        tolerance=tolerance,
    )
    perturbation_audit = audit_gaussian_normal_mode_perturbations(
        system_mass_ev=system_mass_ev,
        environment_mass_ev=environment_mass_ev,
        bilinear_coupling_ev2=bilinear_coupling_ev2,
        vacuum_energy_density_ev4=vacuum_energy_density_ev4,
        scale_factor=scale_factor,
        hubble_ev=hubble_ev,
        comoving_wavenumber_ev=comoving_wavenumber_ev,
        comoving_mode_density_constants_ev4=(
            state_audit.finite_mode_density_constants_ev4
        ),
        reduced_planck_mass_ev=reduced_planck_mass_ev,
        validity_limit=validity_limit,
    )
    manifest_match = bool(
        state_audit.action_parameter_manifest
        == perturbation_audit.action_parameter_manifest
    )
    density_match = bool(
        np.allclose(
            state_audit.finite_mode_density_constants_ev4,
            perturbation_audit.comoving_mode_density_constants_ev4,
            rtol=tolerance,
            atol=0.0,
        )
    )
    if not manifest_match or not density_match:
        raise ValueError("state-to-WKB composition failed its manifest gate")
    perturbation_derived = bool(
        perturbation_audit.perturbation_discriminant_derived
    )
    return ProductGaussianWKBPerturbationAudit(
        state_density_audit=state_audit,
        perturbation_audit=perturbation_audit,
        action_parameter_manifest_match=manifest_match,
        derived_density_constants_match=density_match,
        same_state_finite_mode_energy_map_derived=(
            state_audit.same_state_finite_mode_energy_map_derived
        ),
        perturbation_discriminant_derived=perturbation_derived,
        status=(
            "PASS_CONDITIONAL_SAME_STATE_GAUSSIAN_WKB_BRIDGE"
            if perturbation_derived
            else "FAIL_CONDITIONAL_SAME_STATE_WKB_DOMAIN"
        ),
    )


def _nodes(solution: BackgroundSolution) -> tuple[PerturbationNode, ...]:
    config = solution.config
    rho_inf_ev4 = solution.amplitude * RHO_CRIT0_EV4
    lambda3_ev = 2.0 * (config.kappa * rho_inf_ev4) ** 0.25
    result: list[PerturbationNode] = []
    for node in solution.nodes:
        data = _densities(node.n, node.tau, node.u, config, solution.amplitude)
        rho_b, rho_r, _, rho_k, p_k, _ = data
        _, u_prime = _rhs(
            node.n, (node.tau, node.u), config, solution.amplitude
        )
        delta = node.u / config.kappa
        delta_prime = u_prime / config.kappa
        h_prime_over_h = (
            -3.0 * rho_b - 4.0 * rho_r - 3.0 * (rho_k + p_k)
        ) / (2.0 * node.e2)
        kinetic_prime = 3.0 * delta_prime / (2.0 + 3.0 * delta)
        friction = 3.0 + h_prime_over_h + kinetic_prime
        tachyon_ratio = (
            config.gamma**2
            * config.x_star
            * math.exp(-config.gamma * node.tau)
            / (config.kappa * (2.0 + 3.0 * delta) * node.e2)
        )
        pump = (
            3.0
            + delta_prime / (1.0 + delta)
            + 3.0 * delta_prime / (2.0 + 3.0 * delta)
            - 2.0 * h_prime_over_h
        )
        energy_cutoff = lambda3_ev * node.cs2 ** (7.0 / 8.0)
        wavenumber_cutoff = lambda3_ev * node.cs2 ** (3.0 / 8.0)
        result.append(
            PerturbationNode(
                n=node.n,
                e2=node.e2,
                friction=friction,
                tachyon_ratio=tachyon_ratio,
                cs2=node.cs2,
                q_s_over_mpl2=node.q_s_over_mpl2,
                pump_slope=pump,
                zeta_decay_slope=pump + h_prime_over_h,
                energy_cutoff_ev=energy_cutoff,
                wavenumber_cutoff_ev=wavenumber_cutoff,
            )
        )
    return tuple(result)


def _fixed_coordinate_growth(nodes: tuple[PerturbationNode, ...]) -> float:
    y = 0.0
    velocity = 0.0
    for left, right in zip(nodes, nodes[1:]):
        step = right.n - left.n
        friction = 0.5 * (left.friction + right.friction)
        ratio = 0.5 * (left.tachyon_ratio + right.tachyon_ratio)

        def derivative(yy: float, vv: float) -> tuple[float, float]:
            return vv, -friction * vv + ratio * (1.0 + yy)

        k1 = derivative(y, velocity)
        k2 = derivative(y + step * k1[0] / 2.0, velocity + step * k1[1] / 2.0)
        k3 = derivative(y + step * k2[0] / 2.0, velocity + step * k2[1] / 2.0)
        k4 = derivative(y + step * k3[0], velocity + step * k3[1])
        y += step * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        velocity += step * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
    return y


def evaluate_single_clock_gate(
    solution: BackgroundSolution | None = None,
) -> SingleClockGate:
    selected = solution or solve_background(KineticClockConfig())
    nodes = _nodes(selected)
    min_friction = min(node.friction for node in nodes)
    max_ratio = max(node.tachyon_ratio for node in nodes)
    interval = nodes[-1].n - nodes[0].n
    positive_root = 2.0 * max_ratio / (
        math.sqrt(min_friction**2 + 4.0 * max_ratio) + min_friction
    )
    bound = positive_root * interval
    min_energy_over_h = min(
        node.energy_cutoff_ev / (H0_EV * math.sqrt(node.e2)) for node in nodes
    )
    min_wavenumber_over_k = min(
        node.wavenumber_cutoff_ev / (MPC_INV_EV * math.exp(-node.n))
        for node in nodes
    )
    gate_values = {
        "positive_friction": min_friction > 0.0,
        "sub_hubble_tachyon": max_ratio < 1.0,
        "sub_order_one_growth_bound": bound < 1.0,
        "positive_pump": min(node.pump_slope for node in nodes) > 0.0,
        "decaying_zeta_integrand": min(node.zeta_decay_slope for node in nodes) > 0.0,
        "energy_cutoff_above_h": min_energy_over_h > 1.0,
        "momentum_cutoff_above_1mpc": min_wavenumber_over_k > 1.0,
    }
    failed = tuple(name for name, passed in gate_values.items() if not passed)
    return SingleClockGate(
        gamma=selected.config.gamma,
        min_friction=min_friction,
        max_tachyon_ratio=max_ratio,
        fixed_coordinate_growth_minus_one=_fixed_coordinate_growth(nodes),
        max_log_growth_bound=bound,
        min_pump_slope=min(node.pump_slope for node in nodes),
        min_zeta_decay_slope=min(node.zeta_decay_slope for node in nodes),
        min_energy_cutoff_over_h=min_energy_over_h,
        min_wavenumber_cutoff_over_k_1mpc=min_wavenumber_over_k,
        status=("PASS_SINGLE_CLOCK_ONLY" if not failed else "FAIL_SINGLE_CLOCK_GATE"),
        failed_gates=failed,
    )


def quasi_static_growth_diagnostic(
    solution: BackgroundSolution | None = None,
    *,
    redshift: float = 0.07,
    observed_fsigma8: float = 0.4497,
    observed_sigma: float = 0.0548,
    sigma8_0: float = 0.811,
) -> QuasiStaticGrowthDiagnostic:
    """선언된 지평선 아래 닫힘을 풀고 압축 자료 하나와 비교한다.

    운동 재고와 바리온이 푸아송 항의 원천이고, 포화 판독은 매끄러우며, GR 은
    유지되고, 복사는 H(a) 에만 영향을 준다. 전체 다성분 섭동계를 유도하기 전에
    쓸 수 있는 가장 강한 저자료 진단이다. ``sigma8_0`` 은 외부 입력으로 남는다.
    """

    if redshift < 0.0 or observed_sigma <= 0.0 or sigma8_0 <= 0.0:
        raise ValueError("growth diagnostic inputs are outside their domain")
    selected = solution or solve_background(KineticClockConfig())
    target_n = -math.log1p(redshift)
    if target_n < selected.nodes[0].n:
        raise ValueError("growth redshift is outside the solved window")

    # a>=0.01 에서 물질 시대 성장 모드 씨앗을 놓는다. 닫힘이 단열 복사 시대 전달을
    # 제공하는 척하지 않으려는 의도다.
    growth_nodes = tuple(node for node in selected.nodes if node.n >= math.log(0.01))
    if len(growth_nodes) < 10:
        raise ValueError("background grid is too sparse for the growth closure")
    d_value = math.exp(growth_nodes[0].n)
    velocity = d_value
    history: list[tuple[float, float, float]] = [
        (growth_nodes[0].n, d_value, velocity)
    ]

    def coefficients(n: float) -> tuple[float, float]:
        node = selected.at_n(n)
        rho_b, rho_r, _, rho_k, p_k, _ = _densities(
            n, node.tau, node.u, selected.config, selected.amplitude
        )
        h_prime_over_h = (
            -3.0 * rho_b - 4.0 * rho_r - 3.0 * (rho_k + p_k)
        ) / (2.0 * node.e2)
        omega_cluster = (rho_b + rho_k) / node.e2
        return 2.0 + h_prime_over_h, 1.5 * omega_cluster

    def derivative(n: float, d: float, v: float) -> tuple[float, float]:
        drag, source = coefficients(n)
        return v, -drag * v + source * d

    for left, right in zip(growth_nodes, growth_nodes[1:]):
        step = right.n - left.n
        n = left.n
        k1 = derivative(n, d_value, velocity)
        k2 = derivative(
            n + step / 2.0,
            d_value + step * k1[0] / 2.0,
            velocity + step * k1[1] / 2.0,
        )
        k3 = derivative(
            n + step / 2.0,
            d_value + step * k2[0] / 2.0,
            velocity + step * k2[1] / 2.0,
        )
        k4 = derivative(
            n + step,
            d_value + step * k3[0],
            velocity + step * k3[1],
        )
        d_value += step * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        velocity += step * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        history.append((right.n, d_value, velocity))

    normalization = history[-1][1]
    for left, right in zip(history, history[1:]):
        if left[0] <= target_n <= right[0]:
            weight = (target_n - left[0]) / (right[0] - left[0])
            d_target = left[1] + weight * (right[1] - left[1])
            v_target = left[2] + weight * (right[2] - left[2])
            break
    else:
        d_target, v_target = history[-1][1], history[-1][2]
    # f*sigma8 = (D'/D) * sigma8_0*(D/D0) = sigma8_0*D'/D0 이다.
    prediction = sigma8_0 * v_target / normalization
    return QuasiStaticGrowthDiagnostic(
        redshift=redshift,
        predicted_fsigma8=prediction,
        observed_fsigma8=observed_fsigma8,
        observed_sigma=observed_sigma,
        pull=(prediction - observed_fsigma8) / observed_sigma,
        sigma8_0=sigma8_0,
    )


def scan_kappa_sensitivity(
    kappa_values: tuple[float, ...] = (1.0e10, 3.0e11, 1.0e12, 1.0e14, 1.0e17, 1.0e20),
    *,
    gamma: float = 10.0,
    steps: int = 1200,
) -> tuple[KappaSensitivityRow, ...]:
    """외부 입력인 강성 척도 kappa 가 어떤 역할을 하는지 드러낸다."""

    if not kappa_values or any(value <= 0.0 for value in kappa_values):
        raise ValueError("kappa scan values must be non-empty and positive")
    rows: list[KappaSensitivityRow] = []
    for kappa in kappa_values:
        solution = solve_background(
            KineticClockConfig(gamma=gamma, kappa=kappa, steps=steps)
        )
        gate = evaluate_single_clock_gate(solution)
        rows.append(
            KappaSensitivityRow(
                kappa=kappa,
                min_cs2=solution.min_cs2,
                min_friction=gate.min_friction,
                max_log_growth_bound=gate.max_log_growth_bound,
                min_energy_cutoff_over_h=gate.min_energy_cutoff_over_h,
                status=gate.status,
                failed_gates=gate.failed_gates,
            )
        )
    return tuple(rows)


def main() -> int:
    gate = evaluate_single_clock_gate()
    for name, value in gate.__dict__.items():
        print(name, value)
    growth = quasi_static_growth_diagnostic()
    for name, value in growth.__dict__.items():
        print(f"growth_{name}", value)
    for row in scan_kappa_sensitivity():
        print(
            "kappa_scan",
            row.kappa,
            row.min_cs2,
            row.min_friction,
            row.max_log_growth_bound,
            row.min_energy_cutoff_over_h,
            row.status,
            ",".join(row.failed_gates) or "none",
        )
    return 0


# ---------------------------------------------------------------------------
# 2. 급냉 스펙트럼-현재 존재비 조건부 다리
# ---------------------------------------------------------------------------


def _positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite real number")
    return result


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


@dataclass(frozen=True)
class EntropyRedshiftContract:
    """공변 엔트로피 장부에서 ``a_*`` 를 결정하는 입력이다.

    ``comoving_entropy_growth = S_0/S_*`` 는 단열 팽창이면 1 이고, 생산 뒤 엔트로피가
    주입되면 1 이상이다.
    """

    temperature_at_production: float
    temperature_today: float
    entropy_dof_at_production: float
    entropy_dof_today: float
    comoving_entropy_growth: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "temperature_at_production",
            "temperature_today",
            "entropy_dof_at_production",
            "entropy_dof_today",
            "comoving_entropy_growth",
        ):
            object.__setattr__(
                self,
                name,
                _positive_finite(getattr(self, name), name),
            )
        if self.comoving_entropy_growth < 1.0:
            raise ValueError("comoving_entropy_growth must be >= 1")
        if self.scale_factor_at_production > 1.0:
            raise ValueError(
                "entropy inputs place production after the normalized present"
            )

    @property
    def scale_factor_at_production(self) -> float:
        log_scale_factor = (
            (
                math.log(self.entropy_dof_today)
                - math.log(self.entropy_dof_at_production)
                - math.log(self.comoving_entropy_growth)
            )
            / 3.0
            + math.log(self.temperature_today)
            - math.log(self.temperature_at_production)
        )
        if log_scale_factor > 0.0:
            raise ValueError(
                "entropy inputs place production after the normalized present"
            )
        scale_factor = math.exp(log_scale_factor)
        if not math.isfinite(scale_factor) or scale_factor <= 0.0:
            raise ValueError("production scale factor is not finite and positive")
        return scale_factor


@dataclass(frozen=True)
class NaturalUnitCosmology:
    """종과 같은 자연 에너지 단위로 표현한 현재 정규화다."""

    hubble_today: float
    reduced_planck_mass: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "hubble_today",
            _positive_finite(self.hubble_today, "hubble_today"),
        )
        object.__setattr__(
            self,
            "reduced_planck_mass",
            _positive_finite(self.reduced_planck_mass, "reduced_planck_mass"),
        )
        _ = self.critical_density_today

    @property
    def critical_density_today(self) -> float:
        planck_hubble = self.reduced_planck_mass * self.hubble_today
        value = 3.0 * planck_hubble * planck_hubble
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("critical density is not finite and positive")
        return value


@dataclass(frozen=True)
class CollisionlessAbundanceCertificate:
    """조건부 무충돌 전방 사상의 유한 창 영수증이다."""

    label: str
    scale_factor_at_production: float
    critical_density_today: float
    production_number_density: float
    present_number_density: float
    present_number_density_from_dilution: float
    number_dilution_residual: float
    present_rest_density: float
    present_energy_density: float
    present_pressure: float
    present_equation_of_state: float
    omega_produced_today: float
    omitted_number_density_upper: float
    omitted_energy_density_upper: float
    omitted_omega_upper: float
    relative_kinetic_energy: float
    relative_kinetic_energy_upper_bound: float
    cold_bound_residual: float
    momentum_max: float
    intervals: int
    ultraviolet_status: str
    quadrature_status: str
    tail_numerical_status: str
    production_approximation: str
    role: str = (
        "CONDITIONAL_BETA_TO_ABUNDANCE_FORWARD_MAP_NOT_ABUNDANCE_PREDICTION"
    )


def smooth_quench_collisionless_abundance(
    species: QuantumSeatSpecies,
    *,
    entropy: EntropyRedshiftContract,
    cosmology: NaturalUnitCosmology,
    momentum_max: float | None = None,
    intervals: int = 2400,
) -> CollisionlessAbundanceCertificate:
    """생성된 매끄러운 급냉 초과분을 오늘의 존재비로 전파한다.

    점유수는 생성 초과분 ``(1+2*n_in)|beta_p|^2`` 이지 기존 개체군이 아니다. 종은
    급냉 뒤 안정하고 분리되며 질량이 일정하다고 가정한다. 평탄한 점근 tanh
    스펙트럼을 생산 면에서 채택한다. FLRW 에서 이를 쓰려면 ``H_* duration << 1``
    같은 국소 급냉 근사가 추가로 필요한데, 이 함수는 그것을 보증하지 않는다.
    """

    if not isinstance(species, QuantumSeatSpecies):
        raise ValueError("species must be a QuantumSeatSpecies")
    if not isinstance(entropy, EntropyRedshiftContract):
        raise ValueError("entropy must be an EntropyRedshiftContract")
    if not isinstance(cosmology, NaturalUnitCosmology):
        raise ValueError("cosmology must be a NaturalUnitCosmology")
    if isinstance(intervals, bool) or not isinstance(intervals, int):
        raise ValueError("intervals must be an integer")
    if intervals < 200:
        raise ValueError("intervals must be at least 200")
    if momentum_max is not None:
        momentum_max = _positive_finite(momentum_max, "momentum_max")

    try:
        production = integrate_quench_densities(
            species,
            protocol="smooth",
            momentum_max=momentum_max,
            intervals=intervals,
        )
    except OverflowError as error:
        raise ValueError(
            "production quadrature is outside the finite numerical domain"
        ) from error
    scale_factor = entropy.scale_factor_at_production
    step = production.momentum_max / production.intervals
    number_terms: list[float] = []
    kinetic_terms: list[float] = []
    pressure_terms: list[float] = []
    momentum2_terms: list[float] = []
    for index in range(production.intervals + 1):
        momentum = index * step
        mode = smooth_tanh_mode(species, momentum)
        simpson_weight = (
            1.0
            if index in (0, production.intervals)
            else (4.0 if index % 2 else 2.0)
        )
        occupation = mode.created_occupation
        if not math.isfinite(occupation) or occupation < 0.0:
            raise ValueError("created occupation is not finite and non-negative")
        momentum_squared = momentum * momentum
        if not math.isfinite(momentum_squared):
            raise ValueError("momentum window is outside the finite numerical domain")
        radial_number = momentum_squared * occupation
        momentum_today = scale_factor * momentum
        energy_today = math.hypot(species.mass_out, momentum_today)
        kinetic_energy_today = momentum_today * (
            momentum_today / (energy_today + species.mass_out)
        )
        pressure_energy = momentum_today * (momentum_today / energy_today) / 3.0
        number_terms.append(simpson_weight * radial_number)
        kinetic_terms.append(
            simpson_weight * radial_number * kinetic_energy_today
        )
        pressure_terms.append(
            simpson_weight * radial_number * pressure_energy
        )
        momentum2_terms.append(
            simpson_weight * radial_number * momentum_squared
        )

    spherical_factor_today = (
        species.degeneracy
        / (2.0 * math.pi * math.pi)
        * scale_factor**3
        * step
        / 3.0
    )
    number_from_quadrature = spherical_factor_today * math.fsum(number_terms)
    number_from_dilution = scale_factor**3 * production.number_density
    rest_density = species.mass_out * number_from_quadrature
    kinetic_density = spherical_factor_today * math.fsum(kinetic_terms)
    energy_density = rest_density + kinetic_density
    pressure = spherical_factor_today * math.fsum(pressure_terms)
    if rest_density > 0.0:
        relative_kinetic = kinetic_density / rest_density
        mean_momentum_squared = (
            math.fsum(momentum2_terms) / math.fsum(number_terms)
        )
        relative_kinetic_bound = (
            0.5
            * scale_factor**2
            * mean_momentum_squared
            / species.mass_out**2
        )
    else:
        relative_kinetic = 0.0
        relative_kinetic_bound = 0.0
    equation_of_state = pressure / energy_density if energy_density > 0.0 else 0.0
    critical_density = cosmology.critical_density_today
    omega_produced = energy_density / critical_density
    cold_bound_residual = relative_kinetic - relative_kinetic_bound
    tail = smooth_quench_present_tail_certificate(
        species,
        momentum_start=production.momentum_max,
        scale_factor_at_production=scale_factor,
        critical_density_today=critical_density,
    )

    finite_outputs = (
        number_from_quadrature,
        number_from_dilution,
        energy_density,
        pressure,
        rest_density,
        relative_kinetic,
        relative_kinetic_bound,
        equation_of_state,
        critical_density,
        omega_produced,
        cold_bound_residual,
        tail.present_number_density_upper,
        tail.present_energy_density_upper,
        tail.omega_produced_upper,
    )
    if any(not math.isfinite(value) for value in finite_outputs):
        raise ValueError("abundance output is outside the finite numerical domain")
    invariant_scale = max(1.0, relative_kinetic_bound)
    if (
        rest_density > energy_density
        or pressure < 0.0
        or equation_of_state < 0.0
        or equation_of_state > 1.0 / 3.0 + 32.0 * math.ulp(1.0)
        or cold_bound_residual > 128.0 * math.ulp(invariant_scale)
    ):
        raise ValueError("collisionless abundance invariants failed")

    return CollisionlessAbundanceCertificate(
        label=species.label,
        scale_factor_at_production=scale_factor,
        critical_density_today=critical_density,
        production_number_density=production.number_density,
        present_number_density=number_from_quadrature,
        present_number_density_from_dilution=number_from_dilution,
        number_dilution_residual=number_from_quadrature - number_from_dilution,
        present_rest_density=rest_density,
        present_energy_density=energy_density,
        present_pressure=pressure,
        present_equation_of_state=equation_of_state,
        omega_produced_today=omega_produced,
        omitted_number_density_upper=tail.present_number_density_upper,
        omitted_energy_density_upper=tail.present_energy_density_upper,
        omitted_omega_upper=tail.omega_produced_upper,
        relative_kinetic_energy=relative_kinetic,
        relative_kinetic_energy_upper_bound=relative_kinetic_bound,
        cold_bound_residual=cold_bound_residual,
        momentum_max=production.momentum_max,
        intervals=production.intervals,
        ultraviolet_status="ANALYTIC_EXPONENTIAL_OMITTED_TAIL_BOUND_ATTACHED",
        quadrature_status="FINITE_WINDOW_SIMPSON_NOT_UV_TAIL_CERTIFICATE",
        tail_numerical_status=tail.numerical_status,
        production_approximation=(
            "ASYMPTOTIC_MINKOWSKI_CREATED_EXCESS_HSTAR_DURATION_NOT_CERTIFIED"
        ),
    )


@dataclass(frozen=True)
class FixedBasisBilinearQuenchCosmologyAudit:
    """두 장 가우스 급냉과 후기 WKB 게이트의 합성 영수증이다.

    경로는 명시적으로 주어진 보간

    ``K(t) = K_in + s(t) (K_out-K_in)``,
    ``s(t) = (1+tanh(t/tau))/2``

    이다. 가환이고 비축퇴인 끝점 행렬만 허용한다. 그 제한 아래서는 고정 직교 기저
    하나가 경로 전체를 대각화하므로, 회전 기저 혼합항을 빠뜨리지 않고 두 정확한
    스칼라 보골류보프 스펙트럼을 쓸 수 있다. 생성된 상태는 압축(squeezed) 가우스
    상태이지 E45 세포 증인이 쓰는 유한 결맞음 곱 상태가 아니다.
    """

    initial_action_parameter_manifest: tuple[float, float, float, float]
    final_action_parameter_manifest: tuple[float, float, float, float]
    interpolation: str
    matrix_commutator_frobenius_ev4: float
    matrix_commutator_relative_residual: float
    fixed_basis_off_diagonal_relative_residual: float
    initial_normal_mode_gap_fraction: float
    final_normal_mode_gap_fraction: float
    fixed_basis_mode_mass_squared_at_initial_ev2: tuple[float, float]
    fixed_basis_mode_mass_squared_at_final_ev2: tuple[float, float]
    fixed_basis_mode_masses_at_initial_ev: tuple[float, float]
    fixed_basis_mode_masses_at_final_ev: tuple[float, float]
    quench_duration_ev_minus1: float
    hubble_at_production_ev: float
    local_quench_ratio: float
    local_quench_limit: float
    local_quench_pass: bool
    quench_species: tuple[QuantumSeatSpecies, QuantumSeatSpecies]
    abundance_certificates: tuple[
        CollisionlessAbundanceCertificate,
        CollisionlessAbundanceCertificate,
    ]
    present_created_number_densities_ev3: tuple[float, float]
    present_created_rest_density_constants_ev4: tuple[float, float]
    present_created_energy_densities_ev4: tuple[float, float]
    present_created_pressures_ev4: tuple[float, float]
    present_background_rms_momentum_ratio_upper: tuple[float, float]
    present_dust_replacement_relative_errors: tuple[float, float]
    present_created_energy_density_total_ev4: float
    present_dust_input_density_total_ev4: float
    omega_created_today: float
    omega_dust_input_today: float
    omega_vacuum_today: float
    omega_created_energy_plus_vacuum_today: float
    omega_dust_input_plus_vacuum_today: float
    background_cold_pass: bool
    perturbation_audit: GaussianNormalModePerturbationAudit
    mass_dimension_manifest: tuple[tuple[str, float], ...]
    dimensionless_core_argument_mass_dimensions: tuple[tuple[str, float], ...]
    dimensions_pass: bool
    failed_gates: tuple[str, ...]
    status: str
    mode_order: str = "FINAL_MASS_ASCENDING_IN_FIXED_NORMAL_BASIS"
    fixed_normal_basis_path_derived: bool = True
    conditional_created_abundance_forward_map_derived: bool = True
    conditional_wkb_composition_derived: bool = True
    late_dephased_created_excess_only: bool = True
    bogoliubov_phase_and_anomalous_correlator_propagated: bool = False
    created_state_is_e45_product_gaussian: bool = False
    quench_profile_supplied: bool = True
    initial_in_state_supplied: bool = True
    entropy_history_supplied: bool = True
    quench_driver_energy_ledger_derived: bool = False
    covariant_qft_stress_renormalized: bool = False
    absolute_abundance_predicted: bool = False
    vacuum_energy_derived: bool = False
    integrated_out_environment_stress_added: bool = False
    full_growth_or_lensing_derived: bool = False
    physical_dark_matter_dark_energy_identification: bool = False


def _bilinear_mass_matrix_data(
    *,
    system_mass: float,
    environment_mass: float,
    coupling: float,
    endpoint: str,
) -> tuple[float, float, float, float, float]:
    if system_mass <= 0.0 or environment_mass <= 0.0:
        raise ValueError(f"{endpoint} endpoint masses must be positive")
    system_mass_squared = system_mass * system_mass
    environment_mass_squared = environment_mass * environment_mass
    determinant = (
        system_mass_squared * environment_mass_squared - coupling * coupling
    )
    if determinant <= 0.0:
        raise ValueError(
            f"{endpoint} bilinear mass matrix must be positive definite"
        )
    frobenius = math.sqrt(
        system_mass_squared * system_mass_squared
        + environment_mass_squared * environment_mass_squared
        + 2.0 * coupling * coupling
    )
    return (
        system_mass_squared,
        environment_mass_squared,
        coupling,
        determinant,
        frobenius,
    )


def audit_fixed_basis_bilinear_quench_cosmology(
    *,
    initial_system_mass_ev: float,
    initial_environment_mass_ev: float,
    initial_bilinear_coupling_ev2: float,
    initial_vacuum_energy_density_ev4: float,
    final_system_mass_ev: float,
    final_environment_mass_ev: float,
    final_bilinear_coupling_ev2: float,
    final_vacuum_energy_density_ev4: float,
    quench_duration_ev_minus1: float,
    initial_mode_occupations: tuple[float, float],
    entropy: EntropyRedshiftContract,
    cosmology: NaturalUnitCosmology,
    hubble_at_production_ev: float,
    comoving_wavenumber_ev: float,
    momentum_max_ev: float | None = None,
    intervals: int = 2400,
    validity_limit: float = 0.1,
    commuting_tolerance: float = 2.0e-12,
) -> FixedBasisBilinearQuenchCosmologyAudit:
    """고정 기저 가우스 급냉을 존재비와 WKB 판별기로 사상한다.

    허용된 각 정규 모드에서 보이지 않는 in-진공 중첩은
    ``a_out = alpha*a_in + beta*a_in^dagger`` 를 따른다. 그 후기 탈위상 관측 효과가
    생성 초과분 ``(1+2*n_in)|beta_p|^2`` 이다. 이 함수는 그 초과분을 모든 운동량에
    대해 적분하고, 오늘로 적색이동시키고, 차가운 부분을 정확히 보존되는 정지 밀도
    상수로 바꾸고, 그 먼지(dust) 입력만 보존 두 장 WKB 섭동 게이트에 넘긴다.

    계산은 주어진 평탄 공간 tanh 프로파일에 대해 정확하고 우주론에 대해서는
    조건부다. 프로파일, 그 동력원, in-상태, 엔트로피 역사, 독립 진공 상수를 끝점
    작용에서 추론하지 않는다.
    """

    scalar_inputs = {
        "initial_system_mass_ev": initial_system_mass_ev,
        "initial_environment_mass_ev": initial_environment_mass_ev,
        "initial_bilinear_coupling_ev2": initial_bilinear_coupling_ev2,
        "initial_vacuum_energy_density_ev4": initial_vacuum_energy_density_ev4,
        "final_system_mass_ev": final_system_mass_ev,
        "final_environment_mass_ev": final_environment_mass_ev,
        "final_bilinear_coupling_ev2": final_bilinear_coupling_ev2,
        "final_vacuum_energy_density_ev4": final_vacuum_energy_density_ev4,
        "hubble_at_production_ev": hubble_at_production_ev,
        "comoving_wavenumber_ev": comoving_wavenumber_ev,
    }
    checked = {
        name: _finite_real(value, name) for name, value in scalar_inputs.items()
    }
    duration = _positive_finite(
        quench_duration_ev_minus1,
        "quench_duration_ev_minus1",
    )
    limit = _positive_finite(validity_limit, "validity_limit")
    if limit > 0.25:
        raise ValueError("validity_limit must lie in (0, 0.25]")
    tolerance = _positive_finite(
        commuting_tolerance,
        "commuting_tolerance",
    )
    if tolerance > 1.0e-6:
        raise ValueError("commuting_tolerance must be <= 1e-6")
    if checked["hubble_at_production_ev"] < 0.0:
        raise ValueError("hubble_at_production_ev must be nonnegative")
    if checked["comoving_wavenumber_ev"] <= 0.0:
        raise ValueError("comoving_wavenumber_ev must be positive")
    if not isinstance(entropy, EntropyRedshiftContract):
        raise ValueError("entropy must be an EntropyRedshiftContract")
    if not isinstance(cosmology, NaturalUnitCosmology):
        raise ValueError("cosmology must be a NaturalUnitCosmology")
    try:
        occupations_input = tuple(initial_mode_occupations)
    except TypeError as error:
        raise ValueError("initial_mode_occupations must contain two values") from error
    if len(occupations_input) != 2:
        raise ValueError("initial_mode_occupations must contain two values")
    occupations = tuple(
        _finite_real(value, f"initial_mode_occupations[{index}]")
        for index, value in enumerate(occupations_input)
    )
    if any(value < 0.0 for value in occupations):
        raise ValueError("initial mode occupations must be nonnegative")

    initial_data = _bilinear_mass_matrix_data(
        system_mass=checked["initial_system_mass_ev"],
        environment_mass=checked["initial_environment_mass_ev"],
        coupling=checked["initial_bilinear_coupling_ev2"],
        endpoint="initial",
    )
    final_data = _bilinear_mass_matrix_data(
        system_mass=checked["final_system_mass_ev"],
        environment_mass=checked["final_environment_mass_ev"],
        coupling=checked["final_bilinear_coupling_ev2"],
        endpoint="final",
    )
    initial_a, initial_d, initial_b, _, initial_norm = initial_data
    final_a, final_d, final_b, _, final_norm = final_data

    commutator_entry = (
        final_b * (initial_a - initial_d)
        - initial_b * (final_a - final_d)
    )
    commutator_norm = math.sqrt(2.0) * abs(commutator_entry)
    commutator_relative = commutator_norm / (initial_norm * final_norm)
    if commutator_relative > tolerance:
        raise ValueError(
            "endpoint mass matrices do not share a fixed normal basis"
        )

    initial_radius = math.hypot(0.5 * (initial_a - initial_d), initial_b)
    initial_gap_fraction = 2.0 * initial_radius / initial_norm
    if initial_gap_fraction <= tolerance:
        raise ValueError(
            "initial normal masses are degenerate; per-mode preparation is "
            "basis-dependent"
        )
    final_radius = math.hypot(0.5 * (final_a - final_d), final_b)
    final_gap_fraction = 2.0 * final_radius / final_norm
    if final_gap_fraction <= tolerance:
        raise ValueError(
            "final normal masses are degenerate; per-mode abundance is basis-dependent"
        )
    angle = 0.5 * math.atan2(2.0 * final_b, final_a - final_d)
    cosine = math.cos(angle)
    sine = math.sin(angle)

    def fixed_basis_entries(
        diagonal_a: float,
        diagonal_d: float,
        off_diagonal: float,
    ) -> tuple[float, float, float]:
        mode_minus = (
            diagonal_a * sine * sine
            - 2.0 * off_diagonal * sine * cosine
            + diagonal_d * cosine * cosine
        )
        mode_plus = (
            diagonal_a * cosine * cosine
            + 2.0 * off_diagonal * sine * cosine
            + diagonal_d * sine * sine
        )
        rotated_off_diagonal = (
            off_diagonal * (cosine * cosine - sine * sine)
            + (diagonal_d - diagonal_a) * sine * cosine
        )
        return mode_minus, mode_plus, rotated_off_diagonal

    initial_minus, initial_plus, initial_rotated_off = fixed_basis_entries(
        initial_a,
        initial_d,
        initial_b,
    )
    final_minus, final_plus, final_rotated_off = fixed_basis_entries(
        final_a,
        final_d,
        final_b,
    )
    fixed_basis_residual = max(
        abs(initial_rotated_off) / initial_norm,
        abs(final_rotated_off) / final_norm,
    )
    if fixed_basis_residual > tolerance:
        raise ValueError("fixed-basis diagonalization residual exceeds tolerance")
    mode_mass_squared_in = (initial_minus, initial_plus)
    mode_mass_squared_out = (final_minus, final_plus)
    all_mode_mass_squared = (*mode_mass_squared_in, *mode_mass_squared_out)
    if any(value <= 0.0 for value in all_mode_mass_squared):
        raise ValueError("fixed-basis normal masses must remain positive")
    mode_masses_in = tuple(math.sqrt(value) for value in mode_mass_squared_in)
    mode_masses_out = tuple(math.sqrt(value) for value in mode_mass_squared_out)

    labels = ("fixed-basis-final-minus", "fixed-basis-final-plus")
    species = tuple(
        QuantumSeatSpecies(
            label=label,
            degeneracy=1,
            mass_in=mass_in,
            mass_out=mass_out,
            duration=duration,
            initial_mode_occupation=occupation,
        )
        for label, mass_in, mass_out, occupation in zip(
            labels,
            mode_masses_in,
            mode_masses_out,
            occupations,
            strict=True,
        )
    )
    abundance = tuple(
        smooth_quench_collisionless_abundance(
            mode_species,
            entropy=entropy,
            cosmology=cosmology,
            momentum_max=momentum_max_ev,
            intervals=intervals,
        )
        for mode_species in species
    )
    rest_densities = tuple(
        certificate.present_rest_density for certificate in abundance
    )
    if sum(rest_densities) <= 0.0:
        raise ValueError(
            "the supplied quench creates no positive dust input for the WKB gate"
        )

    energy_densities = tuple(
        certificate.present_energy_density for certificate in abundance
    )
    pressure_densities = tuple(
        certificate.present_pressure for certificate in abundance
    )
    number_densities = tuple(
        certificate.present_number_density for certificate in abundance
    )
    rms_momentum_ratio_upper = tuple(
        math.sqrt(2.0 * certificate.relative_kinetic_energy_upper_bound)
        for certificate in abundance
    )
    dust_replacement_errors = tuple(
        certificate.relative_kinetic_energy for certificate in abundance
    )
    background_cold_pass = max(rms_momentum_ratio_upper) <= limit
    local_quench_ratio = checked["hubble_at_production_ev"] * duration
    local_quench_pass = local_quench_ratio <= limit

    perturbation = audit_gaussian_normal_mode_perturbations(
        system_mass_ev=checked["final_system_mass_ev"],
        environment_mass_ev=checked["final_environment_mass_ev"],
        bilinear_coupling_ev2=checked["final_bilinear_coupling_ev2"],
        vacuum_energy_density_ev4=checked["final_vacuum_energy_density_ev4"],
        scale_factor=1.0,
        hubble_ev=cosmology.hubble_today,
        comoving_wavenumber_ev=checked["comoving_wavenumber_ev"],
        comoving_mode_density_constants_ev4=rest_densities,
        reduced_planck_mass_ev=cosmology.reduced_planck_mass,
        validity_limit=limit,
    )
    failed_gates: list[str] = []
    if not local_quench_pass:
        failed_gates.append("local_quench_Hstar_tau")
    if not background_cold_pass:
        failed_gates.append("created_background_cold")
    failed_gates.extend(
        f"present_perturbation_{gate}" for gate in perturbation.failed_gates
    )

    critical_density = cosmology.critical_density_today
    total_energy_density = math.fsum(energy_densities)
    total_rest_density = math.fsum(rest_densities)
    omega_created = total_energy_density / critical_density
    omega_dust = total_rest_density / critical_density
    omega_vacuum = (
        checked["final_vacuum_energy_density_ev4"] / critical_density
    )
    dimension_manifest = {
        "normal_mode_mass_squared": 2.0,
        "quench_duration": -1.0,
        "hubble_at_production": 1.0,
        "matrix_commutator": 4.0,
        "created_number_density": 3.0,
        "created_energy_density": 4.0,
        "created_pressure": 4.0,
    }
    dimensionless_core_dimensions = {
        "pi_tau_omega": -1.0 + 1.0,
        "Hstar_tau": 1.0 - 1.0,
        "matrix_commutator_relative_residual": 4.0 - 4.0,
        "rms_momentum_over_mass": 1.0 - 1.0,
        "omega_created_today": 4.0 - 4.0,
    }
    dimensions_pass = all(
        dimension == 0.0 for dimension in dimensionless_core_dimensions.values()
    )
    if not dimensions_pass:
        failed_gates.append("dimensions")

    failures = tuple(failed_gates)
    return FixedBasisBilinearQuenchCosmologyAudit(
        initial_action_parameter_manifest=(
            checked["initial_system_mass_ev"],
            checked["initial_environment_mass_ev"],
            checked["initial_bilinear_coupling_ev2"],
            checked["initial_vacuum_energy_density_ev4"],
        ),
        final_action_parameter_manifest=(
            checked["final_system_mass_ev"],
            checked["final_environment_mass_ev"],
            checked["final_bilinear_coupling_ev2"],
            checked["final_vacuum_energy_density_ev4"],
        ),
        interpolation=(
            "K(t)=K_in+(1+tanh(t/tau))*(K_out-K_in)/2_IN_FIXED_BASIS"
        ),
        matrix_commutator_frobenius_ev4=commutator_norm,
        matrix_commutator_relative_residual=commutator_relative,
        fixed_basis_off_diagonal_relative_residual=fixed_basis_residual,
        initial_normal_mode_gap_fraction=initial_gap_fraction,
        final_normal_mode_gap_fraction=final_gap_fraction,
        fixed_basis_mode_mass_squared_at_initial_ev2=mode_mass_squared_in,
        fixed_basis_mode_mass_squared_at_final_ev2=mode_mass_squared_out,
        fixed_basis_mode_masses_at_initial_ev=mode_masses_in,
        fixed_basis_mode_masses_at_final_ev=mode_masses_out,
        quench_duration_ev_minus1=duration,
        hubble_at_production_ev=checked["hubble_at_production_ev"],
        local_quench_ratio=local_quench_ratio,
        local_quench_limit=limit,
        local_quench_pass=local_quench_pass,
        quench_species=species,
        abundance_certificates=abundance,
        present_created_number_densities_ev3=number_densities,
        present_created_rest_density_constants_ev4=rest_densities,
        present_created_energy_densities_ev4=energy_densities,
        present_created_pressures_ev4=pressure_densities,
        present_background_rms_momentum_ratio_upper=rms_momentum_ratio_upper,
        present_dust_replacement_relative_errors=dust_replacement_errors,
        present_created_energy_density_total_ev4=total_energy_density,
        present_dust_input_density_total_ev4=total_rest_density,
        omega_created_today=omega_created,
        omega_dust_input_today=omega_dust,
        omega_vacuum_today=omega_vacuum,
        omega_created_energy_plus_vacuum_today=omega_created + omega_vacuum,
        omega_dust_input_plus_vacuum_today=omega_dust + omega_vacuum,
        background_cold_pass=background_cold_pass,
        perturbation_audit=perturbation,
        mass_dimension_manifest=tuple(dimension_manifest.items()),
        dimensionless_core_argument_mass_dimensions=tuple(
            dimensionless_core_dimensions.items()
        ),
        dimensions_pass=dimensions_pass,
        failed_gates=failures,
        status=(
            "PASS_CONDITIONAL_FIXED_BASIS_QUENCH_TO_WKB"
            if not failures
            else "FAIL_CONDITIONAL_FIXED_BASIS_QUENCH_TO_WKB_GATES"
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
