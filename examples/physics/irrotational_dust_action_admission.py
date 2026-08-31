"""Admit a pre-caustic monokinetic dust flow to a single-clock action.

This module does not construct a gravitational source from a quantum receipt.
It checks a narrower statement.  If an already supplied monokinetic dust flow
is smooth, single-stream, irrotational, and pre-caustic, then it can be written
locally with

    S_d = -1/2 integral sqrt(-g) lambda ((grad tau)^2 + 1),
    u_mu = -partial_mu tau,

and the on-shell action stress is exactly the existing kinetic dust stress.
The two stresses are therefore matched representations of one allocation and
must not be added.  The metric is supplied and has signature (-,+,+,+).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from examples.physics.causal_record_dust_bridge import DustInitialData


DEFAULT_TOLERANCE = 1.0e-12
MINKOWSKI_DIAGONAL = (-1.0, 1.0, 1.0, 1.0)

Vector4 = tuple[float, float, float, float]
Tensor4 = tuple[Vector4, Vector4, Vector4, Vector4]


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _finite_positive(value: float, name: str) -> float:
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


def _close(left: float, right: float, tolerance: float) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def lower_minkowski(vector: Sequence[float]) -> Vector4:
    """Lower a contravariant vector with diag(-1,1,1,1)."""

    vector4 = _vector4(vector, "vector")
    return tuple(  # type: ignore[return-value]
        sign * component for sign, component in zip(MINKOWSKI_DIAGONAL, vector4)
    )


def raise_minkowski(covector: Sequence[float]) -> Vector4:
    """Raise a covector with the inverse Minkowski metric."""

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
    """Return (grad tau)^2 + 1 for the multiplier constraint."""

    return minkowski_covector_inner(
        clock_gradient_covector, clock_gradient_covector
    ) + 1.0


def clock_gradient_from_receipt(
    receipt_gradient_covector: Sequence[float],
    *,
    reference_mass_scale: float,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Vector4:
    """Apply tau=r/M_* and reject receipts that are not unit timelike clocks."""

    scale = _finite_positive(reference_mass_scale, "reference_mass_scale")
    tolerance = _finite_positive(tolerance, "tolerance")
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
    """Return the exact off-shell metric variation of the clock action."""

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
    """Return max |partial_mu u_nu-partial_nu u_mu| in a flat chart."""

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
    """Show that a nonzero overall epsilon is absorbed into the multiplier."""

    epsilon = _finite_positive(epsilon, "epsilon")
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
    """A normalized geodesic dust field that no single scalar clock represents.

    In the region |kappa*y|<1 use
    u^mu=gamma(1,kappa*y,0,0).  The field depends only on y while u^y=0,
    so its flow lines are geodesic and its constant-density current is
    conserved.  Its covector curl is nevertheless nonzero.
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
    equation_of_state: float
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
    """Certify exact local equivalence to the supplied kinetic dust data."""

    tolerance = _finite_positive(tolerance, "tolerance")
    scale = _finite_positive(reference_mass_scale, "reference_mass_scale")
    flow_jacobian = _finite(lagrangian_flow_jacobian, "lagrangian_flow_jacobian")
    proper_time_step = _finite_positive(proper_time_step, "proper_time_step")
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
    mass = _finite_positive(dust.mass, "dust mass")
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
        equation_of_state=0.0,
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
    )


def canonical_dust_data() -> DustInitialData:
    """A nontrivial exact Minkowski dust datum used by the certificate."""

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
    """Return the canonical exact conditional-admission certificate."""

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
