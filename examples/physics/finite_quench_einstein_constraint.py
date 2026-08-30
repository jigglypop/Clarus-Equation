"""One-node scalar Einstein constraints for the finite-quench two-fluid branch.

The metric and velocity conventions are those of Valiviita, Majerotto and
Maartens, arXiv:0804.0232,

    ds^2 = a^2[-(1+2 phi)d tau^2 + (1-2 psi) dx^2],
    theta_A = -k^2 v_A,

in Newtonian gauge.  With n=ln(a), kappa=k/(aH),

    Delta = sum_A delta rho_A/rho_unit,
    U = sum_A (rho_A+P_A) aH v_A/rho_unit,
    C = 4 pi G rho_unit/H^2,

the scalar constraints are

    kappa^2 psi + 3(D_n psi + phi) = -C Delta,
    D_n psi + phi = -C U.

Hence, for kappa>0 and zero total anisotropic stress,

    psi = C(3U-Delta)/kappa^2,
    phi = psi,
    D_n psi = -C U - phi.

This module constructs or audits one Fourier node.  It does not propagate the
constraints, integrate an Einstein-Boltzmann system, derive initial data, or
claim that the two-fluid manifest describes the observed universe.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from examples.physics.finite_quench_barotropic_closure import (
    FiniteQuenchStrictBarotropicClosure,
    StrictBarotropicClosureReceipt,
)
from examples.physics.finite_quench_flat_gr_background import (
    FiniteQuenchTwoFluidFlatGRBackground,
    TwoFluidFlatGRBackgroundReceipt,
)
from examples.physics.kinetic_dark_sector_finite_quench_bridge import (
    FiniteQuenchBridge,
)


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _finite_sum(name: str, *values: float) -> float:
    try:
        result = math.fsum(values)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} left the finite domain") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} left the finite domain")
    return result


def _finite_product(name: str, left: float, right: float) -> float:
    result = left * right
    if not math.isfinite(result):
        raise ValueError(f"{name} left the finite domain")
    return result


def _within_roundoff(residual: float, *terms: float) -> bool:
    scale = max(1.0, *(abs(term) for term in terms))
    return abs(residual) <= 64.0 * math.ulp(scale)


def _comparison(left: float, right: float) -> tuple[float, bool]:
    residual = _finite_sum("cross-receipt comparison", left, -right)
    return residual, _within_roundoff(residual, left, right)


@dataclass(frozen=True)
class ScalarEinsteinConstraintReceipt:
    """Raw residual audit for one nonzero scalar Fourier mode."""

    n: float
    k_over_a_h: float
    k_over_a_h_squared: float
    produced_density_perturbation: float
    reservoir_density_perturbation: float
    total_density_perturbation: float
    produced_momentum_density: float
    reservoir_momentum_density: float
    total_momentum_density: float
    total_normalized_anisotropic_stress: float
    gravity_constraint_coupling: float
    lapse_potential: float
    curvature_potential: float
    curvature_potential_log_derivative: float
    energy_constraint_residual: float
    momentum_constraint_residual: float
    zero_stress_traceless_spatial_residual: float
    combined_constraint_residual: float
    background_cross_residuals: tuple[tuple[str, float], ...]
    closure_cross_residuals: tuple[tuple[str, float], ...]
    background_receipt_matches_bridge: bool
    closure_receipt_matches_bridge: bool
    complete_two_species_perturbation_manifest: bool
    energy_constraint_holds: bool
    momentum_constraint_holds: bool
    zero_stress_traceless_spatial_constraint_holds: bool
    combined_constraint_holds: bool
    all_declared_scalar_constraints_hold: bool
    failure_reasons: tuple[str, ...]
    dimensionless_roles: tuple[tuple[str, str], ...]
    combined_constraint_is_derived_crosscheck: bool = True
    metric_convention: str = "VMM_phi_lapse_psi_curvature"
    velocity_convention: str = "VMM_theta_equals_minus_k_squared_v"
    source: str = (
        "general_flat_GR_constraints_checked_in_"
        "Valiviita_Majerotto_Maartens_2008_Eqs_52_53_55_conventions"
    )
    role: str = (
        "CONDITIONAL_K_POSITIVE_ONE_NODE_SCALAR_EINSTEIN_CONSTRAINT_"
        "NOT_PROPAGATED_OR_INTEGRATED_SOLUTION"
    )


class FiniteQuenchScalarEinsteinConstraint:
    """Construct and independently audit the kappa>0 scalar constraints."""

    def __init__(self, bridge: FiniteQuenchBridge) -> None:
        if not isinstance(bridge, FiniteQuenchBridge):
            raise ValueError("bridge must be a FiniteQuenchBridge")
        self.bridge = bridge

    def _raw_receipts(
        self,
        *,
        background: object,
        closure: object,
    ) -> tuple[
        TwoFluidFlatGRBackgroundReceipt,
        StrictBarotropicClosureReceipt,
        tuple[tuple[str, float], ...],
        tuple[tuple[str, float], ...],
        bool,
        bool,
    ]:
        if not isinstance(background, TwoFluidFlatGRBackgroundReceipt):
            raise ValueError(
                "background must be a TwoFluidFlatGRBackgroundReceipt"
            )
        if not isinstance(closure, StrictBarotropicClosureReceipt):
            raise ValueError("closure must be a StrictBarotropicClosureReceipt")

        raw_background = FiniteQuenchTwoFluidFlatGRBackground(
            self.bridge
        ).audit(
            background.n,
            normalized_hubble_squared=(
                background.hubble_squared_over_eight_pi_g_rho_unit_over_three
            ),
            hubble_log_derivative=background.hubble_log_derivative,
        )
        background_pairs = (
            ("n", background.n, raw_background.n),
            (
                "produced_density",
                background.produced_density,
                raw_background.produced_density,
            ),
            (
                "reservoir_density",
                background.reservoir_density,
                raw_background.reservoir_density,
            ),
            (
                "total_density",
                background.total_density,
                raw_background.total_density,
            ),
            (
                "total_pressure",
                background.total_pressure,
                raw_background.total_pressure,
            ),
            (
                "gravity_constraint_coupling",
                background.gravity_constraint_coupling,
                raw_background.gravity_constraint_coupling,
            ),
            (
                "omega_density_unit",
                background.omega_density_unit,
                raw_background.omega_density_unit,
            ),
            (
                "hubble_log_derivative",
                background.hubble_log_derivative,
                raw_background.hubble_log_derivative,
            ),
        )
        background_residuals: list[tuple[str, float]] = []
        background_numeric_match = True
        for name, left, right in background_pairs:
            residual, holds = _comparison(left, right)
            background_residuals.append((name, residual))
            background_numeric_match = background_numeric_match and holds
        background_match = (
            background_numeric_match
            and raw_background.all_background_constraints_hold
            and background.species_manifest == ("produced", "reservoir")
            and background.external_background_species_assumed_absent
        )

        raw_closure = FiniteQuenchStrictBarotropicClosure(self.bridge).audit(
            n=closure.n,
            produced_density_perturbation=(
                closure.produced_density_perturbation
            ),
            reservoir_density_perturbation=(
                closure.reservoir_density_perturbation
            ),
            produced_pressure_perturbation=(
                closure.produced_pressure_perturbation
            ),
            reservoir_pressure_perturbation=(
                closure.reservoir_pressure_perturbation
            ),
            produced_normalized_anisotropic_stress=(
                closure.produced_normalized_anisotropic_stress
            ),
            reservoir_normalized_anisotropic_stress=(
                closure.reservoir_normalized_anisotropic_stress
            ),
            produced_background_pressure_derivative=(
                closure.produced_background_pressure_derivative
            ),
            reservoir_background_pressure_derivative=(
                closure.reservoir_background_pressure_derivative
            ),
        )
        closure_pairs = (
            ("n", closure.n, background.n),
            (
                "reservoir_equation_of_state",
                closure.reservoir_equation_of_state,
                self.bridge.config.w_reservoir,
            ),
            (
                "produced_pressure_closure",
                raw_closure.produced_pressure_closure_residual,
                0.0,
            ),
            (
                "reservoir_pressure_closure",
                raw_closure.reservoir_pressure_closure_residual,
                0.0,
            ),
            (
                "produced_anisotropic_stress",
                raw_closure.produced_anisotropic_stress_residual,
                0.0,
            ),
            (
                "reservoir_anisotropic_stress",
                raw_closure.reservoir_anisotropic_stress_residual,
                0.0,
            ),
            (
                "produced_background_pressure_derivative",
                raw_closure.produced_background_barotrope_derivative_residual,
                0.0,
            ),
            (
                "reservoir_background_pressure_derivative",
                raw_closure.reservoir_background_barotrope_derivative_residual,
                0.0,
            ),
        )
        closure_residuals: list[tuple[str, float]] = []
        closure_numeric_match = True
        for name, left, right in closure_pairs:
            residual, holds = _comparison(left, right)
            closure_residuals.append((name, residual))
            closure_numeric_match = closure_numeric_match and holds
        closure_match = (
            closure_numeric_match
            and raw_closure.all_strict_barotropic_constraints_hold
        )

        return (
            raw_background,
            raw_closure,
            tuple(background_residuals),
            tuple(closure_residuals),
            background_match,
            closure_match,
        )

    @staticmethod
    def _kappa(k_over_a_h: object) -> tuple[float, float]:
        kappa = _finite_real(k_over_a_h, "k_over_a_h")
        if kappa <= 0.0:
            raise ValueError("algebraic Einstein solver requires k_over_a_h > 0")
        try:
            kappa_squared = kappa**2
        except OverflowError as error:
            raise ValueError("k_over_a_h squared left the finite domain") from error
        if not math.isfinite(kappa_squared) or kappa_squared <= 0.0:
            raise ValueError(
                "k_over_a_h squared must remain positive and finite"
            )
        return kappa, kappa_squared

    def construct(
        self,
        *,
        background: object,
        closure: object,
        k_over_a_h: object,
        produced_momentum_density: object,
        reservoir_momentum_density: object,
    ) -> ScalarEinsteinConstraintReceipt:
        """Construct phi, psi, and D_n psi for one nonzero mode."""

        (
            raw_background,
            raw_closure,
            _,
            _,
            background_match,
            closure_match,
        ) = self._raw_receipts(background=background, closure=closure)
        if not background_match:
            raise ValueError("background receipt does not match this GR branch")
        if not closure_match:
            raise ValueError("closure receipt does not match this fluid branch")
        _, kappa_squared = self._kappa(k_over_a_h)
        momentum_p = _finite_real(
            produced_momentum_density,
            "produced_momentum_density",
        )
        momentum_r = _finite_real(
            reservoir_momentum_density,
            "reservoir_momentum_density",
        )
        delta_total = _finite_sum(
            "total density perturbation",
            raw_closure.produced_density_perturbation,
            raw_closure.reservoir_density_perturbation,
        )
        momentum_total = _finite_sum(
            "total momentum density",
            momentum_p,
            momentum_r,
        )
        three_u_minus_delta = _finite_sum(
            "Einstein constraint numerator",
            3.0 * momentum_total,
            -delta_total,
        )
        numerator = _finite_product(
            "Einstein constraint numerator",
            raw_background.gravity_constraint_coupling,
            three_u_minus_delta,
        )
        curvature = numerator / kappa_squared
        if not math.isfinite(curvature):
            raise ValueError("constructed curvature potential left the finite domain")
        lapse = curvature
        coupling_u = _finite_product(
            "momentum constraint source",
            raw_background.gravity_constraint_coupling,
            momentum_total,
        )
        curvature_prime = _finite_sum(
            "constructed curvature derivative",
            -coupling_u,
            -lapse,
        )
        return self.audit(
            background=background,
            closure=closure,
            k_over_a_h=k_over_a_h,
            produced_momentum_density=momentum_p,
            reservoir_momentum_density=momentum_r,
            lapse_potential=lapse,
            curvature_potential=curvature,
            curvature_potential_log_derivative=curvature_prime,
        )

    def audit(
        self,
        *,
        background: object,
        closure: object,
        k_over_a_h: object,
        produced_momentum_density: object,
        reservoir_momentum_density: object,
        lapse_potential: object,
        curvature_potential: object,
        curvature_potential_log_derivative: object,
    ) -> ScalarEinsteinConstraintReceipt:
        """Audit supplied metric data against all raw scalar constraints."""

        (
            raw_background,
            raw_closure,
            background_residuals,
            closure_residuals,
            background_match,
            closure_match,
        ) = self._raw_receipts(background=background, closure=closure)
        kappa, kappa_squared = self._kappa(k_over_a_h)
        momentum_p = _finite_real(
            produced_momentum_density,
            "produced_momentum_density",
        )
        momentum_r = _finite_real(
            reservoir_momentum_density,
            "reservoir_momentum_density",
        )
        phi = _finite_real(lapse_potential, "lapse_potential")
        psi = _finite_real(curvature_potential, "curvature_potential")
        psi_prime = _finite_real(
            curvature_potential_log_derivative,
            "curvature_potential_log_derivative",
        )
        delta_p = raw_closure.produced_density_perturbation
        delta_r = raw_closure.reservoir_density_perturbation
        delta_total = _finite_sum("total density perturbation", delta_p, delta_r)
        momentum_total = _finite_sum(
            "total momentum density",
            momentum_p,
            momentum_r,
        )
        pi_total = _finite_sum(
            "total anisotropic stress",
            raw_closure.produced_normalized_anisotropic_stress,
            raw_closure.reservoir_normalized_anisotropic_stress,
        )
        coupling = raw_background.gravity_constraint_coupling
        kappa_psi = _finite_product(
            "energy constraint kappa term",
            kappa_squared,
            psi,
        )
        coupling_delta = _finite_product(
            "energy constraint density term",
            coupling,
            delta_total,
        )
        coupling_u = _finite_product(
            "momentum constraint source",
            coupling,
            momentum_total,
        )
        metric_rate = _finite_sum("metric rate", psi_prime, phi)
        energy_residual = _finite_sum(
            "energy constraint residual",
            kappa_psi,
            3.0 * metric_rate,
            coupling_delta,
        )
        momentum_residual = _finite_sum(
            "momentum constraint residual",
            metric_rate,
            coupling_u,
        )
        traceless_residual = _finite_sum(
            "traceless spatial residual",
            psi,
            -phi,
        )
        combined_residual = _finite_sum(
            "combined constraint residual",
            kappa_psi,
            coupling_delta,
            -3.0 * coupling_u,
        )
        energy_holds = _within_roundoff(
            energy_residual,
            kappa_psi,
            3.0 * metric_rate,
            coupling_delta,
        )
        momentum_holds = _within_roundoff(
            momentum_residual,
            metric_rate,
            coupling_u,
        )
        zero_stress_holds = _within_roundoff(
            pi_total,
            raw_closure.produced_normalized_anisotropic_stress,
            raw_closure.reservoir_normalized_anisotropic_stress,
        )
        traceless_holds = zero_stress_holds and _within_roundoff(
            traceless_residual,
            psi,
            phi,
        )
        combined_holds = _within_roundoff(
            combined_residual,
            kappa_psi,
            coupling_delta,
            3.0 * coupling_u,
        )
        manifest_holds = (
            raw_background.species_manifest == ("produced", "reservoir")
            and raw_background.external_background_species_assumed_absent
        )
        all_holds = (
            background_match
            and closure_match
            and manifest_holds
            and energy_holds
            and momentum_holds
            and traceless_holds
            and combined_holds
        )
        failures: list[str] = []
        if not background_match:
            failures.append("BACKGROUND_RECEIPT_MISMATCH")
        if not closure_match:
            failures.append("CLOSURE_RECEIPT_MISMATCH")
        if not manifest_holds:
            failures.append("PERTURBATION_SPECIES_MANIFEST_INCOMPLETE")
        if not energy_holds:
            failures.append("EINSTEIN_00_CONSTRAINT_FAILED")
        if not momentum_holds:
            failures.append("EINSTEIN_0I_CONSTRAINT_FAILED")
        if not traceless_holds:
            failures.append("ZERO_STRESS_TRACELESS_IJ_CONSTRAINT_FAILED")
        if not combined_holds:
            failures.append("COMBINED_EINSTEIN_CONSTRAINT_FAILED")

        return ScalarEinsteinConstraintReceipt(
            n=raw_background.n,
            k_over_a_h=kappa,
            k_over_a_h_squared=kappa_squared,
            produced_density_perturbation=delta_p,
            reservoir_density_perturbation=delta_r,
            total_density_perturbation=delta_total,
            produced_momentum_density=momentum_p,
            reservoir_momentum_density=momentum_r,
            total_momentum_density=momentum_total,
            total_normalized_anisotropic_stress=pi_total,
            gravity_constraint_coupling=coupling,
            lapse_potential=phi,
            curvature_potential=psi,
            curvature_potential_log_derivative=psi_prime,
            energy_constraint_residual=energy_residual,
            momentum_constraint_residual=momentum_residual,
            zero_stress_traceless_spatial_residual=traceless_residual,
            combined_constraint_residual=combined_residual,
            background_cross_residuals=background_residuals,
            closure_cross_residuals=closure_residuals,
            background_receipt_matches_bridge=background_match,
            closure_receipt_matches_bridge=closure_match,
            complete_two_species_perturbation_manifest=manifest_holds,
            energy_constraint_holds=energy_holds,
            momentum_constraint_holds=momentum_holds,
            zero_stress_traceless_spatial_constraint_holds=traceless_holds,
            combined_constraint_holds=combined_holds,
            all_declared_scalar_constraints_hold=all_holds,
            failure_reasons=tuple(failures),
            dimensionless_roles=(
                ("n", "dimensionless_log_scale_factor"),
                ("k_over_a_h", "dimensionless_wavenumber"),
                ("Delta", "density_over_constant_density_unit"),
                ("U", "enthalpy_weighted_dimensionless_velocity"),
                ("C", "four_pi_G_rho_unit_over_H_squared"),
                ("phi", "dimensionless_lapse_potential"),
                ("psi", "dimensionless_curvature_potential"),
                ("D_n_psi", "dimensionless_log_scale_derivative"),
            ),
        )
