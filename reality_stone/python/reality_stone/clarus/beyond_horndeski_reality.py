"""Non-combinability audit for beyond-Horndeski wormhole evidence."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BeyondHorndeskiCandidateAudit:
    name: str
    covariant_action_specified: bool
    degenerate_three_dof_structure: bool
    regular_asymptotically_flat_background: bool
    radial_even_ghost_gradient_gate: bool
    odd_sector_ghost_gradient_gate: bool
    spherical_even_mode_gate: bool
    angular_even_gradient_gate: bool
    slow_tachyon_gate: bool
    gr_weak_field_asymptotics: bool
    robust_luminal_tensor_speed: bool
    ce_action_derivation: bool
    engineering_scale_bridge: bool
    complete_high_energy_linear_stability: bool
    complete_same_model_pass: bool
    verdict: str


@dataclass(frozen=True)
class BeyondHorndeskiPortfolioAudit:
    candidates: tuple[BeyondHorndeskiCandidateAudit, ...]
    gw_speed_relative_bound: float
    background_existence_demonstrated: bool
    no_go_evasion_demonstrated: bool
    all_gates_exist_somewhere_in_portfolio: bool
    one_model_closes_all_gates: bool
    cross_model_evidence_splicing_allowed: bool
    complete_static_stability_criteria_available: bool
    criteria_applied_to_explicit_wormhole: bool
    explicit_wormhole_coefficients_reproducible: bool
    slow_spectrum_reproduction_possible: bool
    current_reality_pass: bool


@dataclass(frozen=True)
class CEHigherDerivativeExtensionAudit:
    standalone_second_derivative_coefficient: float
    highest_derivative_hessian: float
    standalone_operator_degenerate: bool
    ostrogradsky_mode_avoided: bool
    full_dhost_operator_basis_specified: bool
    degeneracy_relations_specified: bool
    matter_frame_specified: bool
    luminal_tensor_condition_specified: bool
    valid_minimal_extension: bool


def beyond_horndeski_reality_audit() -> BeyondHorndeskiPortfolioAudit:
    """Keep background, perturbation, and observational claims model-local."""

    candidates = (
        BeyondHorndeskiCandidateAudit(
            name="2018 spherical EFT construction",
            covariant_action_specified=False,
            degenerate_three_dof_structure=True,
            regular_asymptotically_flat_background=True,
            radial_even_ghost_gradient_gate=True,
            odd_sector_ghost_gradient_gate=True,
            spherical_even_mode_gate=False,
            angular_even_gradient_gate=False,
            slow_tachyon_gate=False,
            gr_weak_field_asymptotics=False,
            robust_luminal_tensor_speed=False,
            ce_action_derivation=False,
            engineering_scale_bridge=False,
            complete_high_energy_linear_stability=False,
            complete_same_model_pass=False,
            verdict="EFT EXISTENCE CONTROL / MICROSCOPIC ACTION OPEN",
        ),
        BeyondHorndeskiCandidateAudit(
            name="2018 explicit covariant example",
            covariant_action_specified=True,
            degenerate_three_dof_structure=True,
            regular_asymptotically_flat_background=True,
            radial_even_ghost_gradient_gate=True,
            odd_sector_ghost_gradient_gate=True,
            spherical_even_mode_gate=False,
            angular_even_gradient_gate=False,
            slow_tachyon_gate=False,
            gr_weak_field_asymptotics=False,
            robust_luminal_tensor_speed=False,
            ce_action_derivation=False,
            engineering_scale_bridge=False,
            complete_high_energy_linear_stability=False,
            complete_same_model_pass=False,
            verdict="PARTIAL STABILITY / NON-GR WEAK FIELD",
        ),
        BeyondHorndeskiCandidateAudit(
            name="2022 covariant high-energy-stable construction",
            covariant_action_specified=True,
            degenerate_three_dof_structure=True,
            regular_asymptotically_flat_background=True,
            radial_even_ghost_gradient_gate=True,
            odd_sector_ghost_gradient_gate=True,
            spherical_even_mode_gate=False,
            angular_even_gradient_gate=True,
            slow_tachyon_gate=False,
            gr_weak_field_asymptotics=False,
            robust_luminal_tensor_speed=False,
            ce_action_derivation=False,
            engineering_scale_bridge=False,
            complete_high_energy_linear_stability=True,
            complete_same_model_pass=False,
            verdict="HIGH-ENERGY STABLE / SLOW TACHYON AND GR LIMIT OPEN",
        ),
        BeyondHorndeskiCandidateAudit(
            name="2021 disformal Lovelock-origin family",
            covariant_action_specified=True,
            degenerate_three_dof_structure=True,
            regular_asymptotically_flat_background=True,
            radial_even_ghost_gradient_gate=False,
            odd_sector_ghost_gradient_gate=False,
            spherical_even_mode_gate=False,
            angular_even_gradient_gate=False,
            slow_tachyon_gate=False,
            gr_weak_field_asymptotics=False,
            robust_luminal_tensor_speed=False,
            ce_action_derivation=False,
            engineering_scale_bridge=False,
            complete_high_energy_linear_stability=False,
            complete_same_model_pass=False,
            verdict="EXPLICIT GLOBAL BACKGROUND / STABILITY OPEN",
        ),
    )
    all_gate_names = (
        "covariant_action_specified",
        "degenerate_three_dof_structure",
        "regular_asymptotically_flat_background",
        "radial_even_ghost_gradient_gate",
        "odd_sector_ghost_gradient_gate",
        "spherical_even_mode_gate",
        "angular_even_gradient_gate",
        "slow_tachyon_gate",
        "gr_weak_field_asymptotics",
        "robust_luminal_tensor_speed",
        "ce_action_derivation",
        "engineering_scale_bridge",
    )
    gates_exist_somewhere = all(
        any(getattr(candidate, gate) for candidate in candidates) for gate in all_gate_names
    )
    one_model_passes = any(candidate.complete_same_model_pass for candidate in candidates)
    return BeyondHorndeskiPortfolioAudit(
        candidates=candidates,
        gw_speed_relative_bound=5.0e-16,
        background_existence_demonstrated=True,
        no_go_evasion_demonstrated=True,
        all_gates_exist_somewhere_in_portfolio=gates_exist_somewhere,
        one_model_closes_all_gates=one_model_passes,
        cross_model_evidence_splicing_allowed=False,
        complete_static_stability_criteria_available=True,
        criteria_applied_to_explicit_wormhole=False,
        explicit_wormhole_coefficients_reproducible=False,
        slow_spectrum_reproduction_possible=False,
        current_reality_pass=False,
    )


def ce_higher_derivative_extension_audit(
    *,
    standalone_second_derivative_coefficient: float = 1.0,
) -> CEHigherDerivativeExtensionAudit:
    """Reject relabeling a lone ``alpha2*(box phi)^2`` term as DHOST."""

    coefficient = float(standalone_second_derivative_coefficient)
    if not math.isfinite(coefficient):
        raise ValueError("the higher-derivative coefficient must be finite")
    hessian = 2.0 * coefficient
    degenerate = hessian == 0.0
    return CEHigherDerivativeExtensionAudit(
        standalone_second_derivative_coefficient=coefficient,
        highest_derivative_hessian=hessian,
        standalone_operator_degenerate=degenerate,
        ostrogradsky_mode_avoided=degenerate,
        full_dhost_operator_basis_specified=False,
        degeneracy_relations_specified=False,
        matter_frame_specified=False,
        luminal_tensor_condition_specified=False,
        valid_minimal_extension=False,
    )
