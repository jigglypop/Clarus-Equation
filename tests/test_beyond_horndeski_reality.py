from __future__ import annotations

from reality_stone.clarus.beyond_horndeski_reality import (
    beyond_horndeski_reality_audit,
    ce_higher_derivative_extension_audit,
)


def test_beyond_horndeski_evades_no_go_but_has_no_complete_model() -> None:
    audit = beyond_horndeski_reality_audit()

    assert audit.background_existence_demonstrated
    assert audit.no_go_evasion_demonstrated
    assert not audit.all_gates_exist_somewhere_in_portfolio
    assert not audit.one_model_closes_all_gates
    assert not audit.cross_model_evidence_splicing_allowed
    assert audit.complete_static_stability_criteria_available
    assert not audit.criteria_applied_to_explicit_wormhole
    assert not audit.explicit_wormhole_coefficients_reproducible
    assert not audit.slow_spectrum_reproduction_possible
    assert not audit.current_reality_pass


def test_explicit_partial_stability_example_keeps_missing_modes_visible() -> None:
    audit = beyond_horndeski_reality_audit()
    candidate = audit.candidates[1]

    assert candidate.covariant_action_specified
    assert candidate.radial_even_ghost_gradient_gate
    assert candidate.odd_sector_ghost_gradient_gate
    assert not candidate.spherical_even_mode_gate
    assert not candidate.angular_even_gradient_gate
    assert not candidate.slow_tachyon_gate
    assert not candidate.gr_weak_field_asymptotics
    assert not candidate.complete_same_model_pass


def test_disformal_global_family_does_not_inherit_other_models_stability() -> None:
    candidate = beyond_horndeski_reality_audit().candidates[3]

    assert candidate.covariant_action_specified
    assert candidate.regular_asymptotically_flat_background
    assert not candidate.radial_even_ghost_gradient_gate
    assert not candidate.odd_sector_ghost_gradient_gate
    assert not candidate.robust_luminal_tensor_speed
    assert not candidate.ce_action_derivation


def test_2022_model_closes_high_energy_but_not_slow_or_real_world_gates() -> None:
    candidate = beyond_horndeski_reality_audit().candidates[2]

    assert candidate.covariant_action_specified
    assert candidate.regular_asymptotically_flat_background
    assert candidate.radial_even_ghost_gradient_gate
    assert candidate.odd_sector_ghost_gradient_gate
    assert candidate.angular_even_gradient_gate
    assert candidate.complete_high_energy_linear_stability
    assert not candidate.slow_tachyon_gate
    assert not candidate.gr_weak_field_asymptotics
    assert not candidate.robust_luminal_tensor_speed
    assert not candidate.ce_action_derivation
    assert not candidate.engineering_scale_bridge
    assert not candidate.complete_same_model_pass


def test_gw170817_bound_is_a_hard_observational_gate() -> None:
    audit = beyond_horndeski_reality_audit()

    assert audit.gw_speed_relative_bound <= 5.0e-16
    assert not any(candidate.robust_luminal_tensor_speed for candidate in audit.candidates)


def test_lone_ce_second_derivative_square_is_not_a_dhost_completion() -> None:
    audit = ce_higher_derivative_extension_audit(
        standalone_second_derivative_coefficient=1.0
    )

    assert audit.highest_derivative_hessian == 2.0
    assert not audit.standalone_operator_degenerate
    assert not audit.ostrogradsky_mode_avoided
    assert not audit.full_dhost_operator_basis_specified
    assert not audit.degeneracy_relations_specified
    assert not audit.valid_minimal_extension
