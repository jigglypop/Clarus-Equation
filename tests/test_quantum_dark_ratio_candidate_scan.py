from __future__ import annotations

import math

import pytest

from examples.physics.quantum_dark_ratio_candidate_scan import (
    certificate,
    dual_rail_blockade_receipt_candidate,
    epoch_ratio,
    fibonacci_completion_counts,
    fibonacci_parry_candidate,
    fibonacci_uniform_history_bridge,
    flag_continuum_counterexample,
    golden_conditional_action_chain,
    interacting_vacuum_ratio_routes,
    observational_dark_sector_diagnostic,
    locally_uniform_edge_counterexample,
    unequal_energy_counterexample,
    weighted_fibonacci_counterexample,
)


def test_flag_density_family_admits_every_probability() -> None:
    for p in (0.0, 0.17, 0.5, 1.0):
        result = flag_continuum_counterexample(p)
        assert result["positive"]
        assert result["trace"] == pytest.approx(1.0)
        assert result["r2"] == pytest.approx(p)
        assert result["r3"] == pytest.approx(1.0)
        assert result["p_mobile_from_flag"] == pytest.approx(p)
        assert result["p_is_free_on_closed_interval"]


def test_unequal_branch_energy_breaks_probability_energy_identification() -> None:
    result = unequal_energy_counterexample(p_mobile=0.3, mobile_energy=2.0, locked_energy=7.0)
    assert not result["equal"]
    assert result["energy_mobile_fraction"] == pytest.approx(0.6 / 5.5)


def test_dust_vacuum_ratio_is_epoch_dependent() -> None:
    result = epoch_ratio(ratio_at_reference=3.0, scale_factor=2.0, reference_scale_factor=1.0)
    assert result["R_of_a"] == pytest.approx(24.0)


def test_fibonacci_perron_eigenvector_stationary_distribution_and_ratio() -> None:
    result = fibonacci_parry_candidate()
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    locked, mobile = result["stationary_distribution_locked_mobile"]
    assert result["perron_lambda"] == pytest.approx(phi)
    assert result["right_eigenvector"] == pytest.approx((phi, 1.0))
    assert result["transition_row_sum_residual"] == pytest.approx(0.0, abs=1e-12)
    assert result["stationarity_residual"] == pytest.approx(0.0, abs=1e-12)
    assert locked + mobile == pytest.approx(1.0)
    assert mobile == pytest.approx(1.0 / (1.0 + phi * phi))
    assert result["R_locked_over_mobile"] == pytest.approx(phi * phi)
    assert not result["hard_exclusion_derived_from_existing_controlled_swap"]
    assert not result["parry_measure_derived_from_existing_quantum_dynamics"]
    assert not result["prediction"]


def test_finite_uniform_histories_converge_to_parry_and_have_cptp_isometry() -> None:
    assert fibonacci_completion_counts(5) == (
        (1, 1), (2, 1), (3, 2), (5, 3), (8, 5), (13, 8)
    )
    result = fibonacci_uniform_history_bridge(horizon=12, energy_star=5.0)
    assert result["finite_uniform_history_theorem_closed"]
    assert result["parry_record_isometry_closed"]
    assert result["uniform_path_probability_residual"] == pytest.approx(0.0, abs=1e-12)
    assert result["transition_convergence_residual"] < 1e-5
    receipt = result["stationary_ensemble_receipt"]
    assert receipt["mobile_energy_expectation"] + receipt["locked_energy_expectation"] == pytest.approx(5.0)
    assert not receipt["per_event_two_simultaneous_receipts"]
    assert not result["conditional_status"]["hard_exclusion_from_fundamental_dynamics_derived"]
    assert not result["conditional_status"]["prediction"]


def test_local_uniform_edges_are_not_global_uniform_histories() -> None:
    result = locally_uniform_edge_counterexample()
    assert result["p_mobile"] == pytest.approx(1.0 / 3.0)
    assert result["different_from_global_uniform_history_measure"]


def test_dual_rail_blockade_has_fibonacci_space_and_one_receipt_per_site() -> None:
    result = dual_rail_blockade_receipt_candidate(length=64, energy_star=3.0)
    assert result["constrained_hilbert_dimension"] == 27777890035288
    assert result["finite_size_residual"] < 0.003
    assert not result["exact_golden_value_is_finite_length_output"]
    assert not result["measure_boundaries"]["general_realtime_pxp_equivalence"]
    assert result["one_receipt_per_site"] and result["no_double_count"]
    assert result["every_configuration_total_receipt_energy"] == pytest.approx(192.0)
    assert not result["fail_closed"]["hard_blockade_derived_from_three_user_claims"]
    assert not result["fail_closed"]["general_pxp_stationary_state_is_uniform"]
    assert not result["fail_closed"]["prediction"]


def test_golden_candidate_passes_same_probability_and_energy_through_action_chain() -> None:
    result = golden_conditional_action_chain(energy_star=10.0, matching_volume=4.0)
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    assert result["p_mobile"] == pytest.approx(1.0 / (1.0 + phi * phi))
    assert result["R_locked_over_mobile"] == pytest.approx(phi * phi)
    assert result["finite_chain_consistent"]
    assert result["interface_compatibility_check_only"]
    assert not result["exact_ratio_derived_by_finite_storage_or_action"]
    assert "infinite-volume Parry" in result["ratio_source"]
    assert result["probability_residual"] == pytest.approx(0.0, abs=1e-12)
    assert result["energy_residual"] == pytest.approx(0.0, abs=1e-12)
    assert not result["unconditional_cosmological_prediction"]


def test_interacting_vacuum_has_stable_late_attractor_but_no_unique_action() -> None:
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    target = phi * phi
    result = interacting_vacuum_ratio_routes(
        target_ratio=target,
        scale_factor=0.01,
        vacuum_density_at_one=2.0,
        primordial_dust_constant=3.0,
    )
    stable = result["stable_late_attractor"]
    unstable = result["unstable_same_fixed_trajectory"]
    assert stable["fixed_point_R_L_over_m"] == pytest.approx(target)
    assert stable["stable"] and stable["linear_eigenvalue_dRprime_dR"] < 0.0
    assert not unstable["stable"]
    assert stable["early_dust_dominates_at_sample"]
    assert stable["matter_continuity_residual"] == pytest.approx(0.0, abs=1e-8)
    assert stable["vacuum_continuity_residual"] == pytest.approx(0.0, abs=1e-8)
    assert stable["total_continuity_residual"] == pytest.approx(0.0, abs=1e-8)
    assert not result["constant_ratio_trajectory"][
        "ordinary_early_matter_era_preserved_if_applied_all_epochs"
    ]
    assert not result["prediction"]


def test_weighted_adjacency_changes_the_golden_ratio_continuously() -> None:
    unweighted = weighted_fibonacci_counterexample(u=1.0, v=1.0)
    weighted = weighted_fibonacci_counterexample(u=2.0, v=1.0)
    assert unweighted["R_locked_over_mobile"] == pytest.approx(((1 + math.sqrt(5)) / 2) ** 2)
    assert weighted["R_locked_over_mobile"] != pytest.approx(unweighted["R_locked_over_mobile"])


def test_observational_dark_sector_diagnostic_excludes_baryons_and_is_not_a_fit() -> None:
    result = observational_dark_sector_diagnostic(
        omega_cdm=0.265,
        omega_dark_energy=0.685,
        source_provenance="Planck 2018 base flat LambdaCDM diagnostic",
    )
    assert result["p_cdm_dark_sector_normalized"] == pytest.approx(0.265 / 0.95)
    assert result["R_dark_energy_over_cdm"] == pytest.approx(0.685 / 0.265)
    assert not result["baryons_included_in_numerator"]
    assert not result["used_for_derivation"]
    assert not result["likelihood_fit_performed"]


def test_certificate_is_fail_closed_and_external_target_is_diagnostic_only() -> None:
    result = certificate(
        omega_cdm=0.265,
        omega_dark_energy=0.685,
        source_provenance="Planck 2018 diagnostic only",
    )
    assert result["status"] == "MAJOR_RATIO_CANDIDATES_AUDITED_NO_UNIQUE_RATIO_DERIVED_GOLDEN_ROUTE_CONDITIONAL"
    assert not result["fail_closed"]["unique_ratio_derived"]
    assert result["weighted_adjacency_counterexample"]["ratio_changes"]
    diagnostic = result["observational_target_diagnostic"]
    assert diagnostic["source_provenance"] == "Planck 2018 diagnostic only"
    assert not diagnostic["used_for_derivation"]


def test_certificate_requires_both_observational_components() -> None:
    with pytest.raises(ValueError, match="supplied together"):
        certificate(omega_cdm=0.265)
