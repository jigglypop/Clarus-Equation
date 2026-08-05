from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from reality_stone.clarus.external_field_to_matter import (
    EXPECTED_SNAPSHOT_SHA256,
    build_external_field_to_matter_reanalysis,
    load_external_field_to_matter_snapshot,
    reanalyse_analogue_dce,
    reanalyse_breit_wheeler,
    reanalyse_metastable_material_phase,
    sha256_file,
)


SNAPSHOT = Path(__file__).parents[1] / "benchmarks" / "external_field_to_matter_v1.json"


def _snapshot() -> dict[str, object]:
    return load_external_field_to_matter_snapshot(SNAPSHOT)


def test_external_snapshot_is_hash_locked() -> None:
    assert sha256_file(SNAPSHOT) == EXPECTED_SNAPSHOT_SHA256
    assert _snapshot()["schema"] == "clarus.external_field_to_matter.v1"


def test_analogue_dce_source_data_reproduce_wavelength_peak_and_antibunching() -> None:
    report = reanalyse_analogue_dce(_snapshot()["analogue_dce"])

    assert report.predicted_signal_wavelength_nm == pytest.approx(954.3126699)
    assert report.predicted_idler_wavelength_nm == pytest.approx(1173.0600603)
    assert report.observed_peak_signal_wavelength_nm == 953.9375
    assert report.observed_peak_idler_wavelength_nm == 1172.888889
    assert report.observed_peak_car == 5.1
    assert report.peak_is_nearest_sampled_cell_to_prediction
    assert report.inverse_wavelength_energy_residual == pytest.approx(2.82329432e-4)
    assert report.best_heralded_car == 4.745
    assert report.best_g2_zero == pytest.approx(0.380952381)
    assert report.reported_error_separation_below_one_sigma == pytest.approx(10.3174603)
    assert report.correlated_g2_point_count == 4
    assert report.all_correlated_g2_points_one_error_below_one
    assert report.zero_car_control_consistent_with_one
    assert report.external_nonclassical_pair_emission_supported


def test_breit_wheeler_hepdata_bins_reproduce_published_total() -> None:
    report = reanalyse_breit_wheeler(_snapshot()["breit_wheeler"])

    assert report.bin_count == 7
    assert report.integrated_cross_section_microbarn == pytest.approx(263.3930128)
    assert report.integrated_stat_uncertainty_microbarn == pytest.approx(1.72378271)
    assert report.integrated_syst_uncorrelated_floor_microbarn == pytest.approx(
        16.49576566
    )
    assert report.integrated_syst_fully_correlated_ceiling_microbarn == pytest.approx(
        19.3188532
    )
    assert report.central_value_reproduced
    assert report.statistical_uncertainty_reproduced
    assert report.published_systematic_compatible_with_unknown_covariance
    assert report.descriptive_zero_separation_sigma == pytest.approx(14.72825688)
    assert report.selected_pair_count == 19_689
    assert report.selected_mass_floor_over_rest_mass_threshold == pytest.approx(4892.37796)
    models = {comparison.model: comparison for comparison in report.model_comparisons}
    assert models["gamma_upc"].integrated_cross_section_microbarn == pytest.approx(
        265.6158502
    )
    assert models["superchic"].integrated_cross_section_microbarn == pytest.approx(
        260.944426
    )
    assert models["starlight"].integrated_cross_section_microbarn == pytest.approx(
        225.0994729
    )
    assert report.charged_form_factor_models_closer_than_starlight
    assert report.external_quasi_real_photon_pair_production_supported


def test_combined_result_preserves_the_missing_clarus_bridge() -> None:
    report = build_external_field_to_matter_reanalysis(_snapshot())

    assert report.independent_external_measurements_reproduced
    assert report.pump_to_nonclassical_photon_pairs_supported
    assert report.electromagnetic_field_to_ordinary_massive_pairs_supported
    assert report.observed_optical_pair_energy_ev == pytest.approx(2.3567937896)
    assert report.electron_positron_rest_threshold_ev == pytest.approx(1.0219979e6)
    assert report.optical_pair_to_rest_threshold_energy_gap > 4.3e5
    assert report.cms_mass_floor_to_optical_pair_energy_gap > 2.1e9
    assert report.material_raw_acquisition_audited
    assert not report.metastable_phase_exact_raw_refit_reproduced
    assert not report.same_apparatus_end_to_end_chain_proven
    assert not report.clarus_field_observed
    assert not report.clarus_coupling_measured
    assert not report.new_material_species_created
    assert not report.pump_off_persistent_structure_proven
    assert not report.free_energy_or_vacuum_energy_extraction_proven
    assert not report.analogue_dce.literal_moving_mirror_dce_proven
    assert not report.analogue_dce.event_level_raw_counts_available
    assert not report.breit_wheeler.ideal_free_on_shell_two_photon_collision_proven
    assert not report.breit_wheeler.detector_event_level_data_public


def test_metastable_material_raw_parameters_are_audited_without_overclaim() -> None:
    report = reanalyse_metastable_material_phase(
        _snapshot()["metastable_material_phase"]
    )

    assert report.intensity_shape == (256, 344, 45)
    assert report.total_pump_fluence_mj_cm2 == pytest.approx(0.6)
    assert report.total_fluence_over_single_pulse_threshold == pytest.approx(1.2)
    assert report.published_coherent_period_fs == pytest.approx(454.5454545)
    assert report.published_coherent_period_error_fs == pytest.approx(20.6611570)
    assert report.samples_per_published_period == pytest.approx(18.1818182)
    assert report.repetition_interval_us == pytest.approx(100.0)
    assert report.repetition_interval_not_longer_than_lifetime_bound
    assert report.fixed_delay_mismatch_ps == pytest.approx(10.0)
    assert report.temperature_mismatch_k == pytest.approx(140.0)
    assert report.public_raw_acquisition_traceable
    assert not report.provenance_metadata_consistent
    assert not report.exact_coherent_frequency_raw_refit_reproduced
    assert not report.persistence_lifetime_raw_refit_reproduced
    assert report.pre_existing_material_phase_rearrangement_supported
    assert not report.new_material_species_created
    assert not report.clarus_field_observed


def test_metastable_material_invalid_shape_and_digest_are_rejected() -> None:
    section = deepcopy(_snapshot()["metastable_material_phase"])
    section["figure_3_acquisition"]["intensity_shape"] = [256, 344]
    with pytest.raises(ValueError, match="three dimensions"):
        reanalyse_metastable_material_phase(section)

    section = deepcopy(_snapshot()["metastable_material_phase"])
    section["source"]["figure_3_nexus_md5"] = "not-a-digest"
    with pytest.raises(ValueError, match="MD5"):
        reanalyse_metastable_material_phase(section)


def test_car_peak_moved_off_prediction_cannot_self_certify() -> None:
    section = deepcopy(_snapshot()["analogue_dce"])
    section["figure_3_car_map"]["rows"][0]["car"][0] = 99.0

    report = reanalyse_analogue_dce(section)

    assert not report.peak_is_nearest_sampled_cell_to_prediction
    assert not report.external_nonclassical_pair_emission_supported


def test_classical_g2_control_is_required() -> None:
    section = deepcopy(_snapshot()["analogue_dce"])
    control = next(
        row
        for row in section["figure_4_experimental_rows"]
        if row["heralded_car"] == 0.0
    )
    control["g2_zero"] = 0.5

    report = reanalyse_analogue_dce(section)

    assert not report.zero_car_control_consistent_with_one
    assert not report.external_nonclassical_pair_emission_supported


def test_breit_wheeler_bin_gap_is_rejected() -> None:
    section = deepcopy(_snapshot()["breit_wheeler"])
    section["figure_4_pair_pt_bins"][2]["low_gev"] = 0.29

    with pytest.raises(ValueError, match="contiguous"):
        reanalyse_breit_wheeler(section)


def test_breit_wheeler_model_tampering_cannot_promote_support() -> None:
    section = deepcopy(_snapshot()["breit_wheeler"])
    for row in section["figure_4_pair_pt_bins"]:
        row["gamma_upc_microbarn_per_gev"] = 0.0
        row["superchic_microbarn_per_gev"] = 0.0

    report = reanalyse_breit_wheeler(section)

    assert not report.charged_form_factor_models_closer_than_starlight
    assert not report.external_quasi_real_photon_pair_production_supported


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True])
def test_nonfinite_or_boolean_measurements_are_rejected(bad: float | bool) -> None:
    section = deepcopy(_snapshot()["breit_wheeler"])
    section["figure_4_pair_pt_bins"][0]["measured_microbarn_per_gev"] = bad

    with pytest.raises(ValueError):
        reanalyse_breit_wheeler(section)


def test_modified_snapshot_fails_before_analysis(tmp_path: Path) -> None:
    modified = deepcopy(_snapshot())
    modified["retrieved_on"] = "2099-01-01"
    path = tmp_path / "modified.json"
    path.write_text(json.dumps(modified), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_external_field_to_matter_snapshot(path)

    assert math.isfinite(len(path.read_bytes()))
