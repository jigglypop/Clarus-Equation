"""Reanalyse public measurements nearest to a field-to-matter claim.

This module does not manufacture a synthetic detector.  It recomputes two
published results and audits one raw acquisition from immutable external data:

* wavelength-selective, non-classical photon-pair emission in an optical
  analogue of the dynamical Casimir effect; and
* the fiducial ``gamma gamma -> e+ e-`` cross section measured by CMS in
  ultraperipheral lead-lead collisions; and
* the acquisition parameters behind a light-driven metastable phase in
  1T-TaS2, while refusing an exact spectral or lifetime replication where the
  public provenance is insufficient.

The two measurements are independent experiments.  They establish separate
pieces of known physics and must not be presented as one end-to-end Clarus
device or as evidence for a new field, a new material species, or free energy.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence


SPEED_OF_LIGHT_M_S = 299_792_458.0
PLANCK_CONSTANT_EV_S = 4.135667696e-15
ELECTRON_MASS_GEV = 0.00051099895
EXPECTED_SNAPSHOT_SHA256 = (
    "8c78bccb0e8d197bcea80c229e068baa72c1c7a2b920c6b8ddd7f473efc0997f"
)


@dataclass(frozen=True)
class AnalogueDCEReanalysis:
    source_archive_sha256: str
    source_is_processed_figure_data: bool
    predicted_signal_wavelength_nm: float
    predicted_idler_wavelength_nm: float
    observed_peak_signal_wavelength_nm: float
    observed_peak_idler_wavelength_nm: float
    observed_peak_car: float
    signal_detuning_nm: float
    idler_detuning_nm: float
    peak_is_nearest_sampled_cell_to_prediction: bool
    inverse_wavelength_energy_residual: float
    best_heralded_car: float
    best_g2_zero: float
    best_g2_zero_error: float
    reported_error_separation_below_one_sigma: float
    correlated_g2_point_count: int
    all_correlated_g2_points_one_error_below_one: bool
    zero_car_control_g2: float
    zero_car_control_consistent_with_one: bool
    external_nonclassical_pair_emission_supported: bool
    literal_moving_mirror_dce_proven: bool
    event_level_raw_counts_available: bool
    pump_powered_not_free_energy: bool


@dataclass(frozen=True)
class ModelIntegralComparison:
    model: str
    integrated_cross_section_microbarn: float
    difference_from_measurement_microbarn: float
    relative_difference: float
    difference_over_published_total_error: float


@dataclass(frozen=True)
class BreitWheelerReanalysis:
    hepdata_payload_sha256: str
    bin_count: int
    integrated_cross_section_microbarn: float
    integrated_stat_uncertainty_microbarn: float
    integrated_syst_uncorrelated_floor_microbarn: float
    integrated_syst_fully_correlated_ceiling_microbarn: float
    published_cross_section_microbarn: float
    published_stat_uncertainty_microbarn: float
    published_syst_uncertainty_microbarn: float
    central_value_reproduced: bool
    statistical_uncertainty_reproduced: bool
    published_systematic_compatible_with_unknown_covariance: bool
    descriptive_zero_separation_sigma: float
    selected_pair_count: int
    selected_mass_floor_over_rest_mass_threshold: float
    model_comparisons: tuple[ModelIntegralComparison, ...]
    charged_form_factor_models_closer_than_starlight: bool
    external_quasi_real_photon_pair_production_supported: bool
    ideal_free_on_shell_two_photon_collision_proven: bool
    detector_event_level_data_public: bool
    persistent_material_created: bool


@dataclass(frozen=True)
class MetastableMaterialPhaseReanalysis:
    summary_xlsx_md5: str
    figure_3_nexus_md5: str
    static_c_phase_nexus_md5: str
    intensity_shape: tuple[int, int, int]
    pump_photon_energy_ev: float
    total_pump_fluence_mj_cm2: float
    single_pulse_threshold_mj_cm2: float
    total_fluence_over_single_pulse_threshold: float
    initial_delay_step_fs: float
    published_coherent_frequency_thz: float
    published_coherent_frequency_error_thz: float
    published_coherent_period_fs: float
    published_coherent_period_error_fs: float
    samples_per_published_period: float
    repetition_interval_us: float
    published_lifetime_upper_bound_us: float
    repetition_interval_not_longer_than_lifetime_bound: bool
    paper_fixed_pump_probe_delay_ps: float
    nexus_fixed_pump_probe_delay_ps: float
    fixed_delay_mismatch_ps: float
    summary_temperature_k: float
    nexus_internal_temperature_k: float
    temperature_mismatch_k: float
    analysis_code_and_fit_bounds_public: bool
    acquisition_level_lifetime_curve_public: bool
    public_raw_acquisition_traceable: bool
    provenance_metadata_consistent: bool
    exact_coherent_frequency_raw_refit_reproduced: bool
    persistence_lifetime_raw_refit_reproduced: bool
    pre_existing_material_phase_rearrangement_supported: bool
    new_material_species_created: bool
    clarus_field_observed: bool


@dataclass(frozen=True)
class ExternalFieldToMatterReanalysis:
    snapshot_sha256: str
    analogue_dce: AnalogueDCEReanalysis
    breit_wheeler: BreitWheelerReanalysis
    metastable_material_phase: MetastableMaterialPhaseReanalysis
    independent_external_measurements_reproduced: bool
    pump_to_nonclassical_photon_pairs_supported: bool
    electromagnetic_field_to_ordinary_massive_pairs_supported: bool
    observed_optical_pair_energy_ev: float
    electron_positron_rest_threshold_ev: float
    optical_pair_to_rest_threshold_energy_gap: float
    cms_mass_floor_to_optical_pair_energy_gap: float
    material_raw_acquisition_audited: bool
    metastable_phase_exact_raw_refit_reproduced: bool
    same_apparatus_end_to_end_chain_proven: bool
    clarus_field_observed: bool
    clarus_coupling_measured: bool
    new_material_species_created: bool
    pump_off_persistent_structure_proven: bool
    free_energy_or_vacuum_energy_extraction_proven: bool
    maximum_supported_claim: str
    unresolved_bridge: str


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    return value


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA-256 hex digest") from exc
    return value.lower()


def _md5(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 32:
        raise ValueError(f"{name} must be an MD5 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be an MD5 hex digest") from exc
    return value.lower()


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_external_field_to_matter_snapshot(
    path: str | Path,
    *,
    expected_sha256: str | None = EXPECTED_SNAPSHOT_SHA256,
) -> dict[str, Any]:
    snapshot_path = Path(path)
    observed_hash = sha256_file(snapshot_path)
    if expected_sha256 is not None:
        expected = _sha256(expected_sha256, name="expected_sha256")
        if observed_hash != expected:
            raise ValueError(
                "external snapshot hash mismatch: "
                f"expected {expected}, observed {observed_hash}"
            )
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("external snapshot must contain a JSON object")
    return payload


def _sideband_candidates_nm(parameters: Mapping[str, Any]) -> tuple[tuple[float, float], ...]:
    pump_nm = _positive(parameters.get("pump_wavelength_nm"), name="pump_wavelength_nm")
    beta2_ps2_km = _finite(
        parameters.get("average_beta2_ps2_per_km"),
        name="average_beta2_ps2_per_km",
    )
    beta4 = _finite(parameters.get("average_beta4_s4_per_m"), name="average_beta4_s4_per_m")
    period_m = _positive(
        parameters.get("fibre_modulation_period_m"),
        name="fibre_modulation_period_m",
    )
    harmonic = _integer(parameters.get("modulation_harmonic"), name="modulation_harmonic", minimum=1)

    beta2 = beta2_ps2_km * 1.0e-27
    quadratic = beta4 / 12.0
    linear = beta2
    constant = -harmonic * 2.0 * math.pi / period_m
    discriminant = linear * linear - 4.0 * quadratic * constant
    if quadratic == 0.0 or discriminant < 0.0:
        raise ValueError("DCE sideband equation does not have the required two finite roots")

    omega_p = 2.0 * math.pi * SPEED_OF_LIGHT_M_S / (pump_nm * 1.0e-9)
    candidates: list[tuple[float, float]] = []
    for sign in (-1.0, 1.0):
        delta_omega_squared = (-linear + sign * math.sqrt(discriminant)) / (
            2.0 * quadratic
        )
        if delta_omega_squared <= 0.0:
            continue
        delta_omega = math.sqrt(delta_omega_squared)
        if delta_omega >= omega_p:
            continue
        signal_nm = (
            2.0 * math.pi * SPEED_OF_LIGHT_M_S / (omega_p + delta_omega) * 1.0e9
        )
        idler_nm = (
            2.0 * math.pi * SPEED_OF_LIGHT_M_S / (omega_p - delta_omega) * 1.0e9
        )
        candidates.append((signal_nm, idler_nm))
    if not candidates:
        raise ValueError("DCE sideband equation has no physical positive-frequency pair")
    return tuple(candidates)


def _nearest_half_spacing(grid: tuple[float, ...], index: int) -> float:
    distances = [abs(grid[index] - value) for j, value in enumerate(grid) if j != index]
    if not distances:
        raise ValueError("a wavelength grid needs at least two points")
    return 0.5 * min(distances)


def reanalyse_analogue_dce(section: Mapping[str, Any]) -> AnalogueDCEReanalysis:
    source = _mapping(section.get("source"), name="analogue_dce.source")
    parameters = _mapping(
        section.get("paper_parameters"), name="analogue_dce.paper_parameters"
    )
    car_map = _mapping(section.get("figure_3_car_map"), name="analogue_dce.figure_3_car_map")
    figure4 = _sequence(
        section.get("figure_4_experimental_rows"),
        name="analogue_dce.figure_4_experimental_rows",
    )

    source_hash = _sha256(source.get("archive_sha256"), name="analogue_dce.archive_sha256")
    _sha256(source.get("figure_3_sha256"), name="analogue_dce.figure_3_sha256")
    _sha256(source.get("figure_4_sha256"), name="analogue_dce.figure_4_sha256")
    nominal_signal = _positive(
        parameters.get("predicted_signal_wavelength_nm"),
        name="predicted_signal_wavelength_nm",
    )
    nominal_idler = _positive(
        parameters.get("predicted_idler_wavelength_nm"),
        name="predicted_idler_wavelength_nm",
    )
    candidates = _sideband_candidates_nm(parameters)
    predicted_signal, predicted_idler = min(
        candidates,
        key=lambda pair: abs(pair[0] - nominal_signal) + abs(pair[1] - nominal_idler),
    )

    signal_grid = tuple(
        _positive(value, name=f"signal_wavelengths_nm[{index}]")
        for index, value in enumerate(
            _sequence(car_map.get("signal_wavelengths_nm"), name="signal_wavelengths_nm")
        )
    )
    if len(signal_grid) < 2 or any(
        right <= left for left, right in zip(signal_grid, signal_grid[1:])
    ):
        raise ValueError("signal wavelength grid must be strictly increasing")

    map_rows = _sequence(car_map.get("rows"), name="figure_3_car_map.rows")
    if len(map_rows) < 2:
        raise ValueError("CAR map needs at least two idler rows")
    idler_grid: list[float] = []
    maximum: tuple[float, int, int] | None = None
    for row_index, raw_row in enumerate(map_rows):
        row = _mapping(raw_row, name=f"figure_3_car_map.rows[{row_index}]")
        idler = _positive(
            row.get("idler_wavelength_nm"),
            name=f"rows[{row_index}].idler_wavelength_nm",
        )
        idler_grid.append(idler)
        car_values = _sequence(row.get("car"), name=f"rows[{row_index}].car")
        if len(car_values) != len(signal_grid):
            raise ValueError("every CAR-map row must match the signal wavelength grid")
        for column_index, raw_car in enumerate(car_values):
            car = _finite(raw_car, name=f"rows[{row_index}].car[{column_index}]")
            if maximum is None or car > maximum[0]:
                maximum = (car, row_index, column_index)
    if len(set(idler_grid)) != len(idler_grid) or maximum is None:
        raise ValueError("idler wavelengths must be unique and the CAR map nonempty")

    peak_car, peak_row, peak_column = maximum
    observed_signal = signal_grid[peak_column]
    observed_idler = idler_grid[peak_row]
    signal_detuning = observed_signal - predicted_signal
    idler_detuning = observed_idler - predicted_idler
    idler_tuple = tuple(idler_grid)
    sampled_cell_match = (
        abs(signal_detuning) <= _nearest_half_spacing(signal_grid, peak_column) + 1.0e-9
        and abs(idler_detuning) <= _nearest_half_spacing(idler_tuple, peak_row) + 1.0e-9
    )
    pump_nm = _positive(parameters.get("pump_wavelength_nm"), name="pump_wavelength_nm")
    energy_residual = (
        (1.0 / observed_signal + 1.0 / observed_idler) / (2.0 / pump_nm) - 1.0
    )

    correlated_points: list[tuple[float, float, float]] = []
    zero_car_controls: list[tuple[float, float]] = []
    for index, raw_row in enumerate(figure4):
        row = _mapping(raw_row, name=f"figure_4_experimental_rows[{index}]")
        _positive(row.get("pump_peak_power_w"), name=f"figure4[{index}].pump_peak_power_w")
        for key in ("car_240_ps", "car_240_ps_std", "car_1_7_ns", "car_1_7_ns_std"):
            _nonnegative(row.get(key), name=f"figure4[{index}].{key}")
        heralded_car = row.get("heralded_car")
        g2 = row.get("g2_zero")
        g2_error = row.get("g2_zero_err")
        values_are_null = heralded_car is None and g2 is None and g2_error is None
        if values_are_null:
            continue
        if heralded_car is None or g2 is None or g2_error is None:
            raise ValueError("heralded CAR, g2, and g2 error must be jointly present or null")
        parsed_car = _nonnegative(heralded_car, name=f"figure4[{index}].heralded_car")
        parsed_g2 = _nonnegative(g2, name=f"figure4[{index}].g2_zero")
        parsed_error = _positive(g2_error, name=f"figure4[{index}].g2_zero_err")
        if parsed_car > 1.0:
            correlated_points.append((parsed_car, parsed_g2, parsed_error))
        if parsed_car == 0.0:
            zero_car_controls.append((parsed_g2, parsed_error))
    if not correlated_points or len(zero_car_controls) != 1:
        raise ValueError("Figure 4 must contain correlated points and one zero-CAR control")

    best_car, best_g2, best_error = max(
        correlated_points,
        key=lambda values: (1.0 - values[1]) / values[2],
    )
    separation = (1.0 - best_g2) / best_error
    all_below = all(g2 + error < 1.0 for _, g2, error in correlated_points)
    control_g2, control_error = zero_car_controls[0]
    control_consistent = abs(control_g2 - 1.0) <= control_error
    supported = (
        sampled_cell_match
        and abs(energy_residual) < 5.0e-4
        and peak_car > 1.0
        and all_below
        and control_consistent
    )
    return AnalogueDCEReanalysis(
        source_archive_sha256=source_hash,
        source_is_processed_figure_data=True,
        predicted_signal_wavelength_nm=predicted_signal,
        predicted_idler_wavelength_nm=predicted_idler,
        observed_peak_signal_wavelength_nm=observed_signal,
        observed_peak_idler_wavelength_nm=observed_idler,
        observed_peak_car=peak_car,
        signal_detuning_nm=signal_detuning,
        idler_detuning_nm=idler_detuning,
        peak_is_nearest_sampled_cell_to_prediction=sampled_cell_match,
        inverse_wavelength_energy_residual=energy_residual,
        best_heralded_car=best_car,
        best_g2_zero=best_g2,
        best_g2_zero_error=best_error,
        reported_error_separation_below_one_sigma=separation,
        correlated_g2_point_count=len(correlated_points),
        all_correlated_g2_points_one_error_below_one=all_below,
        zero_car_control_g2=control_g2,
        zero_car_control_consistent_with_one=control_consistent,
        external_nonclassical_pair_emission_supported=supported,
        literal_moving_mirror_dce_proven=False,
        event_level_raw_counts_available=False,
        pump_powered_not_free_energy=True,
    )


def reanalyse_breit_wheeler(section: Mapping[str, Any]) -> BreitWheelerReanalysis:
    source = _mapping(section.get("source"), name="breit_wheeler.source")
    selection = _mapping(section.get("fiducial_selection"), name="fiducial_selection")
    published = _mapping(section.get("published_summary"), name="published_summary")
    bins = _sequence(section.get("figure_4_pair_pt_bins"), name="figure_4_pair_pt_bins")
    payload_hash = _sha256(
        source.get("hepdata_api_payload_sha256"),
        name="hepdata_api_payload_sha256",
    )
    if len(bins) < 2:
        raise ValueError("Breit-Wheeler pT table needs at least two bins")

    measured_integral = 0.0
    stat_variance = 0.0
    syst_variance = 0.0
    syst_linear = 0.0
    model_integrals = {"gamma_upc": 0.0, "superchic": 0.0, "starlight": 0.0}
    previous_high: float | None = None
    first_low: float | None = None
    for index, raw_bin in enumerate(bins):
        row = _mapping(raw_bin, name=f"figure_4_pair_pt_bins[{index}]")
        low = _nonnegative(row.get("low_gev"), name=f"bins[{index}].low_gev")
        high = _positive(row.get("high_gev"), name=f"bins[{index}].high_gev")
        if high <= low:
            raise ValueError("Breit-Wheeler bin high edge must exceed its low edge")
        if previous_high is not None and not math.isclose(
            low, previous_high, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError("Breit-Wheeler bins must be contiguous")
        if first_low is None:
            first_low = low
        previous_high = high
        width = high - low
        measured = _nonnegative(
            row.get("measured_microbarn_per_gev"),
            name=f"bins[{index}].measured_microbarn_per_gev",
        )
        stat = _nonnegative(
            row.get("stat_microbarn_per_gev"),
            name=f"bins[{index}].stat_microbarn_per_gev",
        )
        syst = _nonnegative(
            row.get("syst_microbarn_per_gev"),
            name=f"bins[{index}].syst_microbarn_per_gev",
        )
        measured_integral += width * measured
        stat_variance += (width * stat) ** 2
        syst_variance += (width * syst) ** 2
        syst_linear += width * syst
        for model in model_integrals:
            density = _nonnegative(
                row.get(f"{model}_microbarn_per_gev"),
                name=f"bins[{index}].{model}_microbarn_per_gev",
            )
            model_integrals[model] += width * density

    pair_pt_max = _positive(selection.get("pair_pt_max_gev"), name="pair_pt_max_gev")
    if first_low != 0.0 or previous_high is None or not math.isclose(
        previous_high, pair_pt_max, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("Breit-Wheeler bins must cover the complete published pT interval")

    integrated_stat = math.sqrt(stat_variance)
    integrated_syst_floor = math.sqrt(syst_variance)
    published_cross_section = _positive(
        published.get("fiducial_cross_section_microbarn"),
        name="published fiducial cross section",
    )
    published_stat = _positive(
        published.get("stat_uncertainty_microbarn"),
        name="published statistical uncertainty",
    )
    published_syst = _positive(
        published.get("syst_uncertainty_microbarn"),
        name="published systematic uncertainty",
    )
    selected_pairs = _integer(
        published.get("exclusive_pair_candidates"),
        name="exclusive_pair_candidates",
        minimum=1,
    )
    central_reproduced = abs(measured_integral - published_cross_section) <= 0.2
    stat_reproduced = abs(integrated_stat - published_stat) <= 0.1
    syst_compatible = integrated_syst_floor <= published_syst <= syst_linear
    total_published_error = math.hypot(published_stat, published_syst)

    comparisons = tuple(
        ModelIntegralComparison(
            model=model,
            integrated_cross_section_microbarn=value,
            difference_from_measurement_microbarn=value - measured_integral,
            relative_difference=(value - measured_integral) / measured_integral,
            difference_over_published_total_error=(value - measured_integral)
            / total_published_error,
        )
        for model, value in model_integrals.items()
    )
    comparison_by_name = {comparison.model: comparison for comparison in comparisons}
    charged_models_closer = (
        abs(comparison_by_name["gamma_upc"].relative_difference)
        < abs(comparison_by_name["starlight"].relative_difference)
        and abs(comparison_by_name["superchic"].relative_difference)
        < abs(comparison_by_name["starlight"].relative_difference)
    )
    mass_floor = _positive(selection.get("pair_mass_min_gev"), name="pair_mass_min_gev")
    supported = (
        central_reproduced
        and stat_reproduced
        and syst_compatible
        and charged_models_closer
        and selected_pairs > 10_000
        and published_cross_section / total_published_error > 10.0
    )
    return BreitWheelerReanalysis(
        hepdata_payload_sha256=payload_hash,
        bin_count=len(bins),
        integrated_cross_section_microbarn=measured_integral,
        integrated_stat_uncertainty_microbarn=integrated_stat,
        integrated_syst_uncorrelated_floor_microbarn=integrated_syst_floor,
        integrated_syst_fully_correlated_ceiling_microbarn=syst_linear,
        published_cross_section_microbarn=published_cross_section,
        published_stat_uncertainty_microbarn=published_stat,
        published_syst_uncertainty_microbarn=published_syst,
        central_value_reproduced=central_reproduced,
        statistical_uncertainty_reproduced=stat_reproduced,
        published_systematic_compatible_with_unknown_covariance=syst_compatible,
        descriptive_zero_separation_sigma=published_cross_section / total_published_error,
        selected_pair_count=selected_pairs,
        selected_mass_floor_over_rest_mass_threshold=mass_floor / (2.0 * ELECTRON_MASS_GEV),
        model_comparisons=comparisons,
        charged_form_factor_models_closer_than_starlight=charged_models_closer,
        external_quasi_real_photon_pair_production_supported=supported,
        ideal_free_on_shell_two_photon_collision_proven=False,
        detector_event_level_data_public=False,
        persistent_material_created=False,
    )


def reanalyse_metastable_material_phase(
    section: Mapping[str, Any],
) -> MetastableMaterialPhaseReanalysis:
    source = _mapping(section.get("source"), name="metastable_material_phase.source")
    acquisition = _mapping(
        section.get("figure_3_acquisition"),
        name="metastable_material_phase.figure_3_acquisition",
    )
    conflicts = _mapping(
        section.get("provenance_conflicts"),
        name="metastable_material_phase.provenance_conflicts",
    )

    summary_md5 = _md5(source.get("summary_xlsx_md5"), name="summary_xlsx_md5")
    figure_3_md5 = _md5(
        source.get("figure_3_nexus_md5"), name="figure_3_nexus_md5"
    )
    static_c_md5 = _md5(
        source.get("static_c_phase_nexus_md5"),
        name="static_c_phase_nexus_md5",
    )
    for key in (
        "dataset_total_bytes",
        "summary_xlsx_bytes",
        "figure_3_nexus_bytes",
        "static_c_phase_nexus_bytes",
    ):
        _integer(source.get(key), name=key, minimum=1)

    raw_shape = _sequence(acquisition.get("intensity_shape"), name="intensity_shape")
    if len(raw_shape) != 3:
        raise ValueError("intensity_shape must contain exactly three dimensions")
    shape = tuple(
        _integer(value, name=f"intensity_shape[{index}]", minimum=1)
        for index, value in enumerate(raw_shape)
    )
    pump_energy = _positive(
        acquisition.get("pump_photon_energy_ev"), name="pump_photon_energy_ev"
    )
    pump_1 = _nonnegative(
        acquisition.get("pump_1_fluence_mj_cm2"), name="pump_1_fluence_mj_cm2"
    )
    pump_2 = _nonnegative(
        acquisition.get("pump_2_fluence_mj_cm2"), name="pump_2_fluence_mj_cm2"
    )
    threshold = _positive(
        acquisition.get("single_pulse_threshold_mj_cm2"),
        name="single_pulse_threshold_mj_cm2",
    )
    delay_step = _positive(
        acquisition.get("initial_delay_step_fs"), name="initial_delay_step_fs"
    )
    frequency = _positive(
        acquisition.get("published_coherent_frequency_thz"),
        name="published_coherent_frequency_thz",
    )
    frequency_error = _positive(
        acquisition.get("published_coherent_frequency_error_thz"),
        name="published_coherent_frequency_error_thz",
    )
    if frequency_error >= frequency:
        raise ValueError("coherent-frequency error must be smaller than the frequency")
    repetition_rate = _positive(
        acquisition.get("laser_repetition_rate_hz"),
        name="laser_repetition_rate_hz",
    )
    lifetime_bound = _positive(
        acquisition.get("published_160k_lifetime_upper_bound_us"),
        name="published_160k_lifetime_upper_bound_us",
    )

    paper_delay = _positive(
        conflicts.get("paper_fixed_pump_probe_delay_ps"),
        name="paper_fixed_pump_probe_delay_ps",
    )
    nexus_delay = _positive(
        conflicts.get("supplement_and_nexus_fixed_pump_probe_delay_ps"),
        name="supplement_and_nexus_fixed_pump_probe_delay_ps",
    )
    summary_temperature = _positive(
        conflicts.get("summary_temperature_k"), name="summary_temperature_k"
    )
    nexus_temperature = _positive(
        conflicts.get("nexus_internal_temperature_k"),
        name="nexus_internal_temperature_k",
    )
    analysis_public = _boolean(
        conflicts.get("analysis_code_and_fit_bounds_public"),
        name="analysis_code_and_fit_bounds_public",
    )
    lifetime_curve_public = _boolean(
        conflicts.get("acquisition_level_lifetime_curve_public"),
        name="acquisition_level_lifetime_curve_public",
    )

    total_fluence = pump_1 + pump_2
    period_fs = 1_000.0 / frequency
    period_error_fs = 1_000.0 * frequency_error / (frequency * frequency)
    repetition_interval_us = 1.0e6 / repetition_rate
    metadata_consistent = math.isclose(
        paper_delay, nexus_delay, rel_tol=0.0, abs_tol=1.0e-12
    ) and math.isclose(
        summary_temperature,
        nexus_temperature,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    acquisition_traceable = (
        shape == (256, 344, 45)
        and total_fluence > threshold
        and period_fs / delay_step >= 10.0
        and repetition_interval_us <= lifetime_bound
    )

    return MetastableMaterialPhaseReanalysis(
        summary_xlsx_md5=summary_md5,
        figure_3_nexus_md5=figure_3_md5,
        static_c_phase_nexus_md5=static_c_md5,
        intensity_shape=(shape[0], shape[1], shape[2]),
        pump_photon_energy_ev=pump_energy,
        total_pump_fluence_mj_cm2=total_fluence,
        single_pulse_threshold_mj_cm2=threshold,
        total_fluence_over_single_pulse_threshold=total_fluence / threshold,
        initial_delay_step_fs=delay_step,
        published_coherent_frequency_thz=frequency,
        published_coherent_frequency_error_thz=frequency_error,
        published_coherent_period_fs=period_fs,
        published_coherent_period_error_fs=period_error_fs,
        samples_per_published_period=period_fs / delay_step,
        repetition_interval_us=repetition_interval_us,
        published_lifetime_upper_bound_us=lifetime_bound,
        repetition_interval_not_longer_than_lifetime_bound=(
            repetition_interval_us <= lifetime_bound
        ),
        paper_fixed_pump_probe_delay_ps=paper_delay,
        nexus_fixed_pump_probe_delay_ps=nexus_delay,
        fixed_delay_mismatch_ps=abs(paper_delay - nexus_delay),
        summary_temperature_k=summary_temperature,
        nexus_internal_temperature_k=nexus_temperature,
        temperature_mismatch_k=abs(summary_temperature - nexus_temperature),
        analysis_code_and_fit_bounds_public=analysis_public,
        acquisition_level_lifetime_curve_public=lifetime_curve_public,
        public_raw_acquisition_traceable=acquisition_traceable,
        provenance_metadata_consistent=metadata_consistent,
        exact_coherent_frequency_raw_refit_reproduced=False,
        persistence_lifetime_raw_refit_reproduced=False,
        pre_existing_material_phase_rearrangement_supported=True,
        new_material_species_created=False,
        clarus_field_observed=False,
    )


def build_external_field_to_matter_reanalysis(
    snapshot: Mapping[str, Any],
    *,
    snapshot_sha256: str = EXPECTED_SNAPSHOT_SHA256,
) -> ExternalFieldToMatterReanalysis:
    if snapshot.get("schema") != "clarus.external_field_to_matter.v1":
        raise ValueError("unsupported external field-to-matter snapshot schema")
    locked_hash = _sha256(snapshot_sha256, name="snapshot_sha256")
    analogue = reanalyse_analogue_dce(
        _mapping(snapshot.get("analogue_dce"), name="analogue_dce")
    )
    pair_section = _mapping(snapshot.get("breit_wheeler"), name="breit_wheeler")
    pair = reanalyse_breit_wheeler(pair_section)
    material = reanalyse_metastable_material_phase(
        _mapping(
            snapshot.get("metastable_material_phase"),
            name="metastable_material_phase",
        )
    )
    reproduced = (
        analogue.external_nonclassical_pair_emission_supported
        and pair.external_quasi_real_photon_pair_production_supported
    )
    photon_energy_ev_nm = PLANCK_CONSTANT_EV_S * SPEED_OF_LIGHT_M_S * 1.0e9
    optical_pair_energy_ev = photon_energy_ev_nm * (
        1.0 / analogue.observed_peak_signal_wavelength_nm
        + 1.0 / analogue.observed_peak_idler_wavelength_nm
    )
    rest_threshold_ev = 2.0 * ELECTRON_MASS_GEV * 1.0e9
    pair_selection = _mapping(
        pair_section.get("fiducial_selection"), name="fiducial_selection"
    )
    pair_mass_floor_gev = _positive(
        pair_selection.get("pair_mass_min_gev"),
        name="pair_mass_min_gev",
    )
    return ExternalFieldToMatterReanalysis(
        snapshot_sha256=locked_hash,
        analogue_dce=analogue,
        breit_wheeler=pair,
        metastable_material_phase=material,
        independent_external_measurements_reproduced=reproduced,
        pump_to_nonclassical_photon_pairs_supported=(
            analogue.external_nonclassical_pair_emission_supported
        ),
        electromagnetic_field_to_ordinary_massive_pairs_supported=(
            pair.external_quasi_real_photon_pair_production_supported
        ),
        observed_optical_pair_energy_ev=optical_pair_energy_ev,
        electron_positron_rest_threshold_ev=rest_threshold_ev,
        optical_pair_to_rest_threshold_energy_gap=rest_threshold_ev / optical_pair_energy_ev,
        cms_mass_floor_to_optical_pair_energy_gap=(
            pair_mass_floor_gev * 1.0e9 / optical_pair_energy_ev
        ),
        material_raw_acquisition_audited=material.public_raw_acquisition_traceable,
        metastable_phase_exact_raw_refit_reproduced=(
            material.exact_coherent_frequency_raw_refit_reproduced
            and material.persistence_lifetime_raw_refit_reproduced
        ),
        same_apparatus_end_to_end_chain_proven=False,
        clarus_field_observed=False,
        clarus_coupling_measured=False,
        new_material_species_created=False,
        pump_off_persistent_structure_proven=False,
        free_energy_or_vacuum_energy_extraction_proven=False,
        maximum_supported_claim=(
            "Independent public measurements support pump-powered, wavelength-selective "
            "non-classical photon-pair emission and standard-QED conversion of quasi-real "
            "electromagnetic fields into ordinary massive lepton pairs. Public raw "
            "1T-TaS2 acquisition parameters also support a pump-driven rearrangement "
            "of a pre-existing material, but not an exact independent phase refit."
        ),
        unresolved_bridge=(
            "No experiment in this package couples a defined Clarus field to a product, "
            "runs the photon-generation and massive-pair stages in one apparatus, or "
            "shows a transferable pump-off material phase or a new mass pole."
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-external-reproduction", action="store_true")
    args = parser.parse_args(argv)
    snapshot = load_external_field_to_matter_snapshot(args.snapshot)
    report = build_external_field_to_matter_reanalysis(snapshot)
    payload = json.dumps(asdict(report), ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return int(
        args.require_external_reproduction
        and not report.independent_external_measurements_reproduced
    )


__all__ = [
    "AnalogueDCEReanalysis",
    "BreitWheelerReanalysis",
    "EXPECTED_SNAPSHOT_SHA256",
    "ExternalFieldToMatterReanalysis",
    "MetastableMaterialPhaseReanalysis",
    "ModelIntegralComparison",
    "build_external_field_to_matter_reanalysis",
    "load_external_field_to_matter_snapshot",
    "main",
    "reanalyse_analogue_dce",
    "reanalyse_breit_wheeler",
    "reanalyse_metastable_material_phase",
    "sha256_file",
]
