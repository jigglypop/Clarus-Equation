from __future__ import annotations

from examples.physics.cosmology_ratio_audit import (
    CE_RATIOS,
    compare_all_density_ratios,
    coverage_verdict,
)


def test_ce_density_ratios_are_close_to_recent_cmb_compressed_sets() -> None:
    comparisons = compare_all_density_ratios()

    assert len(comparisons) >= 4
    assert all(comparison.max_abs_relative_error < 0.04 for comparison in comparisons)


def test_ce_baryon_ratio_stays_near_observed_baryon_fraction() -> None:
    comparisons = compare_all_density_ratios()

    assert abs(CE_RATIOS["omega_b"] - 0.0486) < 2.0e-4
    assert max(abs(comparison.omega_b_diff) for comparison in comparisons) < 0.0012


def test_modern_likelihood_physics_is_not_implemented_by_ratio_audit() -> None:
    verdict = coverage_verdict()

    assert verdict.density_ratios_close
    assert not verdict.has_background_expansion_model
    assert not verdict.has_growth_model_for_s8
    assert not verdict.has_particle_dark_matter_model
    assert not verdict.has_detector_likelihood
    assert verdict.summary == "density ratios match; modern likelihood physics not implemented"
