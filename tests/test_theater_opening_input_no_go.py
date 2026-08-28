import pytest

from examples.physics.theater_opening_input_no_go import (
    theater_opening_input_no_go,
)


def test_equal_endpoint_record_has_two_exact_tanh_spectra() -> None:
    audit = theater_opening_input_no_go()

    assert audit.same_endpoint_record
    assert audit.short_beta_squared == pytest.approx(4.0 / 21.0, abs=2.0e-15)
    assert audit.long_beta_squared == pytest.approx(16.0 / 273.0, abs=2.0e-15)
    assert audit.short_beta_squared > audit.long_beta_squared
    assert not audit.unique_quench_spectrum_follows


def test_initial_state_changes_created_abundance_at_fixed_quench() -> None:
    audit = theater_opening_input_no_go()

    assert audit.occupied_band_stimulation_factor == 7.0
    assert audit.occupied_band_created_occupation == pytest.approx(
        7.0 * audit.vacuum_created_occupation,
        abs=2.0e-15,
    )
    assert audit.compact_support_state_changes_integrated_abundance
    assert not audit.unique_abundance_follows
    assert audit.status == "ZEROD_ENDPOINT_TO_UNIQUE_OPENING_ABUNDANCE_DISPROVED"
    assert audit.claim_ceiling == (
        "COMPLETE_EXACT_TANH_AND_INITIAL_STATE_COUNTEREXAMPLE"
    )
