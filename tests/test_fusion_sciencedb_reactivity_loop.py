from __future__ import annotations

from pathlib import Path
import shutil

import pytest

import reality_stone.clarus.fusion_sciencedb_reactivity_loop as reactivity_loop
from reality_stone.clarus.fusion_sciencedb_payload_loop import (
    SCIENCEDB_REPOSITORY_RELATIVE_DIRECTORY,
)
from reality_stone.clarus.fusion_sciencedb_reactivity_loop import (
    DT_CROSS_SECTION_EXPECTED_SHA256,
    DT_CROSS_SECTION_FILENAME,
    S_FACTOR_LINEAR,
    S_FACTOR_LOG_LINEAR,
    SIGMA_LINEAR,
    SIGMA_LOG_LOG,
    ScienceDBReactivityIntegrityError,
    audit_sciencedb_dt_reactivity,
    current_sciencedb_dt_reactivity_audit,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PAYLOAD = REPOSITORY_ROOT / SCIENCEDB_REPOSITORY_RELATIVE_DIRECTORY


@pytest.fixture(scope="module")
def audit():
    return current_sciencedb_dt_reactivity_audit()


def _copy_payload(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    destination = root / SCIENCEDB_REPOSITORY_RELATIVE_DIRECTORY
    destination.parent.mkdir(parents=True)
    shutil.copytree(SOURCE_PAYLOAD, destination)
    return root, destination


def test_full_payload_integrity_precedes_exact_dt_table_parse(audit) -> None:
    assert audit.payload_audit.payload_integrity_gate_pass
    assert audit.payload_integrity_verified_before_dt_parse
    assert audit.dt_table_parsed_from_integrity_verified_raw_bytes
    assert audit.dt_cross_section_filename == DT_CROSS_SECTION_FILENAME
    assert audit.dt_cross_section_expected_sha256 == DT_CROSS_SECTION_EXPECTED_SHA256
    assert audit.dt_cross_section_runtime_sha256 == DT_CROSS_SECTION_EXPECTED_SHA256
    assert audit.dt_table_row_count == 54


def test_deuteron_lab_to_cm_conversion_is_explicit_and_grid_is_contained(audit) -> None:
    expected_factor = audit.triton_mass_mev / (audit.deuteron_mass_mev + audit.triton_mass_mev)
    assert audit.deuteron_lab_to_cm_energy_factor == pytest.approx(
        expected_factor,
        rel=1.0e-15,
    )
    assert audit.deuteron_lab_to_cm_energy_factor == pytest.approx(
        0.599615903669184,
        rel=1.0e-15,
    )
    assert audit.table_lab_energy_min_mev == pytest.approx(1.0e-4)
    assert audit.table_lab_energy_max_mev == pytest.approx(30.0)
    assert audit.table_cm_energy_min_kev == pytest.approx(0.0599615903669184)
    assert audit.table_cm_energy_max_kev == pytest.approx(17_988.47711007552)
    assert audit.integration_min_energy_kev == 0.5
    assert audit.integration_max_energy_kev == 550.0
    assert audit.integration_grid_inside_table_domain


def test_10kev_maxwellian_values_are_reproduced_for_all_four_controls(audit) -> None:
    assert audit.temperature_kev == 10.0
    assert audit.energy_grid_points == 4_001
    expected = {
        SIGMA_LOG_LOG: 1.1159902109009393e-16,
        SIGMA_LINEAR: 1.1428935248009758e-16,
        S_FACTOR_LOG_LINEAR: 1.1301177307995228e-16,
        S_FACTOR_LINEAR: 1.1398618033626703e-16,
    }
    assert tuple(item.method for item in audit.interpolation_envelopes) == tuple(expected)
    for envelope in audit.interpolation_envelopes:
        assert envelope.central_reactivity_cm3_s == pytest.approx(
            expected[envelope.method],
            rel=2.0e-13,
        )


def test_fully_correlated_pointwise_err_endpoint_envelopes_are_explicit(audit) -> None:
    expected_endpoints = {
        SIGMA_LOG_LOG: (1.1122332754784632e-16, 1.1197471242186420e-16),
        SIGMA_LINEAR: (1.1390548064126958e-16, 1.1467322431892553e-16),
        S_FACTOR_LOG_LINEAR: (1.1263136927570176e-16, 1.1339217466211774e-16),
        S_FACTOR_LINEAR: (1.1360407326924747e-16, 1.1436828740328658e-16),
    }
    for envelope in audit.interpolation_envelopes:
        lower, upper = expected_endpoints[envelope.method]
        assert envelope.all_points_minus_err_reactivity_cm3_s == pytest.approx(
            lower,
            rel=2.0e-13,
        )
        assert envelope.all_points_plus_err_reactivity_cm3_s == pytest.approx(
            upper,
            rel=2.0e-13,
        )
        assert (
            envelope.all_points_minus_err_reactivity_cm3_s
            < envelope.central_reactivity_cm3_s
            < envelope.all_points_plus_err_reactivity_cm3_s
        )
        assert envelope.pointwise_err_shift_is_fully_correlated_endpoint_control
        assert not envelope.covariance_confidence_interval_derived

    assert audit.conservative_method_and_err_lower_cm3_s == pytest.approx(
        expected_endpoints[SIGMA_LOG_LOG][0]
    )
    assert audit.conservative_method_and_err_upper_cm3_s == pytest.approx(
        expected_endpoints[SIGMA_LINEAR][1]
    )


def test_bosch_hale_comparison_uses_closed_fit_and_same_kernel_control(audit) -> None:
    assert audit.bosch_hale_closed_reactivity_cm3_s == pytest.approx(
        1.1361654705836233e-16,
        rel=2.0e-13,
    )
    assert audit.bosch_hale_same_kernel_reactivity_cm3_s == pytest.approx(
        1.1418109771972635e-16,
        rel=2.0e-13,
    )
    assert audit.bosch_hale_same_kernel_to_closed_ratio == pytest.approx(
        1.004969,
        rel=2.0e-6,
    )
    assert audit.sigma_log_log.central_to_bosch_hale_closed_ratio == pytest.approx(
        0.982242674852352,
        rel=2.0e-13,
    )
    assert audit.sigma_linear.central_to_bosch_hale_closed_ratio == pytest.approx(
        1.0059217203757271,
        rel=2.0e-13,
    )


def test_interpolation_choice_alone_exceeds_one_percent(audit) -> None:
    assert audit.sigma_interpolation_relative_spread == pytest.approx(
        0.024107123554710563,
        rel=2.0e-13,
    )
    assert audit.s_factor_interpolation_relative_spread == pytest.approx(
        0.008622174750106515,
        rel=2.0e-13,
    )
    assert audit.all_method_central_relative_spread == pytest.approx(
        audit.sigma_interpolation_relative_spread
    )
    assert not audit.interpolation_spread_below_one_percent


def test_energy_grid_refinement_is_far_below_the_interpolation_spread(audit) -> None:
    assert audit.refined_energy_grid_points == 8_001
    assert audit.grid_refinement_max_relative_residual == pytest.approx(
        5.189090000156922e-7,
        rel=2.0e-8,
    )
    assert audit.grid_refinement_tolerance == 1.0e-5
    assert audit.grid_refinement_gate_pass
    assert audit.grid_refinement_max_relative_residual < (
        audit.all_method_central_relative_spread / 10_000.0
    )


def test_covariance_spin_and_physical_one_percent_gates_remain_fail_closed(audit) -> None:
    assert audit.only_pointwise_scalar_err_available
    assert not audit.numeric_covariance_matrix_available
    assert not audit.initial_state_spin_operator_available
    assert not audit.unpolarized_sub_one_percent_certification_gate_pass
    assert not audit.physical_state_resolved_one_percent_branch_gate_pass
    assert audit.maximum_supported_stage == (
        "integrity-pinned unpolarized Maxwellian sensitivity control"
    )
    assert "fail closed" in audit.status


def test_any_payload_tamper_blocks_dt_parse(monkeypatch, tmp_path: Path) -> None:
    root, payload_dir = _copy_payload(tmp_path)
    unrelated_to_dt_table = payload_dir / "4He(n,d)T-CS.txt"
    payload = unrelated_to_dt_table.read_bytes()
    unrelated_to_dt_table.write_bytes(payload[:-1] + bytes((payload[-1] ^ 1,)))

    parse_called = False

    def forbidden_parse(_raw_bytes: bytes):
        nonlocal parse_called
        parse_called = True
        raise AssertionError("D-T parser ran before full payload integrity")

    monkeypatch.setattr(reactivity_loop, "_parse_dt_table", forbidden_parse)
    with pytest.raises(ScienceDBReactivityIntegrityError, match="before D-T table parsing"):
        audit_sciencedb_dt_reactivity(repository_root=root)
    assert not parse_called


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("temperature_kev", 0.1, "outside"),
        ("temperature_kev", float("nan"), "positive and finite"),
        ("temperature_kev", True, "real number"),
        ("energy_grid_points", 100, "at least 101"),
        ("energy_grid_points", 1_000.5, "integer"),
        ("energy_grid_points", True, "integer"),
    ],
)
def test_invalid_numeric_inputs_are_rejected(keyword: str, value, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        audit_sciencedb_dt_reactivity(**{keyword: value})


@pytest.mark.parametrize(
    "relative_path",
    ["../outside", "C:/outside", "C:outside", "a\\b", "a//b", "a/../b"],
)
def test_unsafe_payload_directory_fails_before_parse(relative_path: str) -> None:
    with pytest.raises(ScienceDBReactivityIntegrityError, match="before D-T table parsing"):
        audit_sciencedb_dt_reactivity(repository_relative_directory=relative_path)
