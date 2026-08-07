from __future__ import annotations

import math

import pytest

from examples.physics.ce_residual_forward_model import (
    CEForwardParams,
    RECOMBINATION_REDSHIFT_UNIT,
    RECOMBINATION_XE_CONVENTION,
    RecombinationHistoryMetadata,
    baryon_drag_inertia_ratio,
    drag_optical_depth_benchmark,
    drag_optical_depth_rate_per_redshift,
    early_hubble_rate_s_inverse,
    hydrogen_nuclei_number_density_today_m3,
    load_recombination_history_table,
    parse_recombination_history_bytes,
    recombination_history_file_sha256,
    sha256_hexdigest,
    sound_horizon_at_redshift_mpc,
)


Y_P = 0.245


def _metadata(
    params: CEForwardParams,
    *,
    omega_b_h2: float | None = None,
) -> RecombinationHistoryMetadata:
    return RecombinationHistoryMetadata(
        solver_family="CLASS",
        solver_version="manufactured-test-1",
        recombination_backend="analytic x_e fixture, not a solver",
        source_label="manufactured CLASS-like z,x_e table",
        redshift_unit=RECOMBINATION_REDSHIFT_UNIT,
        electron_fraction_convention=RECOMBINATION_XE_CONVENTION,
        helium_mass_fraction_y_p=Y_P,
        h0_km_s_mpc=params.h0,
        omega_b_h2=params.omega_b_h2 if omega_b_h2 is None else omega_b_h2,
        omega_m_h2=params.omega_m_h2,
        tcmb_k=params.tcmb_k,
        n_eff=params.n_eff,
    )


def _linear_rate_payload(
    params: CEForwardParams,
    *,
    target_z_drag: float,
    redshifts: tuple[float, ...],
    descending: bool = False,
) -> bytes:
    """Manufacture x_e so d(tau_drag)/dz=K(1+z) at every table node."""
    coefficient = 1.0 / (
        target_z_drag + 0.5 * target_z_drag * target_z_drag
    )
    rows = []
    for z_value in redshifts:
        unit_rate = drag_optical_depth_rate_per_redshift(
            z_value,
            1.0,
            params,
            Y_P,
        )
        x_e = coefficient * (1.0 + z_value) / unit_rate
        rows.append(f"{z_value:.17g} {x_e:.17g}")
    if descending:
        rows.reverse()
    return ("# z x_e=n_e/n_H; manufactured analytic benchmark\n" + "\n".join(rows) + "\n").encode(
        "utf-8"
    )


def test_drag_rate_point_value_locks_class_convention_and_si_units() -> None:
    params = CEForwardParams()

    assert math.isclose(
        hydrogen_nuclei_number_density_today_m3(params, Y_P),
        0.18746908014092686,
        rel_tol=1e-14,
    )
    assert math.isclose(
        baryon_drag_inertia_ratio(1000.0, params),
        0.6713590540686533,
        rel_tol=1e-14,
    )
    assert math.isclose(
        early_hubble_rate_s_inverse(1000.0, params),
        4.391583072970779e-14,
        rel_tol=1e-14,
    )
    assert math.isclose(
        drag_optical_depth_rate_per_redshift(
            1000.0,
            0.1,
            params,
            Y_P,
        ),
        0.01270646494695847,
        rel_tol=1e-14,
    )


def test_hashed_history_recovers_analytic_tau_crossing_and_sound_horizon() -> None:
    params = CEForwardParams()
    target_z_drag = 1059.25
    redshifts = tuple(float(z_value) for z_value in range(1601))
    payload = _linear_rate_payload(
        params,
        target_z_drag=target_z_drag,
        redshifts=redshifts,
    )
    digest = sha256_hexdigest(payload)
    history = parse_recombination_history_bytes(
        payload,
        metadata=_metadata(params),
        expected_sha256=digest,
        redshift_column=0,
        electron_fraction_column=1,
    )

    benchmark = drag_optical_depth_benchmark(history, params)
    coefficient = 1.0 / (
        target_z_drag + 0.5 * target_z_drag * target_z_drag
    )
    expected_tau_max = coefficient * (1600.0 + 0.5 * 1600.0**2)

    assert history.source_sha256 == digest
    assert history.original_grid_order == "ascending"
    assert benchmark.history_metadata == history.metadata
    assert benchmark.input_redshift_column == 0
    assert benchmark.input_electron_fraction_column == 1
    assert benchmark.input_delimiter is None
    assert math.isclose(benchmark.z_drag, target_z_drag, abs_tol=2e-10)
    assert math.isclose(benchmark.tau_drag_at_z_max, expected_tau_max, rel_tol=2e-14)
    assert math.isclose(
        benchmark.rd_mpc,
        sound_horizon_at_redshift_mpc(params, target_z_drag),
        rel_tol=2e-14,
    )
    assert math.isclose(benchmark.rd_mpc, 147.649769047, rel_tol=1e-11)
    assert benchmark.crossing_bracket == (1059.0, 1060.0)
    assert benchmark.crossing_bracket_width == 1.0
    assert benchmark.rd_unit == "Mpc_comoving"
    assert all(
        tau_next >= tau_now
        for tau_now, tau_next in zip(
            benchmark.tau_drag_grid,
            benchmark.tau_drag_grid[1:],
        )
    )
    assert "tau_drag(z_d)=1" in benchmark.convention
    assert "not itself a CLASS/CAMB/HyRec" in benchmark.status


def test_descending_solver_grid_and_file_loader_are_order_invariant(tmp_path) -> None:
    params = CEForwardParams()
    target_z_drag = 1059.25
    redshifts = tuple(float(z_value) for z_value in range(1601))
    ascending_payload = _linear_rate_payload(
        params,
        target_z_drag=target_z_drag,
        redshifts=redshifts,
    )
    descending_payload = _linear_rate_payload(
        params,
        target_z_drag=target_z_drag,
        redshifts=redshifts,
        descending=True,
    )
    history_path = tmp_path / "class_thermodynamics.dat"
    history_path.write_bytes(descending_payload)
    descending_history = load_recombination_history_table(
        history_path,
        metadata=_metadata(params),
        expected_sha256=recombination_history_file_sha256(history_path),
        redshift_column=0,
        electron_fraction_column=1,
    )
    ascending_history = parse_recombination_history_bytes(
        ascending_payload,
        metadata=_metadata(params),
        expected_sha256=sha256_hexdigest(ascending_payload),
        redshift_column=0,
        electron_fraction_column=1,
    )

    descending_result = drag_optical_depth_benchmark(descending_history, params)
    ascending_result = drag_optical_depth_benchmark(ascending_history, params)

    assert descending_history.original_grid_order == "descending"
    assert descending_history.redshift == ascending_history.redshift
    assert descending_history.electron_fraction == ascending_history.electron_fraction
    assert math.isclose(
        descending_result.z_drag,
        ascending_result.z_drag,
        abs_tol=1e-12,
    )
    assert math.isclose(
        descending_result.rd_mpc,
        ascending_result.rd_mpc,
        rel_tol=1e-14,
    )


def test_history_adapter_rejects_hash_units_grid_and_cosmology_mismatches() -> None:
    params = CEForwardParams()
    valid_payload = b"0 1\n500 0.1\n1000 0.01\n1500 1\n"

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        parse_recombination_history_bytes(
            valid_payload,
            metadata=_metadata(params),
            expected_sha256="0" * 64,
            redshift_column=0,
            electron_fraction_column=1,
        )

    with pytest.raises(ValueError, match="redshift_unit"):
        RecombinationHistoryMetadata(
            solver_family="CLASS",
            solver_version="test",
            recombination_backend="test",
            source_label="test",
            redshift_unit="Mpc",
            electron_fraction_convention=RECOMBINATION_XE_CONVENTION,
            helium_mass_fraction_y_p=Y_P,
            h0_km_s_mpc=params.h0,
            omega_b_h2=params.omega_b_h2,
            omega_m_h2=params.omega_m_h2,
            tcmb_k=params.tcmb_k,
            n_eff=params.n_eff,
        )

    with pytest.raises(ValueError, match="electron_fraction_convention"):
        RecombinationHistoryMetadata(
            solver_family="CLASS",
            solver_version="test",
            recombination_backend="test",
            source_label="test",
            redshift_unit=RECOMBINATION_REDSHIFT_UNIT,
            electron_fraction_convention="free_electrons_per_baryon",
            helium_mass_fraction_y_p=Y_P,
            h0_km_s_mpc=params.h0,
            omega_b_h2=params.omega_b_h2,
            omega_m_h2=params.omega_m_h2,
            tcmb_k=params.tcmb_k,
            n_eff=params.n_eff,
        )

    nonmonotonic_payload = b"0 1\n500 0.1\n400 0.2\n1500 1\n"
    with pytest.raises(ValueError, match="strictly monotonic"):
        parse_recombination_history_bytes(
            nonmonotonic_payload,
            metadata=_metadata(params),
            expected_sha256=sha256_hexdigest(nonmonotonic_payload),
            redshift_column=0,
            electron_fraction_column=1,
        )

    missing_zero_payload = b"1 1\n500 0.1\n1000 0.01\n1500 1\n"
    with pytest.raises(ValueError, match="include z=0"):
        parse_recombination_history_bytes(
            missing_zero_payload,
            metadata=_metadata(params),
            expected_sha256=sha256_hexdigest(missing_zero_payload),
            redshift_column=0,
            electron_fraction_column=1,
        )

    history = parse_recombination_history_bytes(
        valid_payload,
        metadata=_metadata(params, omega_b_h2=1.01 * params.omega_b_h2),
        expected_sha256=sha256_hexdigest(valid_payload),
        redshift_column=0,
        electron_fraction_column=1,
    )
    with pytest.raises(ValueError, match="cosmology mismatch for omega_b_h2"):
        drag_optical_depth_benchmark(history, params)


def test_history_adapter_rejects_unbracketed_or_coarse_tau_crossing() -> None:
    params = CEForwardParams()
    zero_payload = b"0 0\n500 0\n1000 0\n1500 0\n"
    zero_history = parse_recombination_history_bytes(
        zero_payload,
        metadata=_metadata(params),
        expected_sha256=sha256_hexdigest(zero_payload),
        redshift_column=0,
        electron_fraction_column=1,
    )
    with pytest.raises(ValueError, match="tau_drag=1 is not bracketed"):
        drag_optical_depth_benchmark(zero_history, params)

    coarse_redshifts = (0.0, 500.0, 1000.0, 1500.0)
    coarse_payload = _linear_rate_payload(
        params,
        target_z_drag=1059.25,
        redshifts=coarse_redshifts,
    )
    coarse_history = parse_recombination_history_bytes(
        coarse_payload,
        metadata=_metadata(params),
        expected_sha256=sha256_hexdigest(coarse_payload),
        redshift_column=0,
        electron_fraction_column=1,
    )
    with pytest.raises(ValueError, match="crossing grid is too coarse"):
        drag_optical_depth_benchmark(coarse_history, params)
