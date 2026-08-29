from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from examples.physics.causal_record_dust_bridge import (
    ExitPhaseMark,
    match_exit_antichain,
    monokinetic_dust_data,
)
from examples.physics.lattice_scalar_transition_bridge import (
    LayeredCausalEdge,
    LayeredRecordEvent,
    certify_layered_cauchy_embedding,
)
import examples.physics.partitioned_dark_sector_flrw as bridge
from examples.physics.partitioned_dark_sector_flrw import (
    construct_record_complement_as_vacuum,
    propagate_partitioned_dust_vacuum_flat_flrw,
)
from examples.physics.quantum_instrument_record_kernel import (
    KrausBranch,
    RecordInstrument,
    build_energy_resolved_instrument_tree,
    construct_luders_energy_instrument,
)


def _partition_fixture():
    hamiltonian = 5.0 * np.eye(2, dtype=np.complex128)
    energy_only = construct_luders_energy_instrument(hamiltonian)
    left = np.diag([1.0, 0.0]).astype(np.complex128)
    right = np.diag([0.0, 1.0]).astype(np.complex128)
    kernel = build_energy_resolved_instrument_tree(
        hamiltonian,
        np.diag([0.4, 0.6]),
        (
            RecordInstrument(
                "root",
                "declared-degenerate-position-pointer",
                (
                    KrausBranch("left", "left exit", left),
                    KrausBranch("right", "right exit", right),
                ),
            ),
        ),
        initial_weight=5.0,
    )
    assert energy_only.spectral_energies == (5.0,)
    embedding = certify_layered_cauchy_embedding(
        (
            LayeredRecordEvent("root", 0, (0, 0, 0)),
            LayeredRecordEvent("left", 1, (0, 0, 0)),
            LayeredRecordEvent("right", 1, (1, 0, 0)),
        ),
        (
            LayeredCausalEdge("root", "left"),
            LayeredCausalEdge("root", "right"),
        ),
        exit_nodes=kernel.terminal_nodes,
        lattice_spacing=1.0,
        clock_step=1.0,
        causal_speed=1.0,
        spatial_shape=(2, 1, 1),
    )
    marks = tuple(
        ExitPhaseMark(
            node,
            embedding.coordinate(node)[1:],
            5.0,
            (0.0, 0.0, 0.0),
            0.4,
        )
        for node in kernel.terminal_nodes
    )
    matching = match_exit_antichain(
        kernel.flow,
        kernel.nodes,
        marks,
        cell_volume=embedding.spatial_volume,
    )
    dust = monokinetic_dust_data(matching)
    receipt = construct_record_complement_as_vacuum(
        matching,
        dust,
        source_receipt_id="record:root",
        dust_allocation_id="record:root:residual",
        vacuum_allocation_id="record:root:complement",
    )
    return matching, dust, receipt


def test_one_shared_receipt_prevents_cross_module_double_counting() -> None:
    _, _, receipt = _partition_fixture()

    assert receipt.total_record_energy == pytest.approx(25.0)
    assert receipt.dust_energy == pytest.approx(10.0)
    assert receipt.vacuum_energy == pytest.approx(15.0)
    assert receipt.unassigned_energy == pytest.approx(0.0)
    assert receipt.dust_fraction == pytest.approx(0.4)
    assert receipt.vacuum_fraction == pytest.approx(0.6)
    assert receipt.record_partition_residual == pytest.approx(0.0)
    assert receipt.vacuum_channel_residual == pytest.approx(0.0)
    assert receipt.two_channel_partition_closed


def test_mixed_dust_vacuum_background_obeys_exact_flrw_equations() -> None:
    _, _, receipt = _partition_fixture()
    result = propagate_partitioned_dust_vacuum_flat_flrw(
        receipt,
        newton_constant=1.0,
        evaluation_scale_factor_ratio=2.0,
        global_constant_vacuum_action_adopted=True,
    )

    assert result.dust_density == pytest.approx(5.0 / 8.0)
    assert result.vacuum_density == pytest.approx(7.5)
    assert result.total_density == pytest.approx(8.125)
    assert result.effective_equation_of_state == pytest.approx(-7.5 / 8.125)
    assert result.normalized_hubble_rate**2 == pytest.approx(0.4 / 8.0 + 0.6)
    assert result.reconstructed_scale_factor_ratio == pytest.approx(2.0)
    assert result.continuity_equation_residual == pytest.approx(0.0)
    assert result.friedmann_equation_residual == pytest.approx(0.0)
    assert result.raychaudhuri_equation_residual == pytest.approx(0.0)
    assert result.acceleration_equation_residual == pytest.approx(0.0)
    assert result.conditional_mixed_background_closed


def test_dimensionless_time_solution_and_transition_scales() -> None:
    _, _, receipt = _partition_fixture()
    result = propagate_partitioned_dust_vacuum_flat_flrw(
        receipt,
        newton_constant=0.5,
        evaluation_scale_factor_ratio=3.0,
        global_constant_vacuum_action_adopted=True,
    )

    expected_dimensionless_time = 2.0 / (3.0 * math.sqrt(0.6)) * (
        math.asinh(math.sqrt(0.6 / 0.4) * 3.0**1.5)
        - math.asinh(math.sqrt(0.6 / 0.4))
    )
    assert result.dimensionless_elapsed_time == pytest.approx(
        expected_dimensionless_time
    )
    assert result.matter_vacuum_equality_scale_factor_ratio == pytest.approx(
        (0.4 / 0.6) ** (1.0 / 3.0)
    )
    assert result.acceleration_transition_scale_factor_ratio == pytest.approx(
        (0.4 / 1.2) ** (1.0 / 3.0)
    )


def test_vacuum_is_constructed_from_exact_record_complement(monkeypatch) -> None:
    matching, dust, _ = _partition_fixture()
    original = bridge.match_constant_vacuum_on_cauchy_slice
    observed: dict[str, float] = {}

    def spy(**arguments):
        observed.update(arguments)
        return original(**arguments)

    monkeypatch.setattr(bridge, "match_constant_vacuum_on_cauchy_slice", spy)
    construct_record_complement_as_vacuum(
        matching,
        dust,
        source_receipt_id="record:root",
        dust_allocation_id="record:root:residual",
        vacuum_allocation_id="record:root:vacuum",
    )
    assert observed["battery_energy"] == pytest.approx(
        matching.complement_energy_density * matching.cell_volume
    )
    assert observed["residual_efficiency"] == pytest.approx(1.0)


@pytest.mark.parametrize("corruption", ("density", "stress"))
def test_malformed_vacuum_transition_is_rejected(monkeypatch, corruption: str) -> None:
    matching, dust, _ = _partition_fixture()
    original = bridge.match_constant_vacuum_on_cauchy_slice

    def malformed(**arguments):
        valid = original(**arguments)
        if corruption == "density":
            return replace(valid, vacuum_density=2.0 * valid.vacuum_density)
        malformed_stress = np.asarray(valid.vacuum_stress).copy()
        malformed_stress[1, 1] = 0.0
        return replace(valid, vacuum_stress=malformed_stress)

    monkeypatch.setattr(bridge, "match_constant_vacuum_on_cauchy_slice", malformed)
    with pytest.raises(ValueError, match="vacuum density|vacuum stress"):
        construct_record_complement_as_vacuum(
            matching,
            dust,
            source_receipt_id="record:root",
            dust_allocation_id="record:root:residual",
            vacuum_allocation_id="record:root:vacuum",
        )


def test_tiny_vacuum_scale_still_rejects_total_relative_error(monkeypatch) -> None:
    matching, dust, _ = _partition_fixture()
    scale = 1.0e-13
    tiny_matching = replace(
        matching,
        residual_energy_density=matching.residual_energy_density * scale,
        complement_energy_density=matching.complement_energy_density * scale,
        total_energy_density=matching.total_energy_density * scale,
    )
    tiny_dust = replace(
        dust,
        rest_energy_density=dust.rest_energy_density * scale,
        stress=tuple(
            tuple(value * scale for value in row) for row in dust.stress
        ),
    )
    original = bridge.match_constant_vacuum_on_cauchy_slice

    def zeroed(**arguments):
        valid = original(**arguments)
        return replace(
            valid,
            vacuum_density=0.0,
            vacuum_stress=np.zeros((4, 4)),
        )

    monkeypatch.setattr(bridge, "match_constant_vacuum_on_cauchy_slice", zeroed)
    with pytest.raises(ValueError, match="vacuum density"):
        construct_record_complement_as_vacuum(
            tiny_matching,
            tiny_dust,
            source_receipt_id="record:tiny",
            dust_allocation_id="record:tiny:residual",
            vacuum_allocation_id="record:tiny:vacuum",
        )


def test_allocation_identifiers_must_be_disjoint() -> None:
    matching, dust, _ = _partition_fixture()
    with pytest.raises(ValueError, match="disjoint"):
        construct_record_complement_as_vacuum(
            matching,
            dust,
            source_receipt_id="record:root",
            dust_allocation_id="same",
            vacuum_allocation_id="same",
        )


def test_mixed_propagation_requires_global_action_and_zero_transfer() -> None:
    _, _, receipt = _partition_fixture()
    with pytest.raises(ValueError, match="global constant-vacuum action"):
        propagate_partitioned_dust_vacuum_flat_flrw(
            receipt,
            newton_constant=1.0,
            evaluation_scale_factor_ratio=2.0,
            global_constant_vacuum_action_adopted=False,
        )
    with pytest.raises(ValueError, match="Q=0"):
        propagate_partitioned_dust_vacuum_flat_flrw(
            receipt,
            newton_constant=1.0,
            evaluation_scale_factor_ratio=2.0,
            global_constant_vacuum_action_adopted=True,
            dust_vacuum_transfer_rate_density=0.1,
        )


def test_noncomoving_or_pressured_dust_is_rejected() -> None:
    matching, dust, _ = _partition_fixture()
    moving = replace(dust, gamma=2.0)
    with pytest.raises(ValueError, match="comoving rest dust"):
        construct_record_complement_as_vacuum(
            matching,
            moving,
            source_receipt_id="record:root",
            dust_allocation_id="dust",
            vacuum_allocation_id="vacuum",
        )


def test_conditional_background_does_not_claim_selection_or_prediction() -> None:
    _, _, receipt = _partition_fixture()
    result = propagate_partitioned_dust_vacuum_flat_flrw(
        receipt,
        newton_constant=1.0,
        evaluation_scale_factor_ratio=2.0,
        global_constant_vacuum_action_adopted=True,
    )

    assert not result.partition_fraction_selected_by_ce_dynamics
    assert not result.absolute_dark_density_predicted
    assert not result.vacuum_action_derived_from_one_slice
    assert not result.renormalized_quantum_stress_derived
    assert not result.perturbations_and_structure_growth_derived
    assert not result.ce_specific_independent_observational_prediction_derived
