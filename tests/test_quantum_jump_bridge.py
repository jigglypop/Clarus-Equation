from __future__ import annotations

import numpy as np
import pytest

from reality_stone.clarus.quantum_jump_bridge import (
    CONDITIONAL_FAIL,
    CONDITIONAL_PASS,
    STRUCTURAL_SCOPE,
    audit_kossakowski,
    audit_no_jump_sector,
    audit_population_coherence_leakage,
    classical_offdiagonal_rates,
    next_generation_from_constant_rates,
    structural_bridge_report,
)


def _transition(
    dimension: int,
    *,
    source: int,
    target: int,
    rate: float,
) -> np.ndarray:
    jump = np.zeros((dimension, dimension), dtype=np.complex128)
    jump[target, source] = np.sqrt(rate)
    return jump


def _classical_cycle(rate: float = 2.0) -> np.ndarray:
    return np.asarray(
        [
            _transition(3, source=0, target=1, rate=rate),
            _transition(3, source=1, target=2, rate=rate),
            _transition(3, source=2, target=0, rate=rate),
        ]
    )


def test_kossakowski_audit_requires_both_hermiticity_and_psd() -> None:
    valid = audit_kossakowski([[1.0, 0.5j], [-0.5j, 1.0]])
    nonhermitian = audit_kossakowski([[1.0, 1.0], [0.0, 1.0]])
    indefinite = audit_kossakowski([[1.0, 0.0], [0.0, -0.1]])

    assert valid.hermitian
    assert valid.positive_semidefinite
    assert valid.structural_pass
    assert not nonhermitian.hermitian
    assert not nonhermitian.structural_pass
    assert indefinite.hermitian
    assert not indefinite.positive_semidefinite
    assert not indefinite.structural_pass


def test_row_source_orientation_is_source_row_target_column() -> None:
    jumps = np.asarray(
        [
            _transition(3, source=0, target=2, rate=4.0),
            _transition(3, source=2, target=1, rate=9.0),
        ]
    )

    rates = classical_offdiagonal_rates(jumps)

    assert rates[0, 2] == pytest.approx(4.0)
    assert rates[2, 1] == pytest.approx(9.0)
    assert rates[2, 0] == 0.0
    assert rates[1, 2] == 0.0
    assert np.all(np.diag(rates) == 0.0)


def test_classical_rank_one_jumps_close_population_block() -> None:
    audit = audit_population_coherence_leakage(
        np.diag([0.0, 1.0, 2.0]),
        _classical_cycle(),
    )

    assert audit.population_to_coherence_norm < 1.0e-12
    assert audit.coherence_to_population_norm < 1.0e-12
    assert audit.populations_invariant
    assert audit.populations_autonomous
    assert audit.classical_closed


def test_coherent_hamiltonian_is_a_population_coherence_leakage_counterexample() -> None:
    audit = audit_population_coherence_leakage(
        [[0.0, 1.0], [1.0, 0.0]],
        [_transition(2, source=1, target=0, rate=1.0)],
    )

    assert audit.population_to_coherence_norm > 0.0
    assert audit.coherence_to_population_norm > 0.0
    assert not audit.classical_closed


def test_common_collective_jump_is_a_leakage_counterexample() -> None:
    collective = np.zeros((3, 3), dtype=np.complex128)
    collective[0, 1] = 1.0
    collective[0, 2] = 1.0

    audit = audit_population_coherence_leakage(
        np.zeros((3, 3)),
        [collective],
    )

    assert audit.population_to_coherence_norm > 0.0
    assert audit.coherence_to_population_norm > 0.0
    assert not audit.classical_closed


def test_uniform_escape_rate_has_invariant_constant_hazard_full_sector() -> None:
    audit = audit_no_jump_sector(
        np.diag([0.0, 1.0, 2.0]),
        _classical_cycle(rate=2.0),
        np.eye(3),
    )

    assert audit.hazard == pytest.approx(2.0)
    assert audit.invariance_residual < 1.0e-12
    assert audit.constant_hazard_residual < 1.0e-12
    assert audit.structural_pass


def test_nonuniform_gamma_fails_constant_hazard_gate() -> None:
    jump_one = np.diag([1.0, 0.0])
    jump_two = np.diag([0.0, np.sqrt(2.0)])

    audit = audit_no_jump_sector(
        np.zeros((2, 2)),
        [jump_one, jump_two],
        np.eye(2),
    )

    assert audit.invariant
    assert audit.hazard == pytest.approx(1.5)
    assert audit.constant_hazard_residual == pytest.approx(0.5)
    assert not audit.constant_hazard
    assert not audit.structural_pass


def test_next_generation_scales_source_rows_not_target_columns() -> None:
    birth_rates = np.array([[0.0, 2.0], [3.0, 0.0]])
    lifetimes = np.array([10.0, 100.0])

    next_generation = next_generation_from_constant_rates(
        birth_rates,
        lifetimes,
    )

    assert np.array_equal(
        next_generation,
        np.array([[0.0, 20.0], [300.0, 0.0]]),
    )


def test_report_pass_is_explicitly_conditional_not_a_ce_sm_derivation() -> None:
    jumps = _classical_cycle(rate=2.0)
    rates = classical_offdiagonal_rates(jumps)
    report = structural_bridge_report(
        kossakowski_matrix=2.0 * np.eye(3),
        hamiltonian=np.diag([0.0, 1.0, 2.0]),
        jump_operators=jumps,
        sector_projector=np.eye(3),
        birth_rates=rates,
        mean_lifetimes=np.full(3, 0.5),
    )
    payload = report.to_dict()

    assert report.scope == STRUCTURAL_SCOPE
    assert report.structural_status == CONDITIONAL_PASS
    assert not report.ce_sm_derivation_complete
    assert not report.poisson_branching_derived
    assert np.allclose(
        payload["next_generation_matrix"],
        (
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
        ),
        atol=1.0e-12,
    )
    assert "does not derive" in report.conclusion
    assert "CE+SM action" in report.assumptions_not_audited[0]
    assert any(
        "consistency between the Kossakowski" in assumption
        for assumption in report.assumptions_not_audited
    )


def test_report_fails_when_coherent_dynamics_leak_out_of_population_block() -> None:
    jumps = np.asarray(
        [
            _transition(2, source=0, target=1, rate=1.0),
            _transition(2, source=1, target=0, rate=1.0),
        ]
    )
    report = structural_bridge_report(
        kossakowski_matrix=np.eye(2),
        hamiltonian=[[0.0, 1.0], [1.0, 0.0]],
        jump_operators=jumps,
        sector_projector=np.eye(2),
        birth_rates=classical_offdiagonal_rates(jumps),
        mean_lifetimes=np.ones(2),
    )

    assert report.structural_status == CONDITIONAL_FAIL
    assert not report.leakage.classical_closed
    assert not report.ce_sm_derivation_complete
    assert not report.poisson_branching_derived


def test_birth_rate_validation_rejects_signed_inputs() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        next_generation_from_constant_rates(
            [[0.0, -1.0], [1.0, 0.0]],
            [1.0, 1.0],
        )
