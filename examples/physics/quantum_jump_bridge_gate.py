"""Run conditional quantum-jump structural gates without claiming CE+SM closure."""

from __future__ import annotations

import numpy as np

from reality_stone.clarus.quantum_jump_bridge import (
    audit_population_coherence_leakage,
    classical_offdiagonal_rates,
    structural_bridge_report,
)


def transition(
    dimension: int,
    *,
    source: int,
    target: int,
    rate: float,
) -> np.ndarray:
    jump = np.zeros((dimension, dimension), dtype=np.complex128)
    jump[target, source] = np.sqrt(rate)
    return jump


def main() -> None:
    classical_jumps = np.asarray(
        [
            transition(3, source=0, target=1, rate=2.0),
            transition(3, source=1, target=2, rate=2.0),
            transition(3, source=2, target=0, rate=2.0),
        ]
    )
    classical_rates = classical_offdiagonal_rates(classical_jumps)
    report = structural_bridge_report(
        kossakowski_matrix=2.0 * np.eye(3),
        hamiltonian=np.diag([0.0, 1.0, 2.0]),
        jump_operators=classical_jumps,
        sector_projector=np.eye(3),
        birth_rates=classical_rates,
        mean_lifetimes=np.full(3, 0.5),
    )

    coherent = audit_population_coherence_leakage(
        [[0.0, 1.0], [1.0, 0.0]],
        [transition(2, source=1, target=0, rate=1.0)],
    )
    collective_jump = np.zeros((3, 3), dtype=np.complex128)
    collective_jump[0, 1] = 1.0
    collective_jump[0, 2] = 1.0
    collective = audit_population_coherence_leakage(
        np.zeros((3, 3)),
        [collective_jump],
    )

    print("CE QUANTUM-JUMP BRIDGE STRUCTURAL GATE")
    print(f"  scope                       {report.scope}")
    print(f"  status                      {report.structural_status}")
    print(f"  row-source rates            {report.classical_offdiagonal_rates}")
    print(f"  next-generation A           {report.next_generation_matrix}")
    print(f"  no-jump hazard              {report.no_jump.hazard:.12g}")
    print(f"  CE+SM derivation complete   {report.ce_sm_derivation_complete}")
    print(f"  Poisson branching derived   {report.poisson_branching_derived}")
    print(
        "  coherent leakage norm       "
        f"{coherent.population_to_coherence_norm:.6g}"
    )
    print(
        "  collective leakage norm     "
        f"{collective.population_to_coherence_norm:.6g}"
    )
    print(f"  conclusion                  {report.conclusion}")


if __name__ == "__main__":
    main()
