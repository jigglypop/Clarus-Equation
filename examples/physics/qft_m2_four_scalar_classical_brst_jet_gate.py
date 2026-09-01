'''Production runner for the exact full-M1 classical BRST jet gate.'''

from __future__ import annotations

from dataclasses import asdict
import json

from examples.physics.qft_m2_four_scalar_classical_brst_jet import (
    evaluate_four_scalar_classical_brst_jet_gate,
)


def main() -> None:
    receipt = evaluate_four_scalar_classical_brst_jet_gate()
    if not receipt.declared_exact_classical_brst_jet_gate_passed:
        raise SystemExit('exact classical BRST jet gate failed')
    print(json.dumps(asdict(receipt), sort_keys=True))


if __name__ == '__main__':
    main()
