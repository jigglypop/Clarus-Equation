'''Production runner for the E70-C classical action/gauge-fixing gate.'''

from __future__ import annotations

from dataclasses import asdict
import json

from examples.physics.qft_m2_m1_classical_action_gaugefixing import (
    evaluate_m1_classical_action_gaugefixing_gate,
)


def main() -> None:
    receipt = evaluate_m1_classical_action_gaugefixing_gate()
    if not receipt.declared_m1_classical_action_gaugefixing_gate_passed:
        raise SystemExit('M1 classical action/gauge-fixing gate failed')
    print(json.dumps(asdict(receipt), sort_keys=True))


if __name__ == '__main__':
    main()
