'''Run the exact E70-A M2 admission closure gate.'''

from __future__ import annotations

from dataclasses import asdict
import json

from examples.physics.qft_m2_quantum_admission_closure import (
    evaluate_m2_admission_closure_gate,
)


def main() -> None:
    receipt = evaluate_m2_admission_closure_gate()
    print(json.dumps(asdict(receipt), sort_keys=True))
    if not receipt.declared_m2_admission_closure_gate_passed:
        raise SystemExit('declared E70-A M2 admission closure gate failed')


if __name__ == '__main__':
    main()
