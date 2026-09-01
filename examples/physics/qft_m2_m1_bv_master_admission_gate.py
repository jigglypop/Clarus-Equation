'''Production runner for the E70-D BV master-admission gate.'''

from __future__ import annotations

from dataclasses import asdict
import json

from examples.physics.qft_m2_m1_bv_master_admission import (
    evaluate_m1_bv_master_admission_gate,
)


def main() -> None:
    receipt = evaluate_m1_bv_master_admission_gate()
    if not receipt.declared_m1_bv_master_admission_gate_passed:
        raise SystemExit('M1 BV master-admission gate failed')
    print(json.dumps(asdict(receipt), sort_keys=True))


if __name__ == '__main__':
    main()
