'''Production runner for the E70-J first-order M1 bulk assembly gate.'''

from __future__ import annotations

from dataclasses import asdict
import json

from examples.physics.qft_m2_m1_first_order_bulk_assembly import (
    evaluate_m1_first_order_bulk_assembly_gate,
)


def main() -> None:
    receipt = evaluate_m1_first_order_bulk_assembly_gate()
    if not receipt.declared_m1_first_order_bulk_assembly_gate_passed:
        raise SystemExit('E70-J first-order M1 bulk assembly gate failed')
    print(json.dumps(asdict(receipt), default=str, sort_keys=True))


if __name__ == '__main__':
    main()
