'''Production runner for the E70-F 4D densitized scalar local BV gate.'''

from __future__ import annotations

from dataclasses import asdict
import json

from examples.physics.qft_m2_m1_4d_densitized_scalar_bv import (
    evaluate_m1_4d_densitized_scalar_bv_gate,
)


def main() -> None:
    receipt = evaluate_m1_4d_densitized_scalar_bv_gate()
    if not receipt.declared_m1_4d_densitized_scalar_bv_gate_passed:
        raise SystemExit('E70-F 4D densitized scalar local BV gate failed')
    print(json.dumps(asdict(receipt), default=str, sort_keys=True))


if __name__ == '__main__':
    main()
