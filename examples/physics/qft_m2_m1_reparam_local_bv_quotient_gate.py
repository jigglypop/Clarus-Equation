'''Production runner for the E70-E bounded local BV quotient gate.'''

from __future__ import annotations

from dataclasses import asdict
import json

from examples.physics.qft_m2_m1_reparam_local_bv_quotient import (
    evaluate_m1_reparam_local_bv_quotient_gate,
)


def main() -> None:
    receipt = evaluate_m1_reparam_local_bv_quotient_gate()
    if not receipt.declared_m1_reparam_local_bv_quotient_gate_passed:
        raise SystemExit('M1-aligned reparametrization local BV quotient gate failed')
    print(json.dumps(asdict(receipt), sort_keys=True))


if __name__ == '__main__':
    main()
