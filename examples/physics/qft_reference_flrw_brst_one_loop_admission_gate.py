'''Run the finite order-hbar ST-cohomology admission receipt.'''

from __future__ import annotations

from dataclasses import asdict
import json

from examples.physics.qft_reference_flrw_brst_one_loop_admission import (
    evaluate_finite_one_loop_st_admission_gate,
)


def main() -> None:
    receipt = evaluate_finite_one_loop_st_admission_gate()
    print(json.dumps(asdict(receipt), sort_keys=True), flush=True)
    if not receipt.declared_finite_one_loop_st_admission_gate_passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
