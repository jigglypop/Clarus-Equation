'''Run the exact E69-G Sym2 potential bulk-quotient gate.'''

from __future__ import annotations

from dataclasses import asdict
import json

from examples.physics.qft_reference_flrw_sym2_potential_bulk_quotient import (
    evaluate_sym2_potential_bulk_gate,
)


def main() -> None:
    receipt = evaluate_sym2_potential_bulk_gate()
    print(json.dumps(asdict(receipt), sort_keys=True))
    if not receipt.declared_finite_sym2_potential_bulk_gate_passed:
        raise SystemExit('declared E69-G Sym2 potential bulk gate failed')


if __name__ == '__main__':
    main()
