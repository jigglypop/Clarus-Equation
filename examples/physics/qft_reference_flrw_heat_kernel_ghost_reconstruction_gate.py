'''Run the exact E69-C source-coefficient assembly gate.'''

from __future__ import annotations

from dataclasses import asdict
import json

from examples.physics.qft_reference_flrw_heat_kernel_ghost_reconstruction import (
    evaluate_heat_kernel_ghost_reconstruction_gate,
)


def main() -> None:
    receipt = evaluate_heat_kernel_ghost_reconstruction_gate()
    print(json.dumps(asdict(receipt), sort_keys=True))
    if not receipt.declared_source_coefficient_assembly_gate_passed:
        raise SystemExit('declared E69-C source-coefficient assembly gate failed')


if __name__ == '__main__':
    main()
