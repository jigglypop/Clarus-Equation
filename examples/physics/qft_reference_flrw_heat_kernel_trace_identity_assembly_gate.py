'''Run the exact E69-D source trace-identity assembly gate.'''

from __future__ import annotations

from dataclasses import asdict
import json

from examples.physics.qft_reference_flrw_heat_kernel_trace_identity_assembly import (
    evaluate_trace_identity_assembly_gate,
)


def main() -> None:
    receipt = evaluate_trace_identity_assembly_gate()
    print(json.dumps(asdict(receipt), sort_keys=True))
    if not receipt.declared_source_trace_identity_assembly_gate_passed:
        raise SystemExit(
            'declared E69-D source trace-identity assembly gate failed'
        )


if __name__ == '__main__':
    main()
