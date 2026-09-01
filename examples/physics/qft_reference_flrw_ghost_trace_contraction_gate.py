'''Run the exact E69-E finite ghost trace-contraction gate.'''

from __future__ import annotations

from dataclasses import asdict
import json

from examples.physics.qft_reference_flrw_ghost_trace_contraction import (
    evaluate_ghost_trace_contraction_gate,
)


def main() -> None:
    receipt = evaluate_ghost_trace_contraction_gate()
    print(json.dumps(asdict(receipt), sort_keys=True))
    if not receipt.declared_finite_ghost_trace_contraction_gate_passed:
        raise SystemExit('declared E69-E ghost trace-contraction gate failed')


if __name__ == '__main__':
    main()
