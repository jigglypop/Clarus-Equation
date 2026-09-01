'''Run the exact E69-F Sym2(V)+scalar curvature-trace gate.'''

from __future__ import annotations

from dataclasses import asdict
import json

from examples.physics.qft_reference_flrw_sym2_curvature_trace import (
    evaluate_sym2_curvature_trace_gate,
)


def main() -> None:
    receipt = evaluate_sym2_curvature_trace_gate()
    print(json.dumps(asdict(receipt), sort_keys=True))
    if not receipt.declared_finite_sym2_curvature_trace_gate_passed:
        raise SystemExit('declared E69-F Sym2 curvature-trace gate failed')


if __name__ == '__main__':
    main()
