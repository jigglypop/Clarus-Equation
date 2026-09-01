'''Run the exact E69-I finite operator-trace synthesis gate.'''

from __future__ import annotations

from dataclasses import asdict
import json

from examples.physics.qft_reference_flrw_operator_trace_synthesis import (
    evaluate_operator_trace_synthesis_gate,
)


def main() -> None:
    receipt = evaluate_operator_trace_synthesis_gate()
    print(json.dumps(asdict(receipt), sort_keys=True))
    if not receipt.declared_operator_trace_synthesis_gate_passed:
        raise SystemExit('declared E69-I operator-trace synthesis gate failed')


if __name__ == '__main__':
    main()
