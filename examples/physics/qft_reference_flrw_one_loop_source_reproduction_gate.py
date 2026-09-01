'''Run the source-locked v7 one-loop counterterm reproduction receipt.'''

from __future__ import annotations

from dataclasses import asdict
import json

from examples.physics.qft_reference_flrw_one_loop_source_reproduction import (
    evaluate_one_loop_source_reproduction_gate,
)


def main() -> None:
    receipt = evaluate_one_loop_source_reproduction_gate()
    print(json.dumps(asdict(receipt), sort_keys=True), flush=True)
    if not receipt.declared_one_loop_source_reproduction_gate_passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
