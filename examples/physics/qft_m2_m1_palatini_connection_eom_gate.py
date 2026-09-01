'''Production runner for the E70-I Palatini connection-EOM gate.'''

from __future__ import annotations

from dataclasses import asdict
import json

from examples.physics.qft_m2_m1_palatini_connection_eom import (
    evaluate_m1_palatini_connection_eom_gate,
)


def main() -> None:
    receipt = evaluate_m1_palatini_connection_eom_gate()
    if not receipt.declared_m1_palatini_connection_eom_gate_passed:
        raise SystemExit('E70-I Palatini connection-EOM gate failed')
    print(json.dumps(asdict(receipt), default=str, sort_keys=True))


if __name__ == '__main__':
    main()
