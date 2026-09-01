'''Reproduce the E68 quartic-contact plus rotating-exchange gate.'''

from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from examples.physics.qft_reference_flrw_quartic_contact_gate import (
    reference_state,
)
from examples.physics.qft_reference_flrw_quartic_exchange import (
    evaluate_scalar_quartic_exchange_gate,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--wavenumbers',
        nargs='+',
        type=float,
        default=(0.05, 0.1, 0.2, 0.4),
    )
    parser.add_argument('--interval', type=float, default=0.5)
    parser.add_argument('--phase-points', type=int, default=256)
    parser.add_argument('--grid-phase-points', type=int, default=512)
    parser.add_argument(
        '--simpson-subintervals',
        nargs=2,
        type=int,
        default=(2048, 4096),
    )
    parser.add_argument('--square-subintervals', type=int, default=512)
    args = parser.parse_args()
    state, parameters = reference_state()
    receipts = []
    for wavenumber in args.wavenumbers:
        receipt = evaluate_scalar_quartic_exchange_gate(
            state,
            parameters,
            base_wavenumber_bar=wavenumber,
            interval_bar=args.interval,
            phase_points=args.phase_points,
            grid_phase_points=args.grid_phase_points,
            simpson_subintervals=tuple(args.simpson_subintervals),
            square_subintervals=args.square_subintervals,
        )
        receipts.append(asdict(receipt))
        print(json.dumps(receipts[-1], sort_keys=True), flush=True)
    print(
        json.dumps(
            {
                'all_declared_rotating_exchange_gates_passed': all(
                    item['declared_rotating_exchange_gate_passed']
                    for item in receipts
                ),
                'receipt_count': len(receipts),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == '__main__':
    main()
