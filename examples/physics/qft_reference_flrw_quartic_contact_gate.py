'''Reproduce the E68 scalar quartic-contact Legendre admission gate.'''

from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from examples.physics.qft_reference_flrw_background import (
    ReferenceFlrwParameters,
    ReferenceFlrwState,
    expanding_h_from_constraint,
)
from examples.physics.qft_reference_flrw_quartic_contact import (
    evaluate_scalar_quartic_contact_gate,
)


def reference_state() -> tuple[ReferenceFlrwState, ReferenceFlrwParameters]:
    parameters = ReferenceFlrwParameters(
        m_planck_over_mu_x=10.0,
        lambda_over_mu_x_squared=0.01,
    )
    u = 0.3
    b = 0.2
    state = ReferenceFlrwState(
        n=0.0,
        h=expanding_h_from_constraint(u=u, b=b, parameters=parameters),
        clock=0.0,
        u=u,
        b=b,
    )
    return state, parameters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--wavenumbers',
        nargs='+',
        type=float,
        default=(0.05, 0.1, 0.2, 0.4),
    )
    parser.add_argument(
        '--quartic-steps',
        nargs=3,
        type=float,
        default=(8.0e-2, 4.0e-2, 2.0e-2),
    )
    parser.add_argument('--cubic-epsilon', type=float, default=2.5e-3)
    parser.add_argument('--phase-points', type=int, default=256)
    parser.add_argument('--grid-phase-points', type=int, default=512)
    args = parser.parse_args()
    state, parameters = reference_state()
    receipts = []
    for wavenumber in args.wavenumbers:
        receipt = evaluate_scalar_quartic_contact_gate(
            state,
            parameters,
            base_wavenumber_bar=wavenumber,
            quartic_steps=tuple(args.quartic_steps),
            cubic_epsilon=args.cubic_epsilon,
            phase_points=args.phase_points,
            grid_phase_points=args.grid_phase_points,
        )
        receipts.append(asdict(receipt))
        print(json.dumps(receipts[-1], sort_keys=True), flush=True)
    print(
        json.dumps(
            {
                'all_declared_quartic_contact_gates_passed': all(
                    item['declared_quartic_contact_gate_passed']
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
