'''Production runner for the E70-H Palatini curvature BRST gate.'''

from __future__ import annotations

from dataclasses import asdict
import json

from examples.physics.qft_m2_m1_palatini_curvature_brst import (
    evaluate_m1_palatini_curvature_brst_gate,
)


def main() -> None:
    receipt = evaluate_m1_palatini_curvature_brst_gate()
    if not receipt.declared_m1_palatini_curvature_brst_gate_passed:
        raise SystemExit('E70-H Palatini curvature BRST gate failed')
    print(json.dumps(asdict(receipt), default=str, sort_keys=True))


if __name__ == '__main__':
    main()
