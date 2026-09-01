'''Production runner for the E70-G metric-density compatibility BV gate.'''

from __future__ import annotations

from dataclasses import asdict
import json

from examples.physics.qft_m2_m1_metric_density_compatibility_bv import (
    evaluate_m1_metric_density_compatibility_bv_gate,
)


def main() -> None:
    receipt = evaluate_m1_metric_density_compatibility_bv_gate()
    if not receipt.declared_m1_metric_density_compatibility_bv_gate_passed:
        raise SystemExit('E70-G metric-density compatibility BV gate failed')
    print(json.dumps(asdict(receipt), default=str, sort_keys=True))


if __name__ == '__main__':
    main()
