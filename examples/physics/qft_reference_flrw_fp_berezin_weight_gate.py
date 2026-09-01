'''Run the exact E69-H finite FP/Berezin weight gate.'''

from __future__ import annotations

from dataclasses import asdict
import json

from examples.physics.qft_reference_flrw_fp_berezin_weight import (
    evaluate_fp_berezin_gate,
)


def main() -> None:
    receipt = evaluate_fp_berezin_gate()
    print(json.dumps(asdict(receipt), sort_keys=True))
    if not receipt.declared_finite_fp_berezin_gate_passed:
        raise SystemExit('declared E69-H finite FP/Berezin gate failed')


if __name__ == '__main__':
    main()
