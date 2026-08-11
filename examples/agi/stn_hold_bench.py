from __future__ import annotations

import json

from reality_stone.clarus.stn_hold_benchmark import evaluate_stn_hold


def main() -> None:
    print(json.dumps(evaluate_stn_hold(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
