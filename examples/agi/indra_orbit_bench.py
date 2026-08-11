from __future__ import annotations

import json

from reality_stone.clarus.indra_causal_quotient import evaluate_orbit_scaling


def main() -> None:
    print(json.dumps(evaluate_orbit_scaling(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
