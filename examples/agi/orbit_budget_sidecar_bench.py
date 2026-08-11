from __future__ import annotations

import json

from reality_stone.clarus.orbit_budget_benchmark import evaluate_orbit_budget_sidecar


if __name__ == "__main__":
    print(json.dumps(evaluate_orbit_budget_sidecar(), indent=2, sort_keys=True))
