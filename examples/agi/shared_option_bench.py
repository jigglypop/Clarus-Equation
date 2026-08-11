from __future__ import annotations

import json

from reality_stone.clarus.shared_option_benchmark import evaluate_shared_options


def main() -> None:
    print(json.dumps(evaluate_shared_options(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
