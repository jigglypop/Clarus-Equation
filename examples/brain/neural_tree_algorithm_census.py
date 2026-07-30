"""Inspect the finite tree-algorithm reverse-engineering census."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reality_stone.clarus.neural_tree_algorithm_census import (
    PARTIAL,
    TESTABLE,
    UNAVAILABLE,
    load_neural_tree_algorithm_census,
)


def _parser(repository_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Print the preregistered behavioral-equivalence census for "
            "tree and hierarchical neural-code candidates."
        )
    )
    parser.add_argument(
        "--census",
        type=Path,
        default=(
            repository_root
            / "benchmarks"
            / "neural_tree_algorithm_census_v1.json"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the validated census as deterministic JSON",
    )
    return parser


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    args = _parser(repository_root).parse_args()
    census = load_neural_tree_algorithm_census(args.census)
    if args.json:
        print(
            json.dumps(
                census.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return

    print("FINITE NEURAL TREE-ALGORITHM CENSUS")
    print(f"  status              {census.method_status}")
    print("  equivalence classes 5")
    print(f"  family count        {len(census.families)}")
    for status in (TESTABLE, PARTIAL, UNAVAILABLE):
        print(
            f"  {status:<19}"
            f"{len(census.families_with_status(status))}"
        )
    print("  screening order")
    for index, family_id in enumerate(census.screening_order, start=1):
        family = census.family(family_id)
        print(
            f"    {index:>2}. {family.family_id:<40}"
            f"{family.status:<11} {family.current_round}"
        )
    print("  unavailable branches")
    for family in census.families_with_status(UNAVAILABLE):
        print(f"    {family.family_id}: {family.required_observations}")
    print("  claim locks")
    for lock in census.claim_locks:
        print(f"    {lock}")


if __name__ == "__main__":
    main()
