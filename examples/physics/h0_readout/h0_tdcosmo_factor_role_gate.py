"""Validate TDCOSMO role maps against declared likelihood factors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from h0_tdcosmo_hdf5_to_json import closure_roles


def iter_tdcosmo_files(path: Path) -> list[Path]:
    return sorted(path.glob("tdcosmo*_covariance.json"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default=str(Path(__file__).with_name("h0_fisher_io_examples")),
        help="directory containing generated TDCOSMO covariance JSON files",
    )
    args = parser.parse_args()
    files = iter_tdcosmo_files(Path(args.path))
    if not files:
        raise SystemExit("No tdcosmo*_covariance.json files found")

    print("# H0 TDCOSMO Factor Role Gate")
    print()
    print("| file | role basis | has population MST | inferred local | declared local | status |")
    print("|---|---|---|---|---|---|")

    failed = 0
    for file in files:
        payload = json.loads(file.read_text(encoding="utf-8"))
        nodes = [str(node) for node in payload["nodes"]]
        factors = payload.get("likelihood_factors", [])
        if not factors:
            failed += 1
            print(f"| {file.name} | missing | -- | -- | -- | FAIL |")
            continue
        inferred_local, inferred_global, mode = closure_roles(
            nodes,
            str(payload.get("source", {}).get("chain_path", "")),
            "auto",
            factors,
        )
        declared_local = sorted(str(node) for node in payload["local_nodes"])
        declared_global = sorted(str(node) for node in payload["global_nodes"])
        has_population_mst = any(
            factor.get("closure_scope") == "population_global"
            and any(str(node).startswith("lambda_mst") for node in factor.get("nodes", []))
            for factor in factors
        )
        ok = (
            inferred_local == declared_local
            and inferred_global == declared_global
            and mode == str(payload.get("conductance_mode", "path"))
        )
        if not ok:
            failed += 1
        print(
            f"| {file.name} | likelihood_factors | {has_population_mst} | "
            f"{', '.join(inferred_local)} | {', '.join(declared_local)} | {'PASS' if ok else 'FAIL'} |"
        )

    print()
    if failed:
        raise SystemExit(1)
    print("Verdict: declared role maps are reproducible from likelihood factor metadata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
