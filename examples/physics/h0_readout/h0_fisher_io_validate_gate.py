"""Validate Fisher/covariance JSON files for the H0 readout pipeline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from h0_fisher_matrix_io_gate import channel_from_payload


def iter_json_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        candidate
        for candidate in path.glob("*.json")
        if candidate.is_file() and candidate.name != "manifest.json"
    )


def validate_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = ["nodes", "observable", "local_nodes", "global_nodes", "matrix_type", "matrix"]
    for key in required:
        if key not in payload:
            errors.append(f"missing required field: {key}")
    if errors:
        return errors

    nodes = payload.get("nodes")
    if not isinstance(nodes, list) or not nodes or len({str(node) for node in nodes}) != len(nodes):
        errors.append("nodes must be a nonempty list of unique names")
        return errors

    node_set = {str(node) for node in nodes}
    observable = str(payload.get("observable"))
    if observable not in node_set:
        errors.append("observable must be in nodes")

    for field in ["local_nodes", "global_nodes"]:
        values = payload.get(field)
        if not isinstance(values, list):
            errors.append(f"{field} must be a list")
            continue
        missing = sorted(str(node) for node in values if str(node) not in node_set)
        if missing:
            errors.append(f"{field} contains unknown nodes: {', '.join(missing)}")

    factors = payload.get("likelihood_factors", [])
    if factors:
        if not isinstance(factors, list):
            errors.append("likelihood_factors must be a list")
        else:
            for index, factor in enumerate(factors):
                if not isinstance(factor, dict):
                    errors.append(f"likelihood_factors[{index}] must be an object")
                    continue
                for key in ["name", "closure_scope", "nodes"]:
                    if key not in factor:
                        errors.append(f"likelihood_factors[{index}] missing field: {key}")
                factor_nodes = factor.get("nodes", [])
                if not isinstance(factor_nodes, list):
                    errors.append(f"likelihood_factors[{index}].nodes must be a list")
                    continue
                missing = sorted(str(node) for node in factor_nodes if str(node) not in node_set)
                if missing:
                    errors.append(
                        f"likelihood_factors[{index}].nodes contains unknown nodes: {', '.join(missing)}"
                    )

    matrix_type = str(payload.get("matrix_type", "")).lower()
    if matrix_type not in {"fisher", "covariance"}:
        errors.append("matrix_type must be 'fisher' or 'covariance'")

    conductance_mode = str(payload.get("conductance_mode", "path")).lower()
    if conductance_mode not in {"path", "direct"}:
        errors.append("conductance_mode must be 'path' or 'direct'")

    matrix = payload.get("matrix")
    n = len(nodes)
    if not isinstance(matrix, list) or len(matrix) != n:
        errors.append("matrix must be square with size len(nodes)")
        return errors
    parsed: list[list[float]] = []
    for i, row in enumerate(matrix):
        if not isinstance(row, list) or len(row) != n:
            errors.append("matrix must be square with size len(nodes)")
            return errors
        parsed_row = []
        for j, value in enumerate(row):
            try:
                number = float(value)
            except (TypeError, ValueError):
                errors.append(f"matrix[{i}][{j}] is not numeric")
                number = float("nan")
            if not math.isfinite(number):
                errors.append(f"matrix[{i}][{j}] is not finite")
            parsed_row.append(number)
        parsed.append(parsed_row)

    for i in range(n):
        if parsed[i][i] <= 0:
            errors.append(f"matrix diagonal must be positive at {nodes[i]}")
        for j in range(i + 1, n):
            if abs(parsed[i][j] - parsed[j][i]) > 1e-10:
                errors.append(f"matrix must be symmetric at ({nodes[i]}, {nodes[j]})")

    if errors:
        return errors

    try:
        channel_from_payload(payload)
    except Exception as exc:  # noqa: BLE001 - report validation failure without traceback.
        errors.append(f"pipeline ingestion failed: {exc}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default=str(Path(__file__).with_name("h0_fisher_io_examples")),
        help="JSON file or directory of channel JSON files",
    )
    args = parser.parse_args()

    files = iter_json_files(Path(args.path))
    if not files:
        raise SystemExit(f"No JSON files found at {args.path}")

    print("# H0 Fisher IO Validate Gate")
    print()
    print("| file | status | notes |")
    print("|---|---|---|")
    failed = 0
    for file in files:
        try:
            payload = json.loads(file.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                errors = ["top-level JSON must be an object"]
            else:
                errors = validate_payload(payload)
        except Exception as exc:  # noqa: BLE001 - keep validator concise.
            errors = [str(exc)]
        if errors:
            failed += 1
            notes = "; ".join(errors)
            print(f"| {file.name} | FAIL | {notes} |")
        else:
            print(f"| {file.name} | PASS | ready |")

    print()
    print(f"validated = {len(files)}")
    print(f"failed = {failed}")
    if failed:
        raise SystemExit(1)
    print("Verdict: all Fisher/covariance JSON inputs passed validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
