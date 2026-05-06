"""Convert a labelled CSV Fisher/covariance table into the H0 readout JSON schema.

Expected CSV shape:

    ,obs,local_anchor,global_prior
    obs,1.0,0.2,0.2
    local_anchor,0.2,1.0,0.0
    global_prior,0.2,0.0,1.0

The header row determines the node order. The first column must repeat the same
labels in the same order.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_labelled_matrix(path: Path) -> tuple[list[str], list[list[float]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.reader(handle) if row and any(cell.strip() for cell in row)]
    if len(rows) < 2:
        raise ValueError("labelled CSV must contain a header and at least one data row")

    nodes = [cell.strip() for cell in rows[0][1:] if cell.strip()]
    if not nodes:
        raise ValueError("header row must contain node labels")

    matrix: list[list[float]] = []
    row_labels: list[str] = []
    for row in rows[1:]:
        if len(row) != len(nodes) + 1:
            raise ValueError("each data row must contain one row label plus len(nodes) values")
        row_labels.append(row[0].strip())
        matrix.append([float(cell.strip()) for cell in row[1:]])

    if row_labels != nodes:
        raise ValueError("row labels must match header labels in the same order")
    return nodes, matrix


def split_nodes(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="channel name")
    parser.add_argument("--observable", required=True, help="observable/readout node")
    parser.add_argument("--local-nodes", default="", help="comma-separated local endpoint nodes")
    parser.add_argument("--global-nodes", default="", help="comma-separated global prior/ruler nodes")
    parser.add_argument("--matrix-type", choices=["fisher", "covariance"], required=True)
    parser.add_argument("--labelled-csv", required=True, help="labelled CSV matrix path")
    parser.add_argument("--h0-obs", type=float)
    parser.add_argument("--h0-sigma", type=float)
    parser.add_argument("--output", required=True, help="output JSON path")
    args = parser.parse_args()

    nodes, matrix = read_labelled_matrix(Path(args.labelled_csv))
    payload: dict[str, object] = {
        "name": args.name,
        "nodes": nodes,
        "observable": args.observable,
        "local_nodes": split_nodes(args.local_nodes),
        "global_nodes": split_nodes(args.global_nodes),
        "matrix_type": args.matrix_type,
        "matrix": matrix,
    }
    if args.h0_obs is not None:
        payload["h0_obs"] = args.h0_obs
    if args.h0_sigma is not None:
        payload["h0_sigma"] = args.h0_sigma

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
