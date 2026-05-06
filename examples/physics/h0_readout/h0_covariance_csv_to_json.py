"""Convert a CSV Fisher/covariance matrix into the H0 readout JSON schema."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_matrix(path: Path) -> list[list[float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    matrix: list[list[float]] = []
    for row in rows:
        if not row or all(not cell.strip() for cell in row):
            continue
        matrix.append([float(cell.strip()) for cell in row])
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="channel name")
    parser.add_argument("--nodes", required=True, help="comma-separated node names in matrix order")
    parser.add_argument("--observable", required=True, help="observable/readout node")
    parser.add_argument("--local-nodes", default="", help="comma-separated local endpoint nodes")
    parser.add_argument("--global-nodes", default="", help="comma-separated global prior/ruler nodes")
    parser.add_argument("--matrix-type", choices=["fisher", "covariance"], required=True)
    parser.add_argument("--matrix-csv", required=True, help="CSV matrix path")
    parser.add_argument("--h0-obs", type=float)
    parser.add_argument("--h0-sigma", type=float)
    parser.add_argument("--output", required=True, help="output JSON path")
    args = parser.parse_args()

    nodes = [item.strip() for item in args.nodes.split(",") if item.strip()]
    local_nodes = [item.strip() for item in args.local_nodes.split(",") if item.strip()]
    global_nodes = [item.strip() for item in args.global_nodes.split(",") if item.strip()]

    payload: dict[str, object] = {
        "name": args.name,
        "nodes": nodes,
        "observable": args.observable,
        "local_nodes": local_nodes,
        "global_nodes": global_nodes,
        "matrix_type": args.matrix_type,
        "matrix": read_matrix(Path(args.matrix_csv)),
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
